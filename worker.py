"""
Scheduled maintenance for the connector layer. Run as a Railway CRON JOB
(a scheduled one-off command in the SAME Railway project, not a second
always-on service) -- see the setup steps at the bottom of this file.

WHY THIS EXISTS
---------------
Before this, every background job in this service was a `threading.Thread`
inside the web process, tracked in an in-memory dict (SYNC_JOBS,
INGEST_JOBS). That dies on every Railway redeploy, can't be inspected across
processes, and — the actual gap this file closes — NOTHING ran on a timer.
Slack's Events API webhook has always captured live messages into
ingest_items in real time (see connector_slack.slack_events), but turning
those captured messages into actual knowledge (filtration) only ever ran
when someone clicked "Sync" in the Integrations panel. So "continuous
ingestion" was true up to the capture step and false after it. This script
is what makes it true end to end: run it on a schedule and captured
messages get filtered into notes without anyone clicking anything.

Each run is one process that does its work and exits — no loop, no
`while True`, no persistent state beyond what it writes to `sync_runs`
(so a run's outcome is inspectable in the DB even with no dashboard for it
yet, and a crashed run doesn't corrupt anything the next run depends on).

WHAT IT DOES, IN ORDER
-----------------------
1. run_pending_filtration() — for every active connection with pending
   ingest_items, runs the existing filtration pipeline (brain_connectors.
   run_filtration). Slack-shaped (chat → keep/discard → notes). Google Chat
   connections reach this too — see connector_google_chat.py, which calls
   run_filtration() itself rather than going through this generic path,
   since it also needs to normalize+store items first.
2. run_google_calendar_polling() / run_google_meet_polling() /
   run_google_chat_polling() — the three Google Workspace surfaces that DO
   run on a schedule (see 09_company_brain_roadmap.md's Google Workspace
   scope-lock decision). Google Drive does NOT get a scheduled poll step —
   Drive is reference-only now (connector_google.resolve_drive_reference),
   triggered reactively by Meet/Chat note creation, never on a timer. An
   earlier version of this file DID poll Drive folders on a schedule; that
   step was removed, not just disabled, when bulk Drive ingestion was
   neutralized per the scope lock.
3. refresh_expiring_tokens() — connections whose token_expires_at is
   approaching get a REAL refresh for providers that implement one
   (connector_google.refresh_access_token, the first real implementation —
   Slack bot tokens never expire so never reach this path). A provider with
   no refresh function yet gets a warning log instead of a fake refresh.
4. flag_expiring_webhooks() — same log-only treatment for webhook_subscriptions.
   Not used by Slack (Events API has no expiry) or Google (polling, not
   push, for every surface); will matter for Microsoft Graph (~3 days) if
   that connector is ever built.

Every step is wrapped so one bad connection can't stop the run; every
outcome is written to sync_runs for later inspection.
"""
import sys
import traceback
from datetime import datetime, timezone, timedelta

import brain_connectors as bc
import connector_google
import connector_google_calendar
import connector_google_meet
import connector_google_chat
import connector_slack
import connector_zoom

# How soon is "expiring soon" for the flag-only checks below.
TOKEN_EXPIRY_WARNING_WINDOW = timedelta(hours=2)
WEBHOOK_EXPIRY_WARNING_WINDOW = timedelta(hours=24)


def _start_run(kind: str, connection_id: str = None, workspace_id: str = None) -> str:
    row = bc.supabase.table("sync_runs").insert({
        "kind": kind, "connection_id": connection_id, "workspace_id": workspace_id,
        "status": "running",
    }).execute().data
    return row[0]["id"]


def _finish_run(run_id: str, status: str, stats: dict = None, error: str = None) -> None:
    bc.supabase.table("sync_runs").update({
        "status": status,
        "stats": stats or {},
        "error": error,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", run_id).execute()


def run_pending_filtration() -> dict:
    """
    The main job. Every active connection with pending ingest_items gets a
    filtration pass, same pipeline POST /connectors/sync already triggers
    on demand — this is what makes it run without a human clicking it.

    Excludes provider='google_drive' connections: those now cover Calendar/
    Meet/Chat/Drive-reference (see connections.config.enabled_surfaces), and
    Chat is the only one of the four that produces 'pending' ingest_items --
    it fully owns its own filtration cycle inside
    connector_google_chat.poll_connection() (which knows to call
    run_filtration with provider="google_chat", not the connection's own
    provider column). Letting THIS generic loop also pick up a google_drive
    connection would call run_filtration(..., "google_drive", ...) on
    Chat-sourced items, mislabelling every resulting note's provider.
    """
    # access_token_enc is needed here (not just id/workspace_id/provider) so
    # a Slack connection can resolve real chat.getPermalink calls for its
    # kept messages -- see connector_slack.build_permalink_resolver.
    connections = bc.supabase.table("connections").select("id, workspace_id, provider, access_token_enc") \
        .eq("status", "active").neq("provider", "google_drive").execute().data or []

    processed, failed = 0, 0
    for conn in connections:
        pending_count = bc.supabase.table("ingest_items").select("id", count="exact") \
            .eq("connection_id", conn["id"]).eq("status", "pending").limit(1).execute().count or 0
        if not pending_count:
            continue

        run_id = _start_run("filtration", conn["id"], conn["workspace_id"])
        try:
            resolver = connector_slack.build_permalink_resolver(conn)
            result = bc.run_filtration(conn["workspace_id"], conn["id"], conn["provider"],
                                       resolve_permalink=resolver)
            _finish_run(run_id, "completed", stats=result)
            processed += 1
            print(f"[worker] filtration OK connection={conn['id']} provider={conn['provider']} {result}")
        except Exception as e:
            _finish_run(run_id, "failed", error=str(e))
            failed += 1
            print(f"[worker] filtration FAILED connection={conn['id']}: {e}")
            print(traceback.format_exc())

    return {"connections_checked": len(connections), "processed": processed, "failed": failed}


def _connections_with_surface(surface: str) -> list[dict]:
    """Every active google_drive-provider connection with `surface` in its
    config.enabled_surfaces -- the shared selection logic all three Google
    Workspace pollers below use, mirroring connector_google.get_active_connection
    but returning the full list rather than one connection."""
    connections = bc.supabase.table("connections").select("id, workspace_id, config") \
        .eq("provider", "google_drive").eq("status", "active").execute().data or []
    return [c for c in connections if surface in (c.get("config") or {}).get("enabled_surfaces", [])]


def _run_surface_poll(kind: str, surface: str, poll_fn) -> dict:
    """Shared runner for the three Google Workspace poll steps below --
    same _start_run/_finish_run/per-connection-isolation shape as every
    other poller in this file."""
    connections = _connections_with_surface(surface)
    processed = failed = 0
    for conn in connections:
        run_id = _start_run(kind, conn["id"], conn["workspace_id"])
        try:
            result = poll_fn(conn["id"], conn["workspace_id"])
            _finish_run(run_id, "completed", stats=result)
            processed += 1
            print(f"[worker] {kind} OK connection={conn['id']} {result}")
        except Exception as e:
            _finish_run(run_id, "failed", error=str(e))
            failed += 1
            print(f"[worker] {kind} FAILED connection={conn['id']}: {e}")
            print(traceback.format_exc())

    return {"connections_checked": len(connections), "processed": processed, "failed": failed}


def run_google_calendar_polling() -> dict:
    """Every active Calendar-enabled connection gets its recent/upcoming
    events synced as structured metadata. See connector_google_calendar.py."""
    return _run_surface_poll("calendar_poll", "calendar", connector_google_calendar.poll_connection)


def run_google_meet_polling() -> dict:
    """Every active Meet-enabled connection gets its recent conferences'
    transcripts captured as durable knowledge. See connector_google_meet.py."""
    return _run_surface_poll("meet_poll", "meet", connector_google_meet.poll_connection)


def run_google_chat_polling() -> dict:
    """Every active Chat-enabled connection gets its recent messages fetched,
    normalized, and run through the existing Slack-shaped filtration
    pipeline. See connector_google_chat.py."""
    return _run_surface_poll("chat_poll", "chat", connector_google_chat.poll_connection)


# Provider -> real refresh function, for providers that have one implemented.
# A provider absent from this dict just gets the warning log below instead of
# a refresh attempt — that is the honest state for a provider with no tested
# refresh flow yet (currently: none besides google_drive; Slack bot tokens
# never expire so never appear here at all).
_TOKEN_REFRESHERS = {
    "google_drive": connector_google.refresh_access_token,
    "zoom": connector_zoom.refresh_access_token,
}


def refresh_expiring_tokens() -> dict:
    """
    Refreshes connections whose access token expires soon, for providers
    with a real refresh implementation; logs a warning for the rest. See
    _TOKEN_REFRESHERS above and connector_google.refresh_access_token for
    the first real one.
    """
    cutoff = (datetime.now(timezone.utc) + TOKEN_EXPIRY_WARNING_WINDOW).isoformat()
    expiring = bc.supabase.table("connections").select("*") \
        .eq("status", "active").not_.is_("token_expires_at", "null") \
        .lte("token_expires_at", cutoff).execute().data or []

    refreshed = failed = warned = 0
    for conn in expiring:
        refresher = _TOKEN_REFRESHERS.get(conn["provider"])
        if not refresher:
            warned += 1
            print(f"[worker] TOKEN EXPIRING SOON: connection={conn['id']} provider={conn['provider']} "
                  f"workspace={conn['workspace_id']} expires_at={conn['token_expires_at']} "
                  f"-- no refresh implemented for '{conn['provider']}' yet, connection will start failing")
            continue
        try:
            new_token = refresher(conn)
            if new_token:
                refreshed += 1
                print(f"[worker] token refreshed OK connection={conn['id']} provider={conn['provider']}")
            else:
                failed += 1  # refresher already marked the connection 'error' and logged why
        except Exception as e:
            failed += 1
            print(f"[worker] token refresh threw for connection={conn['id']}: {e}")
            print(traceback.format_exc())

    return {"expiring_soon": len(expiring), "refreshed": refreshed, "failed": failed, "no_refresher": warned}


def flag_expiring_webhooks() -> dict:
    """Same treatment as flag_expiring_tokens, for webhook_subscriptions."""
    cutoff = (datetime.now(timezone.utc) + WEBHOOK_EXPIRY_WARNING_WINDOW).isoformat()
    expiring = bc.supabase.table("webhook_subscriptions").select("id, connection_id, provider, resource, expires_at") \
        .lte("expires_at", cutoff).execute().data or []

    for sub in expiring:
        print(f"[worker] WEBHOOK SUBSCRIPTION EXPIRING SOON: {sub['provider']} resource={sub['resource']} "
              f"connection={sub['connection_id']} expires_at={sub['expires_at']} "
              f"-- no renewal implemented for '{sub['provider']}' yet")

    return {"expiring_soon": len(expiring)}


def main() -> int:
    run_id = _start_run("filtration")  # umbrella run for the whole pass
    overall_ok = True
    summary = {}

    for name, fn in (
        ("filtration", run_pending_filtration),
        ("calendar_poll", run_google_calendar_polling),
        ("meet_poll", run_google_meet_polling),
        ("chat_poll", run_google_chat_polling),
        ("token_refresh", refresh_expiring_tokens),
        ("webhook_check", flag_expiring_webhooks),
    ):
        try:
            summary[name] = fn()
        except Exception as e:
            overall_ok = False
            summary[name] = {"error": str(e)}
            print(f"[worker] step '{name}' FAILED: {e}")
            print(traceback.format_exc())

    _finish_run(run_id, "completed" if overall_ok else "failed", stats=summary)
    print(f"[worker] run complete: {summary}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())


# ── Railway Cron Job setup (Tanmay, one-time) ───────────────────────────────────
# This runs in the SAME Railway project as the main API service, as an
# additional service of type "Cron Job" — not a second always-on process.
#
# 1. Railway dashboard -> your project -> "+ New" -> "Cron Job"
# 2. Source: same GitHub repo (Heisenbergai/KBingestion), same branch (main)
# 3. Schedule: every 10 minutes to start -- "*/10 * * * *"
#    (tune later; nothing here is expensive at pilot scale)
# 4. Start command: python worker.py
# 5. Environment: it needs the SAME env vars as the main service --
#    SUPABASE_URL, SUPABASE_SERVICE_KEY, CONNECTOR_ENCRYPTION_KEY at minimum.
#    Railway lets a Cron Job "reference" the main service's variables instead
#    of duplicating them -- use that rather than copy-pasting secrets twice.
# 6. First run: trigger manually from the Railway dashboard once, then check
#    the `sync_runs` table (SQL, or ask Claude) to confirm a row landed with
#    status='completed'.
