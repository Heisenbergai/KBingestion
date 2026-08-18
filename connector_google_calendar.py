"""
Google Calendar connector — structured metadata ONLY (Google Workspace scope
lock, Decision 2). No embeddings, no LLM filtration, no knowledge_notes row,
no attachment/file downloads. Calendar exists to give KNOVA company/meeting
CONTEXT (who's meeting whom, when, about what conference), not to become a
second document repository.

Flow:
  poll_connection() — called on a schedule by worker.py, for every connection
  with "calendar" in config.enabled_surfaces (see connector_google.
  get_active_connection). Lists recent+upcoming events via calendar.events.list,
  upserts each into calendar_events (this project's vector-DB tables, plain
  SUPABASE_SERVICE_KEY write — no app-DB RPC needed, since this is not a
  knowledge_items-adjacent write), then attempts one immutable
  calendar_event_snapshot per event (Phase 5E) from that SAME normalized row
  — never a second, independently-normalized copy of the same state.

Dedup/change-detection: unique on (connection_id, external_event_id), and an
event whose Google-side `updated` timestamp hasn't advanced since the last
poll is skipped for the calendar_events UPSERT specifically — same
"re-process only if source changed" shape Drive's now-neutralized bulk poll
used, applied here to metadata rows instead of document chunks. This
optimization is unchanged by Phase 5E. The snapshot attempt, however, runs
regardless of that skip — calendar_evidence.maybe_create_snapshot() has its
own, separate, fingerprint-based idempotency check, and running it every poll
(not just on a calendar_events change) is what makes a snapshot write that
failed on one poll safely retryable on the next, even if nothing about the
event changed in between.

Snapshot failures are caught and counted, never raised — the calendar_events
write already succeeded by the time a snapshot is attempted, and one bad
snapshot write must not fail the whole poll (same per-item-doesn't-cost-the-
others contract this codebase already uses elsewhere, e.g.
structured_persistence.persist_extracted_primitives). See snapshots_failed
in the returned stats dict, which worker.py's _run_surface_poll already
persists verbatim into sync_runs.stats (jsonb) — no schema change needed.

Cancelled events (Phase 5E): calendar_events.deleted_at is set (once, not
re-bumped on repeat sightings) when Google reports status == "cancelled" for
an event this connector already has a row for. No new snapshot is created
for a cancellation itself — deletion is a lifecycle transition, not new
observable content (matches compute_state_fingerprint's own exclusion of
deleted_at from the fingerprint input). NAMED, NOT FIXED HERE: Google's API
only returns cancelled events at all when showDeleted=true is passed to
events.list, which this connector does not currently set — so in practice
this handling primarily protects against a cancelled INSTANCE of a
recurring, singleEvents-expanded series (which Google can include even
without showDeleted). Adding showDeleted=true to actually see cancelled
one-off events is a separate, live-external-API-behavior decision, not made
in this pass — it changes what a real network call fetches from a
production integration this environment has no live credentials to verify
against, and is intentionally left for a dedicated future review.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx

import brain_connectors as bc
import connector_google as google
import calendar_evidence

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"

# How far back/forward each poll looks. Wide enough to catch anything missed
# by a prior failed run, narrow enough to stay a cheap call every pass.
LOOKBACK = timedelta(days=7)
LOOKAHEAD = timedelta(days=30)


def _calendar_get(path: str, token: str, params: dict = None) -> dict:
    res = httpx.get(f"{CALENDAR_API_BASE}/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or {}, timeout=30)
    if res.status_code != 200:
        raise Exception(f"Google Calendar API error ({res.status_code}): {res.text[:300]}")
    return res.json()


def _list_events(token: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    events, page_token = [], None
    while True:
        params = {
            "timeMin": (now - LOOKBACK).isoformat(),
            "timeMax": (now + LOOKAHEAD).isoformat(),
            "singleEvents": "true",   # expands recurring events into instances
            "orderBy": "startTime",
            "maxResults": 250,
        }
        if page_token:
            params["pageToken"] = page_token
        data = _calendar_get("calendars/primary/events", token, params)
        events.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return events


def _event_start_end(event: dict) -> tuple[Optional[str], Optional[str]]:
    """Handles both timed events (dateTime) and all-day events (date-only)."""
    start = event.get("start", {})
    end = event.get("end", {})
    return (start.get("dateTime") or start.get("date"),
            end.get("dateTime") or end.get("date"))


def poll_connection(connection_id: str, workspace_id: str) -> dict:
    """The main job — see module docstring. Called by worker.py per active,
    calendar-enabled connection."""
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise Exception("Connection not found.")
    conn = conn[0]
    if conn.get("provider") != "google_drive" or conn.get("status") != "active":
        raise Exception("Connection is not an active Google connection.")
    if "calendar" not in (conn.get("config") or {}).get("enabled_surfaces", []):
        raise Exception("Calendar is not enabled for this connection.")

    token = google._valid_access_token(conn)
    if not token:
        raise Exception("Google connection needs to be reconnected.")

    events = _list_events(token)
    processed = skipped = 0
    snapshots_created = snapshots_skipped = snapshots_failed = 0

    for event in events:
        external_id = event.get("id")
        if not external_id:
            skipped += 1
            continue

        if event.get("status") == "cancelled":
            # Current-state layer reflects the deletion; the snapshot layer
            # is untouched (no new snapshot for a cancellation itself -- see
            # module docstring). .is_("deleted_at","null") makes this
            # idempotent: repeat sightings of the same cancelled status
            # never re-bump the timestamp.
            bc.supabase.table("calendar_events").update({
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            }).eq("connection_id", connection_id).eq("external_event_id", external_id) \
              .is_("deleted_at", "null").execute()
            skipped += 1
            continue

        existing = bc.supabase.table("calendar_events").select("id, updated_at_source") \
            .eq("connection_id", connection_id).eq("external_event_id", external_id).execute().data
        already = existing[0] if existing else None
        source_updated = event.get("updated")

        start, end = _event_start_end(event)
        organizer = (event.get("organizer") or {}).get("email")
        attendees = [
            {"email": a.get("email"), "response_status": a.get("responseStatus")}
            for a in event.get("attendees", [])
        ]
        conference_id = ((event.get("conferenceData") or {}).get("conferenceId"))
        meeting_url = event.get("hangoutLink") or (
            (event.get("conferenceData") or {}).get("entryPoints", [{}])[0].get("uri")
            if event.get("conferenceData") else None
        )

        # Built once, used for BOTH the calendar_events upsert below (when
        # the state actually changed) AND the snapshot attempt (always) --
        # never two independently-normalized copies of the same state.
        row = {
            "workspace_id": workspace_id, "connection_id": connection_id,
            "external_event_id": external_id, "title": event.get("summary"),
            "start_time": start, "end_time": end, "organizer": organizer,
            "attendees": attendees, "recurrence_rule": ",".join(event.get("recurrence", [])) or None,
            "meeting_url": meeting_url, "conference_id": conference_id,
            "updated_at_source": source_updated,
        }

        if already and already.get("updated_at_source") == source_updated:
            skipped += 1  # unchanged since last poll -- current-state upsert skipped, as before
        else:
            bc.supabase.table("calendar_events").upsert(
                row, on_conflict="connection_id,external_event_id"
            ).execute()
            processed += 1

        # Snapshot attempt always runs, whether or not the upsert above ran
        # -- maybe_create_snapshot()'s own fingerprint check makes this a
        # safe no-op when nothing observable changed, and is what makes a
        # snapshot write that failed on a prior poll retryable here even if
        # updated_at_source hasn't advanced again since.
        try:
            snap_id = calendar_evidence.snapshot_from_calendar_event_row(row)
            if snap_id:
                snapshots_created += 1
            else:
                snapshots_skipped += 1
        except Exception as e:
            snapshots_failed += 1
            print(f"[google_calendar] snapshot write failed for external_event_id={external_id} "
                  f"(non-fatal -- calendar_events state is correct and unaffected; "
                  f"next poll will retry the snapshot): {e}")

    return {
        "events_seen": len(events), "processed": processed, "skipped": skipped,
        "snapshots_created": snapshots_created, "snapshots_skipped": snapshots_skipped,
        "snapshots_failed": snapshots_failed,
    }
