"""
Zoom connector — meeting transcripts (Phase 3).

Shape: webhook-triggered like Slack, but each event is ONE COMPLETE meeting
transcript, not a stream of small messages needing batching — so this calls
brain_connectors.distill_meeting_transcript() directly (one transcript in,
one note out) rather than batch_conversations()/classify_batch(). The note
is tier 2 (curated meeting record — more trusted than raw chat, less than an
official document), via the existing create_note_and_embed() pipeline.

Flow:
  1. POST /integrations/oauth-url → mint the consent URL (popup). GET /zoom/install
     does the same as a redirect, for non-browser callers.
  2. GET  /zoom/oauth/callback → exchange code, store tokens, run an initial
     best-effort backfill of recent recordings
  3. POST /zoom/events         → Zoom webhook: URL validation handshake, then
     recording.completed → fetch transcript → distill → note. ONE shared URL
     for every customer's Zoom app, same "resolve the right customer's secret
     from the event's account_id BEFORE verifying" pattern as Slack's
     /slack/events (see connector_slack.py for the fuller explanation of why
     that matters under single-tenant credentials).

CREDENTIAL MODEL (single-tenant — see 09_company_brain_roadmap.md): each
CUSTOMER creates their own OAuth app at marketplace.zoom.us and pastes its
client_id / client_secret / a webhook "Secret Token" into the Integrations
panel. One-time setup per customer:
  - marketplace.zoom.us/develop/create → "OAuth" app type (NOT Server-to-
    Server — that type has no user consent step, but this codebase's OAuth
    plumbing already assumes the redirect+consent shape shared with Slack/
    Google, so a regular OAuth app is the better fit here)
  - Scopes: this needs an ADMIN of the Zoom account to authorize it, with
    account-wide recording read access. ⚠️ VERIFY the exact granular scope
    names in your Zoom app's scope picker when setting this up for real —
    Zoom has migrated scope naming more than once and this was written
    without a live Zoom account to confirm against. Look for scopes covering
    "View all user recordings" (admin-level cloud recording read) and
    "View all users" (to resolve account_id). As of this writing the
    documented names are approximately: cloud_recording:read:list_recording_files:admin,
    cloud_recording:read:list_user_recordings:admin, user:read:list_users:admin
  - Redirect URL for OAuth: https://kbingestion-production.up.railway.app/zoom/oauth/callback
  - Add a webhook subscription (Feature → Access → Event Subscriptions),
    Event notification endpoint URL: https://kbingestion-production.up.railway.app/zoom/events
    Subscribe to: "All Recordings have completed" (recording.completed)
  - The app's "Secret Token" (shown next to the event subscription) is what
    goes in the panel's "Signing Secret" field — verifies the webhook, same
    role as Slack's signing secret.
"""
import os
import re
import json
import time
import hmac
import hashlib
import threading
import httpx
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from typing import Optional
from dotenv import load_dotenv

from auth import AuthContext, current_user
import brain_connectors as bc

load_dotenv()

router = APIRouter()

RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
REDIRECT_URI = f"{RAILWAY_BASE}/zoom/oauth/callback"

# See the module docstring's "⚠️ VERIFY" note — these are Zoom's documented
# granular scope names at time of writing, not verified against a live app.
ZOOM_SCOPES = ("cloud_recording:read:list_recording_files:admin "
              "cloud_recording:read:list_user_recordings:admin "
              "user:read:list_users:admin")
BACKFILL_DAYS = 30


def _zoom_credentials(workspace_id: str) -> tuple[str, str]:
    creds = bc.get_provider_credentials(workspace_id, "zoom")
    if not creds:
        raise HTTPException(
            status_code=400,
            detail="This workspace hasn't set up its Zoom app credentials yet. "
                   "Go to Integrations → Zoom → Set up to add them.",
        )
    return creds["client_id"], creds["client_secret"]


def build_install_url(workspace_id: str, user_id: str, connection_id: str) -> str:
    """Mirrors connector_slack.build_install_url — see its docstring. Sealed
    to one specific connection_id (multi-integration management, 2026-08-16)."""
    client_id, _ = _zoom_credentials(workspace_id)
    state = bc.encode_oauth_state(workspace_id, user_id, extra={"connection_id": connection_id})
    return (
        "https://zoom.us/oauth/authorize"
        f"?response_type=code&client_id={client_id}&redirect_uri={REDIRECT_URI}"
        f"&state={state}"
    )


@router.get("/zoom/install")
async def zoom_install(workspace_id: str, connection_id: str, user_id: str = "",
                       auth: AuthContext = Depends(current_user)):
    """Redirect variant, for non-browser callers — see connector_google.google_install."""
    auth.assert_workspace(workspace_id)
    if not bc.get_connection_for_workspace(connection_id, workspace_id):
        raise HTTPException(status_code=404, detail="Connection not found.")
    return RedirectResponse(build_install_url(workspace_id, user_id, connection_id))


def _token_request(workspace_id: str, data: dict) -> dict:
    """
    Zoom requires HTTP Basic Auth (client_id:client_secret) on the token
    endpoint for BOTH the initial exchange and refresh — unlike Slack/Google,
    which take client_id/secret as regular form fields. Easy to get wrong
    since it looks so similar to the other two connectors; called out here
    deliberately.
    """
    client_id, client_secret = _zoom_credentials(workspace_id)
    res = httpx.post("https://zoom.us/oauth/token", data=data,
                     auth=(client_id, client_secret), timeout=30)
    return res.json()


@router.get("/zoom/oauth/callback")
async def zoom_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code for tokens and finalizes the ONE pending
    connection named in state (see connector_slack.slack_callback's
    docstring — same UPDATE-by-connection_id pattern, not an upsert-by-
    team-id guess), then kicks off a best-effort initial backfill."""
    from integrations import oauth_complete_html
    if error:
        return oauth_complete_html("zoom", "error")
    st = bc.decode_oauth_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")
    connection_id = st.get("connection_id")
    if not connection_id:
        print(f"[zoom] oauth callback missing connection_id in state for workspace={workspace_id}")
        return oauth_complete_html("zoom", "error")
    if not bc.get_connection_for_workspace(connection_id, workspace_id):
        print(f"[zoom] oauth callback: connection {connection_id} not found/owned by workspace {workspace_id}")
        return oauth_complete_html("zoom", "error")

    data = _token_request(workspace_id, {
        "grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI,
    })
    if "access_token" not in data:
        print(f"[zoom] oauth exchange failed: {data}")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": f"OAuth exchange failed: {data.get('error', 'unknown')}"}
        ).eq("id", connection_id).execute()
        return oauth_complete_html("zoom", "error")

    userinfo = httpx.get("https://api.zoom.us/v2/users/me",
                         headers={"Authorization": f"Bearer {data['access_token']}"}, timeout=15).json()
    account_id = userinfo.get("account_id", "")
    account_email = userinfo.get("email", "")

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()

    update_row = {
        "external_team_id":   account_id,
        "external_team_name": account_email or account_id,
        "access_token_enc":   bc.encrypt_secret(data["access_token"]),
        "refresh_token_enc":  bc.encrypt_secret(data["refresh_token"]) if data.get("refresh_token") else None,
        "token_expires_at":   expires_at,
        "scopes":             data.get("scope", ZOOM_SCOPES),
        "status":             "active",
        "connected_by":       user_id,
    }
    try:
        bc.supabase.table("connections").update(update_row).eq("id", connection_id).execute()
    except Exception as e:
        # This exact Zoom account is already connected via a DIFFERENT
        # connection row in this workspace (unique index on workspace_id+
        # provider+external_team_id) -- fail visibly, leave the row
        # disconnectable/retryable.
        print(f"[zoom] failed to finalize connection {connection_id}: {e}")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": f"Could not finish connecting: {e}"}
        ).eq("id", connection_id).execute()
        return oauth_complete_html("zoom", "error")

    # Best-effort initial backfill — Zoom is otherwise entirely webhook-driven
    # (recording.completed), so nothing depends on this succeeding. Wrapped
    # so an uncertain endpoint/scope assumption (see module docstring) can't
    # break the connection itself.
    threading.Thread(target=_safe_backfill, args=(connection_id,), daemon=True).start()

    return oauth_complete_html("zoom", "connected")


def refresh_access_token(conn: dict) -> Optional[str]:
    """Real refresh, same shape as connector_google.refresh_access_token."""
    if not conn.get("refresh_token_enc"):
        print(f"[zoom] connection {conn['id']} has no refresh_token — cannot refresh, marking error")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": "No refresh token stored. Reconnect Zoom."}
        ).eq("id", conn["id"]).execute()
        return None

    refresh_token = bc.decrypt_secret(conn["refresh_token_enc"])
    data = _token_request(conn["workspace_id"], {
        "grant_type": "refresh_token", "refresh_token": refresh_token,
    })
    if "access_token" not in data:
        print(f"[zoom] token refresh failed for connection {conn['id']}: {data}")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": f"Token refresh failed: {data.get('error', 'unknown error')}"}
        ).eq("id", conn["id"]).execute()
        return None

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()
    update = {"access_token_enc": bc.encrypt_secret(data["access_token"]), "token_expires_at": expires_at}
    if data.get("refresh_token"):  # Zoom rotates refresh tokens on each use
        update["refresh_token_enc"] = bc.encrypt_secret(data["refresh_token"])
    bc.supabase.table("connections").update(update).eq("id", conn["id"]).execute()
    return data["access_token"]


def _valid_access_token(conn: dict) -> Optional[str]:
    expires_at = conn.get("token_expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp > datetime.now(timezone.utc) + timedelta(minutes=2):
                return bc.decrypt_secret(conn["access_token_enc"])
        except ValueError:
            pass
    return refresh_access_token(conn)


# ── VTT transcript parsing ───────────────────────────────────────────────────────
# Generic WebVTT parsing, not Zoom-specific — reusable by any future
# meeting-recording connector that serves standard WebVTT transcripts.

_VTT_CUE_NUM = re.compile(r"^\d+$")
_VTT_TIMESTAMP = re.compile(r"-->")
_VTT_SPEAKER = re.compile(r"^<v\s+([^>]+)>(.*)$")


def parse_vtt_transcript(vtt_text: str) -> str:
    """
    Turns a WebVTT transcript into readable "Speaker: text" lines — the same
    shape brain_connectors.distill_meeting_transcript() expects. Strips the
    WEBVTT header, cue numbers, and timestamp lines; unwraps <v Speaker>text
    voice tags where present, otherwise emits the line as-is.
    """
    lines = []
    for raw_line in vtt_text.splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or _VTT_CUE_NUM.match(line) or _VTT_TIMESTAMP.search(line):
            continue
        m = _VTT_SPEAKER.match(line)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            if text:
                lines.append(f"{speaker}: {text}")
        elif line:
            lines.append(line)
    return "\n".join(lines)


# ── Recording processing (shared by webhook + backfill) ─────────────────────────

def _process_recording(workspace_id: str, connection_id: str, access_token: str,
                       meeting_uuid: str, topic: str, recording_files: list[dict],
                       occurred_at: Optional[str] = None) -> bool:
    """
    Finds the transcript file among a recording's files, fetches + parses it,
    distills a note, and embeds it. Returns True if a note was created.
    Idempotent via ingest_items (unique on connection_id + external_id=meeting_uuid).
    """
    existing = bc.supabase.table("ingest_items").select("id, status") \
        .eq("connection_id", connection_id).eq("external_id", meeting_uuid).execute().data
    if existing and existing[0]["status"] in ("noted", "discarded"):
        return False  # already processed, whichever way it went

    transcript_file = next((f for f in recording_files if f.get("file_type") == "TRANSCRIPT"), None)
    if not transcript_file:
        bc.supabase.table("ingest_items").upsert({
            "workspace_id": workspace_id, "connection_id": connection_id, "provider": "zoom",
            "external_id": meeting_uuid, "kind": "zoom_meeting", "status": "discarded",
            "raw": {"topic": topic, "reason": "no transcript file"},
        }, on_conflict="connection_id,external_id").execute()
        return False

    res = httpx.get(transcript_file["download_url"],
                    headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    res.raise_for_status()
    transcript = parse_vtt_transcript(res.text)

    note = bc.distill_meeting_transcript(transcript, topic, workspace_id=workspace_id)
    status = "discarded"
    note_id = None
    if note:
        note_id = bc.create_note_and_embed(
            workspace_id, connection_id, "zoom", note,
            source_type="meeting", source_tier=2, occurred_at=occurred_at,
        )
        status = "noted"

    bc.supabase.table("ingest_items").upsert({
        "workspace_id": workspace_id, "connection_id": connection_id, "provider": "zoom",
        "external_id": meeting_uuid, "kind": "zoom_meeting", "status": status, "note_id": note_id,
        "raw": {"topic": topic},
    }, on_conflict="connection_id,external_id").execute()
    return note is not None


def _safe_backfill(connection_id: str) -> None:
    """
    Best-effort: lists the connecting user's own recent recordings and
    processes any with a transcript. Wrapped entirely in try/except because
    the exact account-wide recordings-list endpoint was not verified against
    a live Zoom account (see module docstring) — if this assumption is wrong,
    the connector's PRIMARY path (the recording.completed webhook) is
    completely unaffected. This only covers the connecting admin's own past
    meetings; the webhook covers every meeting in the account going forward.
    """
    try:
        conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data[0]
        token = _valid_access_token(conn)
        if not token:
            return
        since = (datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%d")
        res = httpx.get("https://api.zoom.us/v2/users/me/recordings",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"from": since, "page_size": 100}, timeout=30)
        if res.status_code != 200:
            print(f"[zoom] backfill endpoint returned {res.status_code}, skipping (webhook path unaffected)")
            return
        meetings = res.json().get("meetings", [])
        for m in meetings:
            try:
                _process_recording(
                    conn["workspace_id"], connection_id, token,
                    meeting_uuid=m["uuid"], topic=m.get("topic", "Zoom Meeting"),
                    recording_files=m.get("recording_files", []),
                    occurred_at=m.get("start_time"),
                )
            except Exception as e:
                print(f"[zoom] backfill: failed processing meeting {m.get('uuid')}: {e}")
    except Exception as e:
        print(f"[zoom] backfill failed entirely (non-fatal, webhook path unaffected): {e}")


# ── Webhook ──────────────────────────────────────────────────────────────────────

def _verify_zoom_signature(request: Request, body: bytes, secret_token: str) -> bool:
    """Same v0:{timestamp}:{body} HMAC-SHA256 scheme as Slack, different header names."""
    if not secret_token:
        return True  # not configured — allow (dev); the panel requires one to be set for real use
    ts = request.headers.get("x-zm-request-timestamp", "")
    sig = request.headers.get("x-zm-signature", "")
    if not ts:
        return False
    message = f"v0:{ts}:{body.decode()}"
    mine = "v0=" + hmac.new(secret_token.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mine, sig)


@router.post("/zoom/events")
async def zoom_events(request: Request):
    """
    Zoom webhook. Handles the one-time URL validation handshake (Zoom sends a
    plainToken and expects it back HMAC-signed with the Secret Token — a
    different handshake shape than Slack's simple challenge-echo) and
    recording.completed events.

    ONE shared URL for every customer's Zoom app; the Secret Token is
    resolved from the event's account_id BEFORE verifying, same reasoning as
    connector_slack.slack_events() — verifying against a single global secret
    would fail every customer's events except possibly the first configured.
    """
    body = await request.body()
    payload = json.loads(body or "{}")

    if payload.get("event") == "endpoint.url_validation":
        plain_token = payload.get("payload", {}).get("plainToken", "")
        # This handshake arrives before any workspace is known (Zoom sends it
        # once, at subscription setup, to whichever secret is configured in
        # THAT app) — verified per-customer the same way as a real event,
        # but there's no account_id yet to resolve it from, so this uses
        # whichever workspace most recently registered zoom credentials as a
        # best-effort match. If it's wrong, re-saving credentials for the
        # correct workspace and re-triggering validation from the Zoom
        # dashboard fixes it — a one-time setup step, not a runtime concern.
        secret_token = ""
        recent = bc.supabase.table("provider_credentials").select("workspace_id") \
            .eq("provider", "zoom").order("updated_at", desc=True).limit(1).execute().data
        if recent:
            creds = bc.get_provider_credentials(recent[0]["workspace_id"], "zoom")
            secret_token = creds["webhook_secret"] if creds else ""
        encrypted = hmac.new(secret_token.encode(), plain_token.encode(), hashlib.sha256).hexdigest()
        return JSONResponse({"plainToken": plain_token, "encryptedToken": encrypted})

    account_id = payload.get("payload", {}).get("account_id", "")
    creds = bc.get_provider_credentials_by_external_team("zoom", account_id) if account_id else None
    secret_token = creds["webhook_secret"] if creds and creds.get("webhook_secret") else ""

    if not _verify_zoom_signature(request, body, secret_token):
        raise HTTPException(status_code=401, detail="Bad Zoom signature.")

    if payload.get("event") == "recording.completed":
        obj = payload.get("payload", {}).get("object", {})
        candidates = bc.supabase.table("connections").select("*") \
            .eq("provider", "zoom").eq("external_team_id", account_id) \
            .eq("status", "active").execute().data or []
        # Multi-integration management (2026-08-16): more than one Zoom
        # connection can legitimately share the same account_id (the same
        # Zoom account connected into two different KNOVA workspaces, or
        # twice into one for different purposes -- unique index is
        # workspace_id+provider+external_team_id, so within ONE workspace
        # this can't collide, but across workspaces it genuinely can).
        # Zoom's webhook payload carries no second identifier the way
        # Slack's api_app_id sometimes helps disambiguate -- so ANY
        # ambiguity here fails closed rather than guessing which
        # workspace's brain this recording belongs to. Same contract as
        # connector_slack._resolve_slack_connection.
        conn = None
        if len(candidates) == 1:
            conn = candidates[0]
        elif len(candidates) > 1:
            print(f"[zoom] AMBIGUOUS recording.completed: account_id={account_id} matches "
                  f"{len(candidates)} active connections — refusing to guess, discarding event.")
        if conn:
            token = _valid_access_token(conn)
            if token:
                try:
                    _process_recording(
                        conn["workspace_id"], conn["id"], token,
                        meeting_uuid=obj.get("uuid", ""), topic=obj.get("topic", "Zoom Meeting"),
                        recording_files=obj.get("recording_files", []),
                        occurred_at=obj.get("start_time"),
                    )
                except Exception as e:
                    print(f"[zoom] recording.completed processing failed: {e}")

    return JSONResponse({"ok": True})
