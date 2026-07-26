"""
Webex connector — meeting transcripts (Phase 3, alongside connector_zoom.py).

Same content shape as Zoom: a webhook fires once per completed recording, the
transcript is fetched and distilled into ONE tier-2 note via
brain_connectors.distill_meeting_transcript() + create_note_and_embed() — see
connector_zoom.py's module docstring for why that's a different pipeline
shape than Slack's message-batching.

ONE THING GENUINELY DIFFERENT FROM SLACK/ZOOM/GOOGLE: Webex has no dashboard
field where the customer pastes a fixed "signing secret" for a shared webhook
URL. A Webex webhook is a resource this connector CREATES via API call
(POST /v1/webhooks) right after OAuth completes, and the SECRET is one we
generate and hand to Webex ourselves at that moment. So there is no
`needs_webhook_secret` setup field for Webex (unlike Slack/Zoom) — only
client_id/client_secret. The generated secret is stored encrypted in this
connection's OWN `config` (not provider_credentials, which is for
customer-supplied app config) and resolved per-event via the webhook
payload's `orgId`, not via provider_credentials lookup.

Flow:
  1. POST /integrations/oauth-url → mint the consent URL (popup). GET /webex/install
     does the same as a redirect, for non-browser callers.
  2. GET  /webex/oauth/callback → exchange code, store tokens, REGISTER the
     webhook subscription (generates + stores this connection's own secret),
     then run a best-effort initial backfill
  3. POST /webex/events         → Webex webhook: recording created → fetch
     transcript → distill → note

CREDENTIAL MODEL (single-tenant — see 09_company_brain_roadmap.md): each
CUSTOMER creates their own Integration (NOT a Bot — bots cannot read meeting
transcripts) at developer.webex.com/my-apps and pastes its client_id /
client_secret into the panel. One-time setup per customer:
  - developer.webex.com/my-apps/new → "Integration"
  - Redirect URI: https://kbingestion-production.up.railway.app/webex/oauth/callback
  - Scopes: ⚠️ VERIFY against Webex's current scope list when setting this up
    for real — written without a live Webex account to confirm against. Look
    for scopes covering meeting recordings/transcripts read access; as of
    this writing approximately meeting:recordings_read and
    meeting:transcripts_read, requested at the org-admin level if you want
    every meeting in the org, not just the connecting user's own.
  - No dashboard webhook step needed — this connector registers the webhook
    itself via the Webex API once OAuth completes.
"""
import os
import json
import time
import hmac
import hashlib
import secrets as _secrets_mod
import threading
import httpx
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from typing import Optional
from dotenv import load_dotenv

from auth import AuthContext, current_user
import brain_connectors as bc
from connector_zoom import parse_vtt_transcript  # shared WebVTT parsing, not Zoom-specific

load_dotenv()

router = APIRouter()

RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
REDIRECT_URI = f"{RAILWAY_BASE}/webex/oauth/callback"
WEBHOOK_URL = f"{RAILWAY_BASE}/webex/events"

# See module docstring's "⚠️ VERIFY" note.
WEBEX_SCOPES = "meeting:recordings_read meeting:transcripts_read"
BACKFILL_DAYS = 30


def _webex_credentials(workspace_id: str) -> tuple[str, str]:
    creds = bc.get_provider_credentials(workspace_id, "webex")
    if not creds:
        raise HTTPException(
            status_code=400,
            detail="This workspace hasn't set up its Webex app credentials yet. "
                   "Go to Integrations → Webex → Set up to add them.",
        )
    return creds["client_id"], creds["client_secret"]


def build_install_url(workspace_id: str, user_id: str = "") -> str:
    """Mirrors connector_slack.build_install_url — see its docstring."""
    client_id, _ = _webex_credentials(workspace_id)
    state = bc.encode_oauth_state(workspace_id, user_id)
    return (
        "https://webexapis.com/v1/authorize"
        f"?response_type=code&client_id={client_id}&redirect_uri={REDIRECT_URI}"
        f"&scope={WEBEX_SCOPES}&state={state}"
    )


@router.get("/webex/install")
async def webex_install(workspace_id: str, user_id: str = "",
                        auth: AuthContext = Depends(current_user)):
    """Redirect variant, for non-browser callers — see connector_google.google_install."""
    auth.assert_workspace(workspace_id)
    return RedirectResponse(build_install_url(workspace_id, user_id))


def _token_request(workspace_id: str, data: dict) -> dict:
    """Webex takes client_id/secret as regular form fields (unlike Zoom's Basic Auth)."""
    client_id, client_secret = _webex_credentials(workspace_id)
    res = httpx.post("https://webexapis.com/v1/access_token",
                     data={**data, "client_id": client_id, "client_secret": client_secret},
                     timeout=30)
    return res.json()


@router.get("/webex/oauth/callback")
async def webex_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code, stores the connection, REGISTERS this
    connection's webhook subscription (see _register_webhook), then kicks
    off a best-effort initial backfill."""
    from integrations import oauth_complete_html
    if error:
        return oauth_complete_html("webex", "error")
    st = bc.decode_oauth_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")

    data = _token_request(workspace_id, {
        "grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI,
    })
    if "access_token" not in data:
        print(f"[webex] oauth exchange failed: {data}")
        return oauth_complete_html("webex", "error")

    access_token = data["access_token"]
    me = httpx.get("https://webexapis.com/v1/people/me",
                   headers={"Authorization": f"Bearer {access_token}"}, timeout=15).json()
    org_id = me.get("orgId", "")
    display_name = me.get("displayName") or me.get("emails", [""])[0] or org_id

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()

    row = {
        "workspace_id":       workspace_id,
        "provider":           "webex",
        "external_team_id":   org_id,
        "external_team_name": display_name,
        "access_token_enc":   bc.encrypt_secret(access_token),
        "refresh_token_enc":  bc.encrypt_secret(data["refresh_token"]) if data.get("refresh_token") else None,
        "token_expires_at":   expires_at,
        "scopes":             data.get("scope", WEBEX_SCOPES),
        "status":             "active",
        "connected_by":       user_id,
        "config":             {},
    }
    conn = bc.supabase.table("connections").upsert(
        row, on_conflict="workspace_id,provider,external_team_id"
    ).execute().data[0]

    _register_webhook(conn["id"], access_token)
    threading.Thread(target=_safe_backfill, args=(conn["id"],), daemon=True).start()

    return oauth_complete_html("webex", "connected")


def _register_webhook(connection_id: str, access_token: str) -> None:
    """
    Creates (or replaces) this connection's Webex webhook subscription. The
    secret is generated HERE, by us, and stored encrypted in this
    connection's own config — there is no customer-facing "signing secret"
    field for Webex, unlike Slack/Zoom, because there's no dashboard step
    where one would paste it. Best-effort: a failure here means live capture
    won't work until reconnected, but doesn't break anything else, so it's
    logged rather than raised.
    """
    secret = _secrets_mod.token_urlsafe(32)
    try:
        res = httpx.post(
            "https://webexapis.com/v1/webhooks",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "name": f"knova-recordings-{connection_id[:8]}",
                "targetUrl": WEBHOOK_URL,
                "resource": "recordings",
                "event": "created",
                "secret": secret,
            },
            timeout=30,
        )
        if res.status_code not in (200, 201):
            print(f"[webex] webhook registration failed for connection {connection_id}: {res.text[:300]}")
            return
        webhook_id = res.json().get("id")
        bc.supabase.table("connections").update({
            "config": {"webhook_id": webhook_id, "webhook_secret_enc": bc.encrypt_secret(secret)},
        }).eq("id", connection_id).execute()
    except Exception as e:
        print(f"[webex] webhook registration threw for connection {connection_id}: {e}")


def refresh_access_token(conn: dict) -> Optional[str]:
    """Real refresh, same shape as connector_google.refresh_access_token."""
    if not conn.get("refresh_token_enc"):
        print(f"[webex] connection {conn['id']} has no refresh_token — cannot refresh, marking error")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": "No refresh token stored. Reconnect Webex."}
        ).eq("id", conn["id"]).execute()
        return None

    refresh_token = bc.decrypt_secret(conn["refresh_token_enc"])
    data = _token_request(conn["workspace_id"], {
        "grant_type": "refresh_token", "refresh_token": refresh_token,
    })
    if "access_token" not in data:
        print(f"[webex] token refresh failed for connection {conn['id']}: {data}")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": f"Token refresh failed: {data.get('error', 'unknown error')}"}
        ).eq("id", conn["id"]).execute()
        return None

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()
    update = {"access_token_enc": bc.encrypt_secret(data["access_token"]), "token_expires_at": expires_at}
    if data.get("refresh_token"):
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


# ── Recording processing (shared by webhook + backfill) ─────────────────────────

def _process_recording(workspace_id: str, connection_id: str, access_token: str,
                       recording_id: str, topic: str,
                       occurred_at: Optional[str] = None) -> bool:
    """
    Finds this recording's transcript (if any), fetches + parses it, distills
    a note, embeds it. Idempotent via ingest_items (unique on connection_id +
    external_id=recording_id) — same pattern as connector_zoom._process_recording.
    """
    existing = bc.supabase.table("ingest_items").select("id, status") \
        .eq("connection_id", connection_id).eq("external_id", recording_id).execute().data
    if existing and existing[0]["status"] in ("noted", "discarded"):
        return False

    transcripts = httpx.get(
        "https://webexapis.com/v1/meetingTranscripts",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"meetingId": recording_id}, timeout=30,
    )
    transcript_list = transcripts.json().get("items", []) if transcripts.status_code == 200 else []
    if not transcript_list:
        bc.supabase.table("ingest_items").upsert({
            "workspace_id": workspace_id, "connection_id": connection_id, "provider": "webex",
            "external_id": recording_id, "kind": "webex_meeting", "status": "discarded",
            "raw": {"topic": topic, "reason": "no transcript available"},
        }, on_conflict="connection_id,external_id").execute()
        return False

    dl = httpx.get(
        f"https://webexapis.com/v1/meetingTranscripts/{transcript_list[0]['id']}/download",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"format": "vtt"}, timeout=60,
    )
    dl.raise_for_status()
    transcript = parse_vtt_transcript(dl.text)

    note = bc.distill_meeting_transcript(transcript, topic, workspace_id=workspace_id)
    status = "discarded"
    note_id = None
    if note:
        note_id = bc.create_note_and_embed(
            workspace_id, connection_id, "webex", note,
            source_type="meeting", source_tier=2, occurred_at=occurred_at,
        )
        status = "noted"

    bc.supabase.table("ingest_items").upsert({
        "workspace_id": workspace_id, "connection_id": connection_id, "provider": "webex",
        "external_id": recording_id, "kind": "webex_meeting", "status": status, "note_id": note_id,
        "raw": {"topic": topic},
    }, on_conflict="connection_id,external_id").execute()
    return note is not None


def _safe_backfill(connection_id: str) -> None:
    """Best-effort initial backfill — see connector_zoom._safe_backfill for
    the identical reasoning (webhook is the primary path; this just covers
    the gap between account creation and connecting)."""
    try:
        conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data[0]
        token = _valid_access_token(conn)
        if not token:
            return
        since = (datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        res = httpx.get("https://webexapis.com/v1/recordings",
                        headers={"Authorization": f"Bearer {token}"},
                        params={"from": since, "max": 100}, timeout=30)
        if res.status_code != 200:
            print(f"[webex] backfill endpoint returned {res.status_code}, skipping (webhook path unaffected)")
            return
        for r in res.json().get("items", []):
            try:
                _process_recording(
                    conn["workspace_id"], connection_id, token,
                    recording_id=r.get("meetingId", r.get("id", "")),
                    topic=r.get("topic", "Webex Meeting"),
                    occurred_at=r.get("createTime"),
                )
            except Exception as e:
                print(f"[webex] backfill: failed processing recording {r.get('id')}: {e}")
    except Exception as e:
        print(f"[webex] backfill failed entirely (non-fatal, webhook path unaffected): {e}")


# ── Webhook ──────────────────────────────────────────────────────────────────────

def _verify_webex_signature(body: bytes, signature: str, secret: str) -> bool:
    """Webex signs with HMAC-SHA1 (not SHA256 like Slack/Zoom) over the raw body."""
    if not secret:
        return True  # dev fallback; every real webhook has one (we generated it ourselves)
    mine = hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
    return hmac.compare_digest(mine, signature)


@router.post("/webex/events")
async def webex_events(request: Request):
    """
    Webex webhook. Resolved per-event via the payload's orgId — but unlike
    Slack/Zoom, the secret being verified against isn't a customer-supplied
    app-level secret; it's the one THIS connection generated and registered
    for itself in _register_webhook, stored in connections.config. So
    resolution here queries `connections` directly rather than
    provider_credentials.
    """
    body = await request.body()
    payload = json.loads(body or "{}")
    org_id = payload.get("orgId", "")

    conn = bc.supabase.table("connections").select("*") \
        .eq("provider", "webex").eq("external_team_id", org_id) \
        .eq("status", "active").execute().data
    if not conn:
        return JSONResponse({"ok": True})  # unknown org — nothing to do, still 200
    conn = conn[0]

    secret = bc.decrypt_secret((conn.get("config") or {}).get("webhook_secret_enc", ""))
    signature = request.headers.get("x-spark-signature", "")
    if not _verify_webex_signature(body, signature, secret):
        raise HTTPException(status_code=401, detail="Bad Webex signature.")

    if payload.get("resource") == "recordings" and payload.get("event") == "created":
        data = payload.get("data", {})
        token = _valid_access_token(conn)
        if token:
            try:
                _process_recording(
                    conn["workspace_id"], conn["id"], token,
                    recording_id=data.get("meetingId", data.get("id", "")),
                    topic=data.get("topic", "Webex Meeting"),
                    occurred_at=data.get("createTime"),
                )
            except Exception as e:
                print(f"[webex] recording created processing failed: {e}")

    return JSONResponse({"ok": True})
