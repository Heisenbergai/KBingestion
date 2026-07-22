"""
Slack connector (Phase 2, first integration).

Flow:
  1. GET  /slack/install         → redirect user to Slack's OAuth consent
  2. GET  /slack/oauth/callback  → exchange code, store encrypted token, back to app
  3. GET  /slack/channels        → list channels for the admin to pick
  4. POST /slack/channels/select → save selection + backfill (background) + filtration
  5. POST /slack/events          → Slack Events API webhook: live messages → ingest_items

Everything Slack-specific lives here; the shared pipeline (capture, filtration,
notes, embedding) is in brain_connectors.py.

One-time setup (Tanmay, api.slack.com/apps):
  - Create app, add Bot Token scopes: channels:read, channels:history, users:read, team:read
  - OAuth redirect URL: https://kbingestion-production.up.railway.app/slack/oauth/callback
  - Enable Events API, request URL: https://kbingestion-production.up.railway.app/slack/events,
    subscribe to bot event: message.channels
  - Railway env: SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, SLACK_SIGNING_SECRET,
    APP_REDIRECT_URL (where to send the user after connecting, e.g. the Library page),
    CONNECTOR_ENCRYPTION_KEY
"""
import os
import json
import time
import hmac
import hashlib
import threading
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

import brain_connectors as bc

load_dotenv()

router = APIRouter()

SLACK_CLIENT_ID     = os.getenv("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
APP_REDIRECT_URL = os.getenv("APP_REDIRECT_URL", "https://knova.lovable.app")
REDIRECT_URI = f"{RAILWAY_BASE}/slack/oauth/callback"

SLACK_SCOPES = "channels:read,channels:history,users:read,team:read"
BACKFILL_DAYS = 90


# ── OAuth state (workspace_id + user_id, Fernet-signed so it can't be forged) ──
def _encode_state(workspace_id: str, user_id: str) -> str:
    return bc.encrypt_secret(json.dumps({"w": workspace_id, "u": user_id, "t": int(time.time())}))


def _decode_state(state: str) -> dict:
    try:
        return json.loads(bc.decrypt_secret(state))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")


# ── Slack Web API helpers ───────────────────────────────────────────────────────
def _slack_get(method: str, token: str, params: dict) -> dict:
    res = httpx.get(f"https://slack.com/api/{method}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params, timeout=30)
    data = res.json()
    if not data.get("ok"):
        raise HTTPException(status_code=502, detail=f"Slack {method} error: {data.get('error')}")
    return data


def _user_name_map(token: str) -> dict:
    """user_id → display name, for readable transcripts (best-effort)."""
    names = {}
    try:
        cursor = None
        for _ in range(10):  # cap pages
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = _slack_get("users.list", token, params)
            for u in data.get("members", []):
                prof = u.get("profile", {})
                names[u["id"]] = prof.get("real_name") or prof.get("display_name") or u.get("name", u["id"])
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except Exception as e:
        print(f"[slack] user map failed (non-fatal): {e}")
    return names


# ── Routes ──────────────────────────────────────────────────────────────────────

@router.get("/slack/install")
async def slack_install(workspace_id: str, user_id: str = ""):
    """Redirects the admin to Slack's consent screen."""
    if not SLACK_CLIENT_ID:
        raise HTTPException(status_code=500, detail="SLACK_CLIENT_ID not configured on the server.")
    state = _encode_state(workspace_id, user_id)
    url = (f"https://slack.com/oauth/v2/authorize?client_id={SLACK_CLIENT_ID}"
           f"&scope={SLACK_SCOPES}&redirect_uri={REDIRECT_URI}&state={state}")
    return RedirectResponse(url)


@router.get("/slack/oauth/callback")
async def slack_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code for a token and stores the connection."""
    if error:
        return RedirectResponse(f"{APP_REDIRECT_URL}?slack=error")
    st = _decode_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")

    res = httpx.post("https://slack.com/api/oauth.v2.access", data={
        "client_id": SLACK_CLIENT_ID, "client_secret": SLACK_CLIENT_SECRET,
        "code": code, "redirect_uri": REDIRECT_URI,
    }, timeout=30)
    data = res.json()
    if not data.get("ok"):
        print(f"[slack] oauth exchange failed: {data.get('error')}")
        return RedirectResponse(f"{APP_REDIRECT_URL}?slack=error")

    team = data.get("team", {})
    access_token = data.get("access_token")  # bot token (xoxb-…)

    row = {
        "workspace_id":       workspace_id,
        "provider":           "slack",
        "external_team_id":   team.get("id"),
        "external_team_name": team.get("name"),
        "access_token_enc":   bc.encrypt_secret(access_token),
        "bot_user_id":        data.get("bot_user_id"),
        "scopes":             data.get("scope", SLACK_SCOPES),
        "status":             "active",
        "connected_by":       user_id,
        "config":             {},
    }
    # upsert so reconnecting the same Slack workspace refreshes the token
    bc.supabase.table("connections").upsert(
        row, on_conflict="workspace_id,provider,external_team_id"
    ).execute()

    return RedirectResponse(f"{APP_REDIRECT_URL}?slack=connected")


def _get_conn_token(connection_id: str) -> tuple[dict, str]:
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]
    token = bc.decrypt_secret(conn["access_token_enc"])
    return conn, token


@router.get("/slack/channels")
async def slack_channels(connection_id: str):
    """Public channels the admin can choose to ingest from."""
    conn, token = _get_conn_token(connection_id)
    data = _slack_get("conversations.list", token,
                      {"types": "public_channel", "limit": 200, "exclude_archived": "true"})
    channels = [{"id": c["id"], "name": c["name"], "num_members": c.get("num_members", 0),
                 "is_member": c.get("is_member", False)}
                for c in data.get("channels", [])]
    selected = (conn.get("config") or {}).get("channels", [])
    return {"channels": channels, "selected": [c["id"] for c in selected]}


class SelectChannelsRequest(BaseModel):
    connection_id: str
    channels: list[dict]   # [{id, name}]


@router.post("/slack/channels/select")
async def slack_select_channels(body: SelectChannelsRequest):
    """Saves the channel selection and kicks off a background backfill + filtration."""
    conn, token = _get_conn_token(body.connection_id)

    bc.supabase.table("connections").update(
        {"config": {"channels": body.channels}}
    ).eq("id", body.connection_id).execute()

    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    bc.SYNC_JOBS[job_id] = {"job_id": job_id, "connection_id": body.connection_id,
                            "status": "processing", "stage": "backfilling",
                            "messages_captured": 0, "notes_created": 0}

    def _backfill():
        try:
            names = _user_name_map(token)
            total = 0
            oldest = time.time() - BACKFILL_DAYS * 86400
            for ch in body.channels:
                captured = _backfill_channel(conn, token, ch, names, oldest)
                total += captured
                bc.SYNC_JOBS[job_id]["messages_captured"] = total
            bc.SYNC_JOBS[job_id]["stage"] = "filtering"
            result = bc.run_filtration(conn["workspace_id"], conn["id"], "slack",
                                       job=bc.SYNC_JOBS[job_id])
            bc.SYNC_JOBS[job_id].update({"status": "completed", "stage": "completed", **result})
        except Exception as e:
            import traceback; print(f"[slack] backfill failed: {e}"); print(traceback.format_exc())
            bc.SYNC_JOBS[job_id].update({"status": "failed", "error": str(e)})

    threading.Thread(target=_backfill, daemon=True).start()
    return {"success": True, "job_id": job_id, "status": "processing"}


def _backfill_channel(conn: dict, token: str, channel: dict, names: dict, oldest: float) -> int:
    """Pulls up to BACKFILL_DAYS of a channel's history into ingest_items."""
    channel_id, channel_name = channel["id"], channel.get("name", channel["id"])
    items, cursor, pages = [], None, 0
    while pages < 20:  # cap ~20k messages/channel on backfill
        params = {"channel": channel_id, "limit": 200, "oldest": str(oldest)}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _slack_get("conversations.history", token, params)
        except HTTPException as e:
            print(f"[slack] history error on #{channel_name}: {e.detail}")
            break
        for m in data.get("messages", []):
            if m.get("type") != "message" or m.get("subtype"):
                continue  # skip joins/leaves/bot noise
            items.append(_normalize_message(m, channel_id, channel_name, names))
        cursor = data.get("response_metadata", {}).get("next_cursor")
        pages += 1
        if not cursor:
            break
    return bc.save_ingest_items(conn["workspace_id"], conn["id"], "slack", items)


def _normalize_message(m: dict, channel_id: str, channel_name: str, names: dict) -> dict:
    ts = m.get("ts", "")
    user = m.get("user", "")
    iso = None
    try:
        iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, TypeError):
        pass
    return {
        "external_id": f"{channel_id}:{ts}",
        "kind": "message",
        "raw": {
            "channel": channel_id, "channel_name": channel_name,
            "user": user, "user_name": names.get(user, user),
            "text": m.get("text", ""), "ts": ts, "iso_ts": iso,
            "thread_ts": m.get("thread_ts"),
        },
    }


# ── Events API webhook (live messages) ──────────────────────────────────────────

def _verify_slack_signature(request: Request, body: bytes) -> bool:
    """Verifies the request genuinely came from Slack (signing secret)."""
    if not SLACK_SIGNING_SECRET:
        return True  # not configured — allow (dev); set the secret in prod
    ts = request.headers.get("X-Slack-Request-Timestamp", "")
    sig = request.headers.get("X-Slack-Signature", "")
    if not ts or abs(time.time() - int(ts)) > 60 * 5:
        return False
    base = f"v0:{ts}:{body.decode()}"
    mine = "v0=" + hmac.new(SLACK_SIGNING_SECRET.encode(), base.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mine, sig)


@router.post("/slack/events")
async def slack_events(request: Request):
    """
    Slack Events API endpoint. Handles the one-time url_verification handshake
    and live message events (captured into ingest_items; filtration runs on the
    next /connectors/sync or a scheduled job, not per-message).
    """
    body = await request.body()
    payload = json.loads(body or "{}")

    if payload.get("type") == "url_verification":
        return PlainTextResponse(payload.get("challenge", ""))

    if not _verify_slack_signature(request, body):
        raise HTTPException(status_code=401, detail="Bad Slack signature.")

    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        if event.get("type") == "message" and not event.get("subtype") and not event.get("bot_id"):
            team_id = payload.get("team_id")
            conn = bc.supabase.table("connections").select("*") \
                .eq("provider", "slack").eq("external_team_id", team_id) \
                .eq("status", "active").execute().data
            if conn:
                conn = conn[0]
                selected = {c["id"] for c in (conn.get("config") or {}).get("channels", [])}
                ch = event.get("channel")
                if ch in selected or not selected:
                    item = _normalize_message(event, ch,
                                              (conn.get("config") or {}).get("channel_names", {}).get(ch, ch), {})
                    bc.save_ingest_items(conn["workspace_id"], conn["id"], "slack", [item])
    # Slack requires a fast 200
    return JSONResponse({"ok": True})
