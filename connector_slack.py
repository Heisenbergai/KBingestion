"""
Slack connector (Phase 2, first integration).

Flow:
  1. POST /integrations/oauth-url → (authenticated) mint the consent URL, open it
                                    in a popup. GET /slack/install does the same
                                    thing as a redirect, for non-browser callers.
  2. GET  /slack/oauth/callback  → exchange code, store encrypted token, back to app
  3. GET  /slack/channels        → list channels for the admin to pick
  4. POST /slack/channels/select → save selection + backfill (background) + filtration
  5. POST /slack/events          → Slack Events API webhook: live messages → ingest_items

Steps 1, 3 and 4 require the caller's Supabase token and check membership of the
owning workspace (see auth.py). Step 2 is authenticated by the Fernet-signed
state minted in step 1; step 5 by Slack's own request signature.

Everything Slack-specific lives here; the shared pipeline (capture, filtration,
notes, embedding) is in brain_connectors.py.

CREDENTIAL MODEL (single-tenant, decided 2026-07-26 — see 09_company_brain_roadmap.md):
Each CUSTOMER creates their own Slack app at api.slack.com/apps and pastes its
client_id / client_secret / signing_secret into the Integrations panel
(POST /integrations/credentials), not Tanmay. This avoids Slack's app-directory
review, which single-workspace installs don't need. Per customer, in their app:
  - Add Bot Token scopes: channels:read, channels:history, users:read, team:read
  - OAuth redirect URL: https://kbingestion-production.up.railway.app/slack/oauth/callback
    (fixed — same for every customer, since it's Railway's URL, not theirs)
  - Enable Events API, request URL: https://kbingestion-production.up.railway.app/slack/events
    (also fixed/shared — see slack_events() for how one shared URL still verifies
    each customer's own signing secret), subscribe to bot event: message.channels

Env vars SLACK_CLIENT_ID / SLACK_CLIENT_SECRET / SLACK_SIGNING_SECRET are now only
a FALLBACK, for the one connection created before this model existed (Default
Workspace, connected 2026-07-22). CONNECTOR_ENCRYPTION_KEY (Fernet) and
APP_REDIRECT_URL remain required Railway env vars regardless.
"""
import os
import json
import time
import hmac
import hashlib
import threading
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from auth import AuthContext, current_user

import brain_connectors as bc

load_dotenv()

router = APIRouter()

# Env vars are now only a FALLBACK, for the one Slack connection created before
# the per-workspace credential model existed (Default Workspace, connected
# 2026-07-22). Every new connection resolves its own workspace's app via
# provider_credentials (see _slack_credentials below) — single-tenant model,
# each customer registers their own Slack app. SLACK_SIGNING_SECRET stays
# global/env-only: it verifies the /slack/events webhook, which arrives
# before any workspace is known, so it cannot be looked up per-workspace.
_ENV_SLACK_CLIENT_ID     = os.getenv("SLACK_CLIENT_ID", "")
_ENV_SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
APP_REDIRECT_URL = os.getenv("APP_REDIRECT_URL", "https://knova.lovable.app")
REDIRECT_URI = f"{RAILWAY_BASE}/slack/oauth/callback"

SLACK_SCOPES = "channels:read,channels:history,users:read,team:read"
BACKFILL_DAYS = 90


# ── OAuth state ─────────────────────────────────────────────────────────────────
# Shared helper now lives in brain_connectors.py (encode_oauth_state/
# decode_oauth_state) — every OAuth connector uses the same Fernet-signed
# {workspace_id, user_id, timestamp} shape.
_encode_state = bc.encode_oauth_state
_decode_state = bc.decode_oauth_state


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


def get_permalink(token: str, channel: str, ts: str) -> Optional[str]:
    """
    Phase 2B provenance fix (2026-08-15): real Slack chat.getPermalink call --
    the only reliable source for a working link back to a specific message,
    per Slack's own docs. NEVER manually constructs a URL (workspace domain
    slug isn't reliably derivable from what's already stored, and Slack
    itself warns manual construction has edge cases -- Enterprise Grid,
    custom domains). Returns None on any failure (missing channel/ts, API
    error, network failure) -- the caller (run_filtration) treats a missing
    permalink as "real metadata kept, no link available," never fabricates
    one, and never lets this failure block note creation.
    """
    if not channel or not ts:
        return None
    try:
        data = _slack_get("chat.getPermalink", token, {"channel": channel, "message_ts": ts})
        permalink = data.get("permalink")
        return permalink if isinstance(permalink, str) and permalink else None
    except Exception as e:
        print(f"[slack] permalink lookup failed for {channel}:{ts} (non-fatal): {e}")
        return None


def build_permalink_resolver(conn: dict):
    """
    Returns a `raw_message_dict -> Optional[str]` closure bound to this
    connection's real decrypted token, for brain_connectors.run_filtration's
    `resolve_permalink` parameter -- or None if this connection can't
    resolve permalinks (wrong provider, no token) so callers that don't
    know the provider in advance (worker.py, the generic /connectors/sync
    route) can build one uniformly without a provider-specific branch of
    their own beyond "is this a Slack connection."
    """
    if conn.get("provider") != "slack":
        return None
    token_enc = conn.get("access_token_enc")
    if not token_enc:
        return None
    token = bc.decrypt_secret(token_enc)

    def _resolve(raw: dict) -> Optional[str]:
        return get_permalink(token, raw.get("channel"), raw.get("ts"))

    return _resolve


# ── Routes ──────────────────────────────────────────────────────────────────────

def _slack_client_credentials(workspace_id: str) -> tuple[str, str]:
    """
    (client_id, client_secret) for this workspace's own Slack app, falling
    back to the env vars only for connections that predate the per-workspace
    credential model (Default Workspace, connected 2026-07-22, before
    provider_credentials existed).
    """
    creds = bc.get_provider_credentials(workspace_id, "slack")
    if creds:
        return creds["client_id"], creds["client_secret"]
    if _ENV_SLACK_CLIENT_ID and _ENV_SLACK_CLIENT_SECRET:
        return _ENV_SLACK_CLIENT_ID, _ENV_SLACK_CLIENT_SECRET
    raise HTTPException(
        status_code=400,
        detail="This workspace hasn't set up its Slack app credentials yet. "
               "Go to Integrations → Slack → Set up to add them.",
    )


def build_install_url(workspace_id: str, user_id: str = "") -> str:
    """
    The Slack consent URL for one workspace, with the workspace baked into a
    Fernet-signed state so the callback cannot be pointed at someone else's
    workspace. Shared with POST /integrations/oauth-url, which is what the
    browser actually uses — a popup navigation cannot carry an Authorization
    header, so the URL is minted by an authenticated call and only then opened.
    """
    client_id, _ = _slack_client_credentials(workspace_id)
    state = _encode_state(workspace_id, user_id)
    return (f"https://slack.com/oauth/v2/authorize?client_id={client_id}"
            f"&scope={SLACK_SCOPES}&redirect_uri={REDIRECT_URI}&state={state}")


@router.get("/slack/install")
async def slack_install(workspace_id: str, user_id: str = "",
                        auth: AuthContext = Depends(current_user)):
    """
    Redirects the admin to Slack's consent screen.

    Authenticated: without this check anyone could mint a valid state for a
    workspace they do not belong to and bind their own Slack team to it —
    injecting messages into that company's brain. Browser popups should use
    POST /integrations/oauth-url instead, which returns the same URL as JSON.
    """
    auth.assert_workspace(workspace_id)
    return RedirectResponse(build_install_url(workspace_id, user_id))


@router.get("/slack/oauth/callback")
async def slack_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code for a token and stores the connection."""
    from integrations import oauth_complete_html
    if error:
        return oauth_complete_html("slack", "error")
    st = _decode_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")
    client_id, client_secret = _slack_client_credentials(workspace_id)

    res = httpx.post("https://slack.com/api/oauth.v2.access", data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "redirect_uri": REDIRECT_URI,
    }, timeout=30)
    data = res.json()
    if not data.get("ok"):
        print(f"[slack] oauth exchange failed: {data.get('error')}")
        return oauth_complete_html("slack", "error")

    team = data.get("team", {})
    access_token = data.get("access_token")  # bot token (xoxb-…)

    row = {
        "workspace_id":       workspace_id,
        "provider":           "slack",
        "external_team_id":   team.get("id"),
        "external_team_name": team.get("name"),
        "access_token_enc":   bc.encrypt_secret(access_token),
        "bot_user_id":        data.get("bot_user_id"),
        # Slack's own per-installation app identifier (real field on every
        # oauth.v2.access response, per Slack's documented OAuth v2 shape) --
        # stored so a live webhook event (which also carries this as
        # api_app_id) can disambiguate WHICH connection it belongs to when
        # the same Slack team is connected to more than one Knova workspace
        # (real, legitimate case: connections' true uniqueness is
        # (workspace_id, provider, external_team_id), not external_team_id
        # alone -- see 2026-08-15 routing fix). Only helps when the two
        # installs used genuinely different Slack apps; if they share the
        # same app (e.g. both used the env-var fallback), app_id is
        # identical too and the event stays genuinely ambiguous -- see
        # _resolve_slack_connection's fail-closed handling for that case.
        "app_id":             data.get("app_id"),
        "scopes":             data.get("scope", SLACK_SCOPES),
        "status":             "active",
        "connected_by":       user_id,
        "config":             {},
    }
    # upsert so reconnecting the same Slack workspace refreshes the token
    bc.supabase.table("connections").upsert(
        row, on_conflict="workspace_id,provider,external_team_id"
    ).execute()

    return oauth_complete_html("slack", "connected")


def _get_conn_token(connection_id: str) -> tuple[dict, str]:
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]
    token = bc.decrypt_secret(conn["access_token_enc"])
    return conn, token


@router.get("/slack/channels")
async def slack_channels(connection_id: str,
                         auth: AuthContext = Depends(current_user)):
    """
    Public channels the admin can choose to ingest from.

    The connection UUID is an opaque id, not a credential — the workspace that
    owns it is resolved from the row and authorised before any Slack call is
    made with that workspace's bot token.
    """
    conn, token = _get_conn_token(connection_id)
    auth.assert_workspace(conn["workspace_id"])
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
async def slack_select_channels(body: SelectChannelsRequest,
                                auth: AuthContext = Depends(current_user)):
    """
    Saves the channel selection and kicks off a background backfill + filtration.

    Authorised on the connection's owning workspace: this both reads Slack
    history and writes notes into that workspace's brain, so it is the most
    damaging of the three Slack routes to leave open.
    """
    conn, token = _get_conn_token(body.connection_id)
    auth.assert_workspace(conn["workspace_id"])

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
            result = bc.run_filtration(
                conn["workspace_id"], conn["id"], "slack",
                job=bc.SYNC_JOBS[job_id],
                resolve_permalink=lambda raw: get_permalink(token, raw.get("channel"), raw.get("ts")),
            )
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

def _verify_slack_signature(request: Request, body: bytes, signing_secret: str) -> bool:
    """Verifies the request genuinely came from Slack, against ONE specific
    app's signing secret (see slack_events() for how that secret is chosen)."""
    if not signing_secret:
        return True  # not configured — allow (dev); set one in prod
    ts = request.headers.get("X-Slack-Request-Timestamp", "")
    sig = request.headers.get("X-Slack-Signature", "")
    if not ts or abs(time.time() - int(ts)) > 60 * 5:
        return False
    base = f"v0:{ts}:{body.decode()}"
    mine = "v0=" + hmac.new(signing_secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mine, sig)


def _resolve_slack_connection(team_id: Optional[str], api_app_id: Optional[str] = None) -> Optional[dict]:
    """
    Resolves the SINGLE KNOVA connection a Slack event belongs to.

    2026-08-15 routing fix. external_team_id is NOT globally unique --
    connections' real uniqueness is (workspace_id, provider,
    external_team_id) (the actual unique index; see slack_callback's
    upsert), so the SAME Slack team can legitimately be authorized into
    MULTIPLE Knova workspaces (e.g. the same admin connecting the same
    Slack org into two different client accounts, or during onboarding/
    testing -- a real, confirmed live case, not hypothetical). Previously
    this function's job was done inline by taking connections[0] from a
    query that could return more than one row -- a silent, undeterministic
    misrouting risk: a live message could have been filed into the WRONG
    workspace's brain depending on row order.

    Disambiguation: if more than one active connection shares team_id, try
    Slack's own api_app_id (stored on the connection as `app_id` at OAuth
    time, present on every Events API payload) -- this only helps when the
    ambiguous connections were authorized through genuinely DIFFERENT Slack
    apps. If they share the same app too (e.g. both used the env-var
    fallback app -- the real, confirmed state of today's two ambiguous
    connections), app_id can't disambiguate either.

    FAILS CLOSED: zero or more-than-one unresolvable match returns None.
    Callers MUST treat None as "do not ingest this event" -- never fall
    back to picking any row. The opposite bias from most fail-open
    functions in this codebase, deliberately: an unrouted event costs
    nothing (Slack redelivers, or the backfill covers it later); a
    MISROUTED event puts one company's Slack message in another company's
    knowledge base, which cannot be silently undone.
    """
    if not team_id:
        return None
    candidates = bc.supabase.table("connections").select("*") \
        .eq("provider", "slack").eq("external_team_id", team_id) \
        .eq("status", "active").execute().data or []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    if api_app_id:
        matching = [c for c in candidates if c.get("app_id") == api_app_id]
        if len(matching) == 1:
            return matching[0]

    print(f"[slack] AMBIGUOUS event: team_id={team_id} api_app_id={api_app_id} matches "
          f"{len(candidates)} active connections ({[c['id'] for c in candidates]}) and could not "
          f"be disambiguated -- discarding this event rather than guessing which Knova workspace "
          f"it belongs to.")
    return None


@router.post("/slack/events")
async def slack_events(request: Request):
    """
    Slack Events API endpoint. ONE shared URL for every customer's Slack app
    (Railway has one deployment), but each customer's own app signs its
    events with ITS OWN signing secret — verifying against a single global
    secret would fail every real customer's events except possibly the first
    one configured, silently breaking live ingestion for everyone else. So the
    signing secret is resolved from the payload's team_id (and, if ambiguous,
    api_app_id -- see _resolve_slack_connection) BEFORE verifying, not read
    from env, except as a fallback for connections that predate per-workspace
    credentials or whose team_id is genuinely ambiguous.

    Handles the one-time url_verification handshake (no team_id, no signature
    needed — Slack's own onboarding step) and live message events (captured
    into ingest_items; filtration runs on the next /connectors/sync or the
    scheduled worker, not per-message).
    """
    body = await request.body()
    payload = json.loads(body or "{}")

    if payload.get("type") == "url_verification":
        return PlainTextResponse(payload.get("challenge", ""))

    team_id = payload.get("team_id")
    api_app_id = payload.get("api_app_id")
    # Resolved ONCE, reused for both the signing-secret lookup and the
    # event-routing decision below -- previously these were two SEPARATE
    # lookups (get_provider_credentials_by_external_team for the secret,
    # a raw connections query for routing) that could each independently
    # pick a different row under ambiguity, disagreeing with each other.
    conn = _resolve_slack_connection(team_id, api_app_id) if team_id else None

    signing_secret = SLACK_SIGNING_SECRET
    if conn:
        creds = bc.get_provider_credentials(conn["workspace_id"], "slack")
        if creds and creds.get("webhook_secret"):
            signing_secret = creds["webhook_secret"]

    if not _verify_slack_signature(request, body, signing_secret):
        raise HTTPException(status_code=401, detail="Bad Slack signature.")

    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        if event.get("type") == "message" and not event.get("subtype") and not event.get("bot_id"):
            if conn is None:
                print(f"[slack] discarding live message event -- no unambiguous connection for "
                      f"team_id={team_id} api_app_id={api_app_id}")
            else:
                selected = {c["id"] for c in (conn.get("config") or {}).get("channels", [])}
                ch = event.get("channel")
                if ch in selected or not selected:
                    item = _normalize_message(event, ch,
                                              (conn.get("config") or {}).get("channel_names", {}).get(ch, ch), {})
                    bc.save_ingest_items(conn["workspace_id"], conn["id"], "slack", [item])
    # Slack requires a fast 200
    return JSONResponse({"ok": True})
