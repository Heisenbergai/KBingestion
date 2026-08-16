"""
Google connector — shared OAuth/token plumbing for the four Google Workspace
surfaces (Calendar, Meet, Chat, Drive), plus Drive's own reference-only
resolution.

LOCKED PRODUCT RULE (Google Workspace scope lock, 2026-08-15): KNOVA captures
evidence/context from Google Workspace; it must NOT become a mirror of
Google Drive. Calendar/Meet/Chat feed durable knowledge directly (see
connector_google_calendar.py, connector_google_meet.py, connector_google_chat.py).
Drive is REFERENCE ONLY here — no folder polling, no bulk knowledge_items
creation, no Storage copies, ever. A Drive file is only ever looked up one at
a time, by ID, when something else (a Meet transcript, a Chat message)
already mentions it. See resolve_drive_reference() below.

An earlier version of this module DID bulk-poll selected Drive folders and
create a real knowledge_items row per file (see git history / the Phase 2
Google Drive Canonical Ingestion work) — that behavior has been neutralized
per the scope-lock decision, not because it was broken (it was fully tested
and live-verified), but because product decided Drive must not become a
second file repository. drive_app_db.py's RPCs and Storage-upload function
are left in place, unused by this module for now, in case a future explicit
product decision authorizes individual-file import — see that module's
docstring.

ONE CONNECTION PER WORKSPACE covers all four surfaces (Correction 1 of the
scope lock: provider stays "google_drive", not renamed to "google_workspace").
`connections.config.enabled_surfaces` (a list of "calendar"|"meet"|"chat"|
"drive") controls which OAuth scopes get requested and which pollers treat
this connection as theirs — see SURFACE_SCOPES / scopes_for_surfaces() below.

CREDENTIAL MODEL (single-tenant, same as Slack — see 09_company_brain_roadmap.md):
Each CUSTOMER creates their own project in Google Cloud Console and pastes its
client_id / client_secret into the Integrations panel.
"""
import os
import re
import httpx
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from auth import AuthContext, current_user
import brain_connectors as bc

load_dotenv()

router = APIRouter()

RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
REDIRECT_URI = f"{RAILWAY_BASE}/google/oauth/callback"

# Verified against current Google API documentation (not invented) during the
# Google Workspace scope-lock design pass. Each surface requests only the
# narrowest read-only scope that covers what KNOVA actually captures for it —
# see the per-surface product-behavior sections in 09_company_brain_roadmap.md.
SURFACE_SCOPES: dict[str, list[str]] = {
    "calendar": ["https://www.googleapis.com/auth/calendar.events.readonly"],
    "meet":     ["https://www.googleapis.com/auth/meetings.space.readonly"],
    "chat":     ["https://www.googleapis.com/auth/chat.messages.readonly",
                 "https://www.googleapis.com/auth/chat.spaces.readonly"],
    # drive.readonly (not the narrower drive.file) because reference resolution
    # looks up an arbitrary file by ID that the connecting account can read,
    # not just files explicitly picked through Google's Picker widget.
    "drive":    ["https://www.googleapis.com/auth/drive.readonly"],
}
VALID_SURFACES = set(SURFACE_SCOPES.keys())


def scopes_for_surfaces(enabled_surfaces: list[str]) -> str:
    """
    Builds the OAuth scope string from ONLY the enabled surfaces — a disabled
    surface's scope is never requested, so Google's consent screen only ever
    shows what this workspace actually turned on. Unknown surface names are
    silently ignored rather than raising, since this is called from user-
    supplied input at connect time and a typo shouldn't crash the flow.
    """
    scopes: list[str] = []
    for surface in enabled_surfaces:
        scopes.extend(SURFACE_SCOPES.get(surface, []))
    return " ".join(dict.fromkeys(scopes))  # de-duped, order-stable


def _google_credentials(workspace_id: str) -> tuple[str, str]:
    creds = bc.get_provider_credentials(workspace_id, "google_drive")
    if not creds:
        raise HTTPException(
            status_code=400,
            detail="This workspace hasn't set up its Google app credentials yet. "
                   "Go to Integrations → Google Workspace → Set up to add them.",
        )
    return creds["client_id"], creds["client_secret"]


def build_install_url(workspace_id: str, user_id: str = "",
                      enabled_surfaces: Optional[list[str]] = None) -> str:
    """
    Mirrors connector_slack.build_install_url — see its docstring. Scope
    string is built from enabled_surfaces (defaults to just "drive" if the
    caller doesn't specify). The chosen surfaces are also sealed into the
    Fernet-signed OAuth state (Google's redirect_uri is fixed and registered
    with Google in advance, so it can't carry a dynamic query param the way
    a plain link could) — google_callback() reads them back out of state,
    not from any callback query param.
    """
    client_id, _ = _google_credentials(workspace_id)
    enabled = [s for s in (enabled_surfaces or ["drive"]) if s in VALID_SURFACES] or ["drive"]
    state = bc.encode_oauth_state(workspace_id, user_id, extra={"surfaces": enabled})
    scope = scopes_for_surfaces(enabled)
    return (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={client_id}&redirect_uri={REDIRECT_URI}"
        f"&response_type=code&scope={scope}"
        "&access_type=offline&prompt=consent"  # force a refresh_token every time,
        # not just on first-ever consent — needed since worker.py's token refresh
        # (see below) depends on always having one.
        f"&state={state}"
    )


@router.get("/google/install")
async def google_install(workspace_id: str, user_id: str = "", surfaces: str = "drive",
                         auth: AuthContext = Depends(current_user)):
    """Redirect variant of the install URL, for non-browser callers. `surfaces`
    is a comma-separated list, e.g. "calendar,meet,chat,drive"."""
    auth.assert_workspace(workspace_id)
    enabled = [s.strip() for s in surfaces.split(",") if s.strip() in VALID_SURFACES]
    return RedirectResponse(build_install_url(workspace_id, user_id, enabled or ["drive"]))


@router.get("/google/oauth/callback")
async def google_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code for tokens and stores the connection, with
    enabled_surfaces (read back from the signed state — see build_install_url)
    persisted into config so every poller/resolver knows which surfaces this
    connection actually consented to."""
    from integrations import oauth_complete_html
    if error:
        return oauth_complete_html("google_drive", "error")
    st = bc.decode_oauth_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")
    enabled = [s for s in st.get("surfaces", ["drive"]) if s in VALID_SURFACES] or ["drive"]
    client_id, client_secret = _google_credentials(workspace_id)

    res = httpx.post("https://oauth2.googleapis.com/token", data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
    }, timeout=30)
    data = res.json()
    if "access_token" not in data:
        print(f"[google] oauth exchange failed: {data}")
        return oauth_complete_html("google_drive", "error")

    if not data.get("refresh_token"):
        print(f"[google] WARNING: no refresh_token in response for workspace={workspace_id}")

    userinfo = httpx.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {data['access_token']}"}, timeout=15,
    ).json()
    account_email = userinfo.get("email", "")

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()

    row = {
        "workspace_id":       workspace_id,
        "provider":           "google_drive",
        "external_team_id":   account_email,       # one Google account = one "team" here
        "external_team_name": account_email,
        "access_token_enc":   bc.encrypt_secret(data["access_token"]),
        "refresh_token_enc":  bc.encrypt_secret(data["refresh_token"]) if data.get("refresh_token") else None,
        "token_expires_at":   expires_at,
        "scopes":             data.get("scope", ""),
        "status":             "active",
        "connected_by":       user_id,
        "config":             {"enabled_surfaces": enabled},
    }
    bc.supabase.table("connections").upsert(
        row, on_conflict="workspace_id,provider,external_team_id"
    ).execute()

    return oauth_complete_html("google_drive", "connected")


def refresh_access_token(conn: dict) -> Optional[str]:
    """
    Real token refresh. Returns the new access token, or None if refresh
    failed (connection marked 'error' so the admin sees it needs
    reconnecting rather than failing silently forever).
    """
    if not conn.get("refresh_token_enc"):
        print(f"[google] connection {conn['id']} has no refresh_token — cannot refresh, marking error")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": "No refresh token stored. Reconnect Google Workspace."}
        ).eq("id", conn["id"]).execute()
        return None

    client_id, client_secret = _google_credentials(conn["workspace_id"])
    refresh_token = bc.decrypt_secret(conn["refresh_token_enc"])

    res = httpx.post("https://oauth2.googleapis.com/token", data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    }, timeout=30)
    data = res.json()
    if "access_token" not in data:
        print(f"[google] token refresh failed for connection {conn['id']}: {data}")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": f"Token refresh failed: {data.get('error', 'unknown error')}"}
        ).eq("id", conn["id"]).execute()
        return None

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data.get("expires_in", 3600))).isoformat()
    bc.supabase.table("connections").update({
        "access_token_enc": bc.encrypt_secret(data["access_token"]),
        "token_expires_at": expires_at,
    }).eq("id", conn["id"]).execute()
    return data["access_token"]


def _valid_access_token(conn: dict) -> Optional[str]:
    """Returns a usable access token, refreshing first if it's expired or about to be."""
    expires_at = conn.get("token_expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp > datetime.now(timezone.utc) + timedelta(minutes=2):
                return bc.decrypt_secret(conn["access_token_enc"])
        except ValueError:
            pass
    return refresh_access_token(conn)


def get_active_connection(workspace_id: str, required_surface: str) -> Optional[dict]:
    """
    Shared fail-closed lookup used by every Google Workspace poller
    (Calendar/Meet/Chat) and by resolve_drive_reference(): a connection is
    only usable for a given surface if it's a real google_drive-provider
    connection, active, belonging to the right workspace, AND has that
    surface explicitly enabled in config.enabled_surfaces. This is the
    security-boundary lookup — see 09_company_brain_roadmap.md's Google
    Workspace scope-lock notes on why this check cannot be pushed down into
    any RPC (connections lives in a different Supabase project from
    knowledge_items/app-DB tables).
    """
    if required_surface not in VALID_SURFACES:
        return None
    conns = bc.supabase.table("connections").select("*") \
        .eq("workspace_id", workspace_id).eq("provider", "google_drive") \
        .eq("status", "active").execute().data or []
    for conn in conns:
        enabled = (conn.get("config") or {}).get("enabled_surfaces", [])
        if required_surface in enabled:
            return conn
    return None


# ── Drive API helpers (retained — reference resolution needs single-file
#    lookups even though bulk folder polling is neutralized) ───────────────────

def _drive_get(path: str, token: str, params: dict = None) -> dict:
    res = httpx.get(f"https://www.googleapis.com/drive/v3/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or {}, timeout=30)
    if res.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Google Drive API error: {res.text[:300]}")
    return res.json()


# Matches drive.google.com/file/d/{id}, docs.google.com/document|spreadsheets|
# presentation/d/{id}, and a bare drive.google.com/open?id={id} form.
_DRIVE_LINK_RE = re.compile(
    r"https?://(?:drive|docs)\.google\.com/(?:file/d/|document/d/|spreadsheets/d/"
    r"|presentation/d/|open\?id=)([a-zA-Z0-9_-]{10,})"
)


def extract_drive_file_ids(text: str) -> list[str]:
    """De-duped Drive file IDs found in arbitrary text (a Meet transcript or
    Chat message body) — the trigger for reference resolution. Order-stable,
    no dedup-losing set() so a caller can log "found N links" meaningfully."""
    return list(dict.fromkeys(_DRIVE_LINK_RE.findall(text)))


def resolve_drive_reference(workspace_id: str, file_id: str,
                            linked_object_type: str, linked_object_id: str) -> Optional[dict]:
    """
    THE ONLY Drive read path left in this codebase that touches a file's
    content-adjacent metadata, and even this never reads the file's bytes —
    single-file GET by ID, title/URL/modifiedTime only. Creates (or, on a
    repeat resolution of the same file for the same linked object, reuses —
    see the UNIQUE constraint on external_references) exactly one reference
    row. No knowledge_items row, no Storage write, ever.

    Returns the reference dict, or None if no active Drive-enabled
    connection exists for this workspace, or the file isn't reachable
    (deleted, no permission) — logged, never raised, since a broken Drive
    link inside an otherwise-good Meet/Chat note must not cost the note.
    """
    conn = get_active_connection(workspace_id, "drive")
    if not conn:
        print(f"[google] no active Drive-enabled connection for workspace {workspace_id}, skipping reference")
        return None
    token = _valid_access_token(conn)
    if not token:
        return None
    try:
        meta = _drive_get(f"files/{file_id}", token,
                          {"fields": "id,name,webViewLink,modifiedTime"})
    except Exception as e:
        print(f"[google] could not resolve Drive reference {file_id}: {e}")
        return None

    row = {
        "workspace_id":       workspace_id,
        "provider":           "google_drive",
        "external_file_id":   file_id,
        "title":              meta.get("name"),
        "url":                meta.get("webViewLink"),
        "modified_time":      meta.get("modifiedTime"),
        "linked_object_type": linked_object_type,
        "linked_object_id":   linked_object_id,
    }
    res = bc.supabase.table("external_references").upsert(
        row, on_conflict="linked_object_type,linked_object_id,provider,external_file_id"
    ).execute()
    return res.data[0] if res.data else row


def resolve_drive_references_in_text(workspace_id: str, text: str,
                                     linked_object_type: str, linked_object_id: str) -> list[dict]:
    """Convenience wrapper: scans text for Drive links and resolves each one
    found. Used by connector_google_meet.py / connector_google_chat.py right
    after a note is created, on the note's own body/transcript text."""
    refs = []
    for file_id in extract_drive_file_ids(text):
        ref = resolve_drive_reference(workspace_id, file_id, linked_object_type, linked_object_id)
        if ref:
            refs.append(ref)
    return refs
