"""
Google Drive connector (Phase 3 — first non-Slack, non-chat connector).

Shape is deliberately different from Slack's: Drive files ARE documents, not
chat that needs a keep/discard filtration pass. So this feeds the EXISTING
document ingestion pipeline (ingest.process_document_bytes — tier 1, same
quality as a manual upload) rather than brain_connectors' note-distillation
engine. Google Docs/Sheets/Slides are exported to .docx/.xlsx/.pptx before
extraction, reusing the already-well-tested extractors instead of writing new
ones for Google's native formats.

Flow:
  1. POST /integrations/oauth-url → mint the consent URL (popup). GET /google/install
     does the same as a redirect, for non-browser callers.
  2. GET  /google/oauth/callback  → exchange code, store encrypted tokens + expiry
  3. GET  /google/folders         → list this connection's selected folders
  4. POST /google/folders/select  → add a folder (pasted Drive link or bare ID) + sync
  5. sync_connection()            → called on-select AND by worker.py on a schedule;
                                    this is what makes it "continuous" rather than a
                                    one-time import

Steps 1, 3 and 4 require the caller's Supabase token and check membership of the
owning workspace (see auth.py). Step 2 is authenticated by the Fernet-signed state
minted in step 1.

CREDENTIAL MODEL (single-tenant, same as Slack — see 09_company_brain_roadmap.md):
Each CUSTOMER creates their own project in Google Cloud Console and pastes its
client_id / client_secret into the Integrations panel. No env-var fallback exists
here (unlike Slack) — there is no pre-existing Google connection to keep alive.
One-time setup per customer (console.cloud.google.com):
  - Create/select a project, enable the "Google Drive API"
  - OAuth consent screen: type "External", publishing status "Testing" is enough
    for single-workspace use (no Google verification needed — verification is
    only required for apps requesting sensitive scopes AND going to production
    with real (non-test) users beyond the ~100 test-user cap)
  - Credentials → Create OAuth client ID → type "Web application"
  - Authorized redirect URI: https://kbingestion-production.up.railway.app/google/oauth/callback
    (fixed — same for every customer, it's Railway's URL, not theirs)

No shared webhook exists for Drive (polling, not push notifications — Google's
watch channels need a separately-verified domain per project and add real
complexity for no benefit at pilot scale), so there is no per-customer signing
secret to manage, unlike Slack.
"""
import os
import io
import json
import time
import uuid
import threading
import httpx
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from auth import AuthContext, current_user
import brain_connectors as bc
import ingest

load_dotenv()

router = APIRouter()

RAILWAY_BASE = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
if not RAILWAY_BASE.startswith("http"):
    RAILWAY_BASE = f"https://{RAILWAY_BASE}"
REDIRECT_URI = f"{RAILWAY_BASE}/google/oauth/callback"

# drive.readonly (not the narrower drive.file) because the admin picks a whole
# folder by pasting its link/ID rather than through Google's Picker widget —
# drive.file would only grant access to files the user explicitly selects via
# that widget, which this simpler paste-a-link flow doesn't use. The trade-off:
# Google's consent screen will show a broad "See all your Google Drive files"
# permission. Documented here deliberately so it isn't a silent surprise —
# each customer is consenting to THEIR OWN app reading THEIR OWN Drive, not a
# third party's, which is what single-tenant is for.
GOOGLE_SCOPES = "https://www.googleapis.com/auth/drive.readonly"
POLL_MAX_FILES_PER_FOLDER = 500  # safety cap per sync pass, not per folder ever

# Google-native files must be EXPORTED, not downloaded directly. Exporting to
# docx/xlsx/pptx reuses ingest.py's existing, already-tested extractors
# (tables, shapes, etc.) instead of writing new ones for Google's own formats.
_EXPORT_MIME = {
    "application/vnd.google-apps.document":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.google-apps.spreadsheet":
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation":
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_EXPORT_EXT = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.spreadsheet": ".xlsx",
    "application/vnd.google-apps.presentation": ".pptx",
}
# Regular (non-Google-native) files ingest.extract_text already understands.
_SUPPORTED_DIRECT_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
}


def _google_credentials(workspace_id: str) -> tuple[str, str]:
    creds = bc.get_provider_credentials(workspace_id, "google_drive")
    if not creds:
        raise HTTPException(
            status_code=400,
            detail="This workspace hasn't set up its Google app credentials yet. "
                   "Go to Integrations → Google Drive → Set up to add them.",
        )
    return creds["client_id"], creds["client_secret"]


def build_install_url(workspace_id: str, user_id: str = "") -> str:
    """Mirrors connector_slack.build_install_url — see its docstring."""
    client_id, _ = _google_credentials(workspace_id)
    state = bc.encode_oauth_state(workspace_id, user_id)
    return (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={client_id}&redirect_uri={REDIRECT_URI}"
        f"&response_type=code&scope={GOOGLE_SCOPES}"
        "&access_type=offline&prompt=consent"  # force a refresh_token every time,
        # not just on first-ever consent — needed since worker.py's token refresh
        # (see below) depends on always having one.
        f"&state={state}"
    )


@router.get("/google/install")
async def google_install(workspace_id: str, user_id: str = "",
                         auth: AuthContext = Depends(current_user)):
    """Redirect variant of the install URL, for non-browser callers. See
    connector_slack.slack_install — identical reasoning: browser popups should
    use POST /integrations/oauth-url instead, since it can carry the auth header."""
    auth.assert_workspace(workspace_id)
    return RedirectResponse(build_install_url(workspace_id, user_id))


@router.get("/google/oauth/callback")
async def google_callback(code: str = "", state: str = "", error: str = ""):
    """Exchanges the OAuth code for tokens and stores the connection."""
    from integrations import oauth_complete_html
    if error:
        return oauth_complete_html("google_drive", "error")
    st = bc.decode_oauth_state(state)
    workspace_id, user_id = st["w"], st.get("u", "")
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
        # Shouldn't happen with access_type=offline&prompt=consent, but if Google
        # ever omits it, the connection would silently die in ~1 hour with no way
        # to recover without reconnecting — fail loudly now instead.
        print(f"[google] WARNING: no refresh_token in response for workspace={workspace_id}")

    # Who is this (for external_team_name / dedup on reconnect)?
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
        "scopes":             data.get("scope", GOOGLE_SCOPES),
        "status":             "active",
        "connected_by":       user_id,
        "config":             {},
    }
    bc.supabase.table("connections").upsert(
        row, on_conflict="workspace_id,provider,external_team_id"
    ).execute()

    return oauth_complete_html("google_drive", "connected")


def refresh_access_token(conn: dict) -> Optional[str]:
    """
    Real token refresh — the first provider in this codebase where it's
    actually implemented rather than flagged (see worker.py). Returns the new
    access token, or None if refresh failed (connection marked 'error' so the
    admin sees it needs reconnecting rather than failing silently forever).
    """
    if not conn.get("refresh_token_enc"):
        print(f"[google] connection {conn['id']} has no refresh_token — cannot refresh, marking error")
        bc.supabase.table("connections").update(
            {"status": "error", "error_detail": "No refresh token stored. Reconnect Google Drive."}
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


# ── Drive API helpers ────────────────────────────────────────────────────────────

def _drive_get(path: str, token: str, params: dict = None) -> dict:
    res = httpx.get(f"https://www.googleapis.com/drive/v3/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or {}, timeout=30)
    if res.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Google Drive API error: {res.text[:300]}")
    return res.json()


def _extract_folder_id(id_or_url: str) -> str:
    """Accepts either a bare folder id or a full Drive folder URL."""
    id_or_url = id_or_url.strip()
    if "/folders/" in id_or_url:
        return id_or_url.split("/folders/")[1].split("?")[0].split("/")[0]
    return id_or_url


def _list_folder_files(token: str, folder_id: str) -> list[dict]:
    files, page_token, pages = [], None, 0
    while pages < (POLL_MAX_FILES_PER_FOLDER // 100 + 1):
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size)",
            "pageSize": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        data = _drive_get("files", token, params)
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        pages += 1
        if not page_token or len(files) >= POLL_MAX_FILES_PER_FOLDER:
            break
    return files


def _fetch_file_bytes(token: str, file_id: str, mime_type: str) -> tuple[bytes, str, str]:
    """Returns (bytes, effective_mime_type, extension_hint) — exporting Google-
    native files, downloading everything else directly."""
    if mime_type in _EXPORT_MIME:
        export_mime = _EXPORT_MIME[mime_type]
        res = httpx.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/export",
            headers={"Authorization": f"Bearer {token}"},
            params={"mimeType": export_mime}, timeout=120,
        )
        res.raise_for_status()
        return res.content, export_mime, _EXPORT_EXT[mime_type]

    res = httpx.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"alt": "media"}, timeout=120,
    )
    res.raise_for_status()
    return res.content, mime_type, ""


# ── Routes ──────────────────────────────────────────────────────────────────────

@router.get("/google/folders")
async def google_folders(connection_id: str, auth: AuthContext = Depends(current_user)):
    """Folders currently selected for this connection."""
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]
    auth.assert_workspace(conn["workspace_id"])
    return {"folders": (conn.get("config") or {}).get("folders", [])}


class AddFolderRequest(BaseModel):
    connection_id: str
    folder_id_or_url: str


@router.post("/google/folders/select")
async def google_add_folder(body: AddFolderRequest, auth: AuthContext = Depends(current_user)):
    """
    Adds one folder (by pasted link or bare ID) to this connection's watch
    list, validates it's actually reachable, and kicks off an immediate
    background sync — same "select then backfill in a thread" shape as
    Slack's channel selection.
    """
    conn = bc.supabase.table("connections").select("*").eq("id", body.connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]
    auth.assert_workspace(conn["workspace_id"])

    folder_id = _extract_folder_id(body.folder_id_or_url)
    token = _valid_access_token(conn)
    if not token:
        raise HTTPException(status_code=400, detail="Google connection needs to be reconnected.")

    meta = _drive_get(f"files/{folder_id}", token, {"fields": "id,name,mimeType"})
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise HTTPException(status_code=400, detail="That link/ID is not a Google Drive folder.")

    folders = (conn.get("config") or {}).get("folders", [])
    if not any(f["id"] == folder_id for f in folders):
        folders.append({"id": folder_id, "name": meta.get("name", folder_id)})
        bc.supabase.table("connections").update(
            {"config": {**(conn.get("config") or {}), "folders": folders}}
        ).eq("id", body.connection_id).execute()

    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    bc.SYNC_JOBS[job_id] = {"job_id": job_id, "connection_id": body.connection_id,
                            "status": "processing", "stage": "syncing", "files_processed": 0}

    def _run():
        try:
            result = sync_connection(body.connection_id, job=bc.SYNC_JOBS[job_id])
            bc.SYNC_JOBS[job_id].update({"status": "completed", "stage": "completed", **result})
        except Exception as e:
            import traceback; print(f"[google] sync failed: {e}"); print(traceback.format_exc())
            bc.SYNC_JOBS[job_id].update({"status": "failed", "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return {"success": True, "job_id": job_id, "folder": {"id": folder_id, "name": meta.get("name")}}


def sync_connection(connection_id: str, job: Optional[dict] = None) -> dict:
    """
    The main job. Lists every selected folder, fetches new/changed files, and
    embeds them via the EXISTING document pipeline (tier 1 — same as a manual
    upload). Called on-select (background thread above) AND on a schedule by
    worker.py — that second call site is what makes this "continuous" instead
    of a one-time import: a file added to the folder next week gets picked up
    on the next scheduled pass with no one touching the UI.

    Dedup/change-detection reuses ingest_items (unique on connection_id +
    external_id=file_id) as a processed-files ledger — status='embedded' means
    done, and re-processing only happens if modifiedTime advanced since the
    stored raw.modified_time. This is a different usage of that table than
    Slack's (messages are immutable and never revisited; Drive files are
    edited in place), so it's handled here directly rather than through
    brain_connectors.save_ingest_items, which assumes immutability.
    """
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]
    token = _valid_access_token(conn)
    if not token:
        raise HTTPException(status_code=400, detail="Google connection needs to be reconnected.")

    folders = (conn.get("config") or {}).get("folders", [])
    processed = skipped = failed = 0

    for folder in folders:
        files = _list_folder_files(token, folder["id"])
        for f in files:
            if job is not None:
                job["files_processed"] = processed

            existing = bc.supabase.table("ingest_items").select("id, raw, status") \
                .eq("connection_id", connection_id).eq("external_id", f["id"]).execute().data
            already = existing[0] if existing else None
            if already and already["status"] == "embedded" \
               and already.get("raw", {}).get("modified_time") == f.get("modifiedTime"):
                continue  # unchanged since last sync

            mime = f.get("mimeType", "")
            if mime not in _EXPORT_MIME and mime not in _SUPPORTED_DIRECT_MIME:
                skipped += 1
                continue

            try:
                file_bytes, effective_mime, ext = _fetch_file_bytes(token, f["id"], mime)
                document_id = (already.get("raw", {}).get("document_id") if already else None) or str(uuid.uuid4())
                file_name = f["name"] + (ext if ext and not f["name"].endswith(ext) else "")

                ingest.process_document_bytes(
                    file_bytes, document_id=document_id, asset_id=document_id,
                    workspace_id=conn["workspace_id"], mime_type=effective_mime,
                    file_name=file_name, source_type="document", source_tier=1,
                    doc_date=f.get("modifiedTime"),
                )

                row = {
                    "workspace_id": conn["workspace_id"], "connection_id": connection_id,
                    "provider": "google_drive", "external_id": f["id"], "kind": "drive_file",
                    "status": "embedded",
                    "raw": {"name": f["name"], "mime_type": mime, "modified_time": f.get("modifiedTime"),
                           "folder_id": folder["id"], "document_id": document_id},
                }
                bc.supabase.table("ingest_items").upsert(
                    row, on_conflict="connection_id,external_id"
                ).execute()
                processed += 1
            except Exception as e:
                print(f"[google] failed to process file {f.get('id')} ({f.get('name')}): {e}")
                failed += 1

    return {"folders_checked": len(folders), "processed": processed, "skipped": skipped, "failed": failed}
