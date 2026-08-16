"""
Narrow app-DB write path for Google Drive canonical ingestion.

CURRENT STATUS (Google Workspace scope lock, 2026-08-15): the bulk Drive
ingestion flow that used to call this module (connector_google.sync_connection
et al.) has been NEUTRALIZED per a locked product decision -- KNOVA must not
become a mirror of Google Drive. Nothing in this codebase currently calls
upsert_knowledge_item/soft_delete_knowledge_item/upload_original_file. This
module and its two SECURITY DEFINER RPCs are left in place, fully tested and
live-verified, in case a future explicit product decision authorizes
importing an individual Drive file as a real KNOVA document (as opposed to
the reference-only model connector_google.resolve_drive_reference() now
implements, which never creates a knowledge_items row or Storage copy).

WHY THIS EXISTS (historical, still accurate for if/when it's reused again)
----------------------------------------------------------------------------
knowledge_items lives in the APP database project (a different Supabase
project from SUPABASE_URL/SUPABASE_SERVICE_KEY, which point at the vector DB
-- see auth.py's docstring). Every other write this service makes to the app
DB has always gone through the frontend instead, because this service never
held an app-DB credential capable of writing there. Railway holds a NEW
secret, APP_SUPABASE_SERVICE_KEY, scoped to the app DB project, specifically
so a caller like this (had bulk Drive ingestion remained active) could create/
update knowledge_items without a frontend session in the loop.

THIS MODULE IS THE ONLY PLACE THAT SECRET IS USED. It calls exactly two
SECURITY DEFINER Postgres RPCs (drive_sync_upsert_knowledge_item,
drive_sync_soft_delete_knowledge_item) -- both REVOKEd from PUBLIC/anon/
authenticated and GRANTed to service_role only (verified live against the
real database when this migration was applied). It never issues a raw
table read/write against the app DB. Nothing else in this codebase should
import APP_SUPABASE_SERVICE_KEY directly -- go through the functions here.

SECURITY NOTE ON CONNECTION OWNERSHIP: `connections` (the table that maps a
Drive connection to its owning workspace) lives in the VECTOR DB project, not
this one -- confirmed live, cross-project Postgres joins are not possible.
So these RPCs take workspace_id directly rather than connection_id, already
resolved by the caller's own connection lookup (whatever that caller turns
out to be, if this is ever reused). The RPCs cannot independently re-verify
that resolution -- the trust anchor is the caller's own connection lookup,
same as it already is for workspace_id on every document_chunks write today.
"""
import os
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

APP_SUPABASE_URL = (os.getenv("APP_SUPABASE_URL") or "").rstrip("/")
APP_SUPABASE_SERVICE_KEY = os.getenv("APP_SUPABASE_SERVICE_KEY") or ""

STORAGE_BUCKET = "knowledge-files"


def _configured() -> bool:
    return bool(APP_SUPABASE_URL and APP_SUPABASE_SERVICE_KEY)


def _headers() -> dict:
    # Never log/print APP_SUPABASE_SERVICE_KEY -- only ever placed in these
    # request headers, which httpx does not log.
    return {
        "apikey": APP_SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {APP_SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def upsert_knowledge_item(
    workspace_id: str, document_id: str, title: str, file_name: str,
    file_size: Optional[int], mime_type: Optional[str], file_type: str,
    storage_path: Optional[str],
    sensitivity: Optional[str] = None, authority: Optional[str] = None,
    doc_class: Optional[str] = None, lifecycle_status: Optional[str] = None,
) -> dict:
    """
    Calls drive_sync_upsert_knowledge_item. Returns the resulting
    knowledge_items row (including its EFFECTIVE sensitivity after the
    RPC's own raise-only rule is applied) -- callers use the returned
    sensitivity as the source of truth, not the value they proposed.
    """
    if not _configured():
        raise RuntimeError(
            "APP_SUPABASE_URL / APP_SUPABASE_SERVICE_KEY are not set. "
            "Google Drive sync cannot write knowledge_items without them."
        )
    res = httpx.post(
        f"{APP_SUPABASE_URL}/rest/v1/rpc/drive_sync_upsert_knowledge_item",
        headers=_headers(),
        json={
            "p_workspace_id":     workspace_id,
            "p_document_id":      document_id,
            "p_title":            title,
            "p_file_name":        file_name,
            "p_file_size":        file_size,
            "p_mime_type":        mime_type,
            "p_file_type":        file_type,
            "p_storage_path":     storage_path,
            "p_sensitivity":      sensitivity,
            "p_authority":        authority,
            "p_doc_class":        doc_class,
            "p_lifecycle_status": lifecycle_status,
        },
        timeout=30,
    )
    if res.status_code >= 400:
        raise RuntimeError(f"drive_sync_upsert_knowledge_item failed ({res.status_code}): {res.text[:500]}")
    data = res.json()
    # PostgREST returns a single-row RPC result as a bare object (RETURNS a
    # single composite type), not wrapped in a list.
    return data


def soft_delete_knowledge_item(workspace_id: str, document_id: str) -> None:
    """Calls drive_sync_soft_delete_knowledge_item. Idempotent -- a document
    already deleted (or never created) is a safe no-op on the SQL side."""
    if not _configured():
        raise RuntimeError(
            "APP_SUPABASE_URL / APP_SUPABASE_SERVICE_KEY are not set. "
            "Google Drive sync cannot soft-delete knowledge_items without them."
        )
    res = httpx.post(
        f"{APP_SUPABASE_URL}/rest/v1/rpc/drive_sync_soft_delete_knowledge_item",
        headers=_headers(),
        json={"p_workspace_id": workspace_id, "p_document_id": document_id},
        timeout=30,
    )
    if res.status_code >= 400:
        raise RuntimeError(f"drive_sync_soft_delete_knowledge_item failed ({res.status_code}): {res.text[:500]}")


def upload_original_file(workspace_id: str, document_id: str, file_name: str,
                         file_bytes: bytes, mime_type: Optional[str]) -> str:
    """
    Persists the Drive file's original bytes into the same Storage bucket
    manual uploads use, so storage_path on the knowledge_items row is real
    and every existing Library preview/download consumer keeps working
    unmodified.

    REQUIRED, not best-effort: a Drive document only counts as a first-class
    KNOVA document if it's actually previewable/downloadable through the
    existing Library UI (the whole point of Canonical Ingestion). Raises on
    any failure -- the caller (connector_google.sync_connection) calls this
    BEFORE creating any knowledge_items row or writing any chunks for the
    file, so a raise here means nothing was created for this file at all:
    no orphaned knowledge_items row that can't be previewed, no retrieval
    chunks representing an incomplete document, and ingest_items is never
    touched (the file was never marked 'embedded'), so the next scheduled
    sync pass retries it automatically with no special recovery needed.
    Returns the storage_path on success.
    """
    if not _configured():
        raise RuntimeError(
            "APP_SUPABASE_URL / APP_SUPABASE_SERVICE_KEY are not set. "
            "Google Drive sync cannot persist the original file without them."
        )
    safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in file_name)
    path = f"drive-sync/{workspace_id}/{document_id}_{safe_name}"
    try:
        res = httpx.post(
            f"{APP_SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}",
            headers={
                "apikey": APP_SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {APP_SUPABASE_SERVICE_KEY}",
                "Content-Type": mime_type or "application/octet-stream",
                "x-upsert": "true",
            },
            content=file_bytes,
            timeout=120,
        )
        if res.status_code >= 400:
            raise RuntimeError(f"Storage upload failed ({res.status_code}) for {document_id}: {res.text[:300]}")
        return path
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Storage upload threw for {document_id}: {e}") from e
