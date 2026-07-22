"""
Company Brain — connector framework (Phase 2).

The shared layer under every integration (Slack first, then Google/Zoom):
  - token encryption (Fernet)
  - raw-item capture into ingest_items (dedup)
  - THE FILTRATION ENGINE: the GBrain "signal detector" equivalent — batches
    raw messages into conversations, asks the LLM which contain durable company
    knowledge (vs. noise), and DISTILLS keepers into clean knowledge_notes.
    Only distilled notes get embedded — raw chat logs never pollute the brain.
  - note → document_chunks pipeline (tier-3 by default, so official docs still
    outrank chat in hybrid search)
  - generic REST routes for the frontend (list connections, list/delete notes,
    trigger a sync)

Provider-specific code (OAuth, message fetching) lives in connector_*.py files.
Railway owns all of this end to end — no Lovable DB access required.
"""
import os
import json
import threading
import ai
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

from ingest import chunk_text, embed_chunks

load_dotenv()

router = APIRouter()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# ── Token encryption ────────────────────────────────────────────────────────────
# CONNECTOR_ENCRYPTION_KEY is a Fernet key (generate once:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# ) stored in Railway env. Without it we refuse to store OAuth tokens — never
# persist third-party access tokens in plaintext.
_FERNET = None
def _fernet():
    global _FERNET
    if _FERNET is None:
        from cryptography.fernet import Fernet
        key = os.getenv("CONNECTOR_ENCRYPTION_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="CONNECTOR_ENCRYPTION_KEY is not set — cannot securely store connector tokens.",
            )
        _FERNET = Fernet(key.encode() if isinstance(key, str) else key)
    return _FERNET


def encrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    if not value:
        return ""
    return _fernet().decrypt(value.encode()).decode()


# ── In-memory sync job status (same pattern as ingest) ──────────────────────────
SYNC_JOBS: dict[str, dict] = {}


# ── Raw item capture ────────────────────────────────────────────────────────────

def save_ingest_items(workspace_id: str, connection_id: str, provider: str,
                      items: list[dict]) -> int:
    """
    Inserts normalized raw items, skipping duplicates (unique on
    connection_id+external_id). Returns how many NEW items were stored.
    items: [{external_id, kind, raw}]
    """
    if not items:
        return 0
    rows = [{
        "workspace_id":  workspace_id,
        "connection_id": connection_id,
        "provider":      provider,
        "external_id":   it["external_id"],
        "kind":          it.get("kind", "message"),
        "raw":           it["raw"],
        "status":        "pending",
    } for it in items]
    stored = 0
    # upsert with ignore-duplicates so re-running backfill is safe
    for i in range(0, len(rows), 200):
        batch = rows[i:i + 200]
        try:
            res = supabase.table("ingest_items").upsert(
                batch, on_conflict="connection_id,external_id", ignore_duplicates=True
            ).execute()
            stored += len(res.data or [])
        except Exception as e:
            print(f"[connectors] item upsert error: {e}")
    return stored


# ── THE FILTRATION ENGINE ───────────────────────────────────────────────────────

def batch_conversations(items: list[dict]) -> list[list[dict]]:
    """
    Groups raw message items into conversation units for classification:
      - messages sharing a thread_ts stay together (a thread = one topic)
      - remaining standalone messages in a channel are grouped into rolling
        windows of up to 12 messages
    Each returned batch is a list of the original item dicts (with .raw).
    """
    threads: dict[str, list[dict]] = {}
    loose:   dict[str, list[dict]] = {}   # keyed by channel

    for it in items:
        raw = it.get("raw", {})
        thread_ts = raw.get("thread_ts")
        channel   = raw.get("channel", "unknown")
        if thread_ts:
            threads.setdefault(f"{channel}:{thread_ts}", []).append(it)
        else:
            loose.setdefault(channel, []).append(it)

    batches: list[list[dict]] = [v for v in threads.values() if v]
    for channel, msgs in loose.items():
        msgs.sort(key=lambda m: m.get("raw", {}).get("ts", ""))
        for i in range(0, len(msgs), 12):
            batches.append(msgs[i:i + 12])
    return batches


def _format_batch(batch: list[dict]) -> tuple[str, str]:
    """Renders a batch as a readable transcript; returns (transcript, channel)."""
    lines, channel = [], "unknown"
    for it in batch:
        raw = it.get("raw", {})
        channel = raw.get("channel_name") or raw.get("channel", channel)
        who = raw.get("user_name") or raw.get("user", "someone")
        text = (raw.get("text") or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines), channel


CLASSIFY_SYSTEM = """You are the filter that decides what enters a company's permanent knowledge base.
You will see a short workplace conversation. Decide if it contains DURABLE, REUSABLE company \
knowledge that a colleague might search for later — a decision made, a process or how-to, an \
announcement, a factual answer, or a policy. Casual chatter, greetings, logistics ("running 5 min \
late"), reactions, and banter are NOISE and must be discarded.

If it is worth keeping, rewrite it as a clean, standalone knowledge note written in the third person \
as settled fact — NOT "someone said". Include the concrete substance (numbers, names, decisions).

Respond ONLY with valid JSON, no markdown fences:
{
  "worth_keeping": true,
  "category": "decision" | "process" | "announcement" | "fact" | "qa",
  "title": "concise, searchable title",
  "note": "1-4 sentences of standalone knowledge",
  "participants": ["first names of key people involved"]
}
If it is noise: {"worth_keeping": false}"""


def classify_batch(transcript: str, channel: str) -> Optional[dict]:
    """Runs one conversation through the LLM filter. Returns a note dict or
    None (noise / empty / unparseable — fail safe by discarding)."""
    if not transcript.strip():
        return None
    try:
        verdict = ai.chat_json(
            messages=[{"role": "user",
                       "content": f"Channel: #{channel}\n\nConversation:\n{transcript}"}],
            system=CLASSIFY_SYSTEM, max_tokens=500, temperature=0.2,
        )
    except Exception as e:
        print(f"[filtration] classify failed (discarding): {e}")
        return None
    if not isinstance(verdict, dict) or not verdict.get("worth_keeping"):
        return None
    if not verdict.get("note") or not verdict.get("title"):
        return None
    return {
        "category":     verdict.get("category", "fact"),
        "title":        str(verdict["title"])[:200],
        "body":         str(verdict["note"]),
        "participants": [str(p) for p in (verdict.get("participants") or [])][:10],
    }


def create_note_and_embed(workspace_id: str, connection_id: Optional[str], provider: str,
                          note: dict, source_type: str = "slack", source_tier: int = 3,
                          source_ref: str = None, occurred_at: str = None) -> str:
    """
    Inserts a knowledge_note and embeds its body into document_chunks
    (document_id = note id, so its chunks are searchable via hybrid search
    and deletable together). Returns the note id.
    """
    note_row = {
        "workspace_id":  workspace_id,
        "connection_id": connection_id,
        "provider":      provider,
        "source_type":   source_type,
        "source_tier":   source_tier,
        "category":      note.get("category"),
        "title":         note["title"],
        "body":          note["body"],
        "participants":  note.get("participants", []),
        "source_ref":    source_ref,
        "occurred_at":   occurred_at,
    }
    res = supabase.table("knowledge_notes").insert(note_row).execute()
    note_id = res.data[0]["id"]

    # Embed the note body into the searchable brain (tier 3 by default)
    full_text = f"{note['title']}\n\n{note['body']}"
    chunks = chunk_text(full_text) or [full_text]
    embeddings = embed_chunks(chunks)
    rows = [{
        "document_id":  note_id,
        "asset_id":     note_id,
        "workspace_id": workspace_id,
        "content":      chunks[i],
        "embedding":    embeddings[i],
        "chunk_index":  i,
        "source_type":  source_type,
        "source_tier":  source_tier,
        "doc_date":     occurred_at,
        "metadata": {
            "file_name":    note["title"],
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "workspace_id": workspace_id,
            "source_type":  source_type,
            "note_id":      note_id,
        },
    } for i in range(len(chunks))]
    supabase.table("document_chunks").insert(rows).execute()
    return note_id


def run_filtration(workspace_id: str, connection_id: str, provider: str,
                   job: Optional[dict] = None) -> dict:
    """
    Processes all pending ingest_items for a connection:
    batch → classify → distill keepers into notes → mark items.
    This is the step that turns raw chat into curated company knowledge.
    """
    pending = supabase.table("ingest_items").select("*") \
        .eq("connection_id", connection_id).eq("status", "pending") \
        .limit(2000).execute().data or []

    if job is not None:
        job["items_pending"] = len(pending)

    batches = batch_conversations(pending)
    notes_created = 0
    discarded = 0

    for bi, batch in enumerate(batches):
        transcript, channel = _format_batch(batch)
        note = classify_batch(transcript, channel)
        item_ids = [it["id"] for it in batch]

        if note:
            first_raw = batch[0].get("raw", {})
            note_id = create_note_and_embed(
                workspace_id, connection_id, provider, note,
                source_type=provider if provider != "google_drive" else "document",
                source_tier=3 if provider == "slack" else 2,
                source_ref=first_raw.get("permalink"),
                occurred_at=first_raw.get("iso_ts"),
            )
            supabase.table("ingest_items").update(
                {"status": "noted", "note_id": note_id}
            ).in_("id", item_ids).execute()
            notes_created += 1
        else:
            supabase.table("ingest_items").update(
                {"status": "discarded"}
            ).in_("id", item_ids).execute()
            discarded += len(item_ids)

        if job is not None:
            job["batches_done"] = bi + 1
            job["notes_created"] = notes_created

    return {"batches": len(batches), "notes_created": notes_created, "items_discarded": discarded}


def delete_note(note_id: str) -> None:
    """Removes a note and every chunk it produced (chunks share document_id = note id)."""
    supabase.table("document_chunks").delete().eq("document_id", note_id).execute()
    supabase.table("knowledge_notes").delete().eq("id", note_id).execute()


# ── Generic REST routes (frontend drives these; Railway is the API) ─────────────

@router.get("/connections")
async def list_connections(workspace_id: str):
    """All external connections for a workspace (status shown in Settings)."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    rows = supabase.table("connections").select(
        "id, provider, external_team_name, status, error_detail, config, connected_by, created_at"
    ).eq("workspace_id", workspace_id).execute().data or []
    return {"connections": rows}


@router.delete("/connections/{connection_id}")
async def disconnect(connection_id: str, delete_notes: bool = False):
    """Revokes a connection. Optionally deletes all knowledge notes it produced."""
    if delete_notes:
        notes = supabase.table("knowledge_notes").select("id") \
            .eq("connection_id", connection_id).execute().data or []
        for n in notes:
            delete_note(n["id"])
    supabase.table("connections").update({"status": "revoked"}).eq("id", connection_id).execute()
    return {"success": True, "notes_deleted": delete_notes}


@router.get("/knowledge-notes")
async def list_knowledge_notes(workspace_id: str, limit: int = 100):
    """Distilled notes captured from integrations — shown in Library."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    rows = supabase.table("knowledge_notes").select(
        "id, provider, source_type, category, title, body, participants, source_ref, occurred_at, created_at"
    ).eq("workspace_id", workspace_id).eq("status", "active") \
        .order("created_at", desc=True).limit(limit).execute().data or []
    return {"notes": rows}


@router.delete("/knowledge-notes/{note_id}")
async def delete_knowledge_note(note_id: str):
    """Deletes a note and its chunks (admin curation)."""
    delete_note(note_id)
    return {"success": True}


class SyncRequest(BaseModel):
    connection_id: str


@router.post("/connectors/sync")
async def trigger_filtration(body: SyncRequest):
    """
    Runs filtration over any pending captured items for a connection
    (in the background). Provider-specific FETCH (pulling new messages into
    ingest_items) is triggered by the provider's own sync route or webhook;
    this endpoint distills whatever is already captured.
    """
    conn = supabase.table("connections").select("*").eq("id", body.connection_id).execute().data
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found.")
    conn = conn[0]

    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    SYNC_JOBS[job_id] = {"job_id": job_id, "connection_id": body.connection_id,
                         "status": "processing", "notes_created": 0, "batches_done": 0}

    def _work():
        try:
            result = run_filtration(conn["workspace_id"], conn["id"], conn["provider"],
                                    job=SYNC_JOBS[job_id])
            SYNC_JOBS[job_id].update({"status": "completed", **result})
        except Exception as e:
            import traceback; print(f"[connectors] filtration job failed: {e}"); print(traceback.format_exc())
            SYNC_JOBS[job_id].update({"status": "failed", "error": str(e)})

    threading.Thread(target=_work, daemon=True).start()
    return {"success": True, "job_id": job_id, "status": "processing"}


@router.get("/connectors/sync-status/{job_id}")
async def sync_status(job_id: str):
    job = SYNC_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job
