"""
Google Meet connector — transcript capture as durable KNOVA evidence (Google
Workspace scope lock: Meet's transcript retention on Google's side is
temporary, so KNOVA captures the complete transcript once available and
preserves it independently, the same "durable the moment it's embedded"
principle connector_zoom.py already established for Zoom recordings).

Flow: poll_connection() (worker.py, scheduled, per Meet-enabled connection)
  → conferenceRecords.list (recent conferences)
  → transcripts.list per conference record
  → transcripts.entries.list per transcript → assemble "Speaker: text" lines,
    the same shape brain_connectors.distill_meeting_transcript() already
    expects (see connector_zoom.parse_vtt_transcript's comment)
  → distill_meeting_transcript() → create_note_and_embed() (tier 2, same as
    Zoom) — DIRECT REUSE of both functions, zero changes.
  → resolve_drive_reference() on the transcript text, for any Drive links
    the meeting participants mentioned (e.g. "the doc is linked in chat").

Provenance: knowledge_note_sources' existing generic columns are reused
rather than adding new ones (channel_id/message_ts/thread_ts/source_ref/
occurred_at are all already nullable text/timestamp columns) --
  channel_id  -> conference record ID (groups all entries under one meeting)
  thread_ts   -> transcript ID (groups entries under one transcript session)
  message_ts  -> transcript entry ID (the individual speaker turn)
  source_ref  -> the entry's own resource name (Meet's own stable identifier)
  occurred_at -> the entry's startTime
This keeps Meet's provenance queryable with the exact same shape Slack/Chat
sources already use, at zero schema cost.

Do NOT download unrelated Meet artifacts (recordings, chat-in-meeting
exports) -- only conferenceRecords.transcripts.* is ever called here.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx

import brain_connectors as bc
import connector_google as google

MEET_API_BASE = "https://meet.googleapis.com/v2"
LOOKBACK = timedelta(days=2)  # how far back to look for new conference records each poll


def _meet_get(path: str, token: str, params: dict = None) -> dict:
    res = httpx.get(f"{MEET_API_BASE}/{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params or {}, timeout=30)
    if res.status_code != 200:
        raise Exception(f"Google Meet API error ({res.status_code}): {res.text[:300]}")
    return res.json()


def _list_conference_records(token: str) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - LOOKBACK).isoformat()
    records, page_token = [], None
    while True:
        params = {"filter": f"start_time > \"{cutoff}\"", "pageSize": 50}
        if page_token:
            params["pageToken"] = page_token
        data = _meet_get("conferenceRecords", token, params)
        records.extend(data.get("conferenceRecords", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return records


def _list_transcripts(token: str, conference_record_name: str) -> list[dict]:
    data = _meet_get(f"{conference_record_name}/transcripts", token)
    return data.get("transcripts", [])


def _list_transcript_entries(token: str, transcript_name: str) -> list[dict]:
    entries, page_token = [], None
    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        data = _meet_get(f"{transcript_name}/entries", token, params)
        entries.extend(data.get("transcriptEntries", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return entries


def _assemble_transcript(entries: list[dict]) -> str:
    """"Speaker: text" lines, oldest first -- the shape distill_meeting_transcript()
    expects (matches connector_zoom.parse_vtt_transcript's output shape)."""
    lines = []
    for e in sorted(entries, key=lambda e: e.get("startTime", "")):
        speaker = e.get("participant", "Unknown speaker")
        text = (e.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _process_one_conference(conn: dict, token: str, workspace_id: str, record: dict) -> bool:
    """One conference record's worth of work: fetch its transcript(s), distill,
    embed, provenance, Drive-reference scan. Returns True if a note was created."""
    conference_name = record.get("name")  # e.g. "conferenceRecords/abc123"
    transcripts = _list_transcripts(token, conference_name)
    created_any = False

    for transcript in transcripts:
        transcript_name = transcript.get("name")
        external_id = transcript_name  # globally unique, stable per transcript session

        existing = bc.supabase.table("ingest_items").select("id, status") \
            .eq("connection_id", conn["id"]).eq("external_id", external_id).execute().data
        if existing and existing[0]["status"] in ("noted", "discarded"):
            continue  # already processed this transcript

        entries = _list_transcript_entries(token, transcript_name)
        transcript_text = _assemble_transcript(entries)

        meeting_title = record.get("name", "Google Meet")  # Meet API doesn't expose a
        # human title on the conference record itself (unlike Calendar's `summary`) --
        # a future improvement could cross-reference calendar_events by conference_id
        # to get the real meeting title; using the record name is an honest fallback,
        # not a fabricated one.
        note = bc.distill_meeting_transcript(transcript_text, meeting_title, workspace_id=workspace_id)

        status = "discarded"
        note_id = None
        if note:
            sources = [{
                "channel_id":  conference_name,
                "thread_ts":   transcript_name,
                "message_ts":  e.get("name"),
                "source_ref":  e.get("name"),
                "occurred_at": e.get("startTime"),
            } for e in entries if (e.get("text") or "").strip()]

            note_id = bc.create_note_and_embed(
                workspace_id, conn["id"], "google_meet", note,
                source_type="meeting", source_tier=2,
                occurred_at=record.get("startTime"),
                sources=sources or None,
            )
            google.resolve_drive_references_in_text(
                workspace_id, transcript_text, "knowledge_note", note_id,
                connection_id=conn["id"],
            )
            status = "noted"
            created_any = True

        bc.supabase.table("ingest_items").upsert({
            "workspace_id": workspace_id, "connection_id": conn["id"], "provider": "google_meet",
            "external_id": external_id, "kind": "meet_transcript", "status": status, "note_id": note_id,
            "raw": {"conference_record": conference_name, "transcript": transcript_name},
        }, on_conflict="connection_id,external_id").execute()

    return created_any


def poll_connection(connection_id: str, workspace_id: str) -> dict:
    """The main job — see module docstring. Called by worker.py per active,
    Meet-enabled connection."""
    conn = bc.supabase.table("connections").select("*").eq("id", connection_id).execute().data
    if not conn:
        raise Exception("Connection not found.")
    conn = conn[0]
    if conn.get("provider") != "google_drive" or conn.get("status") != "active":
        raise Exception("Connection is not an active Google connection.")
    if "meet" not in (conn.get("config") or {}).get("enabled_surfaces", []):
        raise Exception("Meet is not enabled for this connection.")

    token = google._valid_access_token(conn)
    if not token:
        raise Exception("Google connection needs to be reconnected.")

    records = _list_conference_records(token)
    processed = failed = notes_created = 0

    for record in records:
        try:
            if _process_one_conference(conn, token, workspace_id, record):
                notes_created += 1
            processed += 1
        except Exception as e:
            failed += 1
            print(f"[google_meet] failed to process conference {record.get('name')}: {e}")

    return {"conferences_seen": len(records), "processed": processed,
            "notes_created": notes_created, "failed": failed}
