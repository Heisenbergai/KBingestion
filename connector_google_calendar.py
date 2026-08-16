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
  knowledge_items-adjacent write).

Dedup/change-detection: unique on (connection_id, external_event_id), and an
event whose Google-side `updated` timestamp hasn't advanced since the last
poll is skipped — same "re-process only if source changed" shape Drive's
now-neutralized bulk poll used, applied here to metadata rows instead of
document chunks.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx

import brain_connectors as bc
import connector_google as google

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

    for event in events:
        external_id = event.get("id")
        if not external_id or event.get("status") == "cancelled":
            skipped += 1
            continue

        existing = bc.supabase.table("calendar_events").select("id, updated_at_source") \
            .eq("connection_id", connection_id).eq("external_event_id", external_id).execute().data
        already = existing[0] if existing else None
        source_updated = event.get("updated")
        if already and already.get("updated_at_source") == source_updated:
            continue  # unchanged since last poll

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

        row = {
            "workspace_id": workspace_id, "connection_id": connection_id,
            "external_event_id": external_id, "title": event.get("summary"),
            "start_time": start, "end_time": end, "organizer": organizer,
            "attendees": attendees, "recurrence_rule": ",".join(event.get("recurrence", [])) or None,
            "meeting_url": meeting_url, "conference_id": conference_id,
            "updated_at_source": source_updated,
        }
        bc.supabase.table("calendar_events").upsert(
            row, on_conflict="connection_id,external_event_id"
        ).execute()
        processed += 1

    return {"events_seen": len(events), "processed": processed, "skipped": skipped}
