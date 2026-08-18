"""
Phase 5E -- immutable Calendar evidence layer.

CURRENT STATE (calendar_events, unchanged, F-72) is the mutable "latest
known state" projection connector_google_calendar.py already maintains.
IMMUTABLE SOURCE EVIDENCE (calendar_event_snapshots) is new: an append-only
history, one row per genuinely-changed observable state, written only when
compute_state_fingerprint() actually differs from the most recent existing
snapshot for that identity -- never on every poll.

Deliberately NOT wired into connector_google_calendar.py::poll_connection()
in this pass. That function's docstring/logic is untouched -- "do not
redesign calendar_events" is read here as "do not change that file's own
write path," and wiring maybe_create_snapshot() into the live poll loop is
a separate decision (does a snapshot-write failure block the poll? does it
run inline or async?) that this pass does not make. See the Phase 5E
report's "next implementation step" for this exact gap, named rather than
silently resolved.

Real, concrete finding from re-reading connector_google_calendar.py fresh
for this pass: `event.get("status") == "cancelled"` is currently just
skipped (`skipped += 1; continue`) -- calendar_events.deleted_at is a real
column with zero write path anywhere in this codebase today. Not fixed
here (that would be redesigning the connector, out of scope) -- named in
the Phase 5E report instead.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc


def _normalize_string(s: Optional[str]) -> Optional[str]:
    """Trim + collapse internal whitespace only. Deliberately preserves
    case -- title/organizer/meeting_url/recurrence_rule/conference_id are
    all human- or system-meaningful strings (a person's name in a title, a
    Meet URL slug) where lowercasing would destroy real information, not
    just cosmetic noise."""
    if s is None:
        return None
    collapsed = re.sub(r"\s+", " ", s.strip())
    return collapsed or None


def _normalize_timestamp(ts) -> Optional[str]:
    """Canonical UTC ISO-8601 -- equivalent instants (different offsets,
    trailing Z vs +00:00, a naive datetime assumed UTC) normalize
    identically. All-day events carry date-only strings (YYYY-MM-DD, per
    connector_google_calendar._event_start_end), which fromisoformat also
    parses correctly -- date-only values stay date-only after normalization
    since there's no time-of-day to convert."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    s = str(ts)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s  # date-only (all-day event) -- no timezone concept to normalize
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_attendees(attendees: Optional[list[dict]]) -> list[dict]:
    """Deduplicated by email (case-insensitive key; original casing of the
    FIRST occurrence is preserved in the stored value), sorted
    deterministically by that same lowercased key. Reordering the same real
    attendee list, or a duplicate entry Google's API occasionally returns,
    never changes the fingerprint -- but a genuine response_status change
    (a real RSVP) does, correctly, since that IS a meaningful observable
    state change, not noise."""
    seen: dict[str, dict] = {}
    for a in (attendees or []):
        email = (a.get("email") or "").strip()
        if not email:
            continue
        key = email.lower()
        if key not in seen:
            seen[key] = {"email": email, "response_status": a.get("response_status")}
    return [seen[k] for k in sorted(seen.keys())]


def compute_state_fingerprint(
    title: Optional[str] = None, start_time=None, end_time=None,
    organizer: Optional[str] = None, attendees: Optional[list[dict]] = None,
    recurrence_rule: Optional[str] = None, meeting_url: Optional[str] = None,
    conference_id: Optional[str] = None,
) -> tuple[str, dict]:
    """Deterministic SHA-256 of the normalized observable state -- same
    canonical-JSON pattern as structured_persistence.compute_primitive_
    fingerprint(). external_event_id is deliberately NOT part of the
    fingerprint input -- it's already a separate identity column
    (workspace_id, connection_id, external_event_id), not observable
    content being hashed, matching how structured_knowledge's own
    fingerprint excludes its (canonical_source_type, canonical_id) identity
    columns. updated_at_source and deleted_at are also excluded --
    Google's own change marker is already known to be an unreliable
    identity signal on its own (the same lesson the primitive-fingerprint
    non-determinism finding already taught this codebase), and deleted_at
    is a lifecycle flag, not observable Calendar content.

    Returns (fingerprint, normalized_dict) -- the normalized dict is what
    actually gets stored on the snapshot row, so the stored content and
    the hash that identifies it are always the exact same values, never a
    hash of one thing and a display of a subtly different raw one.
    """
    normalized = {
        "title": _normalize_string(title),
        "start_time": _normalize_timestamp(start_time),
        "end_time": _normalize_timestamp(end_time),
        "organizer": _normalize_string(organizer),
        "attendees": _normalize_attendees(attendees),
        "recurrence_rule": _normalize_string(recurrence_rule),
        "meeting_url": _normalize_string(meeting_url),
        "conference_id": _normalize_string(conference_id),
    }
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return fingerprint, normalized


def maybe_create_snapshot(
    workspace_id: str, connection_id: str, external_event_id: str,
    title: Optional[str] = None, start_time=None, end_time=None,
    organizer: Optional[str] = None, attendees: Optional[list[dict]] = None,
    recurrence_rule: Optional[str] = None, meeting_url: Optional[str] = None,
    conference_id: Optional[str] = None, observed_updated_at: Optional[str] = None,
) -> Optional[str]:
    """Writes a new immutable snapshot ONLY when the normalized observable
    state differs from the most recent existing snapshot for this exact
    (workspace_id, connection_id, external_event_id) identity. Returns the
    new snapshot's id, or None if nothing changed (no write performed at
    all -- this is the "do not snapshot every poll" contract).

    Idempotent two ways: the lookup-before-write above is the primary
    change-detection gate; the real composite UNIQUE constraint on
    calendar_event_snapshots is a defense-in-depth backstop against a
    genuine concurrent-write race, handled via upsert+ignore_duplicates,
    same pattern structured_persistence.persist_extracted_primitives()
    already uses.
    """
    fingerprint, normalized = compute_state_fingerprint(
        title, start_time, end_time, organizer, attendees,
        recurrence_rule, meeting_url, conference_id,
    )

    latest = bc.supabase.table("calendar_event_snapshots") \
        .select("id,state_fingerprint") \
        .eq("workspace_id", workspace_id).eq("connection_id", connection_id) \
        .eq("external_event_id", external_event_id) \
        .order("created_at", desc=True).limit(1).execute().data

    if latest and latest[0]["state_fingerprint"] == fingerprint:
        return None  # unchanged since the last snapshot -- no write, by design

    row = {
        "workspace_id": workspace_id,
        "connection_id": connection_id,
        "external_event_id": external_event_id,
        "observed_updated_at": observed_updated_at,
        "state_fingerprint": fingerprint,
        "title": normalized["title"],
        "start_time": normalized["start_time"],
        "end_time": normalized["end_time"],
        "organizer": normalized["organizer"],
        "attendees": normalized["attendees"],
        "recurrence_rule": normalized["recurrence_rule"],
        "meeting_url": normalized["meeting_url"],
        "conference_id": normalized["conference_id"],
    }
    inserted = bc.supabase.table("calendar_event_snapshots").upsert(
        row, on_conflict="workspace_id,connection_id,external_event_id,state_fingerprint",
        ignore_duplicates=True,
    ).execute().data

    if inserted:
        return inserted[0]["id"]

    # Race backstop: another caller wrote the identical fingerprint between
    # our lookup and our insert. Re-fetch rather than assume which id won.
    existing = bc.supabase.table("calendar_event_snapshots") \
        .select("id").eq("workspace_id", workspace_id).eq("connection_id", connection_id) \
        .eq("external_event_id", external_event_id).eq("state_fingerprint", fingerprint) \
        .execute().data
    return existing[0]["id"] if existing else None


def snapshot_from_calendar_event_row(row: dict) -> Optional[str]:
    """Convenience wrapper: builds the maybe_create_snapshot() call directly
    from a real calendar_events row's own column shape (as read from the
    table, not from Google's raw API response), so a caller holding the
    current-state row doesn't have to unpack it by hand."""
    return maybe_create_snapshot(
        workspace_id=row["workspace_id"],
        connection_id=row["connection_id"],
        external_event_id=row["external_event_id"],
        title=row.get("title"),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        organizer=row.get("organizer"),
        attendees=row.get("attendees"),
        recurrence_rule=row.get("recurrence_rule"),
        meeting_url=row.get("meeting_url"),
        conference_id=row.get("conference_id"),
        observed_updated_at=row.get("updated_at_source"),
    )
