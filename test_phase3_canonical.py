"""
Phase 3 canonical projection tests.

Most tests use synthetic dicts (deterministic, no dependency on which real
rows currently exist -- e.g. bot_learning notes and unmatched conference_ids
have no convenient real example to point at). Where "use real existing rows"
is genuinely valuable and safe, real values are used:
  - the real Google Chat KEEP note (7a9eaa34-...) and its real
    knowledge_note_sources row, fetched live from the vector DB
  - the real calendar_events row (aa473196-..., conference_id
    "ngn-pjwu-jcn"), fetched live from the vector DB, including a genuine
    live exact-match lookup via resolve_meet_calendar_event_id
  - one real knowledge_items row's real values, hardcoded as a literal dict
    fixture rather than fetched live -- there is no established service-role
    read path into the app DB's knowledge_items table from Python (see
    canonical.py's module docstring), only query.py's RLS-governed
    forwarded-user-token pattern, which a background test has no token for.
    Using the real values directly (not fabricated) is the honest
    middle ground.
  - external_references is genuinely empty workspace-wide right now (the
    Phase 2 historical backfill was deliberately skipped), so its test uses
    a synthetic row -- there is no real one to point at today.

Run with: python -m pytest test_phase3_canonical.py -v
"""
import dataclasses

import pytest

import brain_connectors as bc
import canonical

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"
REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
REAL_CHAT_NOTE_ID = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"
REAL_CALENDAR_EVENT_ID = "aa473196-79dd-4a9c-aefc-f2c80d12ea94"
REAL_CALENDAR_CONFERENCE_ID = "ngn-pjwu-jcn"

# Real knowledge_items row values (queried live once, hardcoded here -- see
# module docstring for why this test file doesn't fetch it live itself).
REAL_DOCUMENT_ROW = {
    "id": "21279d95-356d-47f7-b953-b31d1cec417f",
    "title": "MFG-003_Q2_Manufacturing_Business_Review",
    "description": None,
    "file_name": "MFG-003_Q2_Manufacturing_Business_Review.pptx",
    "file_type": "powerpoint",
    "folder_id": None,
    "uploader_name": "Test Uploader",
    "workspace_id": REAL_WORKSPACE,
    "sensitivity": "confidential",
    "authority": "official",
    "doc_class": "financial",
    "lifecycle_status": "active",
    "effective_from": None,
    "valid_until": None,
    "superseded_by": None,
    "created_at": "2026-08-15T10:34:51.540649+00:00",
    "updated_at": "2026-08-15T10:36:53.704134+00:00",
}


# =====================================================================
# Slack / Chat / Meet / bot_learning note projection
# =====================================================================

def test_slack_note_projection_shape():
    note = {
        "id": "note-1", "workspace_id": "ws-1", "connection_id": "conn-1",
        "provider": "slack", "source_type": "slack", "title": "Firmware release date",
        "body": "Firmware release target is September 12.", "category": "decision",
        "sensitivity": "internal", "authority": "official", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": "2026-08-01T10:00:00Z",
        "created_at": "2026-08-01T10:05:00Z",
    }
    sources = [{"channel_id": "C123", "thread_ts": None, "message_ts": "1000.001",
               "source_ref": "https://slack.com/archives/C123/p1000001", "occurred_at": "2026-08-01T10:00:00Z"}]
    result = canonical.knowledge_note_to_canonical(note, sources)

    assert result.source == "slack"
    assert result.workspace_id == "ws-1"
    assert result.connection_id == "conn-1"
    assert result.content == "Firmware release target is September 12."
    assert result.source_tier == 3
    assert result.conference_id is None
    assert result.calendar_event_id is None
    assert len(result.provenance) == 1
    assert result.provenance[0].container_ref == "C123"
    assert result.provenance[0].item_ref == "1000.001"
    assert result.provenance[0].permalink == "https://slack.com/archives/C123/p1000001"


def test_google_chat_note_projection_real_row():
    """Real vector-DB fetch of the exact note this whole Phase 2 Chat
    Drive-reference investigation was built around."""
    note_rows = bc.supabase.table("knowledge_notes").select("*").eq("id", REAL_CHAT_NOTE_ID).execute().data
    assert note_rows, "fixture note no longer exists -- update the fixture"
    note = note_rows[0]
    sources = bc.supabase.table("knowledge_note_sources").select("*") \
        .eq("note_id", REAL_CHAT_NOTE_ID).execute().data

    result = canonical.knowledge_note_to_canonical(note, sources)

    assert result.id == REAL_CHAT_NOTE_ID
    assert result.source == "google_chat"
    assert result.workspace_id == REAL_WORKSPACE
    assert result.sensitivity == "internal"
    assert result.authority == "official"
    assert result.doc_class == "policy_sop"
    assert result.lifecycle_status == "active"
    assert result.updated_at is None, "knowledge_notes has no updated_at column"
    assert result.effective_from is None
    assert result.valid_until is None
    assert result.superseded_by is None
    assert len(result.provenance) == 1
    assert result.provenance[0].container_ref == "spaces/AAQAU5JKkmE"
    assert result.provenance[0].participant is None, "knowledge_note_sources has no participant column"


def test_google_meet_note_projection_shape_and_conference_id():
    note = {
        "id": "note-meet-1", "workspace_id": "ws-1", "connection_id": "conn-meet-1",
        "provider": "google_meet", "source_type": "meeting", "title": "Q4 sync",
        "body": "Team agreed to ship the Q4 plan.", "category": None,
        "sensitivity": "internal", "authority": "official", "doc_class": "meeting",
        "lifecycle_status": "active", "source_tier": 2,
        "occurred_at": "2026-08-20T10:00:00Z", "created_at": "2026-08-20T10:30:00Z",
    }
    sources = [{
        "channel_id": "conferenceRecords/c1", "thread_ts": "conferenceRecords/c1/transcripts/t1",
        "message_ts": "conferenceRecords/c1/transcripts/t1/entries/e1",
        "source_ref": "conferenceRecords/c1/transcripts/t1/entries/e1",
        "occurred_at": "2026-08-20T10:00:00Z",
    }]
    result = canonical.knowledge_note_to_canonical(note, sources, calendar_event_id="cal-event-1")

    assert result.source == "google_meet"
    assert result.conference_id == "conferenceRecords/c1"
    assert result.calendar_event_id == "cal-event-1"
    assert result.provenance[0].container_ref == "conferenceRecords/c1"
    assert result.provenance[0].thread_ref == "conferenceRecords/c1/transcripts/t1"
    assert result.provenance[0].item_ref == "conferenceRecords/c1/transcripts/t1/entries/e1"


def test_meet_calendar_event_id_only_set_for_meet_provider():
    """A non-Meet note must never carry a calendar_event_id even if one is
    (incorrectly) passed in -- this field is Meet-specific by contract."""
    note = {
        "id": "note-slack-1", "workspace_id": "ws-1", "connection_id": "conn-1",
        "provider": "slack", "source_type": "slack", "title": "x", "body": "y",
        "category": None, "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [], calendar_event_id="should-be-ignored")
    assert result.calendar_event_id is None
    assert result.conference_id is None


def test_bot_learning_note_projection_empty_provenance():
    """bot_learning notes genuinely have zero knowledge_note_sources rows
    (create_note_and_embed is called with sources=None for this provider) --
    provenance=[] is the honest result, not a synthesized anchor from the
    legacy source_ref column."""
    note = {
        "id": "note-bot-1", "workspace_id": "ws-1", "connection_id": None,
        "provider": "bot_learning", "source_type": "note", "title": "Answer to escalated question",
        "body": "The refund policy is 30 days.", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 2, "occurred_at": None,
        "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, sources=None)
    assert result.provenance == []
    assert result.connection_id is None
    assert result.event_time is None


# =====================================================================
# Document projection
# =====================================================================

def test_document_projection_real_row():
    result = canonical.knowledge_item_to_canonical(REAL_DOCUMENT_ROW)

    assert result.id == REAL_DOCUMENT_ROW["id"]
    assert result.source == "document"
    assert result.source_type == "powerpoint"
    assert result.connection_id is None
    assert result.source_tier is None
    assert result.category is None
    assert result.content == "MFG-003_Q2_Manufacturing_Business_Review", \
        "description is null on this real row -- must fall back to title, not fabricate text"
    assert result.sensitivity == "confidential"
    assert result.authority == "official"
    assert result.doc_class == "financial"
    assert result.updated_at == REAL_DOCUMENT_ROW["updated_at"], "knowledge_items DOES have real updated_at, unlike notes"
    assert result.event_time is None
    assert result.source_updated_at is None


def test_document_gets_one_provenance_anchor():
    result = canonical.knowledge_item_to_canonical(REAL_DOCUMENT_ROW)
    assert len(result.provenance) == 1
    assert result.provenance[0].item_ref == REAL_DOCUMENT_ROW["file_name"]
    assert result.provenance[0].participant == REAL_DOCUMENT_ROW["uploader_name"]
    assert result.provenance[0].permalink is None, "an internal upload has no public URL -- never fabricated"


# =====================================================================
# Calendar projection
# =====================================================================

def test_calendar_event_projection_real_row():
    rows = bc.supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data
    assert rows, "fixture calendar event no longer exists -- update the fixture"
    event = rows[0]

    result = canonical.calendar_event_to_canonical(event)

    assert result.source == "calendar"
    assert result.workspace_id == REAL_WORKSPACE
    assert result.event_start is not None
    assert result.event_end is not None
    assert result.event_time is None, "Calendar is interval-shaped -- must use event_start/event_end, not event_time"
    assert result.conference_id is None, "conference_id is a Meet-note concept, calendar_event_to_canonical never sets it"


def test_calendar_classification_fields_remain_null():
    event = {
        "id": "evt-1", "workspace_id": "ws-1", "connection_id": "conn-1",
        "title": "Monday Capacity Review", "start_time": "2026-08-24T11:00:00Z",
        "end_time": "2026-08-24T11:30:00Z", "organizer": "ops@example.com",
        "conference_id": None, "updated_at_source": "2026-08-20T09:00:00Z",
        "created_at": "2026-08-20T09:00:00Z",
    }
    result = canonical.calendar_event_to_canonical(event)
    assert result.sensitivity is None
    assert result.authority is None
    assert result.doc_class is None
    assert result.lifecycle_status is None, "no lifecycle column exists on calendar_events either -- same no-fabrication rule"
    assert result.source_updated_at == "2026-08-20T09:00:00Z"
    assert result.updated_at is None, "calendar_events has no updated_at column (only updated_at_source)"


# =====================================================================
# Temporal rules
# =====================================================================

def test_unavailable_temporal_fields_remain_null():
    note = {
        "id": "note-2", "workspace_id": "ws-1", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": "2026-08-01T00:00:00Z",
        "created_at": "2026-08-01T00:05:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.source_updated_at is None
    assert result.event_start is None
    assert result.event_end is None


def test_no_fake_event_time_from_created_at():
    """A note whose real occurred_at is missing must report event_time as
    None -- never silently substitute created_at."""
    note = {
        "id": "note-3", "workspace_id": "ws-1", "connection_id": None, "provider": "bot_learning",
        "source_type": "note", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 2, "occurred_at": None,
        "created_at": "2026-08-01T12:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.event_time is None
    assert result.created_at == "2026-08-01T12:00:00Z"
    assert result.event_time != result.created_at


def test_source_tier_optional_behavior():
    doc_result = canonical.knowledge_item_to_canonical(REAL_DOCUMENT_ROW)
    cal_result = canonical.calendar_event_to_canonical({
        "id": "evt-2", "workspace_id": "ws-1", "connection_id": None, "title": "x",
        "start_time": None, "end_time": None, "organizer": None, "conference_id": None,
        "updated_at_source": None, "created_at": "2026-08-01T00:00:00Z",
    })
    note_result = canonical.knowledge_note_to_canonical({
        "id": "note-4", "workspace_id": "ws-1", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }, [])

    assert doc_result.source_tier is None
    assert cal_result.source_tier is None
    assert note_result.source_tier == 3


# =====================================================================
# Provenance relabeling
# =====================================================================

def test_provenance_relabeling_no_cross_mixing():
    row = {"channel_id": "CHANNEL-VAL", "thread_ts": "THREAD-VAL", "message_ts": "MESSAGE-VAL",
           "source_ref": "PERMALINK-VAL", "occurred_at": "2026-08-01T00:00:00Z"}
    evidence = canonical._note_source_to_provenance(row)
    assert evidence.container_ref == "CHANNEL-VAL"
    assert evidence.thread_ref == "THREAD-VAL"
    assert evidence.item_ref == "MESSAGE-VAL"
    assert evidence.permalink == "PERMALINK-VAL"
    assert evidence.participant is None


# =====================================================================
# External references stay separate, non-knowledge
# =====================================================================

def test_external_references_remain_separate_from_content_and_provenance():
    """external_references is genuinely empty workspace-wide right now (the
    Phase 2 historical backfill was deliberately skipped) -- synthetic row,
    honestly, since there's no real one to point at today."""
    note = {
        "id": "note-5", "workspace_id": "ws-1", "connection_id": "conn-1", "provider": "google_chat",
        "source_type": "google_chat", "title": "Launch QA gate", "body": "Launch requires QA sign-off.",
        "category": "process", "sensitivity": "internal", "authority": "official", "doc_class": "policy_sop",
        "lifecycle_status": "active", "source_tier": 2, "occurred_at": "2026-08-01T00:00:00Z",
        "created_at": "2026-08-01T00:05:00Z",
    }
    ref_row = {"external_file_id": "file123", "title": "Q4 Deck.pptx",
              "url": "https://docs.google.com/presentation/d/file123/edit",
              "modified_time": "2026-08-01T00:00:00Z", "linked_object_type": "knowledge_note",
              "linked_object_id": "note-5"}
    result = canonical.knowledge_note_to_canonical(note, [], external_refs=[ref_row])

    assert len(result.external_references) == 1
    assert result.external_references[0].file_id == "file123"
    assert "file123" not in result.content
    assert result.provenance == []


def test_document_chunks_remain_retrieval_only():
    """MANDATORY negative test, mirroring the Drive bulk-ingestion negative
    test pattern: this module must not expose any function that turns a
    document_chunks row into a CanonicalKnowledge instance."""
    assert not hasattr(canonical, "document_chunk_to_canonical")
    assert not hasattr(canonical, "chunk_to_canonical")


# =====================================================================
# Meet <-> Calendar exact-match resolution (real live DB, read-only)
# =====================================================================

def test_meet_conference_id_resolves_real_calendar_event_id():
    result = canonical.resolve_meet_calendar_event_id(REAL_WORKSPACE, REAL_CALENDAR_CONFERENCE_ID)
    assert result == REAL_CALENDAR_EVENT_ID


def test_unmatched_conference_id_returns_none():
    result = canonical.resolve_meet_calendar_event_id(REAL_WORKSPACE, "not-a-real-conference-id-xyz")
    assert result is None


def test_missing_conference_id_returns_none_without_a_query():
    assert canonical.resolve_meet_calendar_event_id(REAL_WORKSPACE, None) is None


def test_conference_id_wrong_workspace_returns_none():
    """The real conference_id exists, but under a different (synthetic,
    non-existent) workspace -- must not match across workspaces."""
    result = canonical.resolve_meet_calendar_event_id(TEST_COMPANY_1_WS, REAL_CALENDAR_CONFERENCE_ID)
    assert result is None


# =====================================================================
# workspace_id / connection_id preservation
# =====================================================================

def test_workspace_id_preserved_across_all_source_types():
    note = canonical.knowledge_note_to_canonical({
        "id": "n", "workspace_id": "ws-preserved", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }, [])
    doc = canonical.knowledge_item_to_canonical({**REAL_DOCUMENT_ROW, "workspace_id": "ws-preserved"})
    cal = canonical.calendar_event_to_canonical({
        "id": "e", "workspace_id": "ws-preserved", "connection_id": None, "title": "x",
        "start_time": None, "end_time": None, "organizer": None, "conference_id": None,
        "updated_at_source": None, "created_at": "2026-08-01T00:00:00Z",
    })
    assert note.workspace_id == doc.workspace_id == cal.workspace_id == "ws-preserved"


def test_connection_id_preserved_or_null_by_real_source_shape():
    slack_note = canonical.knowledge_note_to_canonical({
        "id": "n1", "workspace_id": "ws-1", "connection_id": "real-conn-id", "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 3, "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }, [])
    bot_note = canonical.knowledge_note_to_canonical({
        "id": "n2", "workspace_id": "ws-1", "connection_id": None, "provider": "bot_learning",
        "source_type": "note", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "source_tier": 2, "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }, [])
    doc = canonical.knowledge_item_to_canonical(REAL_DOCUMENT_ROW)

    assert slack_note.connection_id == "real-conn-id"
    assert bot_note.connection_id is None
    assert doc.connection_id is None, "knowledge_items has no connection_id column at all"


# =====================================================================
# confidence exclusion
# =====================================================================

def test_confidence_field_does_not_exist_on_canonical_knowledge():
    field_names = {f.name for f in dataclasses.fields(canonical.CanonicalKnowledge)}
    assert "confidence" not in field_names


# =====================================================================
# Module has zero side effects / does not touch existing Phase 1 code
# =====================================================================

def test_canonical_module_has_no_import_side_effects():
    """Phase 3's module is purely additive -- reimporting it must not raise
    and must not require anything beyond what brain_connectors already
    needs (no new env vars, no new required credentials)."""
    import importlib
    importlib.reload(canonical)


def test_canonical_module_does_not_modify_query_module():
    """Sanity check that this pass didn't touch retrieval -- the real proof
    is the full existing test_phase1_retrieval.py suite staying green
    (run separately), this just confirms canonical.py doesn't import or
    monkeypatch query.py at all."""
    import query
    assert not hasattr(query, "CanonicalKnowledge")
    assert not hasattr(query, "canonical")
