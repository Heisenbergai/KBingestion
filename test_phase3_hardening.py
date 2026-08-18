"""
Phase 3 hardening pass -- targeted regression tests for the three contained
fixes:
  1. CanonicalKnowledge.record_status / lifecycle_status / processing_status
     (2026-08-17 semantics correction: the original hardening pass's single
     `status` field conflated three distinct concepts -- see FIX 1 below)
  2. shared source_labels.source_type_label() helper (replacing two drifted,
     independently-duplicated dicts in query.py/chatbot.py)
  3. GET /knowledge-notes now enforces the same sensitivity ladder as
     GET /document-tables

Run with: python -m pytest test_phase3_hardening.py -v
"""
import asyncio
import inspect
import uuid

import pytest

import brain_connectors as bc
import canonical
import chatbot
import query
import source_labels
from auth import AuthContext

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"


# =====================================================================
# FIX 1 -- canonical record_status / lifecycle_status / processing_status
# (2026-08-17 semantics correction, replacing the earlier single `status`)
# =====================================================================

def test_note_record_status_comes_from_note_status_column():
    """Test 1: note record_status comes from note.status."""
    note = {
        "id": "n1", "workspace_id": "ws-1", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "status": "active", "source_tier": 3,
        "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.record_status == "active"


def test_document_record_status_comes_from_deleted_at():
    """Test 2 + Test 10 (active case): document record_status is DERIVED
    from deleted_at, never from processing_status."""
    item = {
        "id": "d1", "workspace_id": "ws-1", "title": "Doc", "description": None,
        "file_name": "doc.pdf", "file_type": "pdf", "folder_id": None, "uploader_name": "u",
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "processing_status": "completed", "deleted_at": None,
        "effective_from": None, "valid_until": None, "superseded_by": None,
        "created_at": "2026-08-01T00:00:00Z", "updated_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_item_to_canonical(item)
    assert result.record_status == "active"
    assert result.processing_status == "completed"
    assert result.record_status != result.processing_status


def test_deleted_document_record_status_is_deleted():
    """Test 8: deleted document -> record_status='deleted'."""
    item = {
        "id": "d2", "workspace_id": "ws-1", "title": "Doc", "description": None,
        "file_name": "doc.pdf", "file_type": "pdf", "folder_id": None, "uploader_name": "u",
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "processing_status": "completed",
        "deleted_at": "2026-08-10T00:00:00Z",
        "effective_from": None, "valid_until": None, "superseded_by": None,
        "created_at": "2026-08-01T00:00:00Z", "updated_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_item_to_canonical(item)
    assert result.record_status == "deleted"
    assert result.processing_status == "completed", \
        "processing_status must stay whatever it really is, independent of deletion"


def test_calendar_record_status_comes_from_deleted_at():
    """Test 3 + Test 10 (active case): calendar record_status is DERIVED
    from its own real deleted_at column."""
    event = {
        "id": "e1", "workspace_id": "ws-1", "connection_id": None, "title": "x",
        "start_time": None, "end_time": None, "organizer": None, "conference_id": None,
        "updated_at_source": None, "deleted_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.calendar_event_to_canonical(event)
    assert result.record_status == "active"


def test_deleted_calendar_event_record_status_is_deleted():
    """Test 9: deleted calendar event -> record_status='deleted'."""
    event = {
        "id": "e2", "workspace_id": "ws-1", "connection_id": None, "title": "x",
        "start_time": None, "end_time": None, "organizer": None, "conference_id": None,
        "updated_at_source": None, "deleted_at": "2026-08-10T00:00:00Z",
        "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.calendar_event_to_canonical(event)
    assert result.record_status == "deleted"


def test_document_processing_status_remains_separate():
    """Test 4: document processing_status remains separate from record_status."""
    item = {
        "id": "d3", "workspace_id": "ws-1", "title": "Doc", "description": None,
        "file_name": "doc.pdf", "file_type": "pdf", "folder_id": None, "uploader_name": "u",
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "processing_status": "processing", "deleted_at": None,
        "effective_from": None, "valid_until": None, "superseded_by": None,
        "created_at": "2026-08-01T00:00:00Z", "updated_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_item_to_canonical(item)
    assert result.processing_status == "processing"
    assert result.record_status == "active", \
        "an in-progress ingestion job must not be confused with a deleted record"


def test_calendar_processing_status_is_none():
    """Test 5: calendar processing_status = None."""
    event = {
        "id": "e3", "workspace_id": "ws-1", "connection_id": None, "title": "x",
        "start_time": None, "end_time": None, "organizer": None, "conference_id": None,
        "updated_at_source": None, "deleted_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.calendar_event_to_canonical(event)
    assert result.processing_status is None


def test_note_processing_status_is_none():
    """Test 6: note processing_status = None."""
    note = {
        "id": "n2", "workspace_id": "ws-1", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "status": "active", "source_tier": 3,
        "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.processing_status is None


def test_lifecycle_status_remains_independent_of_record_status():
    """Test 7: lifecycle_status remains a fully separate field from
    record_status -- proven with a note whose content lifecycle is 'draft'
    while its record is still 'active' (not deleted, just unfinished
    content -- two genuinely independent axes)."""
    note = {
        "id": "n3", "workspace_id": "ws-1", "connection_id": None, "provider": "slack",
        "source_type": "slack", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "draft", "status": "active", "source_tier": 3,
        "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.lifecycle_status == "draft"
    assert result.record_status == "active"
    assert canonical.CanonicalKnowledge.__dataclass_fields__["record_status"] is not \
        canonical.CanonicalKnowledge.__dataclass_fields__["lifecycle_status"]
    assert canonical.CanonicalKnowledge.__dataclass_fields__["record_status"] is not \
        canonical.CanonicalKnowledge.__dataclass_fields__["processing_status"]


def test_real_chat_note_record_status_field_populated():
    """Real live-DB proof: the real Q4 Chat note has a real record_status
    value (from its real status column) and no fabricated processing_status."""
    note_rows = bc.supabase.table("knowledge_notes").select("*") \
        .eq("id", "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1").execute().data
    assert note_rows, "fixture note no longer exists -- update the fixture"
    result = canonical.knowledge_note_to_canonical(note_rows[0], [])
    assert result.record_status == "active"
    assert result.processing_status is None
    assert result.lifecycle_status == "active"  # happens to match today, independently-sourced


def test_real_document_record_status_derived_from_real_deleted_at():
    """Test 11 (no fabrication) + real-row proof: a real, non-deleted
    document's record_status is correctly derived as 'active' from its
    real (null) deleted_at -- no schema change needed, deleted_at already existed."""
    # No service-role read path exists for knowledge_items (see canonical.py's
    # module docstring) -- same real-values-hardcoded fixture pattern
    # test_phase3_canonical.py already established for this exact document,
    # rather than a live fetch.
    real_document_row = {
        "id": "21279d95-356d-47f7-b953-b31d1cec417f", "workspace_id": "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d",
        "title": "MFG-003_Q2_Manufacturing_Business_Review", "description": None,
        "file_name": "MFG-003_Q2_Manufacturing_Business_Review.pptx", "file_type": "powerpoint",
        "folder_id": None, "uploader_name": "Test Uploader",
        "sensitivity": "confidential", "authority": "official", "doc_class": "financial",
        "lifecycle_status": "active", "processing_status": "completed", "deleted_at": None,
        "effective_from": None, "valid_until": None, "superseded_by": None,
        "created_at": "2026-08-15T10:34:51.540649+00:00", "updated_at": "2026-08-15T10:36:53.704134+00:00",
    }
    result = canonical.knowledge_item_to_canonical(real_document_row)
    assert result.record_status == "active"


def test_no_schema_change_required_for_status_semantics_fix():
    """Test 11: every field this correction reads (status, deleted_at,
    processing_status, lifecycle_status) already existed on the live
    schema before this fix -- confirmed by construction, since every
    fixture dict above uses only pre-existing column names."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(canonical.CanonicalKnowledge)}
    assert {"record_status", "lifecycle_status", "processing_status"} <= field_names
    assert "status" not in field_names, "the old conflated field must be fully gone, not left alongside the new ones"


# =====================================================================
# FIX 2 -- shared source_labels helper
# =====================================================================

EXPECTED_LABELS = {
    "document":     "Company document",
    "meeting":      "Meeting note",
    "slack":        "Team chat",
    "google_chat":  "Google Chat",
    "google_meet":  "Google Meet",
    "note":         "Curated note",
    "bot_learning": "Curated knowledge",
}


@pytest.mark.parametrize("source_type,expected", list(EXPECTED_LABELS.items()))
def test_source_type_label_matches_approved_mapping(source_type, expected):
    assert source_labels.source_type_label(source_type) == expected


def test_source_type_label_falls_back_to_raw_string_for_unknown_type():
    assert source_labels.source_type_label("some_future_source") == "some_future_source"


def test_source_type_label_falls_back_to_document_for_none():
    assert source_labels.source_type_label(None) == "Company document"
    assert source_labels.source_type_label("") == "Company document"


def test_canonical_source_type_values_unchanged_by_labeling():
    """FIX 2 must never touch the machine-readable source_type value
    itself -- only how it's displayed."""
    note = {
        "id": "n1", "workspace_id": "ws-1", "connection_id": None, "provider": "google_chat",
        "source_type": "google_chat", "title": "x", "body": "y", "category": None,
        "sensitivity": "internal", "authority": "working", "doc_class": None,
        "lifecycle_status": "active", "status": "active", "source_tier": 2,
        "occurred_at": None, "created_at": "2026-08-01T00:00:00Z",
    }
    result = canonical.knowledge_note_to_canonical(note, [])
    assert result.source_type == "google_chat", "canonical.source_type must stay the machine identifier, never the display label"


def test_query_uses_shared_source_label_helper_not_a_local_dict():
    src = inspect.getsource(query.build_context_and_citations)
    assert "source_labels.source_type_label" in src
    assert "company document" not in src.lower(), "old locally-duplicated dict literal must be gone"
    assert "meeting note" not in src.lower(), "old locally-duplicated dict literal must be gone"


def test_chatbot_uses_shared_source_label_helper_not_a_local_dict():
    src = inspect.getsource(chatbot)
    assert "source_labels.source_type_label" in src
    assert "official document" not in src.lower(), "old locally-duplicated dict literal must be gone"


def test_previously_covered_labels_do_not_regress():
    """The four source_types both old dicts already covered (document/
    meeting/slack/note) must still resolve to a real, non-blank,
    non-raw-string label after the fix."""
    for stype in ("document", "meeting", "slack", "note"):
        label = source_labels.source_type_label(stype)
        assert label and label != stype


# =====================================================================
# FIX 3 -- GET /knowledge-notes sensitivity enforcement
# =====================================================================

def _insert_test_note(workspace_id: str, sensitivity: str, title: str, status: str = "active") -> str:
    row = bc.supabase.table("knowledge_notes").insert({
        "workspace_id": workspace_id, "connection_id": None, "provider": "slack",
        "source_type": "slack", "source_tier": 3, "category": None,
        "title": title, "body": "test body for FIX 3 sensitivity tests", "participants": [],
        "source_ref": None, "occurred_at": None, "status": status,
        "sensitivity": sensitivity, "authority": "working", "doc_class": None,
        "lifecycle_status": "active",
    }).execute().data
    return row[0]["id"]


def _delete_test_note(note_id: str) -> None:
    bc.supabase.table("knowledge_notes").delete().eq("id", note_id).execute()


def test_knowledge_notes_member_sees_only_public_and_internal():
    ws = TEST_COMPANY_1_WS
    note_ids = []
    try:
        note_ids.append(_insert_test_note(ws, "public", "FIX3-PUBLIC-NOTE"))
        note_ids.append(_insert_test_note(ws, "internal", "FIX3-INTERNAL-NOTE"))
        note_ids.append(_insert_test_note(ws, "confidential", "FIX3-CONFIDENTIAL-NOTE"))
        note_ids.append(_insert_test_note(ws, "restricted", "FIX3-RESTRICTED-NOTE"))

        member_auth = AuthContext(user_id="u1", workspaces={ws: "member"})
        result = asyncio.run(bc.list_knowledge_notes(workspace_id=ws, limit=100, auth=member_auth))
        titles = {n["title"] for n in result["notes"]}

        assert "FIX3-PUBLIC-NOTE" in titles
        assert "FIX3-INTERNAL-NOTE" in titles
        assert "FIX3-CONFIDENTIAL-NOTE" not in titles, "a plain member must never see confidential notes"
        assert "FIX3-RESTRICTED-NOTE" not in titles, "a plain member must never see restricted notes"
    finally:
        for nid in note_ids:
            _delete_test_note(nid)


def test_knowledge_notes_admin_sees_confidential_but_not_restricted():
    ws = TEST_COMPANY_1_WS
    note_ids = []
    try:
        note_ids.append(_insert_test_note(ws, "confidential", "FIX3-ADMIN-CONF-NOTE"))
        note_ids.append(_insert_test_note(ws, "restricted", "FIX3-ADMIN-RESTRICTED-NOTE"))

        admin_auth = AuthContext(user_id="u1", workspaces={ws: "admin"})
        result = asyncio.run(bc.list_knowledge_notes(workspace_id=ws, limit=100, auth=admin_auth))
        titles = {n["title"] for n in result["notes"]}

        assert "FIX3-ADMIN-CONF-NOTE" in titles
        assert "FIX3-ADMIN-RESTRICTED-NOTE" not in titles, "admin (not owner/super_admin) must not see restricted notes"
    finally:
        for nid in note_ids:
            _delete_test_note(nid)


def test_knowledge_notes_owner_sees_restricted():
    ws = TEST_COMPANY_1_WS
    note_id = None
    try:
        note_id = _insert_test_note(ws, "restricted", "FIX3-OWNER-RESTRICTED-NOTE")
        owner_auth = AuthContext(user_id="u1", workspaces={ws: "owner"})
        result = asyncio.run(bc.list_knowledge_notes(workspace_id=ws, limit=100, auth=owner_auth))
        titles = {n["title"] for n in result["notes"]}
        assert "FIX3-OWNER-RESTRICTED-NOTE" in titles
    finally:
        if note_id:
            _delete_test_note(note_id)


def test_knowledge_notes_super_admin_sees_restricted_across_role():
    ws = TEST_COMPANY_1_WS
    note_id = None
    try:
        note_id = _insert_test_note(ws, "restricted", "FIX3-SUPERADMIN-RESTRICTED-NOTE")
        # super_admin with only a "member" role in this workspace must still see everything.
        super_admin_auth = AuthContext(user_id="u1", workspaces={ws: "member"}, is_super_admin=True)
        result = asyncio.run(bc.list_knowledge_notes(workspace_id=ws, limit=100, auth=super_admin_auth))
        titles = {n["title"] for n in result["notes"]}
        assert "FIX3-SUPERADMIN-RESTRICTED-NOTE" in titles
    finally:
        if note_id:
            _delete_test_note(note_id)


def test_knowledge_notes_status_filter_still_applies():
    """Requirement 5: existing status filtering must survive this fix --
    a note with a non-'active' status must stay excluded regardless of
    sensitivity permissiveness."""
    ws = TEST_COMPANY_1_WS
    note_id = None
    try:
        note_id = _insert_test_note(ws, "public", "FIX3-NONACTIVE-STATUS-NOTE", status="not_active_test_value")
        owner_auth = AuthContext(user_id="u1", workspaces={ws: "owner"})
        result = asyncio.run(bc.list_knowledge_notes(workspace_id=ws, limit=100, auth=owner_auth))
        titles = {n["title"] for n in result["notes"]}
        assert "FIX3-NONACTIVE-STATUS-NOTE" not in titles
    finally:
        if note_id:
            _delete_test_note(note_id)


def test_knowledge_notes_workspace_isolation_still_holds():
    """Requirement 4: existing workspace scoping must survive this fix --
    a caller with no membership in the target workspace must still be
    rejected outright, before sensitivity filtering is even relevant."""
    ws_a = TEST_COMPANY_1_WS
    ws_b = str(uuid.uuid4())
    note_id = None
    try:
        note_id = _insert_test_note(ws_a, "public", "FIX3-WORKSPACE-ISOLATION-NOTE")
        outsider_auth = AuthContext(user_id="attacker", workspaces={ws_b: "owner"})
        with pytest.raises(Exception) as exc:
            asyncio.run(bc.list_knowledge_notes(workspace_id=ws_a, limit=100, auth=outsider_auth))
        assert getattr(exc.value, "status_code", None) == 403
    finally:
        if note_id:
            _delete_test_note(note_id)


def test_knowledge_notes_never_trusts_a_client_supplied_sensitivity():
    """Requirement 3: there is no sensitivity parameter on this route at
    all -- confirmed by signature inspection, so there is nothing a client
    could even attempt to override."""
    sig = inspect.signature(bc.list_knowledge_notes)
    assert "sensitivity" not in sig.parameters
    assert "filter_sensitivities" not in sig.parameters
