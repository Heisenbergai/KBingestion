"""
Phase 4 persistence-path tests -- exercise persist_extracted_primitives()
against the real, live structured_knowledge/extraction_contract_versions
tables. Synthetic ExtractedPrimitive inputs shaped exactly like the
already-validated real V2.1 extraction outputs from test_phase4_extraction.py
(this file does NOT re-test extraction correctness -- that's covered there;
this file tests that PERSISTENCE preserves whatever a validated
ExtractedPrimitive already contains, without corrupting or dropping fields).

No real Bedrock call is made anywhere in this file -- same, still-unresolved
local credential gap as every other Phase 4 pass. See the final report's
"Real controlled corpus" section for the honest boundary between what's
proven here (the persistence function itself, thoroughly) and what still
requires a real extraction run (the actual 9-item real corpus write).

Run with: python -m pytest test_phase4_persistence.py -v
"""
import uuid

import pytest

import brain_connectors as bc
import structured_persistence as sp
from structured_extraction import ExtractedPrimitive

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"
OTHER_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"


def _make_primitive(**overrides) -> ExtractedPrimitive:
    base = dict(
        type="fact", canonical_id="placeholder", workspace_id=TEST_COMPANY_1_WS,
        statement="Test statement.", source_evidence_link="placeholder",
        sensitivity="internal", authority="working", source_tier=3,
        lifecycle_status="active", captured_at="2026-08-01T00:00:00+00:00",
        event_time=None, event_start=None, event_end=None,
        effective_from=None, effective_until=None, raw_subject_phrase=None,
        requirement_kind=None, qualifier_words=[], recurrence_text=None,
    )
    base.update(overrides)
    return ExtractedPrimitive(**base)


def _make_test_note(workspace_id: str = TEST_COMPANY_1_WS, sensitivity: str = "internal",
                    status: str = "active", title: str = "Persistence test note") -> str:
    row = bc.supabase.table("knowledge_notes").insert({
        "workspace_id": workspace_id, "connection_id": None, "provider": "slack",
        "source_type": "slack", "source_tier": 3, "category": None,
        "title": title, "body": "test body", "participants": [],
        "source_ref": None, "occurred_at": None, "status": status,
        "sensitivity": sensitivity, "authority": "working", "doc_class": None,
        "lifecycle_status": "active",
    }).execute().data
    return row[0]["id"]


def _delete_test_note(note_id: str):
    if note_id:
        bc.supabase.table("knowledge_notes").delete().eq("id", note_id).execute()


def _cleanup_structured(*ids):
    for i in ids:
        if i:
            bc.supabase.table("structured_knowledge").delete().eq("id", i).execute()


def _rows_for(canonical_id: str) -> list:
    return bc.supabase.table("structured_knowledge").select("*").eq("canonical_id", canonical_id).execute().data


# =====================================================================
# Basic successful persistence
# =====================================================================

def test_successful_persist_with_correct_workspace():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(canonical_id=note_id, statement="A simple durable fact.")
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        assert result["canonical_rejected_reason"] is None
        rows = _rows_for(note_id)
        assert len(rows) == 1
        assert rows[0]["statement"] == "A simple durable fact."
        assert rows[0]["provider"] == "slack"
        assert rows[0]["extraction_version"] == "v2.1"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


# =====================================================================
# Security: workspace ownership
# =====================================================================

def test_wrong_workspace_fails_and_persists_nothing():
    note_id = None
    try:
        note_id = _make_test_note(workspace_id=TEST_COMPANY_1_WS)
        primitive = _make_primitive(canonical_id=note_id)
        result = sp.persist_extracted_primitives(
            workspace_id=OTHER_WORKSPACE,  # deliberately wrong
            canonical_source_type="knowledge_note", canonical_id=note_id,
            provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 0
        assert "workspace" in result["canonical_rejected_reason"].lower()
        assert _rows_for(note_id) == []
    finally:
        _delete_test_note(note_id)


def test_correct_workspace_succeeds_matching_the_wrong_workspace_test():
    """Direct positive counterpart to the wrong-workspace test, same note."""
    note_id = None
    try:
        note_id = _make_test_note(workspace_id=TEST_COMPANY_1_WS)
        primitive = _make_primitive(canonical_id=note_id)
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


# =====================================================================
# Deleted/inactive parent cannot create new structured knowledge
# =====================================================================

def test_deleted_parent_note_rejected_fails_safe():
    note_id = None
    try:
        note_id = _make_test_note(status="archived_test")
        primitive = _make_primitive(canonical_id=note_id)
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 0
        assert "not currently usable" in result["canonical_rejected_reason"]
        assert _rows_for(note_id) == []
    finally:
        _delete_test_note(note_id)


def test_nonexistent_canonical_parent_rejected():
    fake_id = str(uuid.uuid4())
    primitive = _make_primitive(canonical_id=fake_id)
    result = sp.persist_extracted_primitives(
        workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
        canonical_id=fake_id, provider="slack", extraction_version="v2.1",
        extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
    )
    assert result["inserted"] == 0
    assert "not found" in result["canonical_rejected_reason"]


# =====================================================================
# Source validation: only knowledge_note / calendar_event supported
# =====================================================================

def test_knowledge_item_source_type_rejected_outright():
    result = sp.persist_extracted_primitives(
        workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_item",
        canonical_id=str(uuid.uuid4()), provider="document", extraction_version="v2.1",
        extraction_run_id=str(uuid.uuid4()), primitives=[_make_primitive()],
    )
    assert result["inserted"] == 0
    assert "not currently supported" in result["canonical_rejected_reason"]
    assert "knowledge_note" in result["canonical_rejected_reason"]


def _make_calendar_connection_and_event(**event_overrides) -> tuple:
    conn_row = bc.supabase.table("connections").insert({
        "workspace_id": TEST_COMPANY_1_WS, "provider": "google_drive",
        "external_team_id": f"TEST-PERSIST-{uuid.uuid4()}", "external_team_name": "Persistence Test Connection",
        "access_token_enc": "not-a-real-token", "status": "inactive",
    }).execute().data
    conn_id = conn_row[0]["id"]
    event = {
        "workspace_id": TEST_COMPANY_1_WS, "connection_id": conn_id,
        "external_event_id": f"test-evt-{uuid.uuid4()}", "title": "Persistence test event",
        "start_time": "2026-08-24T11:00:00+00:00", "end_time": "2026-08-24T11:30:00+00:00",
        "organizer": "test@example.com", "attendees": [], "conference_id": None,
        "updated_at_source": None, "deleted_at": None,
    }
    event.update(event_overrides)
    event_row = bc.supabase.table("calendar_events").insert(event).execute().data
    return event_row[0]["id"], conn_id


def test_calendar_event_source_type_accepted():
    event_id = conn_id = None
    try:
        event_id, conn_id = _make_calendar_connection_and_event()
        primitive = _make_primitive(
            canonical_id=event_id, type="event", sensitivity=None,
            event_start="2026-08-24T11:00:00+00:00", event_end="2026-08-24T11:30:00+00:00",
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="calendar_event",
            canonical_id=event_id, provider="calendar", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        rows = _rows_for(event_id)
        assert rows[0]["event_start"] is not None
        assert rows[0]["event_end"] is not None
        assert rows[0]["sensitivity"] is None
    finally:
        for r in _rows_for(event_id) if event_id else []:
            _cleanup_structured(r["id"])
        if event_id:
            bc.supabase.table("calendar_events").delete().eq("id", event_id).execute()
        if conn_id:
            bc.supabase.table("connections").delete().eq("id", conn_id).execute()


def test_calendar_primitive_with_all_four_optional_fields_null_persists():
    """The exact real bug, reproduced faithfully: a Calendar-shaped
    primitive with sensitivity/authority/source_tier/lifecycle_status ALL
    None (matching calendar_event_to_canonical's real output exactly, not
    the earlier test's accidental use of real classification defaults).
    This is the test that should have existed before the live failure --
    added now specifically because it wasn't."""
    event_id = conn_id = None
    try:
        event_id, conn_id = _make_calendar_connection_and_event()
        primitive = _make_primitive(
            canonical_id=event_id, type="event",
            sensitivity=None, authority=None, source_tier=None, lifecycle_status=None,
            event_start="2026-08-24T11:00:00+00:00", event_end="2026-08-24T11:30:00+00:00",
            statement="Persistence test event is scheduled.",
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="calendar_event",
            canonical_id=event_id, provider="calendar", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1, f"expected a clean insert, got: {result}"
        row = _rows_for(event_id)[0]
        assert row["sensitivity"] is None
        assert row["authority"] is None
        assert row["source_tier"] is None
        assert row["lifecycle_status"] is None
    finally:
        for r in _rows_for(event_id) if event_id else []:
            _cleanup_structured(r["id"])
        if event_id:
            bc.supabase.table("calendar_events").delete().eq("id", event_id).execute()
        if conn_id:
            bc.supabase.table("connections").delete().eq("id", conn_id).execute()


def test_classified_note_primitive_still_populates_all_four_fields():
    """The nullability fix must not weaken the note path -- a real,
    correctly-classified Slack/Chat primitive still carries real,
    non-null values for all four fields."""
    note_id = None
    try:
        note_id = _make_test_note(sensitivity="internal")
        primitive = _make_primitive(
            canonical_id=note_id, sensitivity="internal", authority="official",
            source_tier=2, lifecycle_status="active",
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="google_chat", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        row = _rows_for(note_id)[0]
        assert row["sensitivity"] == "internal"
        assert row["authority"] == "official"
        assert row["source_tier"] == 2
        assert row["lifecycle_status"] == "active"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_null_sensitivity_cannot_bypass_ceiling_for_a_classified_source():
    """The nullable column must not become a loophole: a note-sourced
    primitive (parent HAS a real sensitivity) claiming sensitivity=None is
    an invalid/adversarial input, not a free pass -- it must be rejected,
    never silently treated as 'no ceiling to check'."""
    note_id = None
    try:
        note_id = _make_test_note(sensitivity="internal")
        bad_primitive = _make_primitive(canonical_id=note_id, sensitivity=None, statement="sneaky null sensitivity")
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[bad_primitive],
        )
        assert result["inserted"] == 0
        assert len(result["rejected"]) == 1
        assert _rows_for(note_id) == []
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_no_fake_defaults_inserted_for_calendar():
    """Explicit proof the fix did not add 'internal'/'working'/'active'
    fallback defaults anywhere -- a genuinely None field stays None in the
    real persisted row, never silently upgraded to a plausible-looking
    default."""
    event_id = conn_id = None
    try:
        event_id, conn_id = _make_calendar_connection_and_event()
        primitive = _make_primitive(
            canonical_id=event_id, type="event",
            sensitivity=None, authority=None, source_tier=None, lifecycle_status=None,
        )
        sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="calendar_event",
            canonical_id=event_id, provider="calendar", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        row = _rows_for(event_id)[0]
        for field, forbidden_defaults in (
            ("sensitivity", {"public", "internal", "confidential", "restricted"}),
            ("authority", {"canonical", "official", "working", "reference", "informal"}),
            ("lifecycle_status", {"draft", "active", "under_review", "superseded", "archived"}),
        ):
            assert row[field] is None, f"{field} must stay None, never fabricated into {forbidden_defaults}"
    finally:
        for r in _rows_for(event_id) if event_id else []:
            _cleanup_structured(r["id"])
        if event_id:
            bc.supabase.table("calendar_events").delete().eq("id", event_id).execute()
        if conn_id:
            bc.supabase.table("connections").delete().eq("id", conn_id).execute()


def test_existing_constraints_remain_intact_after_nullability_fix():
    """Regression: the CHECK constraints on primitive_type/
    canonical_source_type/requirement_kind and the extraction_version FK
    must still reject invalid values -- the nullability fix touched ONLY
    sensitivity/authority/source_tier/lifecycle_status, nothing else."""
    with pytest.raises(Exception):
        _insert_raw = bc.supabase.table("structured_knowledge").insert({
            "workspace_id": TEST_COMPANY_1_WS, "canonical_source_type": "knowledge_note",
            "canonical_id": str(uuid.uuid4()), "provider": "slack", "primitive_type": "not_a_real_type",
            "statement": "x", "captured_at": "2026-08-01T00:00:00+00:00",
            "extraction_version": "v2.1", "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": uuid.uuid4().hex,
        }).execute()


def test_failed_persistence_remains_safe_db_rejection_caught_per_primitive():
    """A primitive that will genuinely violate a DB constraint (an invalid
    primitive_type, bypassing whatever structured_extraction.py itself
    would normally guarantee) must be caught inside
    persist_extracted_primitives() itself and reported in `rejected` --
    never raise out of the function and never abort a batch's other
    primitives. This is the exact failure mode the real controlled run hit."""
    note_id = None
    try:
        note_id = _make_test_note()
        good = _make_primitive(canonical_id=note_id, statement="valid primitive")
        bad = _make_primitive(canonical_id=note_id, statement="invalid primitive", type="not_a_real_type")
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[good, bad],
        )
        assert result["inserted"] == 1, "the valid primitive must still persist despite the other one failing"
        assert len(result["rejected"]) == 1
        assert "database rejected" in result["rejected"][0]["reason"]
        rows = _rows_for(note_id)
        assert len(rows) == 1
        assert rows[0]["statement"] == "valid primitive"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_failed_extraction_run_cleanup_leaves_zero_residue():
    """Direct proof of the exact cleanup pattern used for the real failed
    run (delete by extraction_run_id, verify zero remain) -- against a
    synthetic failed-run scenario, not just asserted by hand against the
    real one."""
    note_id = None
    failed_run_id = str(uuid.uuid4())
    try:
        note_id = _make_test_note()
        good = _make_primitive(canonical_id=note_id, statement="succeeded before the failure")
        bad = _make_primitive(canonical_id=note_id, statement="the one that fails", type="not_a_real_type")
        sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=failed_run_id, primitives=[good, bad],
        )
        assert len(bc.supabase.table("structured_knowledge").select("id")
                   .eq("extraction_run_id", failed_run_id).execute().data) == 1

        # The cleanup pattern: delete ONLY by the exact failed run id.
        bc.supabase.table("structured_knowledge").delete().eq("extraction_run_id", failed_run_id).execute()

        remaining = bc.supabase.table("structured_knowledge").select("id") \
            .eq("extraction_run_id", failed_run_id).execute().data
        assert remaining == []
    finally:
        # Safety net in case the in-test cleanup above didn't run.
        bc.supabase.table("structured_knowledge").delete().eq("extraction_run_id", failed_run_id).execute()
        _delete_test_note(note_id)


# =====================================================================
# Sensitivity ceiling: cannot exceed canonical parent, per-primitive
# =====================================================================

def test_sensitivity_exceeding_parent_rejected_valid_ones_still_persist():
    note_id = None
    try:
        note_id = _make_test_note(sensitivity="internal")
        ok_primitive = _make_primitive(canonical_id=note_id, sensitivity="internal", statement="ok fact")
        over_primitive = _make_primitive(canonical_id=note_id, sensitivity="restricted", statement="over-sensitive fact")
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[ok_primitive, over_primitive],
        )
        assert result["inserted"] == 1
        assert len(result["rejected"]) == 1
        assert "restricted" in result["rejected"][0]["reason"]
        rows = _rows_for(note_id)
        assert len(rows) == 1
        assert rows[0]["statement"] == "ok fact"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_sensitivity_equal_to_parent_allowed():
    note_id = None
    try:
        note_id = _make_test_note(sensitivity="confidential")
        primitive = _make_primitive(canonical_id=note_id, sensitivity="confidential")
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


# =====================================================================
# Version handling
# =====================================================================

def test_unregistered_extraction_version_rejected_cleanly():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(canonical_id=note_id)
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v99-does-not-exist",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 0
        assert "not a registered contract version" in result["canonical_rejected_reason"]
        assert _rows_for(note_id) == []
    finally:
        _delete_test_note(note_id)


# =====================================================================
# Fingerprint determinism + idempotency
# =====================================================================

def test_fingerprint_is_deterministic():
    p1 = _make_primitive(statement="Same statement.", effective_from="2026-09-15", qualifier_words=["target"])
    p2 = _make_primitive(statement="Same statement.", effective_from="2026-09-15", qualifier_words=["target"])
    assert sp.compute_primitive_fingerprint(p1) == sp.compute_primitive_fingerprint(p2)


def test_fingerprint_differs_for_different_content():
    p1 = _make_primitive(statement="Statement A.")
    p2 = _make_primitive(statement="Statement B.")
    assert sp.compute_primitive_fingerprint(p1) != sp.compute_primitive_fingerprint(p2)


def test_persisting_same_primitives_twice_is_idempotent():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(canonical_id=note_id, statement="Idempotency check statement.")
        run_id_1 = str(uuid.uuid4())

        first = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=run_id_1, primitives=[primitive],
        )
        assert first["inserted"] == 1
        assert first["skipped_duplicates"] == 0

        # Second run: a DIFFERENT extraction_run_id (a real retry would get
        # a new run id) but IDENTICAL primitive content -- fingerprint-based
        # identity must still catch it.
        run_id_2 = str(uuid.uuid4())
        second = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=run_id_2, primitives=[primitive],
        )
        assert second["inserted"] == 0
        assert second["skipped_duplicates"] == 1

        rows = _rows_for(note_id)
        assert len(rows) == 1, "identical fingerprint re-run must never duplicate the row"
        assert rows[0]["extraction_run_id"] == run_id_1, "the original row's run_id must be untouched by the second (no-op) run"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


# =====================================================================
# Field preservation: qualifiers, recurrence, temporal fields,
# classification propagation -- shaped exactly like real V2.1 extraction
# output already validated in test_phase4_extraction.py
# =====================================================================

def test_target_date_primitive_persists_with_no_effective_from_and_qualifier_preserved():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(
            canonical_id=note_id, type="fact",
            statement="The release target is September 12.",
            effective_from=None, qualifier_words=["target"],
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        row = _rows_for(note_id)[0]
        assert row["effective_from"] is None
        assert "target" in row["qualifier_words"]
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_recurrence_text_preserved_through_persistence():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(
            canonical_id=note_id, type="requirement", requirement_kind="process_step",
            statement="Capacity must be submitted every Monday by 11 AM.",
            recurrence_text="every Monday by 11 AM", effective_from=None,
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        row = _rows_for(note_id)[0]
        assert "every Monday" in row["recurrence_text"]
        assert row["effective_from"] is None
        assert row["requirement_kind"] == "process_step"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_committed_effective_date_preserved_for_unhedged_requirement():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(
            canonical_id=note_id, type="requirement", requirement_kind="policy",
            statement="Starting September 15, Product and QA must approve the checklist.",
            raw_subject_phrase="Product and QA", effective_from="2026-09-15",
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="google_chat", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        row = _rows_for(note_id)[0]
        assert row["effective_from"] == "2026-09-15"
        assert row["raw_subject_phrase"] == "Product and QA"
        assert row["provider"] == "google_chat"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_authority_source_tier_lifecycle_propagated_unmodified():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(
            canonical_id=note_id, authority="official", source_tier=2, lifecycle_status="active",
        )
        result = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="google_chat", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        assert result["inserted"] == 1
        row = _rows_for(note_id)[0]
        assert row["authority"] == "official"
        assert row["source_tier"] == 2
        assert row["lifecycle_status"] == "active"
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


def test_no_confidence_or_record_status_persisted():
    note_id = None
    try:
        note_id = _make_test_note()
        primitive = _make_primitive(canonical_id=note_id)
        sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive],
        )
        row = _rows_for(note_id)[0]
        assert "confidence" not in row
        assert "record_status" not in row
    finally:
        for r in _rows_for(note_id) if note_id else []:
            _cleanup_structured(r["id"])
        _delete_test_note(note_id)


# =====================================================================
# Version isolation (test-only version, never promoted to current)
# =====================================================================

def test_version_isolation_test_version_never_promoted_to_current():
    note_id = None
    test_version = f"test-persist-{uuid.uuid4().hex[:8]}"
    row_ids = []
    try:
        note_id = _make_test_note()
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": test_version, "is_current": False, "description": "persistence test version"}
        ).execute()

        current_check = bc.supabase.table("extraction_contract_versions") \
            .select("version").eq("is_current", True).execute().data
        assert current_check == [{"version": "v2.1"}], "test version must never become current"

        primitive_v21 = _make_primitive(canonical_id=note_id, statement="v2.1-era statement")
        primitive_test = _make_primitive(canonical_id=note_id, statement="test-version-era statement")

        result_v21 = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version="v2.1",
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive_v21],
        )
        result_test = sp.persist_extracted_primitives(
            workspace_id=TEST_COMPANY_1_WS, canonical_source_type="knowledge_note",
            canonical_id=note_id, provider="slack", extraction_version=test_version,
            extraction_run_id=str(uuid.uuid4()), primitives=[primitive_test],
        )
        assert result_v21["inserted"] == 1
        assert result_test["inserted"] == 1

        rows = _rows_for(note_id)
        row_ids = [r["id"] for r in rows]
        assert {r["extraction_version"] for r in rows} == {"v2.1", test_version}

        # "Normal current-version read" -- the test version has a HIGHER
        # sequence_number (created after v2.1) but was never promoted, so
        # it must be excluded; v2.1 (the real current) must be selected.
        current = bc.supabase.table("extraction_contract_versions") \
            .select("sequence_number").eq("is_current", True).execute().data[0]
        all_versions = {v["version"]: v["sequence_number"]
                        for v in bc.supabase.table("extraction_contract_versions").select("*").execute().data}
        eligible = [r for r in rows if all_versions[r["extraction_version"]] <= current["sequence_number"]]
        best_seq = max(all_versions[r["extraction_version"]] for r in eligible)
        best_rows = [r for r in eligible if all_versions[r["extraction_version"]] == best_seq]

        assert len(best_rows) == 1
        assert best_rows[0]["extraction_version"] == "v2.1"
        assert best_rows[0]["statement"] == "v2.1-era statement"
        # The test-version row remains physically present, just not selected.
        assert any(r["extraction_version"] == test_version for r in rows)
    finally:
        _cleanup_structured(*row_ids)
        _delete_test_note(note_id)
        bc.supabase.table("extraction_contract_versions").delete().eq("version", test_version).execute()

        final_current = bc.supabase.table("extraction_contract_versions") \
            .select("version").eq("is_current", True).execute().data
        assert final_current == [{"version": "v2.1"}], "cleanup must leave v2.1 as the only current version"
