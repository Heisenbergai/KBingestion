"""
Phase 3 canonical READ-SIDE CONSUMER INTERFACE tests --
canonical.get_canonical_knowledge(). This is the first real caller of the
Phase 3 projection layer; test_phase3_canonical.py / test_phase3_hardening.py
cover the underlying pure projection functions this interface is built on
top of and are NOT duplicated here.

Real rows used where safely possible: the real Google Chat KEEP note
(7a9eaa34-...) and the real calendar event (aa473196-..., conference_id
"ngn-pjwu-jcn"). Everything else (Meet -- no real transcript-derived note
exists yet, per Phase 2's own known gap -- workspace isolation, deleted-row
exclusion, sensitivity ladder edges) uses synthetic, self-cleaning rows
under TEST_COMPANY_1_WS, matching every other real-DB test file's pattern
in this repo.

Run with: python -m pytest test_phase3_consumer.py -v
"""
import uuid

import pytest

import brain_connectors as bc
import canonical

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"
REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
REAL_CHAT_NOTE_ID = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"
REAL_CALENDAR_EVENT_ID = "aa473196-79dd-4a9c-aefc-f2c80d12ea94"
REAL_CALENDAR_CONFERENCE_ID = "ngn-pjwu-jcn"

ALL_SENSITIVITIES = ["public", "internal", "confidential", "restricted"]


def _insert_test_note(workspace_id: str, sensitivity: str = "internal", title: str = "test note",
                      status: str = "active", provider: str = "slack",
                      connection_id: str = None) -> str:
    row = bc.supabase.table("knowledge_notes").insert({
        "workspace_id": workspace_id, "connection_id": connection_id, "provider": provider,
        "source_type": provider, "source_tier": 2 if provider in ("google_chat", "google_meet") else 3,
        "category": None, "title": title, "body": f"test body for {title}", "participants": [],
        "source_ref": None, "occurred_at": None, "status": status,
        "sensitivity": sensitivity, "authority": "working", "doc_class": None,
        "lifecycle_status": "active",
    }).execute().data
    return row[0]["id"]


def _insert_test_note_source(note_id: str, workspace_id: str, provider: str, channel_id: str) -> str:
    row = bc.supabase.table("knowledge_note_sources").insert({
        "note_id": note_id, "workspace_id": workspace_id, "provider": provider,
        "source_type": provider, "connection_id": None,
        "channel_id": channel_id, "message_ts": "test-ts-1", "thread_ts": None,
        "source_ref": "test-permalink", "occurred_at": "2026-08-20T10:00:00Z",
    }).execute().data
    return row[0]["id"]


def _make_test_connection(workspace_id: str) -> str:
    """calendar_events.connection_id is NOT NULL -- a synthetic, inactive
    (never picked up by the real scheduled worker) connection row, matching
    the exact pattern test_phase1_retrieval.py/test_google_workspace.py
    already use."""
    row = bc.supabase.table("connections").insert({
        "workspace_id": workspace_id, "provider": "google_drive",
        "external_team_id": f"TEST-CONSUMER-{uuid.uuid4()}",
        "external_team_name": "Phase 3 Consumer Test Connection",
        "access_token_enc": "not-a-real-token-never-decrypted-in-these-tests",
        "status": "inactive",
    }).execute().data
    return row[0]["id"]


def _delete_test_connection(connection_id: str) -> None:
    bc.supabase.table("connections").delete().eq("id", connection_id).execute()


def _insert_test_calendar_event(workspace_id: str, deleted: bool = False,
                                conference_id: str = None, title: str = "Test Calendar Event") -> tuple:
    """Returns (event_id, connection_id) -- pass the returned tuple straight
    into _cleanup_events(...) to clean up both rows."""
    conn_id = _make_test_connection(workspace_id)
    row = bc.supabase.table("calendar_events").insert({
        "workspace_id": workspace_id, "connection_id": conn_id,
        "external_event_id": f"test-evt-{uuid.uuid4()}",
        "title": title, "start_time": "2026-08-24T11:00:00Z", "end_time": "2026-08-24T11:30:00Z",
        "organizer": "test@example.com", "attendees": [], "recurrence_rule": None,
        "meeting_url": None, "conference_id": conference_id,
        "updated_at_source": "2026-08-20T09:00:00Z",
        "deleted_at": "2026-08-21T00:00:00Z" if deleted else None,
    }).execute().data
    return row[0]["id"], conn_id


def _cleanup_notes(*note_ids):
    for nid in note_ids:
        if nid:
            bc.supabase.table("knowledge_note_sources").delete().eq("note_id", nid).execute()
            bc.supabase.table("knowledge_notes").delete().eq("id", nid).execute()


def _cleanup_events(*pairs):
    """Each arg is either an event_id (str) or an (event_id, connection_id)
    tuple as returned by _insert_test_calendar_event."""
    for p in pairs:
        if not p:
            continue
        event_id, connection_id = p if isinstance(p, tuple) else (p, None)
        if event_id:
            bc.supabase.table("calendar_events").delete().eq("id", event_id).execute()
        if connection_id:
            _delete_test_connection(connection_id)


# =====================================================================
# 1. Workspace scoping
# =====================================================================

def test_workspace_id_required():
    with pytest.raises(ValueError):
        canonical.get_canonical_knowledge(workspace_id="", sensitivity_ceiling=ALL_SENSITIVITIES)


def test_only_requested_workspace_returned():
    ws_other = str(uuid.uuid4())
    note_a = note_b = None
    try:
        note_a = _insert_test_note(TEST_COMPANY_1_WS, title="WS-A-CONSUMER-NOTE")
        note_b = _insert_test_note(ws_other, title="WS-B-CONSUMER-NOTE")

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["slack"])
        titles = {it.title for it in result.items}
        assert "WS-A-CONSUMER-NOTE" in titles
        assert "WS-B-CONSUMER-NOTE" not in titles
        assert all(it.workspace_id == TEST_COMPANY_1_WS for it in result.items)
    finally:
        _cleanup_notes(note_a, note_b)


# =====================================================================
# 2. Sensitivity enforcement
# =====================================================================

def test_sensitivity_ceiling_required_and_nonempty():
    with pytest.raises(ValueError):
        canonical.get_canonical_knowledge(workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=[])
    with pytest.raises(ValueError):
        canonical.get_canonical_knowledge(workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=None)


def test_sensitivity_ceiling_excludes_disallowed_notes():
    note_pub = note_conf = None
    try:
        note_pub = _insert_test_note(TEST_COMPANY_1_WS, sensitivity="public", title="CONSUMER-PUBLIC-NOTE")
        note_conf = _insert_test_note(TEST_COMPANY_1_WS, sensitivity="confidential", title="CONSUMER-CONF-NOTE")

        member_ceiling = ["public", "internal"]  # matches _resolve_allowed_sensitivities's member tier
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=member_ceiling, sources=["slack"])
        titles = {it.title for it in result.items}
        assert "CONSUMER-PUBLIC-NOTE" in titles
        assert "CONSUMER-CONF-NOTE" not in titles
    finally:
        _cleanup_notes(note_pub, note_conf)


def test_calendar_items_never_sensitivity_filtered():
    """Calendar has no sensitivity concept at all -- must pass through
    regardless of how restrictive the ceiling is, by design."""
    event_id = None
    try:
        event_id = _insert_test_calendar_event(TEST_COMPANY_1_WS, title="CONSUMER-CAL-EVENT")
        narrowest_ceiling = ["public"]
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=narrowest_ceiling, sources=["calendar"])
        titles = {it.title for it in result.items}
        assert "CONSUMER-CAL-EVENT" in titles
    finally:
        _cleanup_events(event_id)


# =====================================================================
# 3. record_status filtering (+ 12, 13: deleted exclusion)
# =====================================================================

def test_non_active_note_status_excluded():
    note_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-NONACTIVE-NOTE", status="not_active_test")
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["slack"])
        titles = {it.title for it in result.items}
        assert "CONSUMER-NONACTIVE-NOTE" not in titles
    finally:
        _cleanup_notes(note_id)


def test_13_deleted_calendar_event_excluded():
    event_id = None
    try:
        event_id = _insert_test_calendar_event(TEST_COMPANY_1_WS, deleted=True, title="CONSUMER-DELETED-CAL-EVENT")
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["calendar"])
        titles = {it.title for it in result.items}
        assert "CONSUMER-DELETED-CAL-EVENT" not in titles
    finally:
        _cleanup_events(event_id)


def test_12_document_source_never_leaks_any_result_deleted_or_not():
    """'document' is never orchestrated by this interface at all (see
    UNAVAILABLE handling, test 11) -- so a deleted document can never leak
    through here either, trivially. The actual record_status=deleted
    DERIVATION for documents is proven at the projection layer in
    test_phase3_hardening.py::test_deleted_document_record_status_is_deleted."""
    result = canonical.get_canonical_knowledge(
        workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["document"])
    assert result.items == []


# =====================================================================
# 4. Source filtering
# =====================================================================

def test_sources_none_means_every_readable_source():
    note_id = event_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-DEFAULT-SOURCES-NOTE")
        event_id = _insert_test_calendar_event(TEST_COMPANY_1_WS, title="CONSUMER-DEFAULT-SOURCES-EVENT")
        result = canonical.get_canonical_knowledge(workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES)
        titles = {it.title for it in result.items}
        assert "CONSUMER-DEFAULT-SOURCES-NOTE" in titles
        assert "CONSUMER-DEFAULT-SOURCES-EVENT" in titles
    finally:
        _cleanup_notes(note_id)
        _cleanup_events(event_id)


def test_explicit_empty_sources_returns_nothing():
    """sources=[] means 'read nothing', never silently reinterpreted as
    'give me everything'."""
    note_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-SHOULD-NOT-APPEAR")
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=[])
        assert result.items == []
    finally:
        _cleanup_notes(note_id)


def test_source_filter_only_returns_requested_sources():
    note_id = event_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-ONLY-CALENDAR-TEST-NOTE")
        event_id = _insert_test_calendar_event(TEST_COMPANY_1_WS, title="CONSUMER-ONLY-CALENDAR-TEST-EVENT")
        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["calendar"])
        titles = {it.title for it in result.items}
        assert "CONSUMER-ONLY-CALENDAR-TEST-EVENT" in titles
        assert "CONSUMER-ONLY-CALENDAR-TEST-NOTE" not in titles
        assert all(it.source == "calendar" for it in result.items)
    finally:
        _cleanup_notes(note_id)
        _cleanup_events(event_id)


def test_unknown_source_reported_not_silently_dropped():
    result = canonical.get_canonical_knowledge(
        workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["not_a_real_source"])
    assert "not_a_real_source" in result.unavailable_sources
    assert result.items == []


# =====================================================================
# 5. Canonical field stability / 15. no source-specific schema leakage / 16. no confidence
# =====================================================================

def test_canonical_field_stability_and_no_schema_leakage():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(canonical.CanonicalKnowledge)}

    # Stable, expected canonical fields.
    for expected in ("id", "workspace_id", "connection_id", "source", "source_type",
                     "title", "content", "sensitivity", "authority", "record_status",
                     "lifecycle_status", "processing_status", "provenance",
                     "external_references"):
        assert expected in field_names

    # No confidence (Test 16).
    assert "confidence" not in field_names

    # No raw physical/source-specific column names leaking onto the
    # canonical object itself (Test 15) -- these belong inside
    # ProvenanceEvidence, relabeled, never as top-level canonical fields.
    for leaked in ("channel_id", "thread_ts", "message_ts", "deleted_at",
                   "processing_started_at", "external_file_id"):
        assert leaked not in field_names


# =====================================================================
# 6. Provenance opt-in
# =====================================================================

def test_provenance_opt_in_default_false():
    note_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-PROVENANCE-DEFAULT-NOTE")
        _insert_test_note_source(note_id, TEST_COMPANY_1_WS, "slack", "C-TEST-CHANNEL")

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["slack"])
        matched = [it for it in result.items if it.title == "CONSUMER-PROVENANCE-DEFAULT-NOTE"]
        assert matched and matched[0].provenance == [], "provenance must default to [] when not opted in"
    finally:
        _cleanup_notes(note_id)


def test_provenance_opt_in_true_returns_canonical_shape():
    note_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-PROVENANCE-TRUE-NOTE")
        _insert_test_note_source(note_id, TEST_COMPANY_1_WS, "slack", "C-TEST-CHANNEL-2")

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES,
            sources=["slack"], include_provenance=True)
        matched = [it for it in result.items if it.title == "CONSUMER-PROVENANCE-TRUE-NOTE"]
        assert matched and len(matched[0].provenance) == 1
        ev = matched[0].provenance[0]
        assert ev.container_ref == "C-TEST-CHANNEL-2"
        assert not hasattr(ev, "channel_id"), "canonical names only -- never the raw physical column name"
    finally:
        _cleanup_notes(note_id)


# =====================================================================
# 7. External-reference opt-in
# =====================================================================

def test_external_references_opt_in_default_false():
    note_id = ref_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-EXTREF-DEFAULT-NOTE")
        ref_row = bc.supabase.table("external_references").insert({
            "workspace_id": TEST_COMPANY_1_WS, "provider": "google_drive",
            "external_file_id": f"test-file-{uuid.uuid4()}", "title": "Test File",
            "url": "https://drive.google.com/file/d/testfile/view", "modified_time": "2026-08-01T00:00:00Z",
            "linked_object_type": "knowledge_note", "linked_object_id": note_id,
        }).execute().data
        ref_id = ref_row[0]["id"]

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["slack"])
        matched = [it for it in result.items if it.title == "CONSUMER-EXTREF-DEFAULT-NOTE"]
        assert matched and matched[0].external_references == []
    finally:
        if ref_id:
            bc.supabase.table("external_references").delete().eq("id", ref_id).execute()
        _cleanup_notes(note_id)


def test_external_references_opt_in_true_not_merged_into_content():
    note_id = ref_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-EXTREF-TRUE-NOTE")
        ref_row = bc.supabase.table("external_references").insert({
            "workspace_id": TEST_COMPANY_1_WS, "provider": "google_drive",
            "external_file_id": f"test-file2-{uuid.uuid4()}", "title": "Test File 2",
            "url": "https://drive.google.com/file/d/testfile2/view", "modified_time": "2026-08-01T00:00:00Z",
            "linked_object_type": "knowledge_note", "linked_object_id": note_id,
        }).execute().data
        ref_id = ref_row[0]["id"]

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES,
            sources=["slack"], include_external_references=True)
        matched = [it for it in result.items if it.title == "CONSUMER-EXTREF-TRUE-NOTE"]
        assert matched and len(matched[0].external_references) == 1
        assert "testfile2" not in matched[0].content
    finally:
        if ref_id:
            bc.supabase.table("external_references").delete().eq("id", ref_id).execute()
        _cleanup_notes(note_id)


# =====================================================================
# 8. Calendar canonical read (real row)
# =====================================================================

def test_calendar_canonical_read_real_row():
    result = canonical.get_canonical_knowledge(
        workspace_id=REAL_WORKSPACE, sensitivity_ceiling=ALL_SENSITIVITIES,
        sources=["calendar"], limit=200)
    matched = [it for it in result.items if it.id == REAL_CALENDAR_EVENT_ID]
    assert matched, "real fixture calendar event no longer in the first 200 -- update the fixture/limit"
    ev = matched[0]
    assert ev.source == "calendar"
    assert ev.event_start is not None
    assert ev.event_end is not None
    assert ev.record_status == "active"
    assert ev.source_updated_at is not None
    assert ev.workspace_id == REAL_WORKSPACE
    assert ev.sensitivity is None, "Calendar must never carry an invented classification"


# =====================================================================
# 9. Chat/Slack canonical read (real row)
# =====================================================================

def test_chat_canonical_read_real_row():
    result = canonical.get_canonical_knowledge(
        workspace_id=REAL_WORKSPACE, sensitivity_ceiling=ALL_SENSITIVITIES,
        sources=["google_chat"], limit=200)
    matched = [it for it in result.items if it.id == REAL_CHAT_NOTE_ID]
    assert matched, "real fixture Chat note no longer in the first 200 -- update the fixture/limit"
    note = matched[0]
    assert note.source == "google_chat"
    assert note.record_status == "active"
    assert note.sensitivity == "internal"
    assert note.authority == "official"


# =====================================================================
# 10. Meet canonical read shape (synthetic -- no real transcript-derived note exists yet)
# =====================================================================

def test_meet_canonical_read_shape_and_conference_link():
    """No real Meet transcript-derived note exists yet (Phase 2's known,
    plan-gated gap) -- synthetic note + synthetic source row, linked to the
    REAL calendar event's real conference_id, proving both the shape and
    the exact-match Meet<->Calendar resolution end to end through this
    interface."""
    note_id = None
    try:
        note_id = _insert_test_note(TEST_COMPANY_1_WS, title="CONSUMER-MEET-SHAPE-NOTE", provider="google_meet")
        _insert_test_note_source(note_id, TEST_COMPANY_1_WS, "google_meet", "conferenceRecords/synthetic-c1")

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["google_meet"])
        matched = [it for it in result.items if it.title == "CONSUMER-MEET-SHAPE-NOTE"]
        assert matched
        assert matched[0].conference_id == "conferenceRecords/synthetic-c1"
        assert matched[0].calendar_event_id is None, "no real calendar event has this synthetic conference_id"
    finally:
        _cleanup_notes(note_id)


def test_meet_calendar_event_id_resolves_when_conference_id_matches_real_event():
    note_id = None
    try:
        note_id = _insert_test_note(REAL_WORKSPACE, title="CONSUMER-MEET-REAL-LINK-NOTE", provider="google_meet")
        _insert_test_note_source(note_id, REAL_WORKSPACE, "google_meet", REAL_CALENDAR_CONFERENCE_ID)

        result = canonical.get_canonical_knowledge(
            workspace_id=REAL_WORKSPACE, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["google_meet"])
        matched = [it for it in result.items if it.title == "CONSUMER-MEET-REAL-LINK-NOTE"]
        assert matched
        assert matched[0].calendar_event_id == REAL_CALENDAR_EVENT_ID
    finally:
        _cleanup_notes(note_id)


def test_no_meet_note_means_no_meet_item_fabricated():
    """If there is no transcript-derived note, there is no Meet
    CanonicalKnowledge item -- proven with a workspace that genuinely has
    none under this synthetic id."""
    ws_empty = str(uuid.uuid4())
    result = canonical.get_canonical_knowledge(
        workspace_id=ws_empty, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["google_meet"])
    assert result.items == []
    assert "google_meet" not in result.unavailable_sources, \
        "an empty result is not the same as an unavailable source -- google_meet IS orchestrated, it just has no rows here"


# =====================================================================
# 11. Document behavior + safe-read limitation
# =====================================================================

def test_document_source_reported_unavailable_not_fabricated():
    result = canonical.get_canonical_knowledge(
        workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES, sources=["document"])
    assert result.items == []
    assert "document" in result.unavailable_sources
    assert "service-role" in result.unavailable_sources["document"].lower()


# =====================================================================
# 14. No cross-workspace results (additional case beyond test #1)
# =====================================================================

def test_no_cross_workspace_leakage_across_all_sources():
    ws_b = str(uuid.uuid4())
    note_id = event_id = None
    try:
        note_id = _insert_test_note(ws_b, title="CONSUMER-CROSSWS-NOTE")
        event_id = _insert_test_calendar_event(ws_b, title="CONSUMER-CROSSWS-EVENT")

        result = canonical.get_canonical_knowledge(
            workspace_id=TEST_COMPANY_1_WS, sensitivity_ceiling=ALL_SENSITIVITIES)
        titles = {it.title for it in result.items}
        assert "CONSUMER-CROSSWS-NOTE" not in titles
        assert "CONSUMER-CROSSWS-EVENT" not in titles
    finally:
        _cleanup_notes(note_id)
        _cleanup_events(event_id)


# =====================================================================
# 17. Retrieval evidence remains external to the canonical object
# =====================================================================

def test_no_embeddings_or_chunk_data_in_canonical_object():
    import dataclasses
    import inspect
    field_names = {f.name for f in dataclasses.fields(canonical.CanonicalKnowledge)}
    assert "embedding" not in field_names
    assert "chunks" not in field_names
    assert "content_tsv" not in field_names

    src = inspect.getsource(canonical.get_canonical_knowledge)
    assert '.table("document_chunks")' not in src, \
        "the consumer interface must never query document_chunks -- that stays retrieval infrastructure"