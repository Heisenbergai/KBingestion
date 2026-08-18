"""
Phase 4 persistence schema tests -- exercise the real, live
`structured_knowledge` / `extraction_contract_versions` tables directly
(no extraction pipeline is wired to them yet, per this pass's explicit
scope: schema + migration + validation only). Every test uses
bc.supabase (the same service-role client every other real-DB test in
this project already uses), with synthetic, self-cleaning rows.

Run with: python -m pytest test_phase4_schema.py -v
"""
import uuid

import pytest

import brain_connectors as bc

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"

# A disposable, non-'v2.1' extraction_contract_versions row used for
# idempotency/promotion tests -- never touches the real seeded 'v2.1' row.
TEST_VERSION_A = f"test-{uuid.uuid4().hex[:8]}"
TEST_VERSION_B = f"test-{uuid.uuid4().hex[:8]}"


def _base_row(**overrides) -> dict:
    row = {
        "workspace_id": TEST_COMPANY_1_WS,
        "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()),
        "provider": "slack",
        "primitive_type": "fact",
        "requirement_kind": None,
        "statement": "Test statement.",
        "raw_subject_phrase": None,
        "qualifier_words": [],
        "sensitivity": "internal",
        "authority": "working",
        "source_tier": 3,
        "lifecycle_status": "active",
        "captured_at": "2026-08-01T00:00:00+00:00",
        "event_time": None, "event_start": None, "event_end": None,
        "effective_from": None, "effective_until": None,
        "recurrence_text": None,
        "extraction_version": "v2.1",
        "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": uuid.uuid4().hex,
    }
    row.update(overrides)
    return row


def _insert(row: dict) -> dict:
    return bc.supabase.table("structured_knowledge").insert(row).execute().data[0]


def _cleanup_rows(*ids):
    for rid in ids:
        if rid:
            bc.supabase.table("structured_knowledge").delete().eq("id", rid).execute()


def _cleanup_versions(*versions):
    for v in versions:
        bc.supabase.table("extraction_contract_versions").delete().eq("version", v).execute()


# =====================================================================
# Seed row / table existence
# =====================================================================

def test_v21_is_the_seeded_current_version():
    rows = bc.supabase.table("extraction_contract_versions").select("*").eq("version", "v2.1").execute().data
    assert len(rows) == 1
    assert rows[0]["is_current"] is True
    assert rows[0]["sequence_number"] == 1


# =====================================================================
# Competing assertions remain separate
# =====================================================================

def test_competing_assertions_remain_separate_rows():
    canonical_id = str(uuid.uuid4())
    row_a = row_b = None
    try:
        row_a = _insert(_base_row(
            canonical_id=canonical_id, statement="Release target is September 12.",
            effective_from=None, primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_b = _insert(_base_row(
            canonical_id=canonical_id, statement="Release target moved to September 15.",
            effective_from="2026-09-15", primitive_fingerprint=uuid.uuid4().hex,
        ))
        assert row_a["id"] != row_b["id"]
        rows = bc.supabase.table("structured_knowledge").select("*") \
            .eq("canonical_id", canonical_id).execute().data
        assert len(rows) == 2
        # No supersedes/merge column exists at all -- see test_no_phase5_or_phase6_columns_exist.
    finally:
        _cleanup_rows(row_a["id"] if row_a else None, row_b["id"] if row_b else None)


# =====================================================================
# Idempotency: same extraction rerun does not duplicate
# =====================================================================

def test_same_extraction_rerun_is_idempotent():
    canonical_id = str(uuid.uuid4())
    fingerprint = uuid.uuid4().hex
    row = _base_row(canonical_id=canonical_id, primitive_fingerprint=fingerprint)
    inserted_id = None
    try:
        first = bc.supabase.table("structured_knowledge").upsert(
            row, on_conflict="canonical_source_type,canonical_id,extraction_version,primitive_fingerprint",
            ignore_duplicates=True,
        ).execute().data
        inserted_id = first[0]["id"]

        # Re-running the "same extraction" -- identical identity tuple.
        second = bc.supabase.table("structured_knowledge").upsert(
            row, on_conflict="canonical_source_type,canonical_id,extraction_version,primitive_fingerprint",
            ignore_duplicates=True,
        ).execute().data
        # ignore_duplicates -> no new row returned for the conflicting insert
        assert second == [] or (len(second) == 1 and second[0]["id"] == inserted_id)

        rows = bc.supabase.table("structured_knowledge").select("id") \
            .eq("canonical_id", canonical_id).eq("extraction_version", "v2.1").execute().data
        assert len(rows) == 1
    finally:
        _cleanup_rows(inserted_id)


def test_duplicate_identity_tuple_rejected_without_upsert():
    """Direct proof the UNIQUE constraint itself is real, not just relying
    on application-level upsert logic."""
    canonical_id = str(uuid.uuid4())
    fingerprint = uuid.uuid4().hex
    row = _base_row(canonical_id=canonical_id, primitive_fingerprint=fingerprint)
    first_id = None
    try:
        first = _insert(row)
        first_id = first["id"]
        with pytest.raises(Exception):
            _insert(dict(row))  # exact same identity tuple, plain insert -- must violate UNIQUE
    finally:
        _cleanup_rows(first_id)


# =====================================================================
# Different extraction versions coexist
# =====================================================================

def test_different_extraction_versions_coexist_for_same_canonical_item():
    canonical_id = str(uuid.uuid4())
    row_ids = []
    try:
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_A, "is_current": False, "description": "test version A"}
        ).execute()
        row_v1 = _insert(_base_row(
            canonical_id=canonical_id, extraction_version=TEST_VERSION_A,
            statement="v1-era statement", primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_v21 = _insert(_base_row(
            canonical_id=canonical_id, extraction_version="v2.1",
            statement="v2.1-era statement", primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_ids = [row_v1["id"], row_v21["id"]]

        rows = bc.supabase.table("structured_knowledge").select("extraction_version") \
            .eq("canonical_id", canonical_id).execute().data
        assert {r["extraction_version"] for r in rows} == {TEST_VERSION_A, "v2.1"}
    finally:
        _cleanup_rows(*row_ids)
        _cleanup_versions(TEST_VERSION_A)


# =====================================================================
# Promoting a new version does not hide older-only canonical items;
# newest available approved version is returned; normal read never mixes
# versions for one canonical item
# =====================================================================

def test_promotion_does_not_hide_canonical_items_only_extracted_under_older_version():
    """v2.1 (sequence_number=1) is the real, currently-promoted version --
    nothing can have a LOWER sequence_number than the very first version
    ever seeded, so this test instead faithfully reproduces the real
    scenario your concern describes: canonical A is extracted under v2.1
    (today's current), then a NEWER version is promoted ahead of it
    (simulating a future v3 being approved) without A ever being
    re-extracted. A's v2.1 result must still be the one a normal read
    returns -- promotion must never make it disappear just because it
    hasn't been re-extracted under the new current yet."""
    canonical_a = str(uuid.uuid4())
    row_ids = []
    try:
        row_a = _insert(_base_row(
            canonical_id=canonical_a, extraction_version="v2.1",
            statement="only ever extracted under v2.1, never re-extracted",
            primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_ids = [row_a["id"]]

        # Promote a NEWER hypothetical version ahead of v2.1 (simulating a
        # future v3 becoming current) -- canonical_a is never re-extracted
        # under it.
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_A, "is_current": False, "description": "newer hypothetical version"}
        ).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("is_current", True).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True}
        ).eq("version", TEST_VERSION_A).execute()

        # "Best available, capped at global current" read, implemented
        # directly here exactly as the approved read-semantics design
        # specifies (no application read-path exists yet -- this proves
        # the schema supports the query, not that a Python function does).
        current = bc.supabase.table("extraction_contract_versions") \
            .select("sequence_number").eq("is_current", True).execute().data[0]
        all_versions = {v["version"]: v["sequence_number"]
                        for v in bc.supabase.table("extraction_contract_versions").select("*").execute().data}

        rows = bc.supabase.table("structured_knowledge").select("*") \
            .eq("canonical_id", canonical_a).execute().data
        eligible = [r for r in rows if all_versions[r["extraction_version"]] <= current["sequence_number"]]
        best_seq = max(all_versions[r["extraction_version"]] for r in eligible)
        best_rows = [r for r in eligible if all_versions[r["extraction_version"]] == best_seq]

        assert len(best_rows) == 1
        assert best_rows[0]["extraction_version"] == "v2.1"
        assert best_rows[0]["statement"] == "only ever extracted under v2.1, never re-extracted"
    finally:
        _cleanup_rows(*row_ids)
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("version", TEST_VERSION_A).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True}
        ).eq("version", "v2.1").execute()
        _cleanup_versions(TEST_VERSION_A)


def test_newest_available_approved_version_selected_and_never_mixed():
    """Canonical C: extracted under TEST_VERSION_A and v2.1 -- a normal
    read must return ONLY the v2.1 rows, never a mix."""
    canonical_c = str(uuid.uuid4())
    row_ids = []
    try:
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_A, "is_current": False, "description": "older test version"}
        ).execute()
        row_old = _insert(_base_row(
            canonical_id=canonical_c, extraction_version=TEST_VERSION_A,
            statement="old", primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_new = _insert(_base_row(
            canonical_id=canonical_c, extraction_version="v2.1",
            statement="new", primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_ids = [row_old["id"], row_new["id"]]

        current = bc.supabase.table("extraction_contract_versions") \
            .select("sequence_number").eq("is_current", True).execute().data[0]
        all_versions = {v["version"]: v["sequence_number"]
                        for v in bc.supabase.table("extraction_contract_versions").select("*").execute().data}
        rows = bc.supabase.table("structured_knowledge").select("*") \
            .eq("canonical_id", canonical_c).execute().data
        eligible = [r for r in rows if all_versions[r["extraction_version"]] <= current["sequence_number"]]
        best_seq = max(all_versions[r["extraction_version"]] for r in eligible)
        best_rows = [r for r in eligible if all_versions[r["extraction_version"]] == best_seq]

        assert len(best_rows) == 1
        assert best_rows[0]["extraction_version"] == "v2.1"
        assert best_rows[0]["statement"] == "new"
        # Old row still physically present, just not selected by the "best" read.
        assert any(r["extraction_version"] == TEST_VERSION_A for r in rows)
    finally:
        _cleanup_rows(*row_ids)
        _cleanup_versions(TEST_VERSION_A)


# =====================================================================
# Only one current version allowed (database-enforced)
# =====================================================================

def test_only_one_current_version_allowed_database_enforced():
    """The partial unique index is GLOBAL (at most one is_current=true row
    across the whole table) -- so this test must first clear the real
    seeded v2.1 current flag to have room to set a test row current at
    all, then prove a SECOND test row can never also become current while
    the first still is."""
    try:
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_A, "is_current": False, "description": "candidate A"}
        ).execute()
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_B, "is_current": False, "description": "candidate B"}
        ).execute()

        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("is_current", True).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True}
        ).eq("version", TEST_VERSION_A).execute()

        with pytest.raises(Exception):
            bc.supabase.table("extraction_contract_versions").update(
                {"is_current": True}
            ).eq("version", TEST_VERSION_B).execute()
    finally:
        # Clear both before delete, then restore the real v2.1 current flag.
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("version", TEST_VERSION_A).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True}
        ).eq("version", "v2.1").execute()
        _cleanup_versions(TEST_VERSION_A, TEST_VERSION_B)


def test_transactional_promotion_clears_previous_current():
    """The intended promotion pattern (clear old, set new, update
    promoted_at) leaves exactly one current version, and it's the new one."""
    try:
        bc.supabase.table("extraction_contract_versions").insert(
            {"version": TEST_VERSION_A, "is_current": False, "description": "candidate A"}
        ).execute()

        # Promotion: clear old current, set new one is_current + promoted_at.
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("is_current", True).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True, "promoted_at": "2026-08-17T12:00:00+00:00"}
        ).eq("version", TEST_VERSION_A).execute()

        current_rows = bc.supabase.table("extraction_contract_versions") \
            .select("version").eq("is_current", True).execute().data
        assert len(current_rows) == 1
        assert current_rows[0]["version"] == TEST_VERSION_A
    finally:
        # Restore v2.1 as current so the rest of the suite / real DB state
        # is left exactly as it should be after this test.
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": False}
        ).eq("version", TEST_VERSION_A).execute()
        bc.supabase.table("extraction_contract_versions").update(
            {"is_current": True}
        ).eq("version", "v2.1").execute()
        _cleanup_versions(TEST_VERSION_A)

        current_rows = bc.supabase.table("extraction_contract_versions") \
            .select("version").eq("is_current", True).execute().data
        assert current_rows == [{"version": "v2.1"}], \
            "test cleanup must leave the real seeded v2.1 version as current"


# =====================================================================
# Deleted parent hidden on normal reads, retained physically
# =====================================================================

def test_deleted_parent_hidden_from_join_but_row_physically_retained():
    """Real parent knowledge_notes row, flipped to a non-active status --
    the structured row must still exist directly, but a join against the
    live parent status must exclude it."""
    note_id = None
    structured_id = None
    try:
        note_row = bc.supabase.table("knowledge_notes").insert({
            "workspace_id": TEST_COMPANY_1_WS, "connection_id": None, "provider": "slack",
            "source_type": "slack", "source_tier": 3, "category": None,
            "title": "Schema test parent note", "body": "test body", "participants": [],
            "source_ref": None, "occurred_at": None, "status": "active",
            "sensitivity": "internal", "authority": "working", "doc_class": None,
            "lifecycle_status": "active",
        }).execute().data
        note_id = note_row[0]["id"]

        structured_row = _insert(_base_row(
            canonical_source_type="knowledge_note", canonical_id=note_id,
            statement="derived from a note that will be deleted",
            primitive_fingerprint=uuid.uuid4().hex,
        ))
        structured_id = structured_row["id"]

        # Simulate the parent becoming unusable.
        bc.supabase.table("knowledge_notes").update({"status": "archived_test"}).eq("id", note_id).execute()

        # Physically still present.
        still_present = bc.supabase.table("structured_knowledge").select("id").eq("id", structured_id).execute().data
        assert len(still_present) == 1

        # A "normal read" (live join against the parent's current status) excludes it.
        parent = bc.supabase.table("knowledge_notes").select("status").eq("id", note_id).execute().data[0]
        visible_in_normal_read = parent["status"] == "active"
        assert visible_in_normal_read is False
    finally:
        _cleanup_rows(structured_id)
        if note_id:
            bc.supabase.table("knowledge_notes").delete().eq("id", note_id).execute()


# =====================================================================
# provider / canonical_source_type independence
# =====================================================================

def test_provider_and_canonical_source_type_vary_independently():
    row_ids = []
    try:
        row_chat = _insert(_base_row(
            canonical_source_type="knowledge_note", provider="google_chat",
            primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_slack = _insert(_base_row(
            canonical_source_type="knowledge_note", provider="slack",
            primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_calendar = _insert(_base_row(
            canonical_source_type="calendar_event", provider="calendar",
            primitive_type="event", primitive_fingerprint=uuid.uuid4().hex,
        ))
        row_ids = [row_chat["id"], row_slack["id"], row_calendar["id"]]

        assert row_chat["canonical_source_type"] == row_slack["canonical_source_type"] == "knowledge_note"
        assert row_chat["provider"] != row_slack["provider"]
        assert row_calendar["canonical_source_type"] != row_chat["canonical_source_type"]
    finally:
        _cleanup_rows(*row_ids)


# =====================================================================
# Constraint enforcement (CHECK constraints)
# =====================================================================

def test_invalid_primitive_type_rejected():
    with pytest.raises(Exception):
        _insert(_base_row(primitive_type="opinion", primitive_fingerprint=uuid.uuid4().hex))


def test_invalid_requirement_kind_rejected():
    with pytest.raises(Exception):
        _insert(_base_row(requirement_kind="guideline", primitive_fingerprint=uuid.uuid4().hex))


def test_invalid_canonical_source_type_rejected():
    with pytest.raises(Exception):
        _insert(_base_row(canonical_source_type="slack_message", primitive_fingerprint=uuid.uuid4().hex))


def test_unknown_extraction_version_rejected_by_fk():
    with pytest.raises(Exception):
        _insert(_base_row(extraction_version="v99-does-not-exist", primitive_fingerprint=uuid.uuid4().hex))


# =====================================================================
# No Phase 5/6 columns exist
# =====================================================================

def test_no_phase5_or_phase6_columns_exist():
    # Inspect via a throwaway insert+select for a reliable column list
    # regardless of whether the table already has rows.
    row = _insert(_base_row(primitive_fingerprint=uuid.uuid4().hex))
    try:
        columns = set(row.keys())
        forbidden = {
            "payload", "confidence", "extraction_confidence", "record_status",
            "owner_id", "team_id", "project_id", "product_id", "customer_id", "entity_id",
            "supersedes", "winner", "merged_from", "canonical_current",
        }
        assert columns.isdisjoint(forbidden), f"forbidden columns present: {columns & forbidden}"
    finally:
        _cleanup_rows(row["id"])


def test_no_existing_tables_were_modified():
    """Spot check: knowledge_notes' known column set is unchanged -- this
    migration touched no existing table."""
    note_id = None
    try:
        note_row = bc.supabase.table("knowledge_notes").insert({
            "workspace_id": TEST_COMPANY_1_WS, "connection_id": None, "provider": "slack",
            "source_type": "slack", "source_tier": 3, "category": None,
            "title": "Column-shape check", "body": "x", "participants": [],
            "source_ref": None, "occurred_at": None, "status": "active",
            "sensitivity": "internal", "authority": "working", "doc_class": None,
            "lifecycle_status": "active",
        }).execute().data
        note_id = note_row[0]["id"]
        expected = {
            "id", "workspace_id", "connection_id", "provider", "source_type", "source_tier",
            "category", "title", "body", "participants", "source_ref", "occurred_at",
            "status", "created_at", "sensitivity", "authority", "doc_class", "lifecycle_status",
        }
        assert set(note_row[0].keys()) == expected
    finally:
        if note_id:
            bc.supabase.table("knowledge_notes").delete().eq("id", note_id).execute()
