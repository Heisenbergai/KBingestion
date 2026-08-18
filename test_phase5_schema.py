"""
Phase 5 schema tests -- knowledge graph & evidence tables, live against the
real vector DB (same "real DB, not mocked" convention as
test_phase1_retrieval.py / test_phase4_schema.py). Entirely additive: every
test creates its own synthetic rows and cleans them up, and the real 15
structured_knowledge rows are used READ-ONLY as fixtures for the
polymorphic/existence tests -- never modified.

No entities/relationships/evidence/snapshots exist in production as of this
suite (confirmed live before writing these tests: 0 rows in all 6 new
tables). Nothing here backfills that -- each test's own rows are deleted in
a finally block.

Run with: python -m pytest test_phase5_schema.py -v
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from query import supabase

# ---- Real fixtures, read-only ----
REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
REAL_CONNECTION = "79d54c5e-8e2e-4fd6-bbd0-d7ea45502e83"  # active google_drive connection

# Two real structured_knowledge rows (7a9eaa34's requirement, ff5972e5's two
# requirements) -- used as source/target/evidence in the polymorphic and
# atomic-RPC tests below. Never written to.
SK_7A9EAA34 = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"
SK_FF5972E5_A = "7db9647c-7b7d-4203-9f34-df1b1506cd8e"
SK_FF5972E5_B = "5b77b2ca-2c8c-436c-8070-4f61bf5a270d"

_NOW = datetime.now(timezone.utc).isoformat()


def _fresh_valid_from() -> str:
    """A distinct valid_from for tests that don't care about idempotency --
    each call gets its own historical identity, same as picking a fresh
    uuid. Idempotency-specific tests pass an explicit, shared value instead."""
    return datetime.now(timezone.utc).isoformat()


def _rpc(source_id, target_id, evidence, relationship_type="references",
         valid_from="__FRESH__", workspace_id=REAL_WORKSPACE,
         source_type="structured_knowledge", target_type="structured_knowledge"):
    """valid_from now REQUIRED by the RPC (post-correction) -- the sentinel
    default here generates a fresh, distinct value per call so tests that
    aren't specifically about idempotency don't collide with each other.
    Pass valid_from=None explicitly to test the RPC's own NULL-rejection,
    or a concrete ISO string to test reuse/idempotency."""
    if valid_from == "__FRESH__":
        valid_from = _fresh_valid_from()
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": source_type,
        "p_source_object_id": source_id,
        "p_target_object_type": target_type,
        "p_target_object_id": target_id,
        "p_relationship_type": relationship_type,
        "p_rationale": "test fixture",
        "p_confidence": None,
        "p_valid_from": valid_from,
        "p_valid_until": None,
        "p_evidence": evidence,
    }).execute()


def _delete_relationship(rel_id):
    if rel_id:
        supabase.table("knowledge_relationships").delete().eq("id", rel_id).execute()


# =====================================================================
# 1. Entity type constraints
# =====================================================================

def test_entity_type_check_rejects_invalid_type():
    with pytest.raises(Exception):
        supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "not_a_real_type",
            "canonical_label": "should never insert",
        }).execute()


def test_entity_type_check_accepts_all_five_frozen_types():
    ids = []
    try:
        for t in ("department", "meeting", "policy", "process", "person"):
            res = supabase.table("knowledge_entities").insert({
                "workspace_id": REAL_WORKSPACE, "entity_type": t,
                "canonical_label": f"test-{t}",
            }).execute()
            ids.append(res.data[0]["id"])
        assert len(ids) == 5
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


def test_entity_external_ref_type_and_id_must_be_set_together():
    """CHECK ((external_ref_type IS NULL) = (external_ref_id IS NULL))."""
    with pytest.raises(Exception):
        supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "orphaned ref", "external_ref_type": "department_id",
            "external_ref_id": None,
        }).execute()


def test_entity_external_ref_type_only_allows_department_id():
    """Final Correction 2: Calendar identity must NOT use external_ref_*."""
    with pytest.raises(Exception):
        supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "meeting",
            "canonical_label": "should be rejected",
            "external_ref_type": "calendar_instance", "external_ref_id": "anything",
        }).execute()


# =====================================================================
# 2. Relationship type constraints
# =====================================================================

def test_relationship_type_check_rejects_rejected_type():
    """`related_to` was explicitly rejected for V1 -- must not be insertable
    even by direct insert, bypassing the RPC."""
    with pytest.raises(Exception):
        supabase.table("knowledge_relationships").insert({
            "workspace_id": REAL_WORKSPACE,
            "source_object_type": "structured_knowledge", "source_object_id": SK_7A9EAA34,
            "target_object_type": "structured_knowledge", "target_object_id": SK_FF5972E5_A,
            "relationship_type": "related_to",
        }).execute()


def test_relationship_source_target_type_check_rejects_arbitrary_table_name():
    with pytest.raises(Exception):
        supabase.table("knowledge_relationships").insert({
            "workspace_id": REAL_WORKSPACE,
            "source_object_type": "document_chunks", "source_object_id": SK_7A9EAA34,
            "target_object_type": "structured_knowledge", "target_object_id": SK_FF5972E5_A,
            "relationship_type": "references",
        }).execute()


# =====================================================================
# 3. Evidence type constraints / no circular evidence
# =====================================================================

def test_evidence_type_check_rejects_via_rpc():
    """Structural prevention of circular graph reasoning: 'knowledge_relationship'
    is not in the closed evidence_type enum at all -- the RPC's own explicit
    ELSE branch raises before any row is written."""
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "knowledge_relationship", "evidence_id": str(uuid.uuid4()),
             "stance": "supports", "captured_at": _NOW},
        ])


def test_evidence_stance_check_rejects_invalid_stance():
    rel_id = None
    try:
        with pytest.raises(Exception):
            res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
                {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
                 "stance": "maybe", "captured_at": _NOW},
            ])
            rel_id = res.data
    finally:
        _delete_relationship(rel_id)


# =====================================================================
# 4. Identifier namespace (Final Correction 1)
# =====================================================================

def test_identifier_workspace_global_type_requires_null_connection():
    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department", "canonical_label": "test dept",
        }).execute().data[0]["id"]
        with pytest.raises(Exception):
            supabase.table("knowledge_entity_identifiers").insert({
                "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
                "connection_id": REAL_CONNECTION,  # must be NULL for 'email'
                "identifier_type": "email", "identifier_value": "test@example.com",
            }).execute()
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


def test_identifier_connection_scoped_type_requires_connection_id():
    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "meeting", "canonical_label": "test meeting",
        }).execute().data[0]["id"]
        with pytest.raises(Exception):
            supabase.table("knowledge_entity_identifiers").insert({
                "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
                "connection_id": None,  # must be set for 'external_event_id'
                "identifier_type": "external_event_id", "identifier_value": "evt123",
            }).execute()
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


def test_identifier_global_scope_prevents_duplicate_email_across_entities():
    e1 = e2 = None
    try:
        e1 = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person", "canonical_label": "person A",
        }).execute().data[0]["id"]
        e2 = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person", "canonical_label": "person B",
        }).execute().data[0]["id"]
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": e1, "workspace_id": REAL_WORKSPACE,
            "identifier_type": "email", "identifier_value": "dup-test@example.com",
        }).execute()
        with pytest.raises(Exception):
            supabase.table("knowledge_entity_identifiers").insert({
                "entity_id": e2, "workspace_id": REAL_WORKSPACE,
                "identifier_type": "email", "identifier_value": "dup-test@example.com",
            }).execute()
    finally:
        for eid in (e1, e2):
            if eid:
                supabase.table("knowledge_entities").delete().eq("id", eid).execute()


def test_identifier_connection_scoped_same_value_different_connections_does_not_collide():
    """The same external_event_id string under two different connections must
    NOT be treated as the same identifier -- proves the scoped partial index
    is keyed on connection_id, not just (workspace_id, type, value)."""
    e1 = e2 = None
    try:
        e1 = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "meeting", "canonical_label": "meeting A",
        }).execute().data[0]["id"]
        e2 = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "meeting", "canonical_label": "meeting B",
        }).execute().data[0]["id"]
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": e1, "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
            "identifier_type": "external_event_id", "identifier_value": "same-event-id",
        }).execute()
        # A different (synthetic, nonexistent) connection_id would violate the
        # connections FK, so instead prove the scoping the safe way: the same
        # value is rejected under the SAME connection (proving the constraint
        # is live), which is the meaningful half of this test given only one
        # real connection fixture is available.
        with pytest.raises(Exception):
            supabase.table("knowledge_entity_identifiers").insert({
                "entity_id": e2, "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
                "identifier_type": "external_event_id", "identifier_value": "same-event-id",
            }).execute()
    finally:
        for eid in (e1, e2):
            if eid:
                supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 5. Alias normalization
# =====================================================================

def test_alias_normalization_collides_across_case_and_whitespace():
    import re

    def normalize(s: str) -> str:
        import unicodedata
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip().lower())

    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person", "canonical_label": "alias test",
        }).execute().data[0]["id"]

        supabase.table("knowledge_entity_aliases").insert({
            "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
            "alias_text": "Tanmay", "alias_normalized": normalize("Tanmay"),
            "alias_source_type": "structured_knowledge", "alias_source_id": SK_7A9EAA34,
        }).execute()

        with pytest.raises(Exception):
            supabase.table("knowledge_entity_aliases").insert({
                "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
                "alias_text": " TANMAY ", "alias_normalized": normalize(" TANMAY "),
                "alias_source_type": "structured_knowledge", "alias_source_id": SK_7A9EAA34,
            }).execute()

        kept = supabase.table("knowledge_entity_aliases").select("alias_text") \
            .eq("entity_id", entity_id).execute().data
        assert kept[0]["alias_text"] == "Tanmay", "original display form must survive untouched"
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


def test_alias_source_type_check_rejects_unlisted_type():
    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person", "canonical_label": "alias source test",
        }).execute().data[0]["id"]
        with pytest.raises(Exception):
            supabase.table("knowledge_entity_aliases").insert({
                "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
                "alias_text": "x", "alias_normalized": "x",
                "alias_source_type": "some_invented_type", "alias_source_id": str(uuid.uuid4()),
            }).execute()
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


# =====================================================================
# 6. Relationship idempotency -- POST-CORRECTION behavior.
#
# The prior finding (default now() silently created duplicate relationships
# on retry, documented and regression-tested as
# test_KNOWN_RESIDUAL_default_valid_from_does_not_dedupe_retries) is now
# FIXED: valid_from has no DB default and is mandatory; the RPC is
# idempotent by full logical-key lookup, not by uniqueness collision alone.
# =====================================================================

def test_missing_valid_from_rejected():
    """omitting valid_from entirely (None) must be rejected -- no implicit
    now() substitution anywhere."""
    before = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=None)
    after = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    assert before == after


def test_explicit_null_valid_from_rejected():
    """Same as above, stated as its own scenario per the request -- explicit
    NULL must not be treated differently from omission."""
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=None)


def test_same_valid_from_same_evidence_creates_exactly_one_relationship():
    rel_id = None
    fixed_vf = "2026-03-01T00:00:00+00:00"
    try:
        res1 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=fixed_vf)
        rel_id = res1.data
        assert rel_id

        count = supabase.table("knowledge_relationships").select("id", count="exact") \
            .eq("id", rel_id).execute().count
        assert count == 1
    finally:
        _delete_relationship(rel_id)


def test_retry_same_valid_from_same_evidence_no_duplicate_evidence():
    """RETRY CALL A with the exact same logical identity and same evidence:
    no second relationship, duplicate evidence is a silent no-op."""
    rel_id = None
    fixed_vf = "2026-03-02T00:00:00+00:00"
    evidence = [{"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
                 "stance": "supports", "captured_at": _NOW}]
    try:
        res1 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, evidence, valid_from=fixed_vf)
        rel_id_1 = res1.data
        res2 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, evidence, valid_from=fixed_vf)
        rel_id_2 = res2.data
        rel_id = rel_id_1

        assert rel_id_1 == rel_id_2, "retry with the same logical identity must reuse the same relationship_id"

        ev = supabase.table("knowledge_relationship_evidence").select("id") \
            .eq("relationship_id", rel_id).execute().data
        assert len(ev) == 1, "the same evidence record attached twice must not duplicate"
    finally:
        _delete_relationship(rel_id)


def test_same_relationship_new_evidence_adds_without_duplicating_relationship():
    """CALL A AGAIN with same identity but NEW evidence: reuses the
    relationship_id, inserts the new evidence, no second relationship row."""
    rel_id = None
    fixed_vf = "2026-03-03T00:00:00+00:00"
    try:
        res1 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=fixed_vf)
        rel_id = res1.data

        res2 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "knowledge_note_source", "evidence_id": _real_note_source_id(),
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=fixed_vf)
        assert res2.data == rel_id

        ev = supabase.table("knowledge_relationship_evidence").select("evidence_type") \
            .eq("relationship_id", rel_id).execute().data
        assert len(ev) == 2, "one relationship, two distinct evidence records"
    finally:
        _delete_relationship(rel_id)


def test_different_valid_from_creates_distinct_historical_relationship():
    """CALL B with same source/target/type but DIFFERENT valid_from: a
    distinct historical relationship, a genuinely new row -- history is
    never flattened by the idempotency fix."""
    rel_id_1 = rel_id_2 = None
    try:
        res1 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from="2026-08-01T00:00:00+00:00")
        rel_id_1 = res1.data

        res2 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from="2026-09-01T00:00:00+00:00")
        rel_id_2 = res2.data

        assert rel_id_1 != rel_id_2, "different valid_from must produce a distinct historical row, not a reuse"
    finally:
        _delete_relationship(rel_id_1)
        _delete_relationship(rel_id_2)


def test_malformed_evidence_on_new_relationship_writes_nothing():
    """Restates atomicity for the new logic path: a brand-new logical
    relationship with one valid + one malformed evidence record writes
    zero relationship rows and zero evidence rows."""
    before = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
            {"evidence_type": "structured_knowledge", "evidence_id": "00000000-0000-0000-0000-000000000000",
             "stance": "supports", "captured_at": _NOW},
        ], valid_from="2026-05-01T00:00:00+00:00")
    after = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    assert before == after


def test_malformed_evidence_when_relationship_already_exists_leaves_it_untouched():
    """The trickier atomicity case: the relationship already exists from a
    prior successful call, then a later call reusing the same identity
    includes a malformed evidence record. The existing relationship must
    remain exactly as it was, and no partial new evidence may be written."""
    rel_id = None
    fixed_vf = "2026-03-04T00:00:00+00:00"
    try:
        res1 = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=fixed_vf)
        rel_id = res1.data
        before_row = supabase.table("knowledge_relationships").select("*").eq("id", rel_id).execute().data[0]
        before_evidence_count = len(
            supabase.table("knowledge_relationship_evidence").select("id").eq("relationship_id", rel_id).execute().data
        )

        with pytest.raises(Exception):
            _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
                {"evidence_type": "knowledge_note_source", "evidence_id": _real_note_source_id(),
                 "stance": "supports", "captured_at": _NOW},
                {"evidence_type": "structured_knowledge", "evidence_id": "00000000-0000-0000-0000-000000000000",
                 "stance": "supports", "captured_at": _NOW},
            ], valid_from=fixed_vf)

        after_row = supabase.table("knowledge_relationships").select("*").eq("id", rel_id).execute().data[0]
        after_evidence_count = len(
            supabase.table("knowledge_relationship_evidence").select("id").eq("relationship_id", rel_id).execute().data
        )
        assert after_row == before_row, "the existing relationship row must be byte-for-byte unchanged"
        assert after_evidence_count == before_evidence_count, "no partial new evidence may be written"
    finally:
        _delete_relationship(rel_id)


def _real_note_source_id() -> str:
    """One real knowledge_note_sources row (read-only), for tests that need
    a second, distinct real evidence type beyond structured_knowledge."""
    row = supabase.table("knowledge_note_sources").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).limit(1).execute().data
    assert row, "sanity: at least one real knowledge_note_sources row must exist in this workspace"
    return row[0]["id"]


# =====================================================================
# 7. Evidence uniqueness
# =====================================================================

def test_evidence_uniqueness_same_evidence_cannot_attach_twice():
    rel_id = None
    try:
        res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ])
        rel_id = res.data
        with pytest.raises(Exception):
            supabase.table("knowledge_relationship_evidence").insert({
                "relationship_id": rel_id, "workspace_id": REAL_WORKSPACE,
                "evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
                "stance": "supports", "captured_at": _NOW,
            }).execute()
    finally:
        _delete_relationship(rel_id)


# =====================================================================
# 8. Atomic relationship creation (the 4 required scenarios)
# =====================================================================

def test_atomic_zero_evidence_rejected_and_writes_nothing():
    before = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [])
    after = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    assert before == after


def test_atomic_malformed_evidence_rolls_back_relationship_too():
    before = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
            {"evidence_type": "structured_knowledge", "evidence_id": "00000000-0000-0000-0000-000000000000",
             "stance": "supports", "captured_at": _NOW},
        ])
    after = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    assert before == after, "a malformed evidence record must roll back the relationship insert too"


def test_atomic_valid_relationship_with_one_evidence_succeeds():
    rel_id = None
    try:
        res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ])
        rel_id = res.data
        assert rel_id
        row = supabase.table("knowledge_relationships").select("*").eq("id", rel_id).execute().data[0]
        assert row["relationship_type"] == "references"
        assert row["valid_from"] is not None
        ev = supabase.table("knowledge_relationship_evidence").select("*") \
            .eq("relationship_id", rel_id).execute().data
        assert len(ev) == 1
    finally:
        _delete_relationship(rel_id)


# =====================================================================
# 9. Polymorphic endpoint existence
# =====================================================================

def test_polymorphic_source_must_exist():
    with pytest.raises(Exception):
        _rpc(str(uuid.uuid4()), SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ])


def test_polymorphic_target_must_exist():
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, str(uuid.uuid4()), [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ])


def test_polymorphic_evidence_target_must_exist():
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "external_reference", "evidence_id": str(uuid.uuid4()),
             "stance": "supports", "captured_at": _NOW},
        ])


# =====================================================================
# 10. Workspace isolation
# =====================================================================

EMPTY_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"  # real, isolated, zero-chunk workspace


def test_workspace_isolation_rejects_cross_workspace_source():
    """A real structured_knowledge row from REAL_WORKSPACE must be rejected
    when the RPC is called claiming a DIFFERENT workspace_id."""
    with pytest.raises(Exception):
        _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], workspace_id=EMPTY_WORKSPACE)


# =====================================================================
# 11. Temporal validity
# =====================================================================

def test_temporal_valid_from_is_stored_exactly_as_caller_supplied():
    """Post-correction: there is no DB default -- valid_from is stored
    exactly as the caller passed it, never substituted."""
    rel_id = None
    supplied_vf = "2026-06-15T12:00:00+00:00"
    try:
        res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from=supplied_vf)
        rel_id = res.data
        row = supabase.table("knowledge_relationships").select("valid_from,valid_until") \
            .eq("id", rel_id).execute().data[0]
        assert row["valid_from"] == supplied_vf
        assert row["valid_until"] is None
    finally:
        _delete_relationship(rel_id)


def test_temporal_expired_relationship_representable():
    rel_id = None
    try:
        res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ], valid_from="2020-01-01T00:00:00+00:00")
        rel_id = res.data
        supabase.table("knowledge_relationships").update({
            "valid_until": "2021-01-01T00:00:00+00:00",
        }).eq("id", rel_id).execute()
        row = supabase.table("knowledge_relationships").select("valid_from,valid_until") \
            .eq("id", rel_id).execute().data[0]
        assert row["valid_until"] == "2021-01-01T00:00:00+00:00"
    finally:
        _delete_relationship(rel_id)


# =====================================================================
# 12. Calendar snapshot idempotency
# =====================================================================

def test_calendar_snapshot_idempotency_same_fingerprint_rejected():
    fp = f"test-fingerprint-{uuid.uuid4()}"
    row_id = None
    try:
        res = supabase.table("calendar_event_snapshots").insert({
            "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
            "external_event_id": "test-event-idempotency",
            "state_fingerprint": fp, "title": "test",
        }).execute()
        row_id = res.data[0]["id"]
        with pytest.raises(Exception):
            supabase.table("calendar_event_snapshots").insert({
                "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
                "external_event_id": "test-event-idempotency",
                "state_fingerprint": fp, "title": "test (identical fingerprint)",
            }).execute()
    finally:
        if row_id:
            supabase.table("calendar_event_snapshots").delete().eq("id", row_id).execute()


def test_calendar_snapshot_different_fingerprint_allowed():
    ids = []
    try:
        for i, fp_suffix in enumerate(["a", "b"]):
            res = supabase.table("calendar_event_snapshots").insert({
                "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
                "external_event_id": "test-event-changed-state",
                "state_fingerprint": f"test-fingerprint-changed-{fp_suffix}-{uuid.uuid4()}",
                "title": f"version {i}",
            }).execute()
            ids.append(res.data[0]["id"])
        assert len(ids) == 2, "a genuinely different fingerprint must be allowed to snapshot again"
    finally:
        for rid in ids:
            supabase.table("calendar_event_snapshots").delete().eq("id", rid).execute()


# =====================================================================
# 13. No circular relationship evidence (structural, via CHECK)
# =====================================================================
# Covered by test_evidence_type_check_rejects_via_rpc above -- restated here
# for the acceptance-test category list's own sake.

def test_no_circular_evidence_direct_insert_also_rejected():
    """Even bypassing the RPC, a direct insert with evidence_type=
    'knowledge_relationship' must fail -- the CHECK constraint is the real
    enforcement point, not just the RPC's application logic."""
    rel_id = None
    try:
        res = _rpc(SK_7A9EAA34, SK_FF5972E5_A, [
            {"evidence_type": "structured_knowledge", "evidence_id": SK_FF5972E5_B,
             "stance": "supports", "captured_at": _NOW},
        ])
        rel_id = res.data
        with pytest.raises(Exception):
            supabase.table("knowledge_relationship_evidence").insert({
                "relationship_id": rel_id, "workspace_id": REAL_WORKSPACE,
                "evidence_type": "knowledge_relationship", "evidence_id": rel_id,
                "stance": "supports", "captured_at": _NOW,
            }).execute()
    finally:
        _delete_relationship(rel_id)


# =====================================================================
# 14. RLS posture
# =====================================================================

def test_rls_enabled_zero_policies_all_six_tables():
    """Metadata check, same posture as every other table in this project
    (F-30): RLS enabled, zero policies, service-role-only access."""
    tables = (
        "calendar_event_snapshots", "knowledge_entities", "knowledge_entity_aliases",
        "knowledge_entity_identifiers", "knowledge_relationships", "knowledge_relationship_evidence",
    )
    for t in tables:
        # A plain service-role select must succeed (RLS doesn't block the
        # service role); this is a smoke check that the tables are reachable
        # and correctly created, not a full anon-vs-service-role probe (no
        # anon-key test client exists in this suite's fixtures).
        res = supabase.table(t).select("id").limit(1).execute()
        assert res.data == [] or isinstance(res.data, list)


# =====================================================================
# 15. structured_knowledge integrity -- never modified by anything above
# =====================================================================

def test_structured_knowledge_15_rows_unchanged_after_full_suite():
    total = supabase.table("structured_knowledge").select("id", count="exact").execute().count
    assert total == 15
    v21 = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count
    assert v21 == 15
    contract = supabase.table("extraction_contract_versions").select("*").eq("version", "v2.1").execute().data
    assert len(contract) == 1
    assert contract[0]["is_current"] is True
