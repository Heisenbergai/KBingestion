"""
Phase 5D graph read/traversal tests -- verifies graph_query.py against both
the real Phase 5 graph (1 relationship, 3 entities, 2 evidence records) and
synthetic fixtures for security/temporal boundary cases the real corpus
cannot exercise (no restricted-sensitivity evidence exists in the real 15
rows -- a gap already named in Phase 5A/5B).

IMPORTANT REAL-DATA FINDING, load-bearing for several tests below: the one
real relationship's valid_from is 2026-09-15T00:00:00+00:00 (matching its
source primitive's own effective_from -- the approval-gate rule isn't
organizationally active until then). "Now" in this environment is
2026-08-18. A strict CURRENT read (as_of defaulting to now()) therefore
correctly returns ZERO relationships for Product today -- this is the
temporal model working exactly as designed, not a bug. Tests that need to
see the real relationship pass an explicit as_of on/after 2026-09-15;
tests proving current-exclusion pass no as_of at all.

Run with: python -m pytest test_phase5d_graph_read.py -v
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest

import graph_query as gq
from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

SK_APPROVAL_REQUIREMENT = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"
PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
OPERATIONS_ENTITY_ID = "1034346e-5731-45b8-9ee5-2e7d1413ca81"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
SURVIVING_RELATIONSHIP_ID = "aca7d788-a356-4ad9-8030-4a96f4bd4da7"
REAL_NOTE_SOURCE_7A9EAA34 = "2419c928-5bf1-4b13-bdd6-f1a2b88b1bfb"

AS_OF_RULE_ACTIVE = datetime(2026, 9, 16, tzinfo=timezone.utc)
OWNER_SENSITIVITIES = gq.resolve_allowed_sensitivities("owner", False)
DEFAULT_SENSITIVITIES = gq.resolve_allowed_sensitivities(None, False)  # ['public','internal']


# =====================================================================
# 1. Entity read
# =====================================================================

def test_entity_read_product():
    e = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert e is not None
    assert e.entity_type == "department"
    assert e.canonical_label == "Product"
    assert e.status == "active"
    assert isinstance(e.identifiers, list)  # Product has none, per Phase 5B's own deliberate choice


# =====================================================================
# 2. Relationship read
# =====================================================================

def test_relationship_read_by_id():
    r = gq.get_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert r is not None
    assert r.relationship_type == "requires_approval_from"
    assert r.source.object_type == "structured_knowledge"
    assert r.source.object_id == SK_APPROVAL_REQUIREMENT
    assert r.target.object_type == "entity"
    assert r.target.object_id == PRODUCT_ENTITY_ID
    assert r.confidence is None
    assert r.valid_from == "2026-09-15T00:00:00+00:00"
    assert r.valid_until is None
    assert r.status == "active"


# =====================================================================
# 3. Evidence explanation
# =====================================================================

def test_evidence_explanation_chain():
    evidence = gq.explain_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert evidence is not None
    assert len(evidence) == 2
    by_kind = {e.evidence_kind: e for e in evidence}
    assert "derived_support" in by_kind and "primary_source" in by_kind
    # derived_support -> the structured_knowledge primitive's own full statement
    assert "and QA" in by_kind["derived_support"].source_reference
    # primary_source -> the real permalink, resolved through knowledge_note_sources -> knowledge_notes
    assert by_kind["primary_source"].source_reference == "https://chat.google.com/room/AAQAU5JKkmE/xt5pNtqhkgU.xt5pNtqhkgU"


# =====================================================================
# 4. Primary vs derived evidence distinction
# =====================================================================

def test_evidence_kinds_are_never_flattened():
    evidence = gq.explain_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)
    kinds = {e.evidence_kind for e in evidence}
    assert kinds == {"primary_source", "derived_support"}
    types = {e.evidence_type for e in evidence}
    assert types == {"structured_knowledge", "knowledge_note_source"}


# =====================================================================
# 5 & 6. Relationship visibility / restricted evidence (synthetic fixtures)
# =====================================================================

def _make_synthetic_security_fixture():
    """Two synthetic entities, two synthetic notes (one restricted, one
    public), one synthetic relationship with one evidence record of each
    sensitivity. Returns a dict of all created ids for cleanup.

    Every step appends to `ids` BEFORE moving to the next -- if any later
    step raises, the except block cleans up exactly what was actually
    created so far and re-raises, rather than leaking partial rows. This
    exists because an earlier version of this fixture leaked 8 synthetic
    entities + 6 synthetic notes into the real workspace when a downstream
    insert failed (missing source_type on knowledge_note_sources) -- found
    live, cleaned up by exact id, and closed here structurally so it can't
    recur silently."""
    ids: dict = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-SOURCE", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-TARGET", "status": "active",
        }).execute().data[0]["id"]

        ids["restricted_note"] = supabase.table("knowledge_notes").insert({
            "workspace_id": REAL_WORKSPACE, "provider": "bot_learning", "source_type": "note",
            "title": "TEST-5D restricted note", "body": "synthetic fixture", "sensitivity": "restricted",
        }).execute().data[0]["id"]
        ids["public_note"] = supabase.table("knowledge_notes").insert({
            "workspace_id": REAL_WORKSPACE, "provider": "bot_learning", "source_type": "note",
            "title": "TEST-5D public note", "body": "synthetic fixture", "sensitivity": "public",
        }).execute().data[0]["id"]

        ids["restricted_source"] = supabase.table("knowledge_note_sources").insert({
            "note_id": ids["restricted_note"], "workspace_id": REAL_WORKSPACE,
            "provider": "bot_learning", "source_type": "bot_learning", "source_ref": "TEST-5D-restricted-permalink",
        }).execute().data[0]["id"]
        ids["public_source"] = supabase.table("knowledge_note_sources").insert({
            "note_id": ids["public_note"], "workspace_id": REAL_WORKSPACE,
            "provider": "bot_learning", "source_type": "bot_learning", "source_ref": "TEST-5D-public-permalink",
        }).execute().data[0]["id"]

        now = datetime.now(timezone.utc).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5D security fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [
                {"evidence_type": "knowledge_note_source", "evidence_id": ids["restricted_source"],
                 "stance": "supports", "captured_at": now},
                {"evidence_type": "knowledge_note_source", "evidence_id": ids["public_source"],
                 "stance": "supports", "captured_at": now},
            ],
        }).execute().data

        return ids
    except Exception:
        _cleanup_synthetic_security_fixture(ids)
        raise


def _cleanup_synthetic_security_fixture(ids: dict):
    """Uses .get() throughout, deliberately -- called both on the success
    path (full dict) and from the except-and-reraise path above (a partial
    dict, whatever was actually created before the failure)."""
    if ids.get("relationship_id"):
        supabase.table("knowledge_relationships").delete().eq("id", ids["relationship_id"]).execute()
    if ids.get("restricted_source"):
        supabase.table("knowledge_note_sources").delete().eq("id", ids["restricted_source"]).execute()
    if ids.get("public_source"):
        supabase.table("knowledge_note_sources").delete().eq("id", ids["public_source"]).execute()
    if ids.get("restricted_note"):
        supabase.table("knowledge_notes").delete().eq("id", ids["restricted_note"]).execute()
    if ids.get("public_note"):
        supabase.table("knowledge_notes").delete().eq("id", ids["public_note"]).execute()
    if ids.get("src_entity"):
        supabase.table("knowledge_entities").delete().eq("id", ids["src_entity"]).execute()
    if ids.get("tgt_entity"):
        supabase.table("knowledge_entities").delete().eq("id", ids["tgt_entity"]).execute()


def test_relationship_visible_via_partial_evidence_low_privilege():
    """A caller who cannot see the restricted evidence must still see the
    relationship (justified by the public evidence alone) -- but must NOT
    see the restricted evidence record itself. This is Decision 7's OR-
    across-visible-evidence model, proven live, not just asserted."""
    ids = _make_synthetic_security_fixture()
    try:
        r = gq.get_relationship(ids["relationship_id"], REAL_WORKSPACE, DEFAULT_SENSITIVITIES)
        assert r is not None, "relationship must be visible -- the public evidence alone justifies it"
        assert len(r.evidence) == 1, "only the public evidence record, never the restricted one"
        assert r.evidence[0].source_reference == "TEST-5D-public-permalink"
    finally:
        _cleanup_synthetic_security_fixture(ids)


def test_restricted_evidence_visible_to_high_privilege_caller():
    ids = _make_synthetic_security_fixture()
    try:
        r = gq.get_relationship(ids["relationship_id"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert r is not None
        assert len(r.evidence) == 2, "an owner-level caller sees both evidence records"
        refs = {e.source_reference for e in r.evidence}
        assert refs == {"TEST-5D-public-permalink", "TEST-5D-restricted-permalink"}
    finally:
        _cleanup_synthetic_security_fixture(ids)


def test_relationship_invisible_when_zero_evidence_visible():
    """A caller below even 'internal' (hypothetically 'public'-only) sees
    neither evidence record for a relationship backed ONLY by restricted
    evidence -- the relationship itself must then be entirely invisible,
    not shown with an empty evidence list. All creation happens inside the
    try block, including the entities/note/source themselves, so a failure
    at any step cleans up exactly what was created so far (see
    _make_synthetic_security_fixture's docstring for why this matters)."""
    ids: dict = {}
    try:
        ids["src"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-ZERO-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-ZERO-TGT", "status": "active",
        }).execute().data[0]["id"]
        ids["note"] = supabase.table("knowledge_notes").insert({
            "workspace_id": REAL_WORKSPACE, "provider": "bot_learning", "source_type": "note",
            "title": "TEST-5D zero-visibility note", "body": "synthetic", "sensitivity": "restricted",
        }).execute().data[0]["id"]
        ids["src_row"] = supabase.table("knowledge_note_sources").insert({
            "note_id": ids["note"], "workspace_id": REAL_WORKSPACE, "provider": "bot_learning",
            "source_type": "bot_learning", "source_ref": "TEST-5D-zero-permalink",
        }).execute().data[0]["id"]

        now = datetime.now(timezone.utc).isoformat()
        ids["rel_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt"],
            "p_relationship_type": "references", "p_rationale": "synthetic", "p_confidence": None,
            "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": ids["src_row"],
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        r = gq.get_relationship(ids["rel_id"], REAL_WORKSPACE, ["public"])
        assert r is None, "zero visible evidence must make the relationship invisible entirely"
    finally:
        if ids.get("rel_id"):
            supabase.table("knowledge_relationships").delete().eq("id", ids["rel_id"]).execute()
        if ids.get("src_row"):
            supabase.table("knowledge_note_sources").delete().eq("id", ids["src_row"]).execute()
        if ids.get("note"):
            supabase.table("knowledge_notes").delete().eq("id", ids["note"]).execute()
        if ids.get("src"):
            supabase.table("knowledge_entities").delete().eq("id", ids["src"]).execute()
        if ids.get("tgt"):
            supabase.table("knowledge_entities").delete().eq("id", ids["tgt"]).execute()


# =====================================================================
# 7. Workspace isolation
# =====================================================================

def test_entity_read_wrong_workspace_returns_none():
    assert gq.get_entity_graph(PRODUCT_ENTITY_ID, OTHER_REAL_WORKSPACE, OWNER_SENSITIVITIES) is None


def test_relationship_read_wrong_workspace_returns_none():
    assert gq.get_relationship(SURVIVING_RELATIONSHIP_ID, OTHER_REAL_WORKSPACE, OWNER_SENSITIVITIES) is None


# =====================================================================
# 8. Current temporal filtering (real data + synthetic boundary fixtures)
# =====================================================================

def test_real_relationship_excluded_from_strict_current_read():
    """Load-bearing real finding: as of 'now' (2026-08-18), the one real
    relationship's valid_from (2026-09-15) is in the future -- a strict
    current read correctly excludes it. Proven against real data, not
    synthetic."""
    e = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)  # as_of defaults to now()
    assert e.inbound_relationships == []


def test_synthetic_temporal_boundaries_ABC():
    """A: valid_from=yesterday, valid_until=tomorrow -> CURRENT read includes it.
    B: valid_from=last month, valid_until=yesterday -> CURRENT read excludes it.
    C: valid_from=tomorrow, valid_until=NULL -> CURRENT read excludes it."""
    now = datetime.now(timezone.utc)
    yesterday, tomorrow = now - timedelta(days=1), now + timedelta(days=1)
    last_month = now - timedelta(days=30)

    ids: dict = {}
    rel_a = rel_b = rel_c = None
    try:
        ids["src"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-TEMPORAL-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5D-TEMPORAL-TGT", "status": "active",
        }).execute().data[0]["id"]
        ids["note"] = supabase.table("knowledge_notes").insert({
            "workspace_id": REAL_WORKSPACE, "provider": "bot_learning", "source_type": "note",
            "title": "TEST-5D temporal note", "body": "synthetic", "sensitivity": "public",
        }).execute().data[0]["id"]
        ids["note_src"] = supabase.table("knowledge_note_sources").insert({
            "note_id": ids["note"], "workspace_id": REAL_WORKSPACE, "provider": "bot_learning",
            "source_type": "bot_learning", "source_ref": "TEST-5D-temporal-permalink",
        }).execute().data[0]["id"]
        src, tgt = ids["src"], ids["tgt"]

        def _make(vf, vu):
            return supabase.rpc("create_relationship_with_evidence", {
                "p_workspace_id": REAL_WORKSPACE,
                "p_source_object_type": "entity", "p_source_object_id": src,
                "p_target_object_type": "entity", "p_target_object_id": tgt,
                "p_relationship_type": "references", "p_rationale": "synthetic temporal test",
                "p_confidence": None, "p_valid_from": vf.isoformat(),
                "p_valid_until": vu.isoformat() if vu else None,
                "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": ids["note_src"],
                                "stance": "supports", "captured_at": now.isoformat()}],
            }).execute().data

        rel_a = _make(yesterday, tomorrow)
        rel_b = _make(last_month, yesterday)
        rel_c = _make(tomorrow, None)

        e = gq.get_entity_graph(src, REAL_WORKSPACE, OWNER_SENSITIVITIES)  # source-side outbound, current
        current_ids = {r.id for r in e.outbound_relationships}
        assert rel_a in current_ids, "A (yesterday..tomorrow) must be CURRENT"
        assert rel_b not in current_ids, "B (last month..yesterday) must be EXCLUDED (expired)"
        assert rel_c not in current_ids, "C (tomorrow..) must be EXCLUDED (not yet valid)"

        # Historical query at a point inside B's window
        e_hist = gq.get_entity_graph(src, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=last_month + timedelta(days=5))
        hist_ids = {r.id for r in e_hist.outbound_relationships}
        assert rel_b in hist_ids, "a historical query inside B's own window must return B"
        assert rel_a not in hist_ids, "A did not exist yet at that point in time"
        assert rel_c not in hist_ids, "C did not exist yet either"
    finally:
        for rid in (rel_a, rel_b, rel_c):
            if rid:
                supabase.table("knowledge_relationships").delete().eq("id", rid).execute()
        if ids.get("note_src"):
            supabase.table("knowledge_note_sources").delete().eq("id", ids["note_src"]).execute()
        if ids.get("note"):
            supabase.table("knowledge_notes").delete().eq("id", ids["note"]).execute()
        if ids.get("src"):
            supabase.table("knowledge_entities").delete().eq("id", ids["src"]).execute()
        if ids.get("tgt"):
            supabase.table("knowledge_entities").delete().eq("id", ids["tgt"]).execute()


# =====================================================================
# 9. Historical temporal filtering -- covered inside test_synthetic_temporal_boundaries_ABC
# =====================================================================

# =====================================================================
# 10. Depth-2 traversal
# =====================================================================

def test_depth_2_traversal_entity_to_structured_knowledge():
    e = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert len(e.inbound_relationships) == 1
    r = e.inbound_relationships[0]
    assert r.source.object_type == "structured_knowledge"
    assert r.source.label is not None and "Product and QA" in r.source.label
    assert r.target.object_type == "entity"
    assert r.target.label == "Product"


def test_depth_2_traversal_structured_knowledge_to_entity():
    sk = gq.get_structured_knowledge_graph(SK_APPROVAL_REQUIREMENT, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert sk is not None
    assert len(sk["outbound_relationships"]) == 1
    r = sk["outbound_relationships"][0]
    assert r.target.label == "Product"


# =====================================================================
# 11. No inferred relationships
# =====================================================================

def test_operations_has_no_graph_relationship():
    e = gq.get_entity_graph(OPERATIONS_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert e.inbound_relationships == []
    assert e.outbound_relationships == []


def test_meeting_has_no_graph_relationship():
    """STALE (Phase 5H): as of Phase 5D the Meeting had zero relationships at
    all. Phase 5H legitimately added two real Person->Meeting activity edges
    (Tanmay --organized-->, John Snow --attended-->) -- see
    test_phase5h_meeting_activity_relationships.py. What this test actually
    still verifies: the Meeting has NO OUTBOUND relationships (it never acts
    as a source -- e.g. no Meeting->Department edge was ever created), and
    its only inbound edges are the two real activity types, never anything
    implying ownership/employment (member_of, owns, manages, etc.)."""
    e = gq.get_entity_graph(MEETING_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert e.outbound_relationships == []
    inbound_types = {r.relationship_type for r in e.inbound_relationships}
    assert inbound_types == {"organized", "attended"}


# =====================================================================
# 12. Deterministic ordering
# =====================================================================

def test_repeated_reads_are_deterministically_ordered():
    e1 = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    e2 = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    ids1 = [r.id for r in e1.inbound_relationships]
    ids2 = [r.id for r in e2.inbound_relationships]
    assert ids1 == ids2

    ev1 = [e.evidence_id for e in gq.explain_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)]
    ev2 = [e.evidence_id for e in gq.explain_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)]
    assert ev1 == ev2


# =====================================================================
# 13. No orphan evidence
# =====================================================================

def test_no_orphan_evidence_in_real_data():
    all_evidence = supabase.table("knowledge_relationship_evidence").select("relationship_id").execute().data
    real_rel_ids = {r["id"] for r in supabase.table("knowledge_relationships").select("id").execute().data}
    assert all(e["relationship_id"] in real_rel_ids for e in all_evidence)


# =====================================================================
# 14, 15, 16. Real Product/Meeting/Operations verification (Part 8)
# =====================================================================

def test_part8_product_lookup_returns_exactly_one_relationship():
    e = gq.get_entity_graph(PRODUCT_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=AS_OF_RULE_ACTIVE)
    assert len(e.inbound_relationships) == 1
    assert len(e.outbound_relationships) == 0


def test_part8_full_statement_includes_qa():
    r = gq.get_relationship(SURVIVING_RELATIONSHIP_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert "QA" in r.source.label, "the full source statement, including QA, must remain intact even though QA has no graph entity"


def test_part8_qa_has_no_entity():
    assert supabase.table("knowledge_entities").select("id", count="exact").eq("canonical_label", "QA").execute().count == 0


def test_part8_meeting_no_relationship():
    """STALE (Phase 5H): see test_meeting_has_no_graph_relationship's
    docstring for the same correction -- the Meeting legitimately gained two
    real inbound activity edges in Phase 5H. Re-asserted here as: still no
    OUTBOUND relationships from the Meeting."""
    assert gq.get_entity_graph(MEETING_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES).outbound_relationships == []


def test_part8_operations_no_relationship():
    assert gq.get_entity_graph(OPERATIONS_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES).outbound_relationships == []


# =====================================================================
# 17-20. Integrity of everything Phase 5D must not touch
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count == 15


def test_three_entities_unchanged():
    rows = supabase.table("knowledge_entities").select("id,canonical_label") \
        .in_("id", [PRODUCT_ENTITY_ID, OPERATIONS_ENTITY_ID, MEETING_ENTITY_ID]).execute().data
    assert len(rows) == 3


def test_one_relationship_unchanged():
    """STALE COUNT (Phase 5H): global relationship count was 1 as of Phase
    5D; Phase 5H legitimately added two real Person->Meeting activity edges
    (organized, attended) -- see test_phase5h_meeting_activity_relationships.py.
    This test now verifies the ORIGINAL requires_approval_from edge
    specifically, by id, rather than a stale total count."""
    r = supabase.table("knowledge_relationships").select("*").eq("id", SURVIVING_RELATIONSHIP_ID).execute().data[0]
    assert r["relationship_type"] == "requires_approval_from"


def test_two_evidence_rows_unchanged():
    """STALE COUNT (Phase 5H): global evidence count was 2 as of Phase 5D.
    Phase 5H legitimately added two more real evidence rows for the new
    Person->Meeting activity relationships -- see
    test_phase5h_meeting_activity_relationships.py. Narrowed to what this
    test actually verifies: SURVIVING_RELATIONSHIP_ID specifically still has
    exactly its original two evidence rows."""
    count = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", SURVIVING_RELATIONSHIP_ID).execute().count
    assert count == 2


def test_no_new_permanent_entities_leaked_from_this_suite():
    """Sanity check that every synthetic fixture this file creates is
    actually cleaned up. Phase 5F later added two real, verified Person
    entities (Tanmay, John Snow) -- included here as expected real state,
    not as something this suite's own fixtures could have leaked."""
    rows = supabase.table("knowledge_entities").select("canonical_label").execute().data
    labels = {r["canonical_label"] for r in rows}
    assert labels == {"Product", "Operations", "Knova Test Meeting 1", "Tanmay", "John Snow"}
