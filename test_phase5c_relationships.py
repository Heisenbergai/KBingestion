"""
Phase 5C relationship tests -- verifies the REAL relationship construction
performed against the live vector DB, corrected in the semantic-precision
pass that followed the first construction round.

Current, corrected real graph:

  fc261a0a (7a9eaa34's approval requirement) --requires_approval_from--> Product

Two relationships from the first construction pass were REMOVED as
semantically redundant/weak, by exact ID, once evidence and constraints were
verified live:

  fc261a0a --references--> Product     (redundant with requires_approval_from,
                                         which already carries the specific,
                                         meaningful relationship)
  5b77b2ca --references--> Operations  (references was being used as a
                                         substitute for a not-yet-approved
                                         actor/ownership relationship type)

`references` in this codebase's frozen V1 semantics means "the source
explicitly refers to an already-existing distinct object as a referenced
object" -- it does not mean "mentions", "names", "acts as", or "performs an
action as". Neither removed edge satisfied that bar on reflection; the
surviving `requires_approval_from` edge does, precisely.

QA was the second approval target named in the real source text but is NOT
a verified graph entity (no matching real departments row) -- no relationship
was ever created toward it, in either construction pass. The full statement,
including QA, remains completely intact and unmodified in
structured_knowledge -- the graph is a deliberately conservative subset of
what the source actually says, not a rewrite of it.

The one surviving relationship (and its two evidence records) is real,
intended graph knowledge -- NOT deleted by this suite's cleanup. Only
synthetic rows this suite creates for negative/security tests are cleaned up.

Run with: python -m pytest test_phase5c_relationships.py -v
"""
import uuid

import pytest

from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"  # real, isolated, zero-chunk workspace

SK_APPROVAL_REQUIREMENT = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"   # 7a9eaa34: "...approval from Product and QA..."
SK_OPERATIONS_REQUIREMENT = "5b77b2ca-2c8c-436c-8070-4f61bf5a270d"  # ff5972e5: "Operations will publish..." -- no longer a relationship source, kept as a fixture id for negative tests

PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
OPERATIONS_ENTITY_ID = "1034346e-5731-45b8-9ee5-2e7d1413ca81"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"

REAL_NOTE_SOURCE_7A9EAA34 = "2419c928-5bf1-4b13-bdd6-f1a2b88b1bfb"
REAL_NOTE_SOURCE_FF5972E5 = "a3d2e8ff-4bf3-4d5e-8f21-b766f28315f3"

SURVIVING_RELATIONSHIP_ID = "aca7d788-a356-4ad9-8030-4a96f4bd4da7"
REMOVED_REFERENCES_PRODUCT_ID = "6a2fb26b-2b38-4a5b-b722-30e12373286d"
REMOVED_REFERENCES_OPERATIONS_ID = "41c2e0a7-3ffa-4e6e-9446-fe0ec01e4588"


def _rpc(source_id, target_id, evidence, relationship_type, valid_from,
         workspace_id=REAL_WORKSPACE, source_type="structured_knowledge", target_type="entity",
         rationale="test", confidence=None):
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": source_type,
        "p_source_object_id": source_id,
        "p_target_object_type": target_type,
        "p_target_object_id": target_id,
        "p_relationship_type": relationship_type,
        "p_rationale": rationale,
        "p_confidence": confidence,
        "p_valid_from": valid_from,
        "p_valid_until": None,
        "p_evidence": evidence,
    }).execute()


# =====================================================================
# 1. requires_approval_from Product remains
# =====================================================================

def test_requires_approval_from_product_still_exists():
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("source_object_id", SK_APPROVAL_REQUIREMENT) \
        .eq("relationship_type", "requires_approval_from").execute().data
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == SURVIVING_RELATIONSHIP_ID
    assert r["target_object_type"] == "entity"
    assert r["target_object_id"] == PRODUCT_ENTITY_ID
    assert r["source_object_type"] == "structured_knowledge"
    assert r["status"] == "active"


# =====================================================================
# 2. No references -> Product remains
# =====================================================================

def test_references_to_product_removed():
    rows = supabase.table("knowledge_relationships").select("id") \
        .eq("source_object_id", SK_APPROVAL_REQUIREMENT) \
        .eq("relationship_type", "references").execute().data
    assert rows == []
    assert supabase.table("knowledge_relationships").select("id") \
        .eq("id", REMOVED_REFERENCES_PRODUCT_ID).execute().data == []


# =====================================================================
# 3. No references -> Operations remains
# =====================================================================

def test_references_to_operations_removed():
    rows = supabase.table("knowledge_relationships").select("id") \
        .eq("target_object_id", OPERATIONS_ENTITY_ID).execute().data
    assert rows == [], "Operations must not be the target of any relationship after this correction pass"
    assert supabase.table("knowledge_relationships").select("id") \
        .eq("id", REMOVED_REFERENCES_OPERATIONS_ID).execute().data == []


# =====================================================================
# 4. QA still has no graph entity
# =====================================================================

def test_qa_still_not_an_entity():
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("canonical_label", "QA").execute().count
    assert count == 0


# =====================================================================
# 5. Procurement still has no graph entity
# =====================================================================

def test_procurement_still_not_an_entity():
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("canonical_label", "Procurement").execute().count
    assert count == 0


# =====================================================================
# 6. No unsupported replacement relationship was created
# =====================================================================

def test_exactly_one_relationship_remains_and_it_is_the_expected_one():
    """STALE (Phase 5H): originally asserted exactly one relationship exists
    in the WHOLE graph. Phase 5H legitimately added two real Person->Meeting
    activity edges (source_object_type='entity') -- see
    test_phase5h_meeting_activity_relationships.py. What this test actually
    verifies -- unchanged by Phase 5H -- is that the semantic-correction pass
    left exactly ONE structured_knowledge-SOURCED relationship: the surviving
    requires_approval_from edge, with the two weak `references` edges still
    gone."""
    rows = supabase.table("knowledge_relationships").select("id,relationship_type") \
        .eq("source_object_type", "structured_knowledge").execute().data
    assert len(rows) == 1
    assert rows[0]["id"] == SURVIVING_RELATIONSHIP_ID
    assert rows[0]["relationship_type"] == "requires_approval_from"


def test_no_relationship_targets_qa():
    all_target_ids = {r["target_object_id"] for r in
                       supabase.table("knowledge_relationships").select("target_object_id")
                       .eq("target_object_type", "entity").execute().data}
    entities = {e["id"]: e["canonical_label"] for e in
                supabase.table("knowledge_entities").select("id,canonical_label").execute().data}
    assert all_target_ids <= entities.keys(), "every relationship target must be a real, existing entity"
    assert all(entities[tid] != "QA" for tid in all_target_ids)


# =====================================================================
# 7. All remaining relationship evidence is intact
# =====================================================================

def test_requires_approval_from_has_valid_evidence():
    ev = supabase.table("knowledge_relationship_evidence").select("*") \
        .eq("relationship_id", SURVIVING_RELATIONSHIP_ID).execute().data
    assert len(ev) == 2
    types = {e["evidence_type"] for e in ev}
    assert types == {"structured_knowledge", "knowledge_note_source"}
    assert all(e["stance"] == "supports" for e in ev)
    assert all(e["revoked_at"] is None for e in ev)
    ids = {e["evidence_id"] for e in ev}
    assert SK_APPROVAL_REQUIREMENT in ids
    assert REAL_NOTE_SOURCE_7A9EAA34 in ids


def test_surviving_relationship_confidence_is_null():
    r = supabase.table("knowledge_relationships").select("confidence") \
        .eq("id", SURVIVING_RELATIONSHIP_ID).execute().data[0]
    assert r["confidence"] is None


def test_surviving_relationship_valid_from_matches_primitive_effective_from():
    r = supabase.table("knowledge_relationships").select("valid_from") \
        .eq("id", SURVIVING_RELATIONSHIP_ID).execute().data[0]
    assert r["valid_from"] == "2026-09-15T00:00:00+00:00"


# =====================================================================
# 8. No orphan evidence exists
# =====================================================================

def test_no_orphan_evidence_after_deletion():
    """Evidence rows for the two removed relationships must have cascaded
    away with them -- confirmed via the real ON DELETE CASCADE FK on
    knowledge_relationship_evidence.relationship_id, not assumed.

    STALE COUNT (Phase 5H): the global evidence count was 2 as of Phase 5C.
    Phase 5H legitimately added two more real evidence rows for the new
    activity relationships -- see test_phase5h_meeting_activity_relationships.py.
    The orphan-freedom check above is unaffected; the count check below is
    narrowed to what this test actually originated to prove: the SURVIVING
    relationship specifically still has exactly its original two evidence
    rows, nothing added or lost."""
    all_evidence = supabase.table("knowledge_relationship_evidence").select("relationship_id").execute().data
    real_relationship_ids = {r["id"] for r in
                              supabase.table("knowledge_relationships").select("id").execute().data}
    orphans = [e for e in all_evidence if e["relationship_id"] not in real_relationship_ids]
    assert orphans == []
    surviving_ev = [e for e in all_evidence if e["relationship_id"] == SURVIVING_RELATIONSHIP_ID]
    assert len(surviving_ev) == 2, "exactly the surviving relationship's two evidence rows, nothing left behind"


# =====================================================================
# 9. structured_knowledge remains exactly 15 rows and unchanged
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    total = supabase.table("structured_knowledge").select("id", count="exact").execute().count
    assert total == 15
    v21 = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count
    assert v21 == 15


# =====================================================================
# 10. The valid relationship remains idempotent
# =====================================================================

def test_surviving_relationship_construction_idempotent():
    before_count = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    res = _rpc(
        SK_APPROVAL_REQUIREMENT, PRODUCT_ENTITY_ID,
        [{"evidence_type": "structured_knowledge", "evidence_id": SK_APPROVAL_REQUIREMENT,
          "stance": "supports", "captured_at": "2026-08-16T20:40:46.595754+00:00"},
         {"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
          "stance": "supports", "captured_at": "2026-08-16T20:40:46.595754+00:00"}],
        "requires_approval_from", "2026-09-15T00:00:00+00:00",
    )
    assert res.data == SURVIVING_RELATIONSHIP_ID

    # STALE PIN (Phase 5H): this no longer pins the global count to 1 --
    # Phase 5H legitimately added two more real relationships. The actual
    # invariant this test proves is untouched: retrying construction with
    # the same logical key changes the total by exactly zero.
    after_count = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    assert before_count == after_count

    ev_count = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", SURVIVING_RELATIONSHIP_ID).execute().count
    assert ev_count == 2, "retry must not duplicate evidence either"


# =====================================================================
# 11. Wrong-workspace security behavior remains unchanged
# =====================================================================

def test_wrong_workspace_relationship_rejected_atomically():
    before = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    before_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        _rpc(
            SK_APPROVAL_REQUIREMENT, PRODUCT_ENTITY_ID,
            [{"evidence_type": "structured_knowledge", "evidence_id": SK_APPROVAL_REQUIREMENT,
              "stance": "supports", "captured_at": "2026-08-16T20:40:46.595754+00:00"}],
            "references", "2026-08-16T20:40:46.595754+00:00",
            workspace_id=OTHER_REAL_WORKSPACE,
        )
    after = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    after_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count
    assert before == after, "wrong-workspace attempt must create zero relationship rows"
    assert before_ev == after_ev, "wrong-workspace attempt must create zero evidence rows"


# =====================================================================
# Structural checks carried forward from the first construction pass
# =====================================================================

def test_no_relationship_created_from_the_meeting_primitives_to_themselves():
    """STALE (Phase 5H): originally asserted zero relationships target the
    Meeting at all. Phase 5H legitimately gave the Meeting two real inbound
    activity edges (Tanmay --organized-->, John Snow --attended-->) -- see
    test_phase5h_meeting_activity_relationships.py, both source_object_type=
    'entity'. What this test actually still verifies -- the original Phase
    5C concern -- is that structured_knowledge PRIMITIVES never targeted the
    Meeting (no structured_knowledge -> Meeting edge exists); that remains
    true and unrelated to Phase 5H's entity-sourced activity edges."""
    count = supabase.table("knowledge_relationships").select("id", count="exact") \
        .eq("target_object_id", MEETING_ENTITY_ID).eq("source_object_type", "structured_knowledge").execute().count
    assert count == 0


def test_unsupported_relationship_type_rejected():
    with pytest.raises(Exception):
        _rpc(
            SK_APPROVAL_REQUIREMENT, PRODUCT_ENTITY_ID,
            [{"evidence_type": "structured_knowledge", "evidence_id": SK_APPROVAL_REQUIREMENT,
              "stance": "supports", "captured_at": "2026-08-16T20:40:46.595754+00:00"}],
            "related_to", "2026-08-16T20:40:46.595754+00:00",
        )


def test_all_relationship_endpoints_resolve_to_real_rows():
    """STALE PIN (Phase 5H): no longer pins the total to 1 or assumes every
    row is structured_knowledge-sourced -- Phase 5H legitimately added two
    real entity-sourced activity edges (see
    test_phase5h_meeting_activity_relationships.py). The check itself still
    applies generically to every row, whichever object_type it uses."""
    rows = supabase.table("knowledge_relationships").select("*").execute().data
    assert len(rows) >= 1
    real_sk_ids = {s["id"] for s in supabase.table("structured_knowledge").select("id").execute().data}
    real_entity_ids = {e["id"] for e in supabase.table("knowledge_entities").select("id").execute().data}
    for r in rows:
        assert r["source_object_type"] in ("structured_knowledge", "entity")
        source_pool = real_sk_ids if r["source_object_type"] == "structured_knowledge" else real_entity_ids
        assert r["source_object_id"] in source_pool
        assert r["target_object_type"] == "entity"
        assert r["target_object_id"] in real_entity_ids


def test_three_phase5b_entities_unchanged():
    """Phase 5F later added two real Person entities (Tanmay, John Snow) --
    this no longer asserts exactly 3 entities total, since that would be
    asserting Phase 5F never happened. It re-asserts what actually matters:
    the original three Phase 5B entities are still present and unchanged."""
    rows = supabase.table("knowledge_entities").select("id,entity_type,canonical_label,status") \
        .in_("id", [PRODUCT_ENTITY_ID, OPERATIONS_ENTITY_ID, MEETING_ENTITY_ID]).execute().data
    assert len(rows) == 3
    by_id = {r["id"]: r for r in rows}
    assert by_id[PRODUCT_ENTITY_ID]["canonical_label"] == "Product"
    assert by_id[OPERATIONS_ENTITY_ID]["canonical_label"] == "Operations"
    assert by_id[MEETING_ENTITY_ID]["canonical_label"] == "Knova Test Meeting 1"
    assert all(r["status"] == "active" for r in rows)


def test_no_aliases_created():
    """Calendar snapshots were legitimately out of scope for Phase 5C and
    now exist as of Phase 5E (see test_phase5e_calendar_evidence.py) -- this
    test no longer asserts zero snapshots, since asserting that would be
    asserting Phase 5E never happened."""
    assert supabase.table("knowledge_entity_aliases").select("id", count="exact").execute().count == 0


def test_relationship_and_evidence_row_counts_are_exactly_expected():
    """STALE COUNTS (Phase 5H): were 1 relationship / 2 evidence as of Phase
    5C. Phase 5H legitimately added two real Person->Meeting activity edges
    (organized, attended), each with its own single real evidence row -- see
    test_phase5h_meeting_activity_relationships.py. This test now verifies
    the ORIGINAL requires_approval_from edge specifically, by id, rather than
    a stale total count."""
    r = supabase.table("knowledge_relationships").select("*").eq("id", SURVIVING_RELATIONSHIP_ID).execute().data
    assert len(r) == 1
    assert r[0]["relationship_type"] == "requires_approval_from"
    ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", SURVIVING_RELATIONSHIP_ID).execute().count
    assert ev == 2
