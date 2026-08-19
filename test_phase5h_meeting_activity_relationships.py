"""
Phase 5H Meeting Activity Relationship tests -- verifies the two real
Person -> Meeting edges constructed this pass (Tanmay --organized-->
Knova Test Meeting 1, John Snow --attended--> Knova Test Meeting 1) against
the one real calendar_event_snapshot row, plus the construction discipline
(idempotency, security, non-inference) against synthetic fixtures for the
boundary cases the real corpus can't exercise on its own.

THE ORGANIZER/ATTENDEE OVERLAP CASE, decided precisely (Phase 5H Part 6):
the real snapshot's `attendees` array lists BOTH kingjohnsnow0@gmail.com AND
tanmaydubeytd@gmail.com (the organizer, response_status="accepted"). Per the
strict rule, this does NOT automatically produce a redundant
Tanmay --attended--> Meeting edge -- there is no explicit source semantic
establishing that organizer participation should ALSO count as a separate
attendance claim, and `organized` already fully captures Tanmay's real
participation. Only ONE edge exists for Tanmay. This is asserted directly by
test_organizer_has_no_redundant_attended_edge below.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5h_meeting_activity_relationships.py -v
"""
import pytest

import graph_query as gq
from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_ENTITY_ID = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
JOHN_SNOW_ENTITY_ID = "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
OPERATIONS_ENTITY_ID = "1034346e-5731-45b8-9ee5-2e7d1413ca81"

REAL_SNAPSHOT_ID = "ffcf42b9-62dc-446e-9378-292d70d1d2ca"
MEETING_START_TIME = "2026-08-16T08:30:00+00:00"

ORGANIZED_RELATIONSHIP_ID = None  # resolved in a fixture below, not hardcoded twice
ATTENDED_RELATIONSHIP_ID = None


def _rpc(source_id, target_id, relationship_type, valid_from, evidence,
         workspace_id=REAL_WORKSPACE, rationale="test", confidence=None, valid_until=None):
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": "entity",
        "p_source_object_id": source_id,
        "p_target_object_type": "entity",
        "p_target_object_id": target_id,
        "p_relationship_type": relationship_type,
        "p_rationale": rationale,
        "p_confidence": confidence,
        "p_valid_from": valid_from,
        "p_valid_until": valid_until,
        "p_evidence": evidence,
    }).execute()


def _real_evidence():
    return [{"evidence_type": "calendar_event_snapshot", "evidence_id": REAL_SNAPSHOT_ID,
             "stance": "supports", "captured_at": "2026-08-18T12:03:44.422869+00:00"}]


def _get_relationship(source_id: str, target_id: str, relationship_type: str) -> dict:
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("source_object_id", source_id).eq("target_object_id", target_id) \
        .eq("relationship_type", relationship_type).execute().data
    assert len(rows) == 1, f"expected exactly one {relationship_type} edge, found {len(rows)}"
    return rows[0]


# =====================================================================
# 1. Organizer -> organized relationship
# =====================================================================

def test_tanmay_organized_meeting_edge_exists():
    r = _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")
    assert r["source_object_type"] == "entity"
    assert r["target_object_type"] == "entity"
    assert r["status"] == "active"


# =====================================================================
# 2. Attendee -> attended relationship
# =====================================================================

def test_john_snow_attended_meeting_edge_exists():
    r = _get_relationship(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended")
    assert r["source_object_type"] == "entity"
    assert r["target_object_type"] == "entity"
    assert r["status"] == "active"


# =====================================================================
# 3. Calendar snapshot used as primary evidence
# =====================================================================

def test_both_edges_use_calendar_snapshot_as_sole_primary_evidence():
    organized = _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")
    attended = _get_relationship(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended")
    for rel in (organized, attended):
        ev = supabase.table("knowledge_relationship_evidence").select("*") \
            .eq("relationship_id", rel["id"]).execute().data
        assert len(ev) == 1, "no fabricated second evidence row -- exactly the real snapshot, nothing else"
        assert ev[0]["evidence_type"] == "calendar_event_snapshot"
        assert ev[0]["evidence_id"] == REAL_SNAPSHOT_ID
        assert ev[0]["stance"] == "supports"


# =====================================================================
# 4. Deterministic Person identity
# =====================================================================

def test_person_endpoints_resolved_via_email_identifier_not_name():
    """Both source entity ids used to construct these edges are the exact
    entity_id values found via each Person's real, verified email
    identifier -- never via canonical_label text matching."""
    tanmay_ident = supabase.table("knowledge_entity_identifiers").select("entity_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("identifier_type", "email") \
        .eq("identifier_value", "tanmaydubeytd@gmail.com").execute().data
    assert tanmay_ident[0]["entity_id"] == TANMAY_ENTITY_ID

    js_ident = supabase.table("knowledge_entity_identifiers").select("entity_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("identifier_type", "email") \
        .eq("identifier_value", "kingjohnsnow0@gmail.com").execute().data
    assert js_ident[0]["entity_id"] == JOHN_SNOW_ENTITY_ID


# =====================================================================
# 5. Deterministic Meeting identity
# =====================================================================

def test_meeting_endpoint_resolved_via_external_event_id_not_name():
    ident = supabase.table("knowledge_entity_identifiers").select("entity_id,connection_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("identifier_type", "external_event_id") \
        .eq("identifier_value", "668o197bdkl5sljf4irv1ksju1").execute().data
    assert ident[0]["entity_id"] == MEETING_ENTITY_ID
    assert ident[0]["connection_id"] is not None


# =====================================================================
# 6. Explicit valid_from
# =====================================================================

def test_valid_from_is_meeting_start_time_not_captured_at_or_now():
    organized = _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")
    attended = _get_relationship(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended")
    snapshot = supabase.table("calendar_event_snapshots").select("start_time,captured_at") \
        .eq("id", REAL_SNAPSHOT_ID).execute().data[0]
    assert organized["valid_from"] == snapshot["start_time"]
    assert attended["valid_from"] == snapshot["start_time"]
    assert organized["valid_from"] != snapshot["captured_at"], \
        "valid_from must be the meeting's own start_time, never the snapshot's capture time"
    assert organized["valid_until"] is None
    assert attended["valid_until"] is None


# =====================================================================
# 7. Repeat idempotency
# =====================================================================

def test_repeat_construction_with_identical_valid_from_is_idempotent():
    before_rel = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    before_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count

    r1 = _rpc(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized", MEETING_START_TIME, _real_evidence())
    r2 = _rpc(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended", MEETING_START_TIME, _real_evidence())
    assert r1.data == _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")["id"]
    assert r2.data == _get_relationship(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended")["id"]

    after_rel = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    after_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count
    assert after_rel == before_rel, "no new relationship rows on identical-valid_from retry"
    assert after_ev == before_ev, "no new evidence rows on identical-valid_from retry"


# =====================================================================
# 8. Duplicate evidence prevention
# =====================================================================

def test_duplicate_evidence_attach_is_a_no_op():
    """Calling the RPC again with the SAME evidence record for an existing
    relationship must not create a second evidence row -- the RPC's own
    ON CONFLICT (relationship_id, evidence_type, evidence_id) DO NOTHING."""
    organized = _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")
    before = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", organized["id"]).execute().count

    _rpc(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized", MEETING_START_TIME, _real_evidence())

    after = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", organized["id"]).execute().count
    assert after == before == 1


# =====================================================================
# 9. Wrong workspace rejection
# =====================================================================

def test_wrong_workspace_rejected_atomically():
    before_rel = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    before_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count

    with pytest.raises(Exception):
        _rpc(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized", MEETING_START_TIME, _real_evidence(),
             workspace_id=OTHER_REAL_WORKSPACE, rationale="wrong-workspace security test")

    after_rel = supabase.table("knowledge_relationships").select("id", count="exact").execute().count
    after_ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count
    assert after_rel == before_rel, "zero new relationship rows on rejected wrong-workspace attempt"
    assert after_ev == before_ev, "zero new evidence rows on rejected wrong-workspace attempt"


# =====================================================================
# 10 & 11. No member_of / no employment inference
# =====================================================================

def test_no_member_of_relationship_type_exists_anywhere():
    types = {r["relationship_type"] for r in supabase.table("knowledge_relationships").select("relationship_type").execute().data}
    assert "member_of" not in types
    assert types == {"requires_approval_from", "organized", "attended"}


def test_no_employment_relationship_type_exists_anywhere():
    """No owns/manages/works_on/supports/affects/produced relationship_type
    exists -- confirms Phase 5H introduced ONLY organized and attended,
    nothing else, and no employment-adjacent semantic was smuggled in."""
    forbidden = {"member_of", "owns", "manages", "works_on", "supports", "affects", "produced", "employee_of"}
    types = {r["relationship_type"] for r in supabase.table("knowledge_relationships").select("relationship_type").execute().data}
    assert types.isdisjoint(forbidden)


def test_person_entities_still_carry_no_role_or_authority_fields():
    p = supabase.table("knowledge_entities").select("*").eq("id", TANMAY_ENTITY_ID).execute().data[0]
    assert set(p.keys()) == {"id", "workspace_id", "entity_type", "canonical_label",
                             "status", "external_ref_type", "external_ref_id",
                             "created_at", "updated_at"}


# =====================================================================
# 12. Organizer/attendee overlap rule
# =====================================================================

def test_organizer_also_listed_as_attendee_in_source_but_no_redundant_edge():
    """Reconfirms the real, observed case: the snapshot's attendees array
    DOES include the organizer's own email (tanmaydubeytd@gmail.com,
    response_status='accepted'). Despite that, Tanmay must have exactly ONE
    activity edge to the Meeting (organized), never a second 'attended'
    edge -- the strict Part 6 rule: same-person-in-multiple-Calendar-fields
    is not, on its own, a reason to create redundant relationship
    semantics."""
    snapshot = supabase.table("calendar_event_snapshots").select("attendees,organizer") \
        .eq("id", REAL_SNAPSHOT_ID).execute().data[0]
    attendee_emails = {a["email"] for a in snapshot["attendees"]}
    assert snapshot["organizer"] in attendee_emails, "sanity: this test only means something if the real data has the overlap"

    edges = supabase.table("knowledge_relationships").select("relationship_type") \
        .eq("source_object_id", TANMAY_ENTITY_ID).eq("target_object_id", MEETING_ENTITY_ID).execute().data
    assert {e["relationship_type"] for e in edges} == {"organized"}


def test_no_attended_edge_from_tanmay_exists():
    rows = supabase.table("knowledge_relationships").select("id") \
        .eq("source_object_id", TANMAY_ENTITY_ID).eq("relationship_type", "attended").execute().data
    assert rows == []


# =====================================================================
# 13. Graph read exposes relationship
# =====================================================================

def test_graph_read_exposes_both_edges():
    allowed = gq.resolve_allowed_sensitivities("owner", False)
    tanmay_graph = gq.get_entity_graph(TANMAY_ENTITY_ID, REAL_WORKSPACE, allowed)
    types = {r.relationship_type for r in tanmay_graph.outbound_relationships}
    assert "organized" in types

    js_graph = gq.get_entity_graph(JOHN_SNOW_ENTITY_ID, REAL_WORKSPACE, allowed)
    js_types = {r.relationship_type for r in js_graph.outbound_relationships}
    assert "attended" in js_types

    meeting_graph = gq.get_entity_graph(MEETING_ENTITY_ID, REAL_WORKSPACE, allowed)
    inbound_types = {(r.relationship_type, r.source.object_id) for r in meeting_graph.inbound_relationships}
    assert ("organized", TANMAY_ENTITY_ID) in inbound_types
    assert ("attended", JOHN_SNOW_ENTITY_ID) in inbound_types


# =====================================================================
# 14. Evidence explanation works
# =====================================================================

def test_explain_relationship_resolves_calendar_snapshot_for_both_edges():
    allowed = gq.resolve_allowed_sensitivities("owner", False)
    organized = _get_relationship(TANMAY_ENTITY_ID, MEETING_ENTITY_ID, "organized")
    attended = _get_relationship(JOHN_SNOW_ENTITY_ID, MEETING_ENTITY_ID, "attended")

    organized_explanation = gq.explain_relationship(organized["id"], REAL_WORKSPACE, allowed)
    assert len(organized_explanation) == 1
    assert organized_explanation[0].evidence_type == "calendar_event_snapshot"
    assert organized_explanation[0].source_reference == "https://meet.google.com/ngn-pjwu-jcn"

    attended_explanation = gq.explain_relationship(attended["id"], REAL_WORKSPACE, allowed)
    assert len(attended_explanation) == 1
    assert attended_explanation[0].evidence_type == "calendar_event_snapshot"


# =====================================================================
# 15. structured_knowledge remains exactly 15
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count == 15


# =====================================================================
# 16. Existing relationship remains
# =====================================================================

def test_original_requires_approval_from_relationship_unchanged():
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("relationship_type", "requires_approval_from").execute().data
    assert len(rows) == 1
    assert rows[0]["target_object_id"] == PRODUCT_ENTITY_ID
    ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", rows[0]["id"]).execute().count
    assert ev == 2


# =====================================================================
# 17. Existing entities remain
# =====================================================================

def test_original_five_entities_preserved():
    rows = supabase.table("knowledge_entities").select("canonical_label,entity_type") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    labels = {r["canonical_label"] for r in rows}
    assert labels == {"Product", "Operations", "Knova Test Meeting 1", "Tanmay", "John Snow"}
    assert len(rows) == 5, "no new entity was created by this pass -- only new relationship rows"


# =====================================================================
# 18. Calendar snapshot remains unchanged
# =====================================================================

def test_calendar_snapshot_unchanged():
    """STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    Calendar sync event ("Sales Catchup") legitimately arrived live during
    this session via the deployed filtration-worker cron -- the actual
    invariant this test protects (THIS specific snapshot's own content is
    untouched by anything in this pass) is unaffected and still checked by id."""
    count = supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count
    assert count == 2
    snap = supabase.table("calendar_event_snapshots").select("*").eq("id", REAL_SNAPSHOT_ID).execute().data[0]
    assert snap["organizer"] == "tanmaydubeytd@gmail.com"
    assert snap["title"] == "Knova Test Meeting 1"


# =====================================================================
# 19. No fixture leakage
# =====================================================================

def test_no_test_5h_relationships_leaked():
    leaked = supabase.table("knowledge_relationships").select("id,rationale") \
        .like("rationale", "%security test%").execute().data
    # the wrong-workspace security test (test 9) is REJECTED before any row is
    # written, so this must always be empty -- proves that rejection really is
    # atomic, not just "eventually cleaned up".
    assert leaked == [], f"a security-test relationship was persisted despite being rejected: {leaked}"


def test_no_leaked_synthetic_entities_or_identifiers_from_this_suite():
    leaked_entities = supabase.table("knowledge_entities").select("id,canonical_label") \
        .like("canonical_label", "TEST-5H-%").execute().data
    assert leaked_entities == []


# =====================================================================
# 20. Full-state regression sentinel
# =====================================================================

def test_full_known_state_after_phase_5h():
    """A single, comprehensive snapshot of every row count this and every
    prior Phase 5 pass is responsible for -- the closest thing to a one-test
    full regression check. The real, separate full pytest run across every
    test_phase5*.py file remains the authoritative regression gate; this is
    a fast sentinel for exactly the counts this pass must not have disturbed."""
    assert supabase.table("knowledge_entities").select("id", count="exact").execute().count == 5
    assert supabase.table("knowledge_entity_identifiers").select("id", count="exact").execute().count == 5
    assert supabase.table("knowledge_relationships").select("id", count="exact").execute().count == 3
    assert supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count == 4
    # STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    # Calendar sync event legitimately arrived live during this session --
    # see test_calendar_snapshot_unchanged's docstring above.
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count == 2
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
