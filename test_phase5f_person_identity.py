"""
Phase 5F Person identity tests -- verifies the two real Person entities
constructed this pass (Tanmay, John Snow) against real app-DB identity data,
and the deterministic-identity discipline (Parts 2/3/5) against synthetic
fixtures for the boundary cases the real corpus can't exercise on its own.

THE JOHN SNOW CASE, decided precisely (see the Phase 5F report for the full
reasoning): `kingjohnsnow0@gmail.com` is a real, Google-verified auth.users
account whose own raw_user_meta_data.name genuinely IS "John Snow" -- found
via the deterministic email match on the real Calendar organizer/attendee
evidence, not via the Slack participant string. The Slack-sourced notes'
`participants: ["John Snow"]` field is NEVER linked, aliased, or merged into
this entity -- the only connection between the two is text-string
resemblance, which the identity rules explicitly exclude as sufficient,
regardless of how likely it is to be factually correct. This is deliberate
discipline, not an oversight, and is asserted directly by
test_slack_john_snow_mention_never_aliased below.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5f_person_identity.py -v
"""
import uuid
import unicodedata
import re
from datetime import datetime, timezone

import pytest

import graph_query as gq
from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_EMAIL = "tanmaydubeytd@gmail.com"
JOHN_SNOW_EMAIL = "kingjohnsnow0@gmail.com"
REAL_CALENDAR_EVENT_MEETING_URL = "https://meet.google.com/ngn-pjwu-jcn"
# A second real Calendar sync event ("Sales Catchup") arrived live during
# the Phase 6D regression pass (2026-08-18 20:00:30 UTC) via the actually-
# deployed filtration-worker cron -- same class of real, unrelated
# production event already documented once before this session (the first
# calendar_events row growing 1->2 mid-session). Tanmay organizes/attends
# this one too, so his own evidence resolution now legitimately returns 2
# rows instead of 1.
REAL_CALENDAR_EVENT_2_MEETING_URL = "https://meet.google.com/rsb-bgch-ycs"


def _normalize_alias(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip().lower())


def _get_person(email: str) -> dict:
    ident = supabase.table("knowledge_entity_identifiers").select("entity_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("identifier_type", "email") \
        .eq("identifier_value", email).execute().data
    assert ident, f"sanity: {email} must have a real identifier row"
    return supabase.table("knowledge_entities").select("*").eq("id", ident[0]["entity_id"]).execute().data[0]


# =====================================================================
# 1. Valid deterministic provider identity creates Person
# =====================================================================

def test_tanmay_person_created_with_verified_identity():
    p = _get_person(TANMAY_EMAIL)
    assert p["entity_type"] == "person"
    assert p["canonical_label"] == "Tanmay"  # real, Google-verified raw_user_meta_data.name -- not the email, not the job title
    assert p["status"] == "active"
    assert p["external_ref_type"] is None and p["external_ref_id"] is None, "Person never uses external_ref_* -- same as Meeting"


def test_john_snow_person_created_with_verified_identity():
    p = _get_person(JOHN_SNOW_EMAIL)
    assert p["entity_type"] == "person"
    assert p["canonical_label"] == "John Snow"
    assert p["status"] == "active"


# =====================================================================
# 2. Bare name does not create Person (on its own)
# =====================================================================

def test_no_person_entity_lacks_a_real_identifier():
    """Every Person entity in the graph must have at least one real
    knowledge_entity_identifiers row -- there is no Person created from a
    bare name alone anywhere in the real construction."""
    people = supabase.table("knowledge_entities").select("id").eq("entity_type", "person").execute().data
    for p in people:
        idents = supabase.table("knowledge_entity_identifiers").select("id").eq("entity_id", p["id"]).execute().data
        assert idents, f"person entity {p['id']} has no identifier -- violates the deterministic-identity contract"


def test_exactly_two_person_entities_exist():
    """STALE NOTE (Phase 5G): this docstring originally said the Google Chat
    resource users/109566945468284233018 "is not stored as a second
    identifier ... It remains documented evidence, not a third entity or a
    new identifier row." That was true as of Phase 5F, when the frozen
    identifier_type enum had no generic provider-user slot. Phase 5G closed
    that exact gap: it added identifier_type='provider_user_id' (connection-
    scoped) and attached users/109566945468284233018 to Tanmay's EXISTING
    entity as a SECOND identifier row (see test_phase5g_identity_contract.py
    ::test_tanmay_gets_real_google_provider_user_id). The invariant that
    still genuinely holds, and is what this test actually asserts, is entity
    COUNT: attaching a second identifier to an existing Person must never
    create a new entity. Two Person entities, not three."""
    count = supabase.table("knowledge_entities").select("id", count="exact").eq("entity_type", "person").execute().count
    assert count == 2


# =====================================================================
# 3. Same identifier repeated is idempotent
# =====================================================================

def test_person_identifier_uniqueness_enforced():
    entity_id = _get_person(TANMAY_EMAIL)["id"]
    with pytest.raises(Exception):
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": entity_id, "workspace_id": REAL_WORKSPACE, "connection_id": None,
            "identifier_type": "email", "identifier_value": TANMAY_EMAIL,
        }).execute()


# =====================================================================
# 4. Same identifier across workspaces does not collide
# =====================================================================

def test_same_email_different_workspace_is_independent():
    """Synthetic: the same real email, anchoring a DIFFERENT synthetic
    Person entity under a different real workspace, must not collide with
    or be confused for the real f7aab311 Person."""
    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": OTHER_REAL_WORKSPACE, "entity_type": "person",
            "canonical_label": "TEST-5F-CROSS-WS", "status": "active",
        }).execute().data[0]["id"]
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": entity_id, "workspace_id": OTHER_REAL_WORKSPACE, "connection_id": None,
            "identifier_type": "email", "identifier_value": TANMAY_EMAIL,
        }).execute()

        real_tanmay = _get_person(TANMAY_EMAIL)
        assert real_tanmay["id"] != entity_id
        assert real_tanmay["workspace_id"] == REAL_WORKSPACE
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


# =====================================================================
# 5 & 6. Same display name, different identifiers stays separate; conflicting identifiers do not merge
# =====================================================================

def test_same_display_name_different_identifiers_stays_separate():
    """Two synthetic Person entities sharing a canonical_label must remain
    two distinct rows -- name equality is never treated as identity
    equality anywhere in this construction."""
    ids = []
    try:
        a = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person",
            "canonical_label": "TEST-5F-SAMENAME", "status": "active",
        }).execute().data[0]["id"]
        ids.append(a)
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": a, "workspace_id": REAL_WORKSPACE, "connection_id": None,
            "identifier_type": "email", "identifier_value": "test-5f-samename-a@example.com",
        }).execute()

        b = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person",
            "canonical_label": "TEST-5F-SAMENAME", "status": "active",
        }).execute().data[0]["id"]
        ids.append(b)
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": b, "workspace_id": REAL_WORKSPACE, "connection_id": None,
            "identifier_type": "email", "identifier_value": "test-5f-samename-b@example.com",
        }).execute()

        assert a != b
        rows = supabase.table("knowledge_entities").select("id").eq("canonical_label", "TEST-5F-SAMENAME").execute().data
        assert len(rows) == 2, "same display name must never collapse two real, distinct identifiers into one entity"
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 7. Alias normalization
# =====================================================================

def test_alias_normalization_still_holds_for_person_entities():
    """No real alias was created for either real Person (see module
    docstring on why the Slack 'John Snow' string is never aliased) -- this
    proves the mechanism itself still works correctly for entity_type=
    'person', using a synthetic alias with real evidentiary backing."""
    entity_id = None
    try:
        entity_id = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "person",
            "canonical_label": "TEST-5F-ALIAS", "status": "active",
        }).execute().data[0]["id"]

        supabase.table("knowledge_entity_aliases").insert({
            "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
            "alias_text": "T. Alias", "alias_normalized": _normalize_alias("T. Alias"),
            "alias_source_type": "structured_knowledge", "alias_source_id": "fc261a0a-4aa7-4224-a2b1-66513a03a05e",
        }).execute()

        with pytest.raises(Exception):
            supabase.table("knowledge_entity_aliases").insert({
                "entity_id": entity_id, "workspace_id": REAL_WORKSPACE,
                "alias_text": " T.  ALIAS ", "alias_normalized": _normalize_alias(" T.  ALIAS "),
                "alias_source_type": "structured_knowledge", "alias_source_id": "fc261a0a-4aa7-4224-a2b1-66513a03a05e",
            }).execute()
    finally:
        if entity_id:
            supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


# =====================================================================
# 8. Wrong-workspace rejection
# =====================================================================

def test_person_entity_not_visible_under_wrong_workspace():
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    result = gq.get_entity_graph(tanmay_id, OTHER_REAL_WORKSPACE, gq.resolve_allowed_sensitivities("owner", False))
    assert result is None


def test_email_identifier_lookup_scoped_to_correct_workspace():
    wrong_ws_match = supabase.table("knowledge_entity_identifiers").select("id") \
        .eq("workspace_id", OTHER_REAL_WORKSPACE).eq("identifier_type", "email") \
        .eq("identifier_value", TANMAY_EMAIL).execute().data
    assert wrong_ws_match == []


# =====================================================================
# 9. Ambiguous John Snow case -- the crux test
# =====================================================================

def test_slack_john_snow_mention_never_aliased():
    """The single most important assertion in this suite: the John Snow
    Person entity (anchored by the real, verified email) must have ZERO
    alias rows -- specifically, the Slack participants' bare "John Snow"
    string was never turned into an alias, despite genuinely being the same
    real name (confirmed via Google-verified account metadata). The
    identity rules correctly withhold that convergence because the Slack
    connector never captured a structural identifier for it -- text
    resemblance alone, however factually likely to be correct, is not
    treated as proof."""
    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    aliases = supabase.table("knowledge_entity_aliases").select("id").eq("entity_id", john_snow_id).execute().data
    assert aliases == []


def test_john_snow_identity_is_anchored_by_email_not_by_slack_text():
    """The ONLY identifier on the John Snow entity is its real, verified
    email -- confirming construction never fell back to using the Slack
    display-name string as an identifier of any kind."""
    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    idents = supabase.table("knowledge_entity_identifiers").select("identifier_type,identifier_value") \
        .eq("entity_id", john_snow_id).execute().data
    assert len(idents) == 1
    assert idents[0]["identifier_type"] == "email"
    assert idents[0]["identifier_value"] == JOHN_SNOW_EMAIL


# =====================================================================
# 10. Verified Person evidence chain
# =====================================================================

def test_tanmay_evidence_resolves_as_calendar_organizer():
    """STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    Calendar sync event ("Sales Catchup") arrived live during this session
    -- Tanmay organizes/attends it too, so his evidence now legitimately
    resolves to 2 real snapshots, not 1. See REAL_CALENDAR_EVENT_2_MEETING_
    URL's own comment for the full explanation."""
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    evidence = gq.get_entity_primary_evidence(tanmay_id, REAL_WORKSPACE)
    assert len(evidence) == 2
    references = {e.source_reference for e in evidence}
    assert references == {REAL_CALENDAR_EVENT_MEETING_URL, REAL_CALENDAR_EVENT_2_MEETING_URL}
    for e in evidence:
        assert e.evidence_kind == "primary_source"
        assert e.evidence_type == "calendar_event_snapshot"
        assert e.stance == "supports"


def test_john_snow_evidence_resolves_as_calendar_attendee():
    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    evidence = gq.get_entity_primary_evidence(john_snow_id, REAL_WORKSPACE)
    assert len(evidence) == 1
    assert evidence[0].evidence_type == "calendar_event_snapshot"
    assert evidence[0].source_reference == REAL_CALENDAR_EVENT_MEETING_URL


# =====================================================================
# 11 & 12. member_of only with real evidence (none here); no guessed role
# =====================================================================

def test_no_employment_relationships_involve_either_person():
    """STALE (Phase 5H): this originally asserted ZERO relationships touch
    either Person at all. Phase 5H legitimately gave both Persons real
    activity edges (Tanmay --organized--> Meeting, John Snow --attended-->
    Meeting) -- see test_phase5h_meeting_activity_relationships.py. member_of
    is still not introduced, and still for the same reason (evidence
    threshold never met, not unsupported) -- what this test now verifies is
    that no relationship_type TOUCHING either Person implies employment,
    membership, ownership, or management; only the two real activity types
    do."""
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    rows = supabase.table("knowledge_relationships").select("relationship_type,source_object_id,target_object_id").execute().data
    touching = [r for r in rows if r["source_object_id"] in (tanmay_id, john_snow_id) or r["target_object_id"] in (tanmay_id, john_snow_id)]
    assert {r["relationship_type"] for r in touching} == {"organized", "attended"}


def test_person_entities_carry_no_role_or_authority_fields():
    """Identity establishes WHO, never organizational role -- knowledge_
    entities has no role/title/authority column at all for any entity_type,
    confirming no such field was fabricated for Person specifically."""
    p = _get_person(TANMAY_EMAIL)
    assert set(p.keys()) == {"id", "workspace_id", "entity_type", "canonical_label",
                             "status", "external_ref_type", "external_ref_id",
                             "created_at", "updated_at"}


# =====================================================================
# 13-16. Existing state unchanged
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count == 15


def test_original_three_entities_preserved():
    rows = supabase.table("knowledge_entities").select("canonical_label,entity_type") \
        .in_("entity_type", ["department", "meeting"]).execute().data
    labels = {r["canonical_label"] for r in rows}
    assert labels == {"Product", "Operations", "Knova Test Meeting 1"}


def test_original_relationship_preserved():
    """STALE (Phase 5H): originally asserted exactly one relationship exists
    globally. Phase 5H legitimately added two real Person->Meeting activity
    edges -- see test_phase5h_meeting_activity_relationships.py. Narrowed to
    what this test actually verifies: the original requires_approval_from
    edge, found by its own type rather than by "the only row", is unchanged."""
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("relationship_type", "requires_approval_from").execute().data
    assert len(rows) == 1
    ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", rows[0]["id"]).execute().count
    assert ev == 2


def test_calendar_snapshot_preserved():
    """STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    Calendar sync event legitimately arrived live during this session --
    see REAL_CALENDAR_EVENT_2_MEETING_URL's comment."""
    count = supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count
    assert count == 2


# =====================================================================
# 17. Fixture cleanup sentinel
# =====================================================================

def test_no_leaked_synthetic_entities():
    rows = supabase.table("knowledge_entities").select("canonical_label").execute().data
    labels = {r["canonical_label"] for r in rows}
    assert labels == {"Product", "Operations", "Knova Test Meeting 1", "Tanmay", "John Snow"}


def test_no_leaked_synthetic_identifiers():
    """STALE COUNT (Phase 5G): this was 4 as of Phase 5F (Meeting's
    external_event_id + conference_id, Tanmay's email, John Snow's email).
    Phase 5G legitimately added a 5th real, permanent row: Tanmay's
    provider_user_id (users/109566945468284233018, connection-scoped to the
    real Google connection) -- see test_phase5g_identity_contract.py::
    test_tanmay_gets_real_google_provider_user_id. The invariant this test
    still actually enforces is that the count matches the known-real set
    exactly, so any THIRD-party leaked row (synthetic or otherwise) would
    still be caught."""
    count = supabase.table("knowledge_entity_identifiers").select("id", count="exact").execute().count
    assert count == 5  # + Tanmay's provider_user_id (Phase 5G)


def test_no_leaked_synthetic_aliases():
    count = supabase.table("knowledge_entity_aliases").select("id", count="exact").execute().count
    assert count == 0
