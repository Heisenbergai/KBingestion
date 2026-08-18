"""
Phase 5B construction tests -- verifies the REAL entity construction that
was performed against the live vector DB: one Meeting entity (from the real
`calendar_events` row) and two Department entities (Product, Operations --
verified against the real app-DB `departments` table for workspace
f7aab311-c7b5-49c8-a8e4-36c89fa0b25d).

IMPORTANT ASYMMETRY, stated up front: the department-table verification
itself was performed via this session's own Supabase MCP tooling, which has
a direct, service-level connection to BOTH Supabase projects (vector DB and
app DB). The application's own runtime code has no such credential for the
app DB -- Railway holds no service-role key for it, which is exactly why
query_routing.py resolves department data via the caller's forwarded bearer
token instead. This test file's `supabase` client (from query.py) is
configured for the vector DB project only, matching the real application's
own access boundary -- so these tests do NOT re-query the app DB live; they
assert against the real department_id values already confirmed once, live,
outside this suite. If the real departments table ever changes, these
fixture IDs would need re-verification the same way they were the first
time -- they are not self-updating.

The three entities and two identifiers created by Phase 5B are real,
intended production data -- NOT deleted by this suite's cleanup. Only
synthetic rows this suite creates for negative/security tests are cleaned
up.

Run with: python -m pytest test_phase5b_construction.py -v
"""
import uuid

import pytest

from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
REAL_CONNECTION = "79d54c5e-8e2e-4fd6-bbd0-d7ea45502e83"
OTHER_REAL_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"  # bot_learning HR note's workspace

# Real, once-verified app-DB department ids (workspace f7aab311) -- see
# module docstring for why this suite doesn't re-query the app DB itself.
REAL_PRODUCT_DEPARTMENT_ID = "15b8bc62-22b3-42a0-8b6a-df0b18601909"
REAL_OPERATIONS_DEPARTMENT_ID = "af04b68f-ca0e-4279-b093-26fbb3575208"

REAL_MEETING_EXTERNAL_EVENT_ID = "668o197bdkl5sljf4irv1ksju1"
REAL_MEETING_CONFERENCE_ID = "ngn-pjwu-jcn"
REAL_CALENDAR_EVENT_ID = "aa473196-79dd-4a9c-aefc-f2c80d12ea94"


# =====================================================================
# 1. Meeting creation from real Calendar identity
# =====================================================================

def test_meeting_entity_created_with_correct_real_identity():
    row = supabase.table("knowledge_entities").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "meeting").execute().data
    assert len(row) == 1, "exactly one Meeting entity must exist"
    m = row[0]
    assert m["canonical_label"] == "Knova Test Meeting 1"
    assert m["status"] == "active"
    assert m["external_ref_type"] is None and m["external_ref_id"] is None, (
        "Meeting must NOT use external_ref_* -- Final Correction 2"
    )


def test_meeting_identifiers_match_real_calendar_row():
    m = supabase.table("knowledge_entities").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "meeting").execute().data[0]
    idents = supabase.table("knowledge_entity_identifiers").select("*") \
        .eq("entity_id", m["id"]).execute().data
    by_type = {i["identifier_type"]: i for i in idents}

    assert by_type["external_event_id"]["identifier_value"] == REAL_MEETING_EXTERNAL_EVENT_ID
    assert by_type["external_event_id"]["connection_id"] == REAL_CONNECTION
    assert by_type["conference_id"]["identifier_value"] == REAL_MEETING_CONFERENCE_ID
    assert by_type["conference_id"]["connection_id"] == REAL_CONNECTION
    assert len(idents) == 2, "exactly external_event_id + conference_id, nothing extra"


# =====================================================================
# 11. Meeting recurrence guard
# =====================================================================

def test_meeting_source_calendar_event_has_no_recurrence():
    """The real calendar_events row must have recurrence_rule IS NULL --
    this is what made Meeting Instance identity safe to use without any
    Series ambiguity. If this ever starts failing, the Meeting Instance
    identity basis for THIS row needs re-review, not a silent re-run."""
    row = supabase.table("calendar_events").select("recurrence_rule") \
        .eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    assert row["recurrence_rule"] is None


# =====================================================================
# 2. Meeting duplicate prevention (idempotency)
# =====================================================================

def test_meeting_construction_idempotent_no_duplicates():
    """Re-running the real construction logic (lookup-by-identifier,
    insert-only-if-not-found) against the already-constructed real data
    must not create a second Meeting entity."""
    existing = supabase.table("knowledge_entity_identifiers").select("entity_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
        .eq("identifier_type", "external_event_id") \
        .eq("identifier_value", REAL_MEETING_EXTERNAL_EVENT_ID).execute().data
    assert len(existing) == 1, "lookup-by-identifier must find exactly the one real Meeting -- this IS the idempotency check the construction logic performs"

    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "meeting").execute().count
    assert count == 1


# =====================================================================
# 3 & 4. Department creation only after real match; name normalization
# =====================================================================

def test_verified_departments_created_with_real_app_db_name_and_id():
    rows = supabase.table("knowledge_entities").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().data
    by_label = {r["canonical_label"]: r for r in rows}

    assert set(by_label.keys()) == {"Product", "Operations"}, (
        "exactly the two verified departments, real app-DB names, nothing invented"
    )
    assert by_label["Product"]["external_ref_type"] == "department_id"
    assert by_label["Product"]["external_ref_id"] == REAL_PRODUCT_DEPARTMENT_ID
    assert by_label["Operations"]["external_ref_type"] == "department_id"
    assert by_label["Operations"]["external_ref_id"] == REAL_OPERATIONS_DEPARTMENT_ID


def test_department_entities_have_no_redundant_identifier_rows():
    """Deliberate design choice: external_ref_type/external_ref_id on
    knowledge_entities IS the department's canonical identity -- a second,
    duplicate identifier row would be exactly the dual-source-of-truth drift
    trap this codebase has already named twice (F-13, F-40)."""
    rows = supabase.table("knowledge_entities").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().data
    for r in rows:
        idents = supabase.table("knowledge_entity_identifiers").select("id") \
            .eq("entity_id", r["id"]).execute().data
        assert idents == [], "Department entities must have zero rows in knowledge_entity_identifiers"


# =====================================================================
# 5. No Department created when app-DB lookup has no match
# =====================================================================

def test_qa_and_procurement_were_not_created():
    """QA and Procurement were the two text-strongest candidates in the
    Phase 5A analysis (QA especially -- corroborated across two independent
    canonical items) but neither matches a real department row for this
    workspace. Text strength alone must not have been enough to create
    them -- this is the single most important assertion in this suite."""
    rows = supabase.table("knowledge_entities").select("canonical_label") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().data
    labels = {r["canonical_label"] for r in rows}
    assert "QA" not in labels
    assert "Procurement" not in labels
    assert "HR" not in labels, "HR was AMBIGUOUS, not CREATE -- must not exist"
    assert "warehouse" not in labels and "assembly" not in labels, "both explicitly DEFERRED"


# =====================================================================
# 6. No Person creation from bare name
# =====================================================================

def test_person_entities_only_exist_with_real_verified_identity():
    """Person construction was completely disabled in Phase 5B specifically
    -- correct and unchanged as history. Phase 5F later authorized VERIFIED
    Person creation (see test_phase5f_person_identity.py) and legitimately
    added two: Tanmay and John Snow, both anchored by real, Google-verified
    emails, neither from a bare name alone. This test no longer asserts zero
    Person entities globally, since that would be asserting Phase 5F never
    happened -- instead it re-asserts the invariant that actually matters:
    every Person entity that exists has a real identifier backing it."""
    people = supabase.table("knowledge_entities").select("id").eq("entity_type", "person").execute().data
    for p in people:
        idents = supabase.table("knowledge_entity_identifiers").select("id").eq("entity_id", p["id"]).execute().data
        assert idents, f"person entity {p['id']} exists with no real identifier"


# =====================================================================
# 7. No Policy/Process auto-instantiation
# =====================================================================

def test_no_policy_or_process_entities_exist():
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .in_("entity_type", ["policy", "process"]).execute().count
    assert count == 0


def test_no_entity_types_outside_the_frozen_v1_set_exist():
    """'person' was legitimately added by Phase 5F (verified identity only,
    see test_phase5b_construction.py::test_person_entities_only_exist_with_
    real_verified_identity above) -- still no policy/process/product/
    project/etc., which remain fully unauthorized."""
    rows = supabase.table("knowledge_entities").select("entity_type").execute().data
    types = {r["entity_type"] for r in rows}
    assert types <= {"department", "meeting", "person"}, f"unexpected entity types present: {types - {'department', 'meeting', 'person'}}"


# =====================================================================
# 8. Workspace isolation
# =====================================================================

def test_all_real_entities_belong_to_the_source_workspace():
    rows = supabase.table("knowledge_entities").select("workspace_id").execute().data
    assert all(r["workspace_id"] == REAL_WORKSPACE for r in rows), (
        "every Phase 5B entity must belong to the one real workspace all 15 corpus rows come from"
    )


def test_department_external_ref_would_reject_cross_workspace_reuse():
    """Synthetic negative test: the SAME real department_id, claimed under a
    DIFFERENT workspace_id, must be a distinct (workspace_id, external_ref)
    pair per the partial unique index -- i.e. it does NOT collide with the
    real Product entity, so it would insert as a separate row (proving the
    index is workspace-scoped, not just external_ref-scoped). Cleaned up
    immediately -- this is a synthetic fixture, not a real entity."""
    new_id = None
    try:
        res = supabase.table("knowledge_entities").insert({
            "workspace_id": OTHER_REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "Product", "status": "active",
            "external_ref_type": "department_id", "external_ref_id": REAL_PRODUCT_DEPARTMENT_ID,
        }).execute()
        new_id = res.data[0]["id"]
        assert new_id != supabase.table("knowledge_entities").select("id") \
            .eq("workspace_id", REAL_WORKSPACE).eq("canonical_label", "Product").execute().data[0]["id"]
    finally:
        if new_id:
            supabase.table("knowledge_entities").delete().eq("id", new_id).execute()


# =====================================================================
# 9. Identifier uniqueness
# =====================================================================

def test_duplicate_meeting_identifier_rejected():
    m = supabase.table("knowledge_entities").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "meeting").execute().data[0]
    with pytest.raises(Exception):
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": m["id"], "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
            "identifier_type": "external_event_id", "identifier_value": REAL_MEETING_EXTERNAL_EVENT_ID,
        }).execute()


# =====================================================================
# 10. Repeat construction idempotency (entity level, both types)
# =====================================================================

def test_department_construction_idempotent_via_on_conflict():
    """Mirrors the real ON CONFLICT DO NOTHING construction logic directly:
    re-inserting the exact same (workspace, type, external_ref) pair must
    not create a duplicate, and must not raise -- DO NOTHING means exactly
    that."""
    before = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().count
    # The real Supabase client's .insert() doesn't expose a raw ON CONFLICT
    # clause, so this proves the invariant the SQL-level ON CONFLICT already
    # guarantees (verified live via direct SQL during construction): the
    # partial unique index makes a duplicate (workspace, type, external_ref)
    # tuple impossible to insert twice.
    with pytest.raises(Exception):
        supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "Product", "status": "active",
            "external_ref_type": "department_id", "external_ref_id": REAL_PRODUCT_DEPARTMENT_ID,
        }).execute()
    after = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().count
    assert before == after == 2


# =====================================================================
# 12. structured_knowledge / calendar_events / canonical data untouched
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    total = supabase.table("structured_knowledge").select("id", count="exact").execute().count
    assert total == 15
    v21 = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count
    assert v21 == 15


def test_extraction_contract_still_sole_current_v21():
    rows = supabase.table("extraction_contract_versions").select("*").execute().data
    assert len(rows) == 1
    assert rows[0]["version"] == "v2.1"
    assert rows[0]["is_current"] is True


def test_source_calendar_event_row_unchanged():
    row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    assert row["external_event_id"] == REAL_MEETING_EXTERNAL_EVENT_ID
    assert row["conference_id"] == REAL_MEETING_CONFERENCE_ID
    assert row["deleted_at"] is None


def test_no_aliases_created_in_this_pass():
    """Phase 5B created zero aliases (neither Department needed one -- see
    test_department_entities_have_no_redundant_identifier_rows -- and the
    Meeting entity has no naming variant in the corpus). Relationships were
    legitimately out of scope for Phase 5B and now exist as of Phase 5C
    (see test_phase5c_relationships.py); Calendar snapshots were legitimately
    out of scope too and now exist as of Phase 5E (see
    test_phase5e_calendar_evidence.py) -- this test no longer asserts zero
    relationships/evidence/snapshots, since asserting that would be asserting
    later phases never happened."""
    assert supabase.table("knowledge_entity_aliases").select("id", count="exact").execute().count == 0
