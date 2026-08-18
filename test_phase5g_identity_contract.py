"""
Phase 5G Provider Identity Contract tests -- verifies the new, connection-
scoped `provider_user_id` identifier_type (added this pass, replacing the
never-used `slack_user_id`), Tanmay's new real identifier row, and the
identity-vs-activity evidence distinction added to
graph_query.get_entity_primary_evidence().

Design decisions this suite asserts (see the Phase 5G report for full
reasoning):
  - provider_user_id is CONNECTION-SCOPED (connection_id NOT NULL), exactly
    like external_event_id/conference_id -- the connection supplies the
    provider/account namespace, so there is no separate google_chat_user_id/
    slack_user_id/etc. per-provider type.
  - email/department_id remain WORKSPACE-GLOBAL (connection_id NULL).
  - slack_user_id was retired in this same migration: live verification
    (zero rows, zero code references anywhere in this repo) proved it was
    safe to consolidate rather than keep as a second overlapping identifier
    representation.
  - Text/name resemblance is NEVER sufficient for identity (John Snow's
    Slack mention is still never aliased to the real John Snow Person --
    reconfirmed here, not just in Phase 5F, because Phase 5G's new
    provider_user_id type is exactly the kind of change that could tempt a
    future pass into "resolving" that case improperly; this suite pins the
    discipline down again on the far side of the schema change).

Every fixture helper builds its id list incrementally with cleanup-on-
failure from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5g_identity_contract.py -v
"""
import pytest

import graph_query as gq
from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_EMAIL = "tanmaydubeytd@gmail.com"
JOHN_SNOW_EMAIL = "kingjohnsnow0@gmail.com"
REAL_CALENDAR_EVENT_MEETING_URL = "https://meet.google.com/ngn-pjwu-jcn"

# Two distinct REAL connection rows in REAL_WORKSPACE (both exist independently
# of this suite -- verified live before writing these tests), used to prove
# provider_user_id's connection-scoping without needing a synthetic connection.
REAL_GOOGLE_CONNECTION = "79d54c5e-8e2e-4fd6-bbd0-d7ea45502e83"
REAL_SLACK_CONNECTION = "35e46bc2-3909-41f8-a8d6-52ab12321d77"

TANMAY_PROVIDER_USER_ID = "users/109566945468284233018"


def _get_person(email: str) -> dict:
    ident = supabase.table("knowledge_entity_identifiers").select("entity_id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("identifier_type", "email") \
        .eq("identifier_value", email).execute().data
    assert ident, f"sanity: {email} must have a real identifier row"
    return supabase.table("knowledge_entities").select("*").eq("id", ident[0]["entity_id"]).execute().data[0]


def _make_synthetic_person(label: str, workspace_id: str = REAL_WORKSPACE) -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": "person",
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _insert_identifier(entity_id: str, workspace_id: str, connection_id, identifier_type: str, identifier_value: str):
    return supabase.table("knowledge_entity_identifiers").insert({
        "entity_id": entity_id, "workspace_id": workspace_id, "connection_id": connection_id,
        "identifier_type": identifier_type, "identifier_value": identifier_value,
    }).execute()


# =====================================================================
# 1. provider_user_id is connection-scoped
# =====================================================================

def test_provider_user_id_requires_connection_id():
    """The CHECK constraint must reject connection_id=NULL for
    provider_user_id -- it is not workspace-global like email/department_id."""
    entity_id = _make_synthetic_person("TEST-5G-SCOPE-1")
    try:
        with pytest.raises(Exception):
            _insert_identifier(entity_id, REAL_WORKSPACE, None, "provider_user_id", "users/test-5g-noconn")
    finally:
        supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


def test_email_still_requires_null_connection_id():
    """The inverse of test 1 -- reconfirms email is still rejected WITH a
    connection_id, i.e. the CHECK constraint's two branches are both intact
    after the migration, not just the new branch."""
    entity_id = _make_synthetic_person("TEST-5G-SCOPE-2")
    try:
        with pytest.raises(Exception):
            _insert_identifier(entity_id, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "email", "test-5g-scope2@example.com")
    finally:
        supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


# =====================================================================
# 2. Same provider_user_id in different connections doesn't collide
# =====================================================================

def test_same_provider_user_id_different_connections_independent():
    ids = []
    try:
        a = _make_synthetic_person("TEST-5G-CROSSCONN-A")
        ids.append(a)
        _insert_identifier(a, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-crossconn")

        b = _make_synthetic_person("TEST-5G-CROSSCONN-B")
        ids.append(b)
        # Same identifier_value, DIFFERENT real connection -- must succeed independently.
        _insert_identifier(b, REAL_WORKSPACE, REAL_SLACK_CONNECTION, "provider_user_id", "users/test-5g-crossconn")

        rows = supabase.table("knowledge_entity_identifiers").select("entity_id,connection_id") \
            .eq("identifier_type", "provider_user_id").eq("identifier_value", "users/test-5g-crossconn").execute().data
        assert {r["entity_id"] for r in rows} == {a, b}
        assert {r["connection_id"] for r in rows} == {REAL_GOOGLE_CONNECTION, REAL_SLACK_CONNECTION}
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 3. Same provider_user_id can't collide incorrectly within one connection
# =====================================================================

def test_same_provider_user_id_same_connection_rejected():
    """The partial unique index on (workspace_id, connection_id,
    identifier_type, identifier_value) must reject a second row with the
    exact same tuple -- this is what stops the SAME provider account from
    silently anchoring two different Person entities under one connection."""
    ids = []
    try:
        a = _make_synthetic_person("TEST-5G-SAMECONN-A")
        ids.append(a)
        _insert_identifier(a, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-sameconn")

        b = _make_synthetic_person("TEST-5G-SAMECONN-B")
        ids.append(b)
        with pytest.raises(Exception):
            _insert_identifier(b, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-sameconn")
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 4. Email remains workspace-global
# =====================================================================

def test_email_identifier_still_workspace_global():
    """Reconfirms email's uniqueness scope is (workspace_id, identifier_type,
    identifier_value) with no connection_id dimension at all -- unaffected
    by the provider_user_id migration."""
    entity_id = _make_synthetic_person("TEST-5G-EMAILSCOPE")
    try:
        with pytest.raises(Exception):
            _insert_identifier(entity_id, REAL_WORKSPACE, None, "email", TANMAY_EMAIL)
    finally:
        supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


# =====================================================================
# 5. provider_user_id + email can coexist on one Person
# =====================================================================

def test_tanmay_has_both_email_and_provider_user_id():
    tanmay = _get_person(TANMAY_EMAIL)
    idents = supabase.table("knowledge_entity_identifiers").select("identifier_type,identifier_value,connection_id") \
        .eq("entity_id", tanmay["id"]).execute().data
    by_type = {i["identifier_type"]: i for i in idents}
    assert set(by_type.keys()) == {"email", "provider_user_id"}
    assert by_type["email"]["identifier_value"] == TANMAY_EMAIL
    assert by_type["email"]["connection_id"] is None
    assert by_type["provider_user_id"]["identifier_value"] == TANMAY_PROVIDER_USER_ID
    assert by_type["provider_user_id"]["connection_id"] == REAL_GOOGLE_CONNECTION


# =====================================================================
# 6. Tanmay gets the real Google provider_user_id
# =====================================================================

def test_tanmay_gets_real_google_provider_user_id():
    """The exact value stored must match the real Google Chat sender.name /
    auth.users.raw_user_meta_data.provider_id cross-reference verified this
    pass -- not a placeholder, not a bare numeric ID (stored as observed:
    the full 'users/{id}' resource name, matching what Google Chat's API
    actually returns)."""
    tanmay = _get_person(TANMAY_EMAIL)
    row = supabase.table("knowledge_entity_identifiers").select("*") \
        .eq("entity_id", tanmay["id"]).eq("identifier_type", "provider_user_id").execute().data
    assert len(row) == 1
    assert row[0]["identifier_value"] == "users/109566945468284233018"
    assert row[0]["workspace_id"] == REAL_WORKSPACE


def test_tanmay_still_exactly_one_person_entity():
    """Adding the second identifier must not have created a duplicate
    Tanmay entity -- the insert reused the existing entity found via the
    email identifier."""
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("canonical_label", "Tanmay").execute().count
    assert count == 1


# =====================================================================
# 7. John Snow remains correctly represented
# =====================================================================

def test_john_snow_unchanged_by_provider_identity_pass():
    """John Snow gets NO provider_user_id in this pass -- no Google/Slack
    provider identity was ever verified for him (unlike Tanmay, whose
    provider_id came from a real auth.users cross-reference). His only
    identifier remains the real, verified email."""
    john_snow = _get_person(JOHN_SNOW_EMAIL)
    idents = supabase.table("knowledge_entity_identifiers").select("identifier_type,identifier_value") \
        .eq("entity_id", john_snow["id"]).execute().data
    assert len(idents) == 1
    assert idents[0]["identifier_type"] == "email"
    assert idents[0]["identifier_value"] == JOHN_SNOW_EMAIL


# =====================================================================
# 8. Slack name alone does not resolve John Snow
# =====================================================================

def test_slack_john_snow_mention_still_never_aliased():
    """Reconfirmed on the far side of the provider_user_id schema change:
    the Slack-sourced 'John Snow' participant text is still never turned
    into an alias or identifier for the real John Snow Person. The new
    provider_user_id type does not retroactively "solve" this case --
    Slack's connector does not currently persist a structural user ID
    anywhere queryable (see Part 7 of the report), so there is nothing new
    for the identity contract to resolve against; text resemblance alone
    remains insufficient, unchanged from Phase 5F."""
    john_snow = _get_person(JOHN_SNOW_EMAIL)
    aliases = supabase.table("knowledge_entity_aliases").select("id").eq("entity_id", john_snow["id"]).execute().data
    assert aliases == []


# =====================================================================
# 9. Same display name + different provider IDs stays separate
# =====================================================================

def test_same_display_name_different_provider_ids_stays_separate():
    ids = []
    try:
        a = _make_synthetic_person("TEST-5G-SAMENAME")
        ids.append(a)
        _insert_identifier(a, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-samename-a")

        b = _make_synthetic_person("TEST-5G-SAMENAME")
        ids.append(b)
        _insert_identifier(b, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-samename-b")

        assert a != b
        rows = supabase.table("knowledge_entities").select("id").eq("canonical_label", "TEST-5G-SAMENAME").execute().data
        assert len(rows) == 2
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 10. Conflicting identifiers do not merge
# =====================================================================

def test_conflicting_provider_ids_never_merge_entities():
    """Two synthetic Persons, each anchored by a DIFFERENT real
    provider_user_id under the same connection, must remain permanently
    distinct rows -- there is no merge/reconciliation mechanism anywhere in
    this construction, by design (Level 6 name-only text is never enough,
    and no automatic entity-merging exists for any identifier level)."""
    ids = []
    try:
        a = _make_synthetic_person("TEST-5G-CONFLICT-A")
        ids.append(a)
        _insert_identifier(a, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-conflict-a")

        b = _make_synthetic_person("TEST-5G-CONFLICT-B")
        ids.append(b)
        _insert_identifier(b, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "provider_user_id", "users/test-5g-conflict-b")

        entity_count = supabase.table("knowledge_entities").select("id", count="exact") \
            .in_("id", ids).execute().count
        assert entity_count == 2, "conflicting identifiers must never collapse into one entity"
    finally:
        for eid in ids:
            supabase.table("knowledge_entities").delete().eq("id", eid).execute()


# =====================================================================
# 11. Wrong-workspace identity cannot resolve
# =====================================================================

def test_provider_user_id_lookup_scoped_to_correct_workspace():
    wrong_ws_match = supabase.table("knowledge_entity_identifiers").select("id") \
        .eq("workspace_id", OTHER_REAL_WORKSPACE).eq("identifier_type", "provider_user_id") \
        .eq("identifier_value", TANMAY_PROVIDER_USER_ID).execute().data
    assert wrong_ws_match == []


def test_tanmay_entity_not_visible_under_wrong_workspace():
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    result = gq.get_entity_graph(tanmay_id, OTHER_REAL_WORKSPACE, gq.resolve_allowed_sensitivities("owner", False))
    assert result is None


# =====================================================================
# 12. No membership/employee relationship inferred
# =====================================================================

def test_no_membership_relationship_inferred_for_either_person():
    """STALE (Phase 5H): originally asserted zero relationships touch either
    Person -- Phase 5H legitimately added real activity edges (organized,
    attended) for exactly this reason (identity -> activity evidence ->
    graph relationship). Adding provider_user_id in Phase 5G itself still
    did not introduce any member_of/employee-style relationship; what this
    test verifies now is that no relationship touching either Person implies
    membership/employment -- only the two real, non-employment activity
    types do."""
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    rows = supabase.table("knowledge_relationships").select("relationship_type,source_object_id,target_object_id").execute().data
    touching = [r for r in rows if r["source_object_id"] in (tanmay_id, john_snow_id) or r["target_object_id"] in (tanmay_id, john_snow_id)]
    assert {r["relationship_type"] for r in touching} == {"organized", "attended"}


def test_no_relationship_type_implies_employment_exists_anywhere():
    """STALE (Phase 5H): originally asserted the only relationship_type in
    the whole graph was requires_approval_from. Phase 5H legitimately added
    'organized' and 'attended' -- both explicitly non-employment activity
    types (see the Phase 5H report). What this test verifies now is that no
    employment/membership-implying type (member_of, works_for, owns,
    manages, etc.) exists anywhere."""
    forbidden = {"member_of", "works_for", "owns", "manages", "works_on", "supports", "affects", "produced", "employee_of"}
    types = {r["relationship_type"] for r in supabase.table("knowledge_relationships").select("relationship_type").execute().data}
    assert types.isdisjoint(forbidden)
    assert types == {"requires_approval_from", "organized", "attended"}


# =====================================================================
# 13. Person primary evidence does not incorrectly imply employment
# =====================================================================

def test_person_evidence_role_is_activity_not_identity():
    """Part 8's core assertion: a Person's Calendar-snapshot evidence must
    be labeled evidence_role='activity' (proves attendance/organizing),
    NEVER 'identity' -- the actual identity proof (auth.users/provider_id
    cross-reference) happened outside this function and is not itself
    stored as queryable evidence anywhere in the graph. Mislabeling this as
    'identity' would let a caller wrongly treat meeting attendance as proof
    of who someone is, which could misread mere participation as
    membership/employment."""
    tanmay_id = _get_person(TANMAY_EMAIL)["id"]
    evidence = gq.get_entity_primary_evidence(tanmay_id, REAL_WORKSPACE)
    assert len(evidence) == 1
    assert evidence[0].evidence_role == "activity"

    john_snow_id = _get_person(JOHN_SNOW_EMAIL)["id"]
    js_evidence = gq.get_entity_primary_evidence(john_snow_id, REAL_WORKSPACE)
    assert len(js_evidence) == 1
    assert js_evidence[0].evidence_role == "activity"


# =====================================================================
# 14. Existing Meeting identifiers unchanged
# =====================================================================

def test_meeting_identifiers_unchanged():
    meeting = supabase.table("knowledge_entities").select("id").eq("workspace_id", REAL_WORKSPACE) \
        .eq("entity_type", "meeting").execute().data
    assert len(meeting) == 1
    idents = supabase.table("knowledge_entity_identifiers").select("identifier_type,connection_id") \
        .eq("entity_id", meeting[0]["id"]).execute().data
    types = {i["identifier_type"] for i in idents}
    assert types == {"external_event_id", "conference_id"}
    for i in idents:
        assert i["connection_id"] is not None


def test_meeting_evidence_role_is_identity():
    """The Meeting's own primary evidence (its calendar_event_snapshot,
    resolved via external_event_id) is correctly labeled evidence_role=
    'identity' -- unlike a Person's Calendar evidence, this snapshot IS
    what the Meeting entity's existence is anchored by, not mere activity."""
    meeting = supabase.table("knowledge_entities").select("id").eq("workspace_id", REAL_WORKSPACE) \
        .eq("entity_type", "meeting").execute().data[0]
    evidence = gq.get_entity_primary_evidence(meeting["id"], REAL_WORKSPACE)
    assert len(evidence) >= 1
    assert all(e.evidence_role == "identity" for e in evidence)


# =====================================================================
# 15. Existing Department identifiers unchanged
# =====================================================================

def test_department_identifiers_unchanged():
    depts = supabase.table("knowledge_entities").select("id,canonical_label") \
        .eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "department").execute().data
    assert {d["canonical_label"] for d in depts} == {"Product", "Operations"}
    for d in depts:
        idents = supabase.table("knowledge_entity_identifiers").select("identifier_type,connection_id") \
            .eq("entity_id", d["id"]).execute().data
        for i in idents:
            assert i["identifier_type"] == "department_id"
            assert i["connection_id"] is None


# =====================================================================
# 16. Existing relationship unchanged
# =====================================================================

def test_original_relationship_and_evidence_unchanged():
    """STALE (Phase 5H): originally asserted exactly one relationship exists
    globally. Phase 5H legitimately added two real Person->Meeting activity
    edges -- see test_phase5h_meeting_activity_relationships.py. Narrowed to
    what this test actually verifies: the original requires_approval_from
    edge, found by its own type, is unchanged."""
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("relationship_type", "requires_approval_from").execute().data
    assert len(rows) == 1
    ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", rows[0]["id"]).execute().count
    assert ev == 2


# =====================================================================
# 17. structured_knowledge exactly 15 rows
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count == 15


# =====================================================================
# 18. Idempotent identity construction
# =====================================================================

def test_tanmay_provider_user_id_insert_is_idempotent():
    """Re-running the exact NOT EXISTS-guarded insert used to construct
    Tanmay's real provider_user_id row must be a no-op the second time --
    matching this codebase's established idempotent-by-logical-key
    convention (same pattern as create_relationship_with_evidence)."""
    before = supabase.table("knowledge_entity_identifiers").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_GOOGLE_CONNECTION) \
        .eq("identifier_type", "provider_user_id").eq("identifier_value", TANMAY_PROVIDER_USER_ID).execute().count
    assert before == 1

    existing = supabase.table("knowledge_entity_identifiers").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_GOOGLE_CONNECTION) \
        .eq("identifier_type", "provider_user_id").eq("identifier_value", TANMAY_PROVIDER_USER_ID).execute().data
    if not existing:
        supabase.table("knowledge_entity_identifiers").insert({
            "entity_id": _get_person(TANMAY_EMAIL)["id"], "workspace_id": REAL_WORKSPACE,
            "connection_id": REAL_GOOGLE_CONNECTION, "identifier_type": "provider_user_id",
            "identifier_value": TANMAY_PROVIDER_USER_ID,
        }).execute()

    after = supabase.table("knowledge_entity_identifiers").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_GOOGLE_CONNECTION) \
        .eq("identifier_type", "provider_user_id").eq("identifier_value", TANMAY_PROVIDER_USER_ID).execute().count
    assert after == 1, "idempotent guard must prevent a duplicate row on re-run"


# =====================================================================
# Supplementary -- migration-level verification (not one of the 18, but
# required by Part 10's migration-safety review: proves the retirement
# itself, not just the new type's behavior)
# =====================================================================

def test_slack_user_id_no_longer_a_valid_identifier_type():
    """slack_user_id was retired in this migration (zero real rows, zero
    code references anywhere in this repo, confirmed live before the
    migration was applied) -- the CHECK constraint must now reject it."""
    entity_id = _make_synthetic_person("TEST-5G-RETIRED-TYPE")
    try:
        with pytest.raises(Exception):
            _insert_identifier(entity_id, REAL_WORKSPACE, REAL_GOOGLE_CONNECTION, "slack_user_id", "U12345")
    finally:
        supabase.table("knowledge_entities").delete().eq("id", entity_id).execute()


def test_no_slack_user_id_rows_exist_anywhere():
    count = supabase.table("knowledge_entity_identifiers").select("id", count="exact") \
        .eq("identifier_type", "slack_user_id").execute().count
    assert count == 0


# =====================================================================
# Fixture-leak sentinel (same discipline as every prior Phase 5 suite)
# =====================================================================

def test_no_test_5g_entities_leaked():
    leaked = supabase.table("knowledge_entities").select("id,canonical_label") \
        .like("canonical_label", "TEST-5G-%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_no_test_5g_identifiers_leaked():
    leaked = supabase.table("knowledge_entity_identifiers").select("id,identifier_value") \
        .like("identifier_value", "%test-5g%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"
