"""
Phase 8D deep drill-down tests.

Everything runs against the real Brain and the real endpoint coroutines —
this is the surface where an object id arrives from a browser, so the tests
drive the actual authorization path rather than a mock of it.

Run with: python -m pytest test_phase8d_drilldown_intelligence.py -v
"""
import inspect
import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from query import supabase
from auth import AuthContext
import graph_query as gq
import semantic_datasets as sd
import dashboard_detail as dd
import dashboard_brain_api as api

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)


def _ctx(workspace=REAL_WORKSPACE, role="owner"):
    return AuthContext(user_id="u", workspaces={workspace: role}, enforced=True, caller="pytest")


def _run(result):
    """Invokes a route handler and returns its result.

    The handlers used to be `async def`, so this wrapped every call in
    asyncio.run. They are plain `def` now -- they only ever did blocking
    Supabase I/O, and declaring that `async` made FastAPI run it ON the event
    loop, serialising every concurrent request in the process.

    This helper accepts BOTH so it is asserting the handler's result, not its
    declaration style. Nothing about what these tests check has changed; the
    calling convention was never the contract under test.
    """
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _drill(kind, object_id, dataset="memories", workspace=REAL_WORKSPACE, role="owner",
           as_of=None, max_hops=1):
    return _run(api.drilldown(
        api.DrillRequest(workspace_id=workspace, dataset=dataset, object_kind=kind,
                         object_id=object_id, as_of=as_of, max_hops=max_hops),
        _ctx(workspace, role)))


def _first(dataset, allowed=OWNER, workspace=REAL_WORKSPACE):
    r = sd.run_query(dataset, workspace, allowed)
    return r.rows[0] if r.rows else None


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# 1-9. Every real object type resolves.
# =====================================================================

@pytest.mark.parametrize("dataset,kind,type_label", [
    ("policies", "memory", "Policy"),
    ("processes", "memory", "Process"),
    ("people", "entity", "Person"),
    ("departments", "entity", "Department"),
    ("meetings", "entity", "Meeting"),
    ("relationships", "relationship", "Relationship"),
    ("evidence", "structured_knowledge", "Evidence"),
    ("learning", "learning", "Learning"),
])
def test_object_detail_resolves(dataset, kind, type_label):
    row = _first(dataset)
    assert row is not None, f"{dataset} has no rows to drill into"
    d = _drill(kind, row["object_id"], dataset=dataset)
    assert d["header"]["type_label"] == type_label
    assert d["header"]["label"]
    # Universal sections are always present, even when empty — an absent
    # section would read as "nothing to say", which is a different claim.
    for section in ("attributes", "changes", "evidence", "affected",
                     "connections", "not_established", "undetectable_changes"):
        assert section in d, f"{dataset} missing {section}"
    assert d["temporal_context"] == "current"


def test_decisions_dataset_has_no_rows_to_drill():
    """Zero decisions is a fact about the data, and the detail surface simply
    never opens — it must not fabricate an object to show."""
    assert _first("decisions") is None


# =====================================================================
# 10-13. Evidence chain and temporal semantics.
# =====================================================================

def test_evidence_chain_reaches_original_source():
    row = _first("policies")
    d = _drill("memory", row["object_id"], dataset="policies")
    assert d["evidence"], "a real policy must carry evidence"
    e = d["evidence"][0]
    assert e["statement"]
    assert e["provider"]
    if e["source_resolved"]:
        # The chain terminates at a real external reference.
        assert e["source_reference"]
        assert str(e["source_reference"]).startswith("http")
    else:
        assert any("original source" in n for n in d["not_established"])


def test_unresolved_source_is_disclosed_not_hidden():
    """Every evidence item declares whether its chain actually terminated."""
    for dataset, kind in (("policies", "memory"), ("evidence", "structured_knowledge")):
        row = _first(dataset)
        d = _drill(kind, row["object_id"], dataset=dataset)
        for e in d["evidence"]:
            assert "source_resolved" in e


def test_temporal_labels_are_semantic_not_generic():
    """The four temporal concepts must stay distinguishable in a memory's
    own attributes — a single generic 'date' would erase the distinction."""
    row = _first("policies")
    d = _drill("memory", row["object_id"], dataset="policies")
    attrs = d["attributes"]
    for key in ("created_at", "valid_from", "valid_until", "superseded_at", "last_confirmed_at"):
        assert key in attrs
    assert "date" not in attrs


def test_historical_detail_carries_its_temporal_context():
    row = _first("policies")
    past = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    # 400 days ago the memory did not exist yet -> identical safe 404.
    with pytest.raises(HTTPException) as e:
        _drill("memory", row["object_id"], dataset="policies", as_of=past)
    assert e.value.status_code == 404

    recent = datetime.now(timezone.utc).isoformat()
    d = _drill("memory", row["object_id"], dataset="policies", as_of=recent)
    assert d["temporal_context"] == recent
    assert d["temporal_context"] != "current"


# =====================================================================
# 14-18. Change explanation, impact, and what must never be inferred.
# =====================================================================

def test_change_explanation_is_deterministic_or_absent():
    row = _first("policies")
    d = _drill("memory", row["object_id"], dataset="policies")
    for c in d["changes"]:
        assert c["explanation"], "a change must carry text"
        # Either a real deterministic explanation, or the honest fallback.
        assert isinstance(c["explanation"], str)
        assert "previous_state" in c and "new_state" in c


def test_reason_not_established_is_a_real_value():
    """The fallback string exists in the module and is used verbatim rather
    than an LLM being asked to invent a cause."""
    import inspect
    src = inspect.getsource(dd)
    assert "Reason not established." in src
    for banned in ("ai.chat", "chat_json", "invoke_model", "bedrock"):
        assert banned not in src


def test_impact_is_bounded_and_evidence_backed():
    row = _first("departments")
    d = _drill("entity", row["object_id"], dataset="departments", max_hops=2)
    assert d["max_hops"] == 2
    for a in d["affected"]:
        assert a["hops"] <= 2
        assert a["object_id"]
        # Impact comes from real relationship traversal, never similarity.
        assert isinstance(a["relationship_types"], list)


def test_traversal_depth_is_rejected_beyond_two():
    row = _first("departments")
    for bad in (0, 3, 10):
        with pytest.raises(HTTPException) as e:
            _drill("entity", row["object_id"], dataset="departments", max_hops=bad)
        assert e.value.status_code == 400


def test_person_never_infers_employment_or_department():
    """Part 8. An email identifier is the only verified identity fact."""
    row = _first("people")
    d = _drill("entity", row["object_id"], dataset="people")
    joined = " ".join(d["not_established"]).lower()
    for claim in ("job title", "department membership", "reporting line", "employment"):
        assert claim in joined, f"person detail must declare '{claim}' not established"
    for forbidden in ("job_title", "department", "manager", "employer", "reports_to"):
        assert forbidden not in d["attributes"]
    assert "verified_email_identifiers" in d["attributes"]


def test_department_never_infers_membership_or_headcount():
    row = _first("departments")
    d = _drill("entity", row["object_id"], dataset="departments")
    joined = " ".join(d["not_established"]).lower()
    for claim in ("membership", "headcount", "ownership"):
        assert claim in joined
    for forbidden in ("members", "headcount", "owner", "member_count"):
        assert forbidden not in d["attributes"]


def test_meeting_attendance_never_implies_membership():
    row = _first("meetings")
    d = _drill("entity", row["object_id"], dataset="meetings")
    joined = " ".join(d["not_established"]).lower()
    assert "attendance does not establish" in joined


def test_meeting_uses_real_calendar_evidence():
    row = _first("meetings")
    d = _drill("entity", row["object_id"], dataset="meetings")
    attrs = d["attributes"]
    if "meeting_title" in attrs:
        # Calendar snapshot resolved -> these come from the real snapshot row.
        for k in ("start_time", "end_time", "organizer"):
            assert k in attrs
    else:
        assert any("calendar record" in n.lower() for n in d["not_established"])


# =====================================================================
# 19-21. Relationship, learning, and the learning/fact boundary.
# =====================================================================

def test_relationship_detail_shows_the_full_triple():
    row = _first("relationships")
    d = _drill("relationship", row["object_id"], dataset="relationships")
    a = d["attributes"]
    for k in ("source", "relationship_type", "target", "valid_from", "status"):
        assert k in a
    assert len(d["connections"]) == 2  # both endpoints are navigable
    assert "-[" in d["header"]["label"] and "]->" in d["header"]["label"]


def test_learning_is_labelled_as_derived_not_fact():
    row = _first("learning")
    d = _drill("learning", row["object_id"], dataset="learning")
    joined = " ".join(d["not_established"]).lower()
    assert "derived pattern, not a fact" in joined
    a = d["attributes"]
    for k in ("learning_type", "reasoning_state", "support_count",
               "observed_from", "observed_to", "review_required"):
        assert k in a


def test_no_relationship_type_is_invented():
    """The frozen ontology holds: nothing in the detail layer introduces a
    membership/ownership edge the graph does not have."""
    import inspect
    src = inspect.getsource(dd)
    for forbidden in ("member_of", "works_on", "owns", "manages", "reports_to", "employed_by"):
        assert f'"{forbidden}"' not in src


# =====================================================================
# 22-27. Security.
# =====================================================================

def test_wrong_workspace_is_rejected():
    """The caller holds a real membership, but not for the workspace being
    requested. That must be a 403 BEFORE any object lookup — the earlier
    version of this test handed the context membership of the very workspace
    it was asking about, so it only ever proved the object wasn't there."""
    row = _first("policies")
    other = str(uuid.uuid4())
    auth = AuthContext(user_id="u", workspaces={REAL_WORKSPACE: "owner"},
                       enforced=True, caller="pytest")
    with pytest.raises(HTTPException) as e:
        _run(api.drilldown(
            api.DrillRequest(workspace_id=other, dataset="policies",
                             object_kind="memory", object_id=row["object_id"]),
            auth))
    assert e.value.status_code == 403


def test_unknown_object_and_hidden_object_are_indistinguishable():
    with pytest.raises(HTTPException) as e:
        _drill("memory", str(uuid.uuid4()), dataset="policies")
    assert e.value.status_code == 404
    assert e.value.detail == "Not found."


def test_restricted_evidence_is_not_disclosed_by_id():
    """Phase 8D security finding, regression-locked.

    graph_query.get_structured_knowledge_graph selects a claim by id +
    workspace and applies the sensitivity ladder only to the relationships
    around it, so it returns a restricted STATEMENT to any caller. This
    endpoint accepts an object id straight from a browser, so it gates the
    claim itself before reading anything — without that, a low-clearance
    caller could read any restricted claim by guessing or replaying an id.
    """
    ws, ids = str(uuid.uuid4()), []
    try:
        sk = supabase.table("structured_knowledge").insert({
            "workspace_id": ws, "canonical_source_type": "knowledge_note",
            "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
            "primitive_type": "fact", "statement": "RESTRICTED-8D secret claim",
            "raw_subject_phrase": "x", "qualifier_words": [], "sensitivity": "restricted",
            "authority": "official", "source_tier": 2, "lifecycle_status": "active",
            "extraction_version": "v2.1", "captured_at": _now_iso(),
            "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"t8d-{uuid.uuid4()}"}).execute().data[0]["id"]
        ids.append(sk)

        with pytest.raises(HTTPException) as e:
            _drill("structured_knowledge", sk, dataset="evidence", workspace=ws, role="member")
        assert e.value.status_code == 404
        assert e.value.detail == "Not found."

        owner_view = _drill("structured_knowledge", sk, dataset="evidence",
                            workspace=ws, role="owner")
        assert "RESTRICTED-8D" in owner_view["header"]["label"]
    finally:
        for i in ids:
            supabase.table("structured_knowledge").delete().eq("id", i).execute()


def test_restricted_memory_detail_is_not_disclosed():
    ws, ids = str(uuid.uuid4()), {"sk": [], "mem": []}
    try:
        sk = supabase.table("structured_knowledge").insert({
            "workspace_id": ws, "canonical_source_type": "knowledge_note",
            "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
            "primitive_type": "fact", "statement": "RESTRICTED-8D memory claim",
            "raw_subject_phrase": "x", "qualifier_words": [], "sensitivity": "restricted",
            "authority": "official", "source_tier": 2, "lifecycle_status": "active",
            "extraction_version": "v2.1", "captured_at": _now_iso(),
            "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"t8d-{uuid.uuid4()}"}).execute().data[0]["id"]
        ids["sk"].append(sk)
        mem = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": ws, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": None,
            "p_valid_until": None, "p_supersedes_memory_id": None,
            "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                             "stance": "supports", "captured_at": _now_iso()}]}).execute().data
        ids["mem"].append(mem)

        with pytest.raises(HTTPException) as e:
            _drill("memory", mem, dataset="policies", workspace=ws, role="member")
        assert e.value.status_code == 404
    finally:
        for m in ids["mem"]:
            supabase.table("memory_evidence").delete().eq("memory_id", m).execute()
            supabase.table("org_memory").delete().eq("id", m).execute()
        for s in ids["sk"]:
            supabase.table("structured_knowledge").delete().eq("id", s).execute()


def test_workspace_isolation_of_detail_content():
    row = _first("policies", workspace=LEAK_WORKSPACE)
    if row is None:
        pytest.skip("leak workspace has no policy to drill")
    d = _drill("memory", row["object_id"], dataset="policies", workspace=LEAK_WORKSPACE)
    blob = str(d).lower()
    for term in ("credential", "procurement", "tanmay"):
        assert term not in blob


def test_unsupported_object_kind_is_rejected():
    for bad in ("project", "milestone", "user", "file"):
        with pytest.raises(HTTPException) as e:
            _drill(bad, str(uuid.uuid4()), dataset="memories")
        assert e.value.status_code == 400


def test_client_cannot_supply_authorization_on_drilldown():
    fields = set(api.DrillRequest.model_fields)
    for forbidden in ("role", "is_super_admin", "allowed_sensitivities", "sensitivity"):
        assert forbidden not in fields


# =====================================================================
# 28-33. Contract, disclosure, and Ask-KNOVA.
# =====================================================================

def test_ask_knova_context_is_identifiers_only():
    """Part 17: grounded context for the EXISTING query path. It must carry
    no answer and nothing that would let a downstream caller skip
    re-authorization."""
    row = _first("policies")
    d = _drill("memory", row["object_id"], dataset="policies")
    ctx = d["ask_context"]
    assert set(ctx) == {"workspace_id", "object_kind", "object_id", "label",
                         "temporal_context", "evidence_ids", "suggested_question"}
    for forbidden in ("allowed_sensitivities", "role", "answer", "prompt", "token"):
        assert forbidden not in ctx


def test_undetectable_changes_always_disclosed():
    import change_detection
    row = _first("policies")
    d = _drill("memory", row["object_id"], dataset="policies")
    joined = " ".join(d["undetectable_changes"])
    for k in change_detection.UNDETECTABLE_CHANGES:
        assert k in joined


def test_no_change_is_stated_not_implied():
    row = _first("evidence")
    d = _drill("structured_knowledge", row["object_id"], dataset="evidence")
    if not d["changes"]:
        assert any("no recorded change" in n.lower() for n in d["not_established"])


def test_detail_returns_no_raw_internal_columns():
    """The detail is a semantic view: no workspace_id, no fingerprints, no
    extraction bookkeeping leaks into a user-facing payload."""
    for dataset, kind in (("policies", "memory"), ("people", "entity"),
                           ("evidence", "structured_knowledge")):
        row = _first(dataset)
        d = _drill(kind, row["object_id"], dataset=dataset)
        for forbidden in ("workspace_id", "primitive_fingerprint", "extraction_run_id",
                           "state_fingerprint", "connection_id"):
            assert forbidden not in d["attributes"], f"{dataset} leaked {forbidden}"


def test_single_drilldown_system():
    """One endpoint, one resolver. A second drill-down path would be the
    thing this phase most needed to avoid."""
    paths = [r.path for r in api.router.routes]
    assert paths.count("/dashboard/drilldown") == 1
    assert sorted(dd.SUPPORTED_OBJECT_KINDS) == [
        "entity", "learning", "memory", "relationship", "structured_knowledge"]


def test_no_project_domain_anywhere_in_detail():
    import inspect
    src = inspect.getsource(dd)
    for forbidden in ("project", "milestone", "due_date", "progress", "AT RISK"):
        assert forbidden.lower() not in src.lower().replace("projection", "")


def test_fixture_cleanup_leaves_no_residue():
    ws = str(uuid.uuid4())
    sk = supabase.table("structured_knowledge").insert({
        "workspace_id": ws, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "temp-8d",
        "raw_subject_phrase": "x", "qualifier_words": [], "sensitivity": "internal",
        "authority": "official", "source_tier": 2, "lifecycle_status": "active",
        "extraction_version": "v2.1", "captured_at": _now_iso(),
        "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"t8d-{uuid.uuid4()}"}).execute().data[0]["id"]
    supabase.table("structured_knowledge").delete().eq("id", sk).execute()
    assert (supabase.table("structured_knowledge").select("id")
            .eq("workspace_id", ws).execute().data or []) == []
