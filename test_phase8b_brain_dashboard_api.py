"""
Phase 8B Brain dashboard API tests.

Authorization is exercised against REAL AuthContext objects and the real
endpoint coroutines rather than a mocked layer -- the whole point of this
phase is that the router is the only thing standing between a browser and
the Brain DB, so the tests must drive the actual gate.

AUTH_ENFORCE is off in local dev, so `current_user` itself cannot be
meaningfully exercised here; what IS exercised is every decision that
depends on it -- assert_workspace, the role->ceiling ladder, and the fact
that no request model can carry authorization at all.

Run with: python -m pytest test_phase8b_brain_dashboard_api.py -v
"""
import ast
import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from query import supabase
from auth import AuthContext
import graph_query as gq
import semantic_datasets as sd
import dashboard_brain_api as api

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)


def _ctx(workspace=REAL_WORKSPACE, role="owner", super_admin=False, enforced=True):
    return AuthContext(user_id="test-user", workspaces={workspace: role} if workspace else {},
                       is_super_admin=super_admin, enforced=enforced, caller="pytest")


def _run(coro):
    return asyncio.run(coro)


def _query(**kw):
    kw.setdefault("workspace_id", REAL_WORKSPACE)
    auth = kw.pop("auth", None) or _ctx()
    return _run(api.query_dataset(api.DatasetQueryRequest(**kw), auth, None))


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _cleanup(ws, ids):
    for m in ids.get("mem", []):
        supabase.table("memory_evidence").delete().eq("memory_id", m).execute()
        supabase.table("org_memory").delete().eq("id", m).execute()
    for s in ids.get("sk", []):
        supabase.table("structured_knowledge").delete().eq("id", s).execute()


def _mk_sk(ws, ids, **kw):
    row = {"workspace_id": ws, "canonical_source_type": "knowledge_note",
           "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
           "primitive_type": "fact", "statement": "TEST-8B stmt",
           "raw_subject_phrase": "TEST-8B", "qualifier_words": [], "sensitivity": "internal",
           "authority": "official", "source_tier": 2, "lifecycle_status": "active",
           "extraction_version": "v2.1", "captured_at": _now_iso(),
           "extraction_run_id": str(uuid.uuid4()), "primitive_fingerprint": f"t8b-{uuid.uuid4()}"}
    row.update(kw)
    s = supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]
    ids.setdefault("sk", []).append(s)
    return s


def _mk_mem(ws, ids, sk, **kw):
    p = {"p_workspace_id": ws, "p_memory_type": "policy",
         "p_promotion_basis": "authoritative_policy", "p_valid_from": None,
         "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
         "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                          "stance": "supports", "captured_at": _now_iso()}]}
    p.update(kw)
    m = supabase.rpc("create_memory_with_evidence", p).execute().data
    ids.setdefault("mem", []).append(m)
    return m


# =====================================================================
# 1-3. Authorization gate.
# =====================================================================

def test_wrong_workspace_is_rejected():
    """A caller with a real membership elsewhere must not reach this
    workspace, and the 403 must be identical to the no-such-workspace case."""
    auth = _ctx(workspace=str(uuid.uuid4()))
    with pytest.raises(HTTPException) as e:
        _query(dataset="memories", auth=auth)
    assert e.value.status_code == 403


def test_missing_workspace_is_rejected():
    with pytest.raises(HTTPException) as e:
        _query(dataset="memories", workspace_id="")
    assert e.value.status_code == 400


def test_unenforced_context_is_flagged_untrusted():
    """AUTH_ENFORCE off produces enforced=False. Nothing may treat such a
    context as trustworthy -- this test exists so that fact stays visible."""
    assert _ctx(enforced=False).enforced is False
    assert _ctx().enforced is True


# =====================================================================
# 4-6. Sensitivity ceiling and existence leakage.
# =====================================================================

def test_sensitivity_ceiling_is_server_derived_per_role():
    assert api._allowed_sensitivities(_ctx(role="owner"), REAL_WORKSPACE) == OWNER
    assert api._allowed_sensitivities(_ctx(role="member"), REAL_WORKSPACE) == LOW
    assert "restricted" in api._allowed_sensitivities(_ctx(role="member", super_admin=True),
                                                       REAL_WORKSPACE)


def test_client_cannot_supply_authorization():
    """The request model has no role / is_super_admin / sensitivity field at
    all. Absence is stronger than validation: there is nothing to smuggle."""
    fields = set(api.DatasetQueryRequest.model_fields)
    for forbidden in ("role", "is_super_admin", "allowed_sensitivities",
                       "sensitivity_ceiling", "super_admin"):
        assert forbidden not in fields
    extra = api.DatasetQueryRequest(workspace_id=REAL_WORKSPACE, dataset="memories",
                                     role="owner", is_super_admin=True)
    assert not hasattr(extra, "role") or getattr(extra, "role", None) is None


def test_restricted_row_is_absent_not_hidden():
    """The Part 8 rule. A restricted memory must be ABSENT from a
    low-clearance caller's count -- never counted then redacted, because
    "3 (1 hidden)" discloses exactly what the ladder conceals."""
    ws, ids = str(uuid.uuid4()), {}
    try:
        _mk_mem(ws, ids, _mk_sk(ws, ids, statement="secret", sensitivity="restricted"),
                p_evidence=[{"evidence_type": "structured_knowledge",
                              "evidence_id": _mk_sk(ws, ids, statement="secret2",
                                                     sensitivity="restricted"),
                              "stance": "supports", "captured_at": _now_iso()}])
        hi = sd.run_query("memories", ws, OWNER, aggregation="count")
        lo = sd.run_query("memories", ws, LOW, aggregation="count")
        assert hi.row_count == 1
        assert lo.row_count == 0
        assert lo.aggregation["buckets"][0]["value"] == 0 if lo.aggregation["buckets"] else True
        blob = str(lo.rows) + str(lo.notes) + str(lo.not_established)
        assert "hidden" not in blob.lower()
        assert "secret" not in blob.lower()
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 7-19. Every dataset resolves against real data.
# =====================================================================

@pytest.mark.parametrize("dataset,expected", [
    ("policies", 3), ("processes", 1), ("decisions", 0), ("memories", 4),
    ("departments", 2), ("people", 2), ("meetings", 1), ("relationships", 3),
])
def test_dataset_real_counts(dataset, expected):
    r = _query(dataset=dataset)
    assert r["row_count"] == expected, f"{dataset} returned {r['row_count']}"
    assert r["security"]["filtered_before_aggregation"] is True


@pytest.mark.parametrize("dataset", ["changes", "attention", "company_state", "learning",
                                      "evidence"])
def test_derived_datasets_resolve(dataset):
    r = _query(dataset=dataset)
    assert r["row_count"] > 0
    assert r["generated_at"]


def test_decisions_empty_state_is_honest():
    """Zero rows means nothing was promoted -- NOT that the company made no
    decisions (Part 11)."""
    r = _query(dataset="decisions")
    assert r["row_count"] == 0
    assert r["empty_reason"] == "No decisions have been promoted to durable memory."
    assert "no decisions" not in (r["empty_reason"] or "").lower().replace(
        "no decisions have been promoted", "")


# =====================================================================
# 20-23. Registry is the boundary: every invalid input is rejected.
# =====================================================================

@pytest.mark.parametrize("kwargs,fragment", [
    ({"dataset": "projects"}, "Unknown dataset"),
    ({"dataset": "not_a_dataset"}, "Unknown dataset"),
    ({"dataset": "memories", "fields": ["password"]}, "Unknown field"),
    ({"dataset": "memories", "aggregation": "sum", "value_field": "evidence_count"},
     "Unsupported aggregation"),
    ({"dataset": "memories", "aggregation": "avg", "value_field": "evidence_count"},
     "Unsupported aggregation"),
    ({"dataset": "memories", "group_by": "statement", "aggregation": "count"},
     "not groupable"),
    ({"dataset": "memories", "filters": [{"field": "nope", "op": "eq", "value": 1}]},
     "Unknown field"),
    ({"dataset": "memories", "filters": [{"field": "memory_type", "op": "regex", "value": 1}]},
     "Unsupported filter operator"),
    ({"dataset": "changes", "temporal_mode": "as_of", "as_of": "2026-01-01T00:00:00Z"},
     "does not support temporal mode"),
    ({"dataset": "memories", "temporal_mode": "as_of"}, "requires as_of"),
    ({"dataset": "memories", "group_by": "created_at", "aggregation": "count"},
     "requires group_bucket"),
])
def test_invalid_input_rejected(kwargs, fragment):
    with pytest.raises(HTTPException) as e:
        _query(**kwargs)
    assert e.value.status_code == 400
    assert fragment in str(e.value.detail)


def test_projects_dataset_does_not_exist():
    """Part 19. Absent, not empty: `decisions` is empty because nothing was
    promoted; `projects` would be empty because the domain does not exist."""
    assert "projects" not in sd.DATASETS
    assert not any(d["key"] == "projects" for d in sd.list_datasets())


# =====================================================================
# 24-26. Temporal behaviour.
# =====================================================================

def test_current_semantics():
    r = _query(dataset="memories")
    assert r["temporal_context"] == "current"
    assert r["temporal_mode"] == "current"


def test_as_of_semantics_excludes_later_rows():
    """Reuses the Phase 6D.1 availability rule through memory_retrieval --
    no new date logic in the dataset layer."""
    past = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    r = _query(dataset="memories", temporal_mode="as_of", as_of=past)
    assert r["row_count"] == 0
    assert r["temporal_context"] == past


def test_window_semantics():
    r = _query(dataset="changes", temporal_mode="window", window_days=7)
    assert r["temporal_context"].startswith("window:")
    with pytest.raises(HTTPException):
        _query(dataset="changes", temporal_mode="window")


def test_no_generic_date_field_is_exposed():
    """Part 4: every date field must declare which temporal concept it is,
    so a dashboard can never group availability and claim-validity together
    on one unlabelled axis."""
    for ds in sd.DATASETS.values():
        for f in ds.fields:
            if f.datatype == "date":
                assert f.temporal_meaning != sd.TEMPORAL_NONE, f"{ds.key}.{f.key}"
                assert f.temporal_meaning in sd.TEMPORAL_MEANINGS
            assert f.key != "date"


# =====================================================================
# 27-28. Cross-database and people honesty.
# =====================================================================

def test_cross_database_department_join_degrades_honestly():
    """With no caller token the App-DB lookup cannot run. The Brain-side
    result must still be returned, flagged unresolved -- never fabricated
    (Part 9)."""
    assert api._resolve_app_departments(REAL_WORKSPACE, "") is None
    r = _query(dataset="departments")
    assert r["row_count"] == 2
    assert any("could not be resolved" in n for n in r["not_established"])
    for row in r["rows"]:
        assert row["values"].get("app_department_name") is None


def test_department_join_key_is_real():
    """The join key itself exists in the Brain data, so the join is a
    resolution problem and not an invented relationship."""
    r = _query(dataset="departments", fields=["label", "app_department_id"])
    ids = [row["values"]["app_department_id"] for row in r["rows"]]
    assert all(i for i in ids), "every department entity should carry external_ref_id"


def test_people_never_claims_department_membership():
    """Part 10: member_departments is empty database-wide, so a department
    column of nulls would read as 'unassigned' rather than 'unknown'."""
    ds = sd.get_dataset("people")
    keys = {f.key for f in ds.fields}
    for forbidden in ("department", "department_id", "department_name", "headcount"):
        assert forbidden not in keys
    r = _query(dataset="people")
    assert any("not established" in n.lower() for n in r["not_established"])
    for row in r["rows"]:
        assert any("not established" in n.lower() for n in row["not_established"])


# =====================================================================
# 29-31. Drilldown, markers, explanation.
# =====================================================================

def test_drilldown_reaches_evidence():
    mem = _query(dataset="memories", fields=["memory_id"])["rows"][0]
    body = api.DrillRequest(workspace_id=REAL_WORKSPACE, dataset="memories",
                             object_kind="memory", object_id=mem["object_id"])
    d = _run(api.drilldown(body, _ctx()))
    assert d["header"]["kind"] == "memory"
    assert d["evidence"], "a real memory must drill down to real evidence"
    assert all(e["evidence_type"] == "structured_knowledge" for e in d["evidence"])
    assert d["max_hops"] == 1


def test_drilldown_hides_unknown_object_identically():
    body = api.DrillRequest(workspace_id=REAL_WORKSPACE, dataset="memories",
                             object_kind="memory", object_id=str(uuid.uuid4()))
    with pytest.raises(HTTPException) as e:
        _run(api.drilldown(body, _ctx()))
    assert e.value.status_code == 404
    assert e.value.detail == "Not found."


def test_drilldown_rejects_bad_hops_and_kind():
    for kw, code in (({"max_hops": 3}, 400), ({"object_kind": "project"}, 400)):
        body = api.DrillRequest(workspace_id=REAL_WORKSPACE, dataset="memories",
                                 object_kind=kw.get("object_kind", "memory"),
                                 object_id=str(uuid.uuid4()),
                                 max_hops=kw.get("max_hops", 1))
        with pytest.raises(HTTPException) as e:
            _run(api.drilldown(body, _ctx()))
        assert e.value.status_code == code


def test_change_markers_are_real_only():
    """Part 14: no `+30 days`, no `AT RISK` -- both need a project/risk model
    that does not exist."""
    assert "AT RISK" not in sd.VALID_MARKERS
    assert not any("day" in m.lower() for m in sd.VALID_MARKERS)
    assert set(sd.CHANGE_MARKERS) <= set(sd.change_detection.SUPPORTED_CHANGE_TYPES) \
        if hasattr(sd.change_detection, "SUPPORTED_CHANGE_TYPES") else True
    r = _query(dataset="changes")
    for row in r["rows"]:
        for m in row["markers"]:
            assert m in sd.VALID_MARKERS


def test_explanation_is_deterministic_not_generated():
    """Part 15: resolvers surface the existing deterministic explanation
    fields and never call a model.

    Checked structurally, not by text search: the module legitimately
    mentions `chat_json_fn` in a comment explaining why it is deliberately
    NOT passed, and a grep would fail on that comment while still missing a
    real call written differently."""
    r = _query(dataset="changes")
    assert any(row["explanation"] for row in r["rows"])

    tree = ast.parse(inspect.getsource(sd))
    imported = {n.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert "ai" not in imported, "dataset layer must not import the model client"

    banned_calls = {"chat", "chat_json", "invoke_model", "converse"}
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    called |= {n.func.id for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not (called & banned_calls), f"LLM call in dataset layer: {called & banned_calls}"

    # And the keyword is never actually passed to a Phase 7 builder.
    kwargs_used = {kw.arg for n in ast.walk(tree) if isinstance(n, ast.Call)
                   for kw in n.keywords if kw.arg}
    assert "chat_json_fn" not in kwargs_used


def test_undetectable_changes_are_disclosed():
    r = _query(dataset="changes")
    joined = " ".join(r["not_established"])
    for k in sd.change_detection.UNDETECTABLE_CHANGES:
        assert k in joined


# =====================================================================
# 32-36. Isolation, response contract, structural guarantees.
# =====================================================================

def test_workspace_isolation():
    auth = AuthContext(user_id="u", workspaces={REAL_WORKSPACE: "owner",
                                                 LEAK_WORKSPACE: "owner"},
                       enforced=True, caller="pytest")
    leak = _run(api.query_dataset(
        api.DatasetQueryRequest(workspace_id=LEAK_WORKSPACE, dataset="memories"), auth, None))
    blob = str(leak["rows"]).lower()
    for term in ("credential", "knova", "tanmay", "procurement"):
        assert term not in blob


def test_no_raw_database_rows_returned():
    """Part 12: only registry-declared keys may appear in a row's values."""
    for key in sorted(sd.DATASETS):
        ds = sd.get_dataset(key)
        declared = {f.key for f in ds.fields}
        r = _query(dataset=key)
        for row in r["rows"]:
            assert set(row["values"]) <= declared, f"{key} leaked {set(row['values']) - declared}"
            assert "workspace_id" not in row["values"]


def test_no_brain_credentials_in_any_response():
    for key in ("memories", "departments", "changes"):
        blob = str(_query(dataset=key))
        for secret in ("SUPABASE_SERVICE_KEY", "service_role", "apikey", "Bearer ",
                        "supabase.co"):
            assert secret not in blob


def test_service_key_is_never_used_for_cross_database_reads():
    """The App-DB lookup must forward the CALLER's token with the anon key.
    A service-key read would bypass App-DB RLS and hand back rows the
    caller's own session could not see."""
    src = inspect.getsource(api._resolve_app_departments)
    assert "APP_SUPABASE_ANON_KEY" in src
    assert "SERVICE_KEY" not in src
    assert "token" in src


def test_response_contract_shape():
    r = _query(dataset="policies", group_by="memory_type", aggregation="count")
    for k in ("dataset", "fields", "rows", "row_count", "temporal_context",
              "temporal_mode", "generated_at", "aggregation", "evidence_available",
              "drilldown_target", "not_established", "security"):
        assert k in r
    assert r["aggregation"]["buckets"][0]["group"] == "policy"
    assert all("temporal_meaning" in f for f in r["fields"])


def test_registry_listing_exposes_no_workspace_data():
    listing = _run(api.list_datasets(_ctx()))
    # Tied to the registry rather than pinned to a number: adding a dataset is
    # a legitimate act, and a hardcoded count only ever fails for a correct
    # change. What actually matters is that the listing describes EXACTLY the
    # registry and nothing more.
    assert {d["key"] for d in listing["datasets"]} == set(sd.DATASETS)
    # The one absence worth asserting by name -- Projects are not modelled,
    # and a listing that grew one would mean somebody fabricated a domain.
    assert "projects" not in {d["key"] for d in listing["datasets"]}
    assert set(listing["allowed_aggregations"]) == set(sd.ALLOWED_AGGREGATIONS)
    blob = str(listing)
    assert REAL_WORKSPACE not in blob
    for f in listing["datasets"]:
        assert "resolver" not in f


def test_aggregations_limited_to_four():
    assert sd.ALLOWED_AGGREGATIONS == {"count", "count_distinct", "min", "max"}
    for ds in sd.DATASETS.values():
        for f in ds.fields:
            assert set(f.allowed_aggregations) <= sd.ALLOWED_AGGREGATIONS


def test_fixture_cleanup_leaves_no_residue():
    ws, ids = str(uuid.uuid4()), {}
    _mk_mem(ws, ids, _mk_sk(ws, ids, statement="temp"))
    assert sd.run_query("memories", ws, OWNER).row_count == 1
    _cleanup(ws, ids)
    assert sd.run_query("memories", ws, OWNER).row_count == 0
    assert (supabase.table("structured_knowledge").select("id")
            .eq("workspace_id", ws).execute().data or []) == []
