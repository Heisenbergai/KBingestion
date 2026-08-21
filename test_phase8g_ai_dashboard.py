"""
Phase 8G -- AI dashboard builder and grounded Explain.

TESTING STRATEGY. The part that protects the product is `validate_intent`,
which is a pure function: no model, no network, no database. So the great
majority of these tests feed it hostile or malformed proposals directly and
assert the verdict -- deterministic, fast, and immune to model variance.

Model behaviour is exercised with injected `chat_json_fn` mocks. Real Bedrock
calls are NOT made here: no AWS credentials exist in this environment, so a
real call raises NoCredentialsError. That is used as a genuine failure path
(test: model unavailable -> deterministic fallback), never simulated as
success.

Run with: python -m pytest test_phase8g_ai_dashboard.py -v
"""
import uuid

import pytest

from query import supabase
import graph_query as gq
import semantic_datasets as sd
import dashboard_ai as da

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)


def mock_model(payload):
    """A model that returns exactly `payload`."""
    def _fn(**kwargs):
        return payload
    return _fn


def failing_model(exc=RuntimeError("model unavailable")):
    def _fn(**kwargs):
        raise exc
    return _fn


# =====================================================================
# 1-7. Natural language -> valid configuration.
# =====================================================================

def test_simple_request_produces_a_valid_widget():
    out = da.generate_dashboard("show active policies", chat_json_fn=mock_model({
        "dashboard_name": "Policies",
        "widgets": [{"dataset": "policies", "visualization": "table",
                      "title": "Active policies"}],
    }))
    assert out["ok"] is True
    assert out["draft"]["widgets"][0]["config"]["dataset"] == "policies"
    assert out["rejected"] == []


def test_multi_widget_request():
    out = da.generate_dashboard("executive dashboard", chat_json_fn=mock_model({
        "dashboard_name": "Executive",
        "widgets": [
            {"dataset": "changes", "aggregation": "count", "visualization": "kpi",
             "temporal_mode": "window", "window_days": 90, "title": "Changes"},
            {"dataset": "policies", "visualization": "table", "title": "Policies"},
            {"dataset": "attention", "visualization": "summary", "title": "Attention"},
        ],
    }))
    assert out["ok"] is True
    assert len(out["draft"]["widgets"]) == 3
    assert out["draft"]["dashboard_name"] == "Executive"


def test_temporal_and_series_and_bucket_request():
    out = da.generate_dashboard("changes by month and type", chat_json_fn=mock_model({
        "dashboard_name": "Change trend",
        "widgets": [{"dataset": "changes", "group_by": "occurred_at",
                      "group_bucket": "month", "series_by": "change_type",
                      "aggregation": "count", "visualization": "line",
                      "temporal_mode": "window", "window_days": 90,
                      "title": "Changes by month and type"}],
    }))
    assert out["ok"] is True
    cfg = out["draft"]["widgets"][0]["config"]
    assert cfg["group_bucket"] == "month" and cfg["series_by"] == "change_type"


def test_ranking_and_percentage_requests():
    out = da.generate_dashboard("top 3 change types as percentages",
                                chat_json_fn=mock_model({
        "dashboard_name": "Top changes",
        "widgets": [{"dataset": "changes", "group_by": "change_type",
                      "aggregation": "count", "top_n": 3, "percent": True,
                      "visualization": "bar", "temporal_mode": "window",
                      "window_days": 90, "title": "Top change types"}],
    }))
    assert out["ok"] is True
    cfg = out["draft"]["widgets"][0]["config"]
    assert cfg["top_n"] == 3 and cfg["percent"] is True


def test_comparison_request_requires_a_window():
    ok = da.generate_dashboard("compare with previous period", chat_json_fn=mock_model({
        "dashboard_name": "Comparison",
        "widgets": [{"dataset": "changes", "aggregation": "count", "compare": True,
                      "temporal_mode": "window", "window_days": 30,
                      "visualization": "kpi", "title": "Changes"}],
    }))
    assert ok["ok"] is True and ok["draft"]["widgets"][0]["config"]["compare"] is True

    bad = da.generate_dashboard("compare", chat_json_fn=mock_model({
        "dashboard_name": "Comparison",
        "widgets": [{"dataset": "policies", "aggregation": "count", "compare": True,
                      "visualization": "kpi", "title": "Policies"}],
    }))
    assert bad["ok"] is False
    assert "comparison requires a time window" in bad["rejected"][0]["reason"].lower()


def test_the_interpretation_is_stated_in_plain_language():
    """Part 9: the user must be able to see what KNOVA understood before
    applying it."""
    out = da.generate_dashboard("changes by month", chat_json_fn=mock_model({
        "dashboard_name": "Trend",
        "widgets": [{"dataset": "changes", "group_by": "occurred_at",
                      "group_bucket": "month", "aggregation": "count",
                      "visualization": "bar", "temporal_mode": "window",
                      "window_days": 90, "title": "Changes by month"}],
    }))
    text = " ".join(out["interpretation"]).lower()
    assert "changes by month" in text
    assert "month" in text and "bar" in text


# =====================================================================
# 8-16. Hostile and malformed model output -- the pure validator.
# =====================================================================

def test_invalid_dataset_is_rejected():
    v = da.validate_intent({"widgets": [{"dataset": "revenue"}]})
    assert not v.ok and "Unknown dataset" in v.rejected[0]["reason"]


def test_project_request_is_refused_honestly():
    """Part 5/26: no fabricated dataset, and a reason a human can act on."""
    v = da.validate_intent({"widgets": [{"dataset": "projects", "visualization": "bar"}]})
    assert not v.ok
    assert "does not currently have a verified Project dataset" in v.rejected[0]["reason"]


def test_invalid_field_is_rejected():
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "group_by": "salary", "aggregation": "count"}]})
    assert not v.ok and "unknown field" in v.rejected[0]["reason"].lower()


def test_invalid_aggregation_is_rejected():
    for bad in ("sum", "avg", "median", "count(*)"):
        v = da.validate_intent({"widgets": [
            {"dataset": "memories", "group_by": "memory_type", "aggregation": bad}]})
        assert not v.ok, bad
        assert "Unsupported aggregation" in v.rejected[0]["reason"]


def test_invalid_visualization_is_rejected():
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "visualization": "sankey"}]})
    assert not v.ok and "Unknown visualization" in v.rejected[0]["reason"]


def test_invalid_combinations_are_caught_by_the_registry_itself():
    """The decisive check: a widget the real API would reject must be rejected
    HERE, so it can never reach a dashboard."""
    # Non-groupable field as a group.
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "group_by": "statement", "aggregation": "count"}]})
    assert not v.ok and "not groupable" in v.rejected[0]["reason"]

    # Date group with no bucket.
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "group_by": "created_at", "aggregation": "count"}]})
    assert not v.ok and "requires group_bucket" in v.rejected[0]["reason"]

    # Percentage over a min/max.
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "group_by": "memory_type", "aggregation": "max",
         "value_field": "created_at", "percent": True}]})
    assert not v.ok and "not a quantity" in v.rejected[0]["reason"]


def test_sql_and_authorization_keys_are_refused_outright():
    """Not merely unknown -- each is an attempt to reach past the semantic
    layer into SQL, storage, or authorization."""
    for key, value in (
        ("sql", "SELECT * FROM org_memory"),
        ("table", "structured_knowledge"),
        ("role", "owner"),
        ("allowed_sensitivities", ["restricted"]),
        ("workspace_id", "someone-elses-workspace"),
        ("eval", "__import__('os')"),
    ):
        v = da.validate_intent({"widgets": [{"dataset": "memories", key: value}]})
        assert not v.ok, key
        assert "forbidden keys" in v.rejected[0]["reason"], key


def test_unknown_keys_are_reported_not_silently_ignored():
    v = da.validate_intent({"widgets": [
        {"dataset": "memories", "sort_direction": "sideways"}]})
    assert not v.ok and "unknown keys" in v.rejected[0]["reason"]


def test_malformed_model_output_is_handled():
    for bad in (None, [], "a string", 42, {"widgets": "not a list"}, {}):
        v = da.validate_intent(bad)
        assert not v.ok
        assert v.rejected or v.clarification_needed


def test_partial_success_keeps_the_valid_widgets():
    """One bad widget must not destroy a good dashboard."""
    v = da.validate_intent({"dashboard_name": "Mixed", "widgets": [
        {"dataset": "policies", "visualization": "table"},
        {"dataset": "projects"},
        {"dataset": "memories", "group_by": "memory_type", "aggregation": "count",
         "visualization": "bar"},
    ]})
    assert len(v.widgets) == 2
    assert len(v.rejected) == 1
    assert v.rejected[0]["index"] == 1


# =====================================================================
# 17-20. Layout, ambiguity, drafts.
# =====================================================================

def test_layout_dimensions_are_clamped_to_the_real_grid():
    v = da.validate_intent({"widgets": [
        {"dataset": "policies", "visualization": "table", "span": 99, "height": "gigantic"},
        {"dataset": "memories", "visualization": "table", "span": 5, "height": "short"},
        {"dataset": "evidence", "visualization": "table", "span": "nonsense"},
    ]})
    assert all(w.span in da.VALID_SPANS for w in v.widgets)
    assert all(w.height in da.VALID_HEIGHTS for w in v.widgets)
    assert v.widgets[1].span in (4, 6)      # 5 snaps to a legal neighbour
    assert v.widgets[2].span == da.DEFAULT_SPAN


def test_ambiguous_request_asks_rather_than_guesses():
    """Part 6: 'show me activity' could mean changes, meetings, evidence or
    attention. Inventing one would be worse than asking."""
    out = da.generate_dashboard("show me activity", chat_json_fn=mock_model({
        "dashboard_name": "Activity",
        "widgets": [],
        "clarification_needed": "Do you mean changes, meetings, or attention items?",
    }))
    assert out["ok"] is False
    assert "changes, meetings, or attention" in out["clarification_needed"]
    assert out["draft"] is None


def test_result_is_a_draft_that_creates_nothing():
    """Part 7/20: the builder proposes. It writes no dashboard, touches no
    sharing, and changes no permission."""
    out = da.generate_dashboard("show policies", chat_json_fn=mock_model({
        "dashboard_name": "P", "widgets": [{"dataset": "policies", "visualization": "table"}],
    }))
    assert set(out) == {"ok", "draft", "interpretation", "rejected", "unavailable",
                         "notes", "clarification_needed", "fallback"}
    # A draft carries configuration only -- no ids, no rows, no sharing.
    for w in out["draft"]["widgets"]:
        assert set(w) == {"config", "title", "span", "height"}
        assert "dashboard_id" not in w["config"]
        assert "shared_with_user_id" not in w["config"]


def test_model_failure_falls_back_to_the_manual_builder():
    """Part 17: the Studio must never depend on the model."""
    out = da.generate_dashboard("anything", chat_json_fn=failing_model())
    assert out["ok"] is False
    assert out["draft"] is None
    assert out["fallback"] == "manual_builder"
    assert "couldn't generate" in out["error"].lower()


def test_real_model_path_is_the_approved_provider_only():
    """Part 2/25: Bedrock/Nova only. No Anthropic client, and no credential
    read of any kind -- the exposed ANTHROPIC_API_KEY must be rotated before
    Anthropic is introduced, and this phase must not quietly pre-empt that.

    Inspected via AST, not text: this module's own docstring EXPLAINS why
    Anthropic is absent, so a text grep would match the explanation itself."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(da))

    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert "anthropic" not in imported
    assert imported <= {"semantic_datasets", "dataclasses", "typing", "ai", "re"}

    # No environment read at all -- so no key, Anthropic's or otherwise.
    env_reads = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr in {"environ", "getenv"}]
    assert env_reads == []

    # The one model entry point is the approved shared client.
    import ai
    assert "nova" in ai.CHAT_MODEL.lower()


# =====================================================================
# 21-30. Explain -- grounding, claim boundary, no invented numbers.
# =====================================================================

def _response(**over):
    base = {
        "dataset": "changes", "row_count": 22, "temporal_context": "current",
        "not_established": ["RELATIONSHIP_REMOVED: no deletion audit exists."],
        "aggregation": {
            "aggregation": "count", "group_by": "change_type", "group_bucket": None,
            "series_by": None, "series_bucket": None, "value_field": None,
            "top_n": None, "top_direction": None, "percent": False,
            "percent_basis": None, "series_values": [],
            "buckets": [
                {"group": "NEW_KNOWLEDGE", "series": None, "value": 14, "row_count": 14},
                {"group": "MEMORY_PROMOTED", "series": None, "value": 4, "row_count": 4},
            ],
        },
    }
    base.update(over)
    return base


def test_explanation_uses_only_real_numbers():
    exp = da.explain_widget(_response(), {"title": "Changes"}, chat_json_fn=mock_model({
        "observed": "The widget shows 22 changes, 14 of them NEW_KNOWLEDGE.",
        "derived": None, "connections": None,
        "unknown": "A cause is not established.",
    }))
    assert exp.source == "model"
    assert "22" in exp.observed and exp.rejected_reasons == []


def test_an_invented_number_is_rejected_entirely():
    """Part 13: the model may repeat a real value; it may never compute one."""
    exp = da.explain_widget(_response(), {"title": "Changes"}, chat_json_fn=mock_model({
        "observed": "The widget shows 22 changes, a 37% increase over last month.",
        "derived": None, "connections": None, "unknown": "x",
    }))
    assert exp.source == "deterministic"
    assert any("invented number" in r for r in exp.rejected_reasons)
    assert "37" not in exp.observed


def test_asserted_causation_is_rejected():
    """Part 12: the Brain can show connection and change. Not cause."""
    for causal in (
        "The spike was caused by the Product team.",
        "Changes rose due to the new policy.",
        "The launch delay resulted in more reviews.",
    ):
        exp = da.explain_widget(_response(), {"title": "Changes"}, chat_json_fn=mock_model({
            "observed": "The widget shows 22 changes.",
            "derived": causal, "connections": None, "unknown": "x",
        }))
        assert exp.source == "deterministic", causal
        assert any("causation" in r for r in exp.rejected_reasons)


def test_a_real_backend_comparison_may_be_quoted():
    """A delta between two REAL periods is real, so the model may state it."""
    exp = da.explain_widget(
        _response(), {"title": "Changes"}, comparison={"value": 8, "label": "previous 30 days"},
        chat_json_fn=mock_model({
            "observed": "The widget shows 22 changes.",
            "derived": "That is 14 more than the previous period's 8.",
            "connections": None, "unknown": "A cause is not established.",
        }))
    assert exp.source == "model" and exp.rejected_reasons == []


def test_historical_context_is_carried_into_the_explanation():
    """Part 14: an as-of widget must be explained as of that date."""
    exp = da.explain_widget(
        _response(temporal_context="2026-07-01T00:00:00+00:00"),
        {"title": "Changes"}, chat_json_fn=failing_model())
    assert exp.temporal_context.startswith("2026-07-01")
    assert "as of 2026-07-01" in exp.observed


def test_deterministic_fallback_is_always_grounded():
    exp = da.explain_widget(_response(), {"title": "Changes"}, chat_json_fn=failing_model())
    assert exp.source == "deterministic" and exp.grounded is True
    assert "22" in exp.observed
    assert "NEW_KNOWLEDGE" in exp.observed          # the real largest group
    assert exp.unknown


def test_malformed_explanation_output_falls_back():
    for bad in ("a string", ["list"], 42, None):
        exp = da.explain_widget(_response(), {"title": "x"}, chat_json_fn=mock_model(bad))
        assert exp.source == "deterministic"


def test_explanation_states_what_is_not_established():
    exp = da.explain_widget(_response(), {"title": "Changes"}, chat_json_fn=failing_model())
    assert "deletion audit" in exp.unknown or "not established" in exp.unknown.lower()


def test_explain_never_calls_a_model_before_authorization():
    """Part 11/15: explain_widget receives an ALREADY-RESOLVED response. It
    cannot query, so it cannot widen what the model sees."""
    import inspect
    src = inspect.getsource(da.explain_widget)
    for forbidden in ("run_query", "supabase", "build_detail", "detect_changes",
                       "allowed_sensitivities"):
        assert forbidden not in src, f"explain_widget must not {forbidden}"


def test_explain_persists_nothing():
    """Part 21: an explanation is derived at view time. It must never mutate
    memory, graph, source data, or a dashboard."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(da))
    mutations = [n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in {"insert", "update", "upsert", "delete", "rpc"}]
    assert mutations == [], f"dashboard_ai must never write: {mutations}"


# =====================================================================
# 31-34. Security and isolation of the explanation context.
# =====================================================================

def test_context_is_bounded_to_the_supplied_result():
    """Part 19: the model gets the widget's own facts, never the corpus."""
    import inspect
    src = inspect.getsource(da.explain_widget)
    assert "facts = {" in src
    assert "[:5]" in src, "evidence must be truncated, not sent wholesale"


def test_restricted_content_cannot_reach_the_model():
    """The response handed in is the caller's OWN authorized result, so a
    low-clearance viewer's explanation is built from a smaller result -- there
    is no path by which the model sees more than the viewer."""
    ws, ids = str(uuid.uuid4()), []
    try:
        for sens in ("public", "restricted"):
            ids.append(supabase.table("structured_knowledge").insert({
                "workspace_id": ws, "canonical_source_type": "knowledge_note",
                "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
                "primitive_type": "fact", "statement": f"{sens.upper()}-8G claim",
                "raw_subject_phrase": "x", "qualifier_words": [], "sensitivity": sens,
                "authority": "official", "source_tier": 2, "lifecycle_status": "active",
                "extraction_version": "v2.1", "captured_at": "2026-08-19T00:00:00Z",
                "extraction_run_id": str(uuid.uuid4()),
                "primitive_fingerprint": f"t8g-{uuid.uuid4()}"}).execute().data[0]["id"])

        seen = {}

        def capture(**kwargs):
            seen["prompt"] = str(kwargs.get("messages"))
            return {"observed": "x", "derived": None, "connections": None, "unknown": "y"}

        low = sd.run_query("evidence", ws, LOW)
        da.explain_widget({"dataset": "evidence", "row_count": low.row_count,
                            "temporal_context": "current", "aggregation": None,
                            "not_established": []},
                           {"title": "Evidence"}, chat_json_fn=capture)

        assert low.row_count == 1, "the low caller sees only the public claim"
        assert "RESTRICTED-8G" not in seen["prompt"]
    finally:
        for i in ids:
            supabase.table("structured_knowledge").delete().eq("id", i).execute()


def test_owner_and_viewer_explanations_come_from_different_results():
    """Part 16: an explanation is never reused across viewers."""
    owner_exp = da.explain_widget(_response(row_count=22), {"title": "x"},
                                  chat_json_fn=failing_model())
    viewer_exp = da.explain_widget(_response(row_count=4), {"title": "x"},
                                   chat_json_fn=failing_model())
    assert "22" in owner_exp.observed
    assert "4" in viewer_exp.observed and "22" not in viewer_exp.observed


def test_no_autonomous_actions_are_possible():
    """Part 27: the builder proposes configuration and nothing else."""
    import inspect
    src = inspect.getsource(da)
    for forbidden in ("dashboard_shares", "logAuditEvent", "create_memory",
                       "knowledge_relationships", "notifications"):
        assert forbidden not in src


def test_fixture_cleanup_leaves_no_residue():
    ws = str(uuid.uuid4())
    i = supabase.table("structured_knowledge").insert({
        "workspace_id": ws, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "temp-8g", "raw_subject_phrase": "x",
        "qualifier_words": [], "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": "2026-08-19T00:00:00Z", "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"t8g-{uuid.uuid4()}"}).execute().data[0]["id"]
    supabase.table("structured_knowledge").delete().eq("id", i).execute()
    assert (supabase.table("structured_knowledge").select("id")
            .eq("workspace_id", ws).execute().data or []) == []


# =====================================================================
# 37-41. The API layer -- where the Part 15 security order is enforced.
# =====================================================================

def test_the_two_endpoints_exist_and_are_the_only_ai_surface():
    import dashboard_brain_api as api
    paths = {r.path for r in api.router.routes}
    assert "/dashboard/ai/build" in paths
    assert "/dashboard/ai/explain" in paths
    # No second, unauthenticated or free-form AI route crept in.
    assert {p for p in paths if "/ai" in p} == {
        "/dashboard/ai/build", "/dashboard/ai/explain"}


def test_neither_request_can_carry_authorization():
    """Part 15: authorization is DERIVED from the verified token. There is no
    field to send it in -- stronger than validating it away."""
    import dashboard_brain_api as api
    for model in (api.AIGenerateRequest, api.AIExplainRequest):
        fields = set(model.model_fields)
        assert fields & {"role", "is_super_admin", "allowed_sensitivities",
                          "sensitivity_ceiling", "user_id"} == set(), model


def test_explain_cannot_be_handed_a_result_to_describe():
    """The decisive API check. If a client could post the numbers, the
    explanation would describe what the CLIENT claimed rather than what this
    caller may actually see. The endpoint takes the question, not the answer."""
    import dashboard_brain_api as api
    fields = set(api.AIExplainRequest.model_fields)
    for answer_shaped in ("result", "response", "rows", "row_count", "buckets",
                           "aggregation_result", "value", "explanation"):
        assert answer_shaped not in fields, answer_shaped
    # It takes the QUESTION -- the same vocabulary /dashboard/query takes.
    assert {"dataset", "group_by", "aggregation", "temporal_mode"} <= fields


def test_both_endpoints_authorize_the_workspace_before_anything_else():
    """authenticate -> workspace -> ceiling -> query -> model, in that order."""
    import ast, inspect
    import dashboard_brain_api as api

    for fn, needs_ceiling in ((api.ai_build, False), (api.ai_explain, True)):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        # Sorted by LINE NUMBER, not ast.walk order -- walk is breadth-first
        # and would report a meaningless sequence.
        seq = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                nm = (n.func.attr if isinstance(n.func, ast.Attribute)
                      else n.func.id if isinstance(n.func, ast.Name) else None)
                if nm:
                    seq.append((n.lineno, n.col_offset, nm))
        order = [nm for _, _, nm in sorted(seq)]

        assert "assert_workspace" in order, fn.__name__
        i_ws = order.index("assert_workspace")

        if needs_ceiling:
            # ceiling derived after membership, query after ceiling,
            # model strictly last.
            i_ceiling = order.index("_allowed_sensitivities")
            i_query = order.index("run_query")
            i_model = order.index("explain_widget")
            assert i_ws < i_ceiling < i_query < i_model, f"{fn.__name__}: {order}"

        model_calls = [c for c in order
                       if c in ("generate_dashboard", "explain_widget")]
        assert model_calls, fn.__name__
        assert i_ws < order.index(model_calls[0]), fn.__name__


def test_generation_never_reads_workspace_data():
    """Part 19: the generator is given the registry SCHEMA, not the corpus --
    so a request that never gets applied still cannot have leaked anything."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(da._registry_summary))
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "run_query" not in calls and "execute" not in calls

    summary = da._registry_summary()
    # It describes every real dataset...
    for key in sd.DATASETS:
        assert key in summary
    # ...and contains no statement, name, or count from the live corpus.
    real = sd.run_query("memories", REAL_WORKSPACE, OWNER)
    assert real.row_count > 0, "this check is vacuous without real rows"
    checked = 0
    for row in real.rows:
        stmt = (row.get("values") or {}).get("statement")
        if isinstance(stmt, str) and len(stmt) > 20:
            assert stmt not in summary
            checked += 1
    assert checked > 0, "no statement was actually compared"
