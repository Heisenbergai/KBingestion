"""
Contract test for query.widget_suggest -> ai.chat_json (G7, 2026-08-13).

WHY THIS EXISTS. query_routing.py's own history (test_routing_contract.py)
proved that mocking ai.chat_json fully can hide a broken call SHAPE -- 18
passing tests once shipped a feature (soft query routing) that threw on
EVERY real call because a string was passed where list[dict] was expected.
This test asserts the actual shape handed to ai.chat_json, using ai.chat()'s
own normalization loop as the oracle -- plus the validation logic that must
REJECT a hallucinated table id / column rather than passing it through to
the frontend, since widget_suggest's whole safety property depends on that
validation actually running.

PLAIN-SCRIPT STYLE, not pytest. test_routing_contract.py uses pytest's
`monkeypatch` fixture, but pytest is not installed in this local
environment (checked: `python3 -m pytest` -> "No module named pytest", not
in requirements.txt) -- meaning that reference test currently cannot be RUN
here either, only read. This file follows test_xlsx_header_guard.py's
convention instead (plain functions, manual attribute swap in try/finally,
a bare asserts + summary runner) specifically so it can actually be
executed and verified in this environment, not just written.

Run: python3 test_widget_suggest_contract.py
"""
import asyncio

import ai
import query


def _chat_normalize(messages):
    """Verbatim copy of the loop at the top of ai.chat() -- the real consumer."""
    normalized = []
    for m in messages:
        if not normalized and m["role"] == "assistant":
            continue
        if normalized and normalized[-1]["role"] == m["role"]:
            normalized[-1]["content"] += "\n" + m["content"]
        else:
            normalized.append(dict(m))
    return normalized


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def in_(self, *a, **k):
        return self

    def execute(self):
        return _FakeResult(self._rows)


class _FakeSupabase:
    def __init__(self, rows):
        self._rows = rows

    def table(self, name):
        return _FakeQuery(self._rows)


class _FakeAuth:
    def __init__(self, user_id="user-1", role="owner", is_super_admin=False):
        self.user_id = user_id
        self._role = role
        self.is_super_admin = is_super_admin

    def assert_workspace(self, workspace_id):
        pass

    def role_in(self, workspace_id):
        return self._role


REAL_TABLES = [
    {"id": "t-finance", "sheet_name": "Budget Summary",
     "headers": ["Department", "Q1 Actual", "Q2 Actual", "Variance"],
     "numeric_columns": ["Q1 Actual", "Q2 Actual", "Variance"]},
    {"id": "t-sales", "sheet_name": "Monthly Sales",
     "headers": ["Month", "Revenue", "Deals Closed"],
     "numeric_columns": ["Revenue", "Deals Closed"]},
]


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _call(description, chat_json_fn, tables=REAL_TABLES, auth=None):
    """Calls the real endpoint function directly, bypassing FastAPI's
    Depends() machinery -- same convention this codebase already uses for
    testing route handlers (see 06's R-C notes)."""
    orig_chat_json, orig_supabase = ai.chat_json, query.supabase
    ai.chat_json = chat_json_fn
    query.supabase = _FakeSupabase(tables)
    try:
        return _run(query.widget_suggest(
            query.WidgetSuggestRequest(workspace_id="ws-1", description=description),
            auth=auth or _FakeAuth(),
        ))
    finally:
        ai.chat_json = orig_chat_json
        query.supabase = orig_supabase


def test_shape_and_happy_path():
    captured = {}

    def fake_chat_json(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        _chat_normalize(messages)  # the assertion that matters: must not raise
        return {"tableId": "t-finance", "valueColumn": "Q2 Actual",
                "groupColumn": "Department", "aggregation": "sum", "chartKind": "bar"}

    out = _call("Q2 budget variance by department", fake_chat_json)

    assert isinstance(captured["messages"], list), "must be list[dict], not a bare string"
    assert captured["messages"][0]["role"] == "user"
    assert "Q2 budget variance" in captured["messages"][0]["content"]
    assert captured["kwargs"].get("feature") == "widget_suggest", "token usage must be attributable"
    assert captured["kwargs"].get("workspace_id") == "ws-1"
    assert out == {"tableId": "t-finance", "valueColumn": "Q2 Actual",
                    "groupColumn": "Department", "aggregation": "sum", "chartKind": "bar"}


def test_fails_open_when_model_errors():
    def boom(*a, **k):
        raise RuntimeError("bedrock down")
    out = _call("revenue trend", boom)
    assert out == query._WIDGET_SUGGEST_EMPTY


def test_hallucinated_table_id_is_rejected():
    fake = lambda *a, **k: {"tableId": "made-up-id", "valueColumn": "Revenue",
                             "groupColumn": None, "aggregation": "sum", "chartKind": "bar"}
    out = _call("anything", fake)
    assert out == query._WIDGET_SUGGEST_EMPTY, \
        "a hallucinated table id must yield NO suggestion, not a partially-wrong one"


def test_hallucinated_value_column_is_dropped_not_passed_through():
    fake = lambda *a, **k: {"tableId": "t-sales", "valueColumn": "Made Up Column",
                             "groupColumn": "Month", "aggregation": "sum", "chartKind": "line"}
    out = _call("sales trend", fake)
    assert out["tableId"] == "t-sales"
    assert out["valueColumn"] is None, "a column outside the sheet's own numeric_columns must be dropped"
    assert out["groupColumn"] == "Month"


def test_group_column_equal_to_value_column_is_dropped():
    fake = lambda *a, **k: {"tableId": "t-sales", "valueColumn": "Revenue",
                             "groupColumn": "Revenue", "aggregation": "sum", "chartKind": "bar"}
    out = _call("revenue", fake)
    assert out["valueColumn"] == "Revenue"
    assert out["groupColumn"] is None, "grouping by the same column as the value must be dropped"


def test_empty_description_never_calls_the_model():
    calls = {"n": 0}
    def counting(*a, **k):
        calls["n"] += 1
        return {}
    out = _call("   ", counting)
    assert calls["n"] == 0, "a blank description must never spend a Bedrock call"
    assert out == query._WIDGET_SUGGEST_EMPTY


def test_empty_catalog_never_calls_the_model():
    calls = {"n": 0}
    def counting(*a, **k):
        calls["n"] += 1
        return {}
    out = _call("revenue trend", counting, tables=[])
    assert calls["n"] == 0, "an empty catalog (no spreadsheets yet) must never spend a Bedrock call"
    assert out == query._WIDGET_SUGGEST_EMPTY


def test_non_dict_model_output_is_treated_as_no_suggestion():
    out = _call("anything", lambda *a, **k: ["not", "a", "dict"])
    assert out == query._WIDGET_SUGGEST_EMPTY


TESTS = [
    test_shape_and_happy_path,
    test_fails_open_when_model_errors,
    test_hallucinated_table_id_is_rejected,
    test_hallucinated_value_column_is_dropped_not_passed_through,
    test_group_column_equal_to_value_column_is_dropped,
    test_empty_description_never_calls_the_model,
    test_empty_catalog_never_calls_the_model,
    test_non_dict_model_output_is_treated_as_no_suggestion,
]

if __name__ == "__main__":
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL: {t.__name__} -- {e}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"ERROR: {t.__name__} -- {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{len(failed)} of {len(TESTS)} FAILED: {failed}")
        raise SystemExit(1)
    print(f"ALL {len(TESTS)} CHECKS PASSED")
