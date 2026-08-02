"""
Contract test for query_routing -> ai.chat_json.

WHY THIS EXISTS. The original routing test suite mocked ai.chat_json entirely
and asserted fail-open behaviour, so it passed while the real call signature
was wrong: route_question passed a bare STRING where chat_json expects
list[dict]. ai.chat() then iterated the string character by character and
raised TypeError on m["role"] for EVERY call. The fail-open handler swallowed
it, so routing silently never boosted anything from the day it shipped, and
every "does it fail open?" test still passed.

The lesson generalises: mocking a collaborator proves your error handling, not
that you are calling it correctly. This test asserts the SHAPE actually handed
to the boundary, using ai.chat()'s own normalization loop as the oracle.
"""
import query_routing


def _chat_normalize(messages):
    """Verbatim copy of the loop at the top of ai.chat() — the real consumer."""
    normalized = []
    for m in messages:
        if not normalized and m["role"] == "assistant":
            continue
        if normalized and normalized[-1]["role"] == m["role"]:
            normalized[-1]["content"] += "\n" + m["content"]
        else:
            normalized.append(dict(m))
    return normalized


def test_route_question_passes_a_shape_ai_chat_can_consume(monkeypatch):
    captured = {}

    def fake_chat_json(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        # The assertion that matters: ai.chat() must be able to consume this.
        _chat_normalize(messages)
        return {"department": "Engineering", "doc_class": "product"}

    monkeypatch.setattr(query_routing.ai, "chat_json", fake_chat_json)

    out = query_routing.route_question(
        "Tell me the Engineering team's performance for Q4",
        ["Engineering", "Sales"],
        workspace_id="ws-1",
    )

    assert isinstance(captured["messages"], list), "must be list[dict], not str"
    assert captured["messages"][0]["role"] == "user"
    assert "Engineering" in captured["messages"][0]["content"]
    # Token usage must be attributable, not anonymous.
    assert captured["kwargs"].get("feature") == "query_routing"
    assert captured["kwargs"].get("workspace_id") == "ws-1"
    assert out == {"department": "Engineering", "doc_class": "product"}


def test_still_fails_open_when_the_model_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bedrock down")

    monkeypatch.setattr(query_routing.ai, "chat_json", boom)
    assert query_routing.route_question("q", ["Engineering"]) == {
        "department": None, "doc_class": None,
    }


def test_hallucinated_department_is_rejected(monkeypatch):
    monkeypatch.setattr(
        query_routing.ai, "chat_json",
        lambda *a, **k: {"department": "Ministry of Magic", "doc_class": "product"},
    )
    out = query_routing.route_question("q", ["Engineering", "Sales"])
    assert out["department"] is None, "unknown department must not survive validation"
    assert out["doc_class"] == "product"
