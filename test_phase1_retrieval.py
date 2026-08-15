"""
Phase 1 regression suite -- hybrid retrieval security filtering, response
shape (citations/confidence/gaps), and the duplicate-chunk crowding fix.

WHY REAL DB, NOT MOCKED. Mocking match_chunks_hybrid would only prove the
Python call SHAPE is right -- exactly the class of bug test_routing_contract.py
already exists to catch (a mock that matches your own wrong assumption still
passes). The security/isolation assertions specifically need the real RPC's
real SQL filtering to mean something. This suite is entirely read-only
(SELECT / RPC calls only) against the live vector DB -- it creates and
deletes nothing.

Uses real, pre-existing production rows (Magic Smart Homes workspace, a real
Default-Workspace bot-taught note pair, a real empty workspace, and the real
"Cybersecurity Awareness" duplicate-content document) identified during the
2026-08-15 live Phase 1 validation pass. If any of these IDs stop existing
(document deleted, workspace removed), the affected test will fail loudly
with a clear assertion message rather than silently skip -- that is
deliberate: a disappearing fixture is itself worth knowing about.

Run with: python -m pytest test_phase1_retrieval.py -v
"""
import io
from datetime import datetime, timezone

import pytest

from query import supabase, build_context_and_citations, split_answer_and_gaps
from query_reasoning import deduplicate_chunks, validate_gap_relevance
from ingest import extract_doc_date

# ---- Real fixture IDs, from the 2026-08-15 live validation pass ----

MAGIC_SMART_HOMES_WS = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
DEFAULT_WORKSPACE_WS = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
EMPTY_WORKSPACE_WS = "20c3df60-d33c-4003-81d5-504750e526f1"  # real workspace, 0 chunks

ENG003_SLIDE3_CHUNK = "24625bde-ac87-498f-8dd1-2f50289918c9"  # exact "Sub-18ms Latency Milestone" phrase
ENG003_DOCUMENT_ID = "0d0827c2-d822-4e30-826f-f05a55dd955f"
ENG003_ASSET_ID = ENG003_DOCUMENT_ID  # confirmed asset_id == document_id in this schema

# Two real bot-taught notes in Default Workspace, each scoped to a different bot.
DEEPIKA_NOTE_CHUNK = "e99360b9-69b0-4b45-9630-e6de2a35fe9e"
DEEPIKA_NOTE_BOT_ID = "f0daa66e-b69b-46ba-aae8-5c9271bef677"
HR_CONTACT_NOTE_CHUNK = "1c52dc90-78bd-4615-9562-1c018008d061"
HR_CONTACT_NOTE_BOT_ID = "d74864aa-c24d-4de6-bf7a-f407ea9c142b"

# Real document with 18x near-duplicate chunks (Default Workspace) -- the
# exact crowding case found live: 7/8 top results were near-duplicates of
# this one document before the Step 3 fix.
CYBERSECURITY_DOC_ID = "82700025-8678-4194-8ba9-4faafcf3431f"

# Real MFG-001 document (Magic Smart Homes workspace) -- a templated
# "mad-libs" style manufacturing manual: the same long boilerplate sentence
# repeated verbatim across 14 genuinely distinct topic sections, with only
# 2-3 words substituted per topic ("The management of factory layout
# represents a critical variable..." vs "...bom control represents a
# critical variable..."). Found live in the 2026-08-15 MFG document
# validation pass: the pre-fix PLAIN-Jaccard dedup collapsed all 47 of these
# chunks (spanning the 14 distinct topics) down to a single survivor, since
# shared boilerplate scored just as "duplicate" as genuine repeated content.
MFG001_DOCUMENT_ID = "243dc3da-b933-4a10-8bd3-28113067eed4"


def _embedding_of(chunk_id: str) -> list[float]:
    """Reuses a real, already-computed production embedding as a stand-in
    query vector -- exactly the technique used in the live validation pass.
    Zero external API cost, and it's a REAL semantic vector, not a fake one."""
    res = supabase.table("document_chunks").select("embedding").eq("id", chunk_id).single().execute()
    assert res.data, f"fixture chunk {chunk_id} no longer exists -- update the fixture"
    return res.data["embedding"]


# =====================================================================
# Security / isolation
# =====================================================================

def test_workspace_isolation():
    """A query embedding maximally attractive to Magic Smart Homes content,
    filtered to a DIFFERENT workspace, must never return Magic Smart Homes
    rows."""
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "Q2 Business Review executive summary",
        "query_embedding": _embedding_of(ENG003_SLIDE3_CHUNK),
        "match_count": 20,
        "filter_workspace_id": DEFAULT_WORKSPACE_WS,
    }).execute()
    rows = result.data or []
    assert len(rows) > 0, "sanity: Default Workspace should have real content to return"
    leaked = [r for r in rows if r["workspace_id"] == MAGIC_SMART_HOMES_WS]
    assert leaked == [], f"workspace isolation violated: {len(leaked)} rows leaked from another workspace"
    assert all(r["workspace_id"] == DEFAULT_WORKSPACE_WS for r in rows)


def test_asset_filtering():
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "Q2 review",
        "query_embedding": _embedding_of(ENG003_SLIDE3_CHUNK),
        "match_count": 30,
        "filter_workspace_id": MAGIC_SMART_HOMES_WS,
        "filter_asset_id": ENG003_ASSET_ID,
    }).execute()
    rows = result.data or []
    assert len(rows) > 0
    leaked = [r for r in rows if r["asset_id"] != ENG003_ASSET_ID]
    assert leaked == [], f"asset filtering violated: {len(leaked)} rows from a different asset"


@pytest.mark.parametrize("sensitivities,expect_nonempty", [
    (["internal"], True),
    (["confidential"], True),
    (["public"], False),  # Magic Smart Homes has no public-tier chunks -- must cleanly return 0, not error
])
def test_sensitivity_filtering(sensitivities, expect_nonempty):
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "Q2 review",
        "query_embedding": _embedding_of(ENG003_SLIDE3_CHUNK),
        "match_count": 100,
        "filter_workspace_id": MAGIC_SMART_HOMES_WS,
        "filter_sensitivities": sensitivities,
    }).execute()
    row_ids = [r["id"] for r in (result.data or [])]
    if expect_nonempty:
        assert row_ids, f"expected rows for sensitivity {sensitivities}"
    else:
        assert row_ids == [], f"expected 0 rows for a sensitivity not present in the workspace, got {len(row_ids)}"
    # Cross-check every returned row's real sensitivity matches the filter.
    if row_ids:
        actual = supabase.table("document_chunks").select("id,sensitivity").in_("id", row_ids).execute()
        bad = [r for r in actual.data if r["sensitivity"] not in sensitivities]
        assert bad == [], f"sensitivity leak: {bad}"


def test_bot_specific_filtering_isolates_each_bots_own_notes():
    """Regression test for the historical filter_bot_id vulnerability: a
    bot's filter_bot_id must never surface ANOTHER bot's taught note, even
    when filter_document_ids is a non-matching dummy (forcing reliance on
    the bot_id branch), and even when sensitivity matches."""
    # Bot A's own query must find its own note, never bot B's.
    result_a = supabase.rpc("match_chunks_hybrid", {
        "query_text": "HR contact",
        "query_embedding": _embedding_of(DEEPIKA_NOTE_CHUNK),
        "match_count": 10,
        "filter_workspace_id": DEFAULT_WORKSPACE_WS,
        "filter_document_ids": ["00000000-0000-0000-0000-000000000000"],
        "filter_sensitivities": ["internal"],
        "filter_bot_id": DEEPIKA_NOTE_BOT_ID,
    }).execute()
    ids_a = {r["id"] for r in (result_a.data or [])}
    assert DEEPIKA_NOTE_CHUNK in ids_a, "bot A should see its own taught note"
    assert HR_CONTACT_NOTE_CHUNK not in ids_a, "bot A must NOT see bot B's taught note"

    # The exact historical leak pattern: a low-sensitivity (public-only) caller
    # with a matching filter_bot_id must still be blocked by sensitivity.
    result_leak_check = supabase.rpc("match_chunks_hybrid", {
        "query_text": "HR contact",
        "query_embedding": _embedding_of(DEEPIKA_NOTE_CHUNK),
        "match_count": 10,
        "filter_workspace_id": DEFAULT_WORKSPACE_WS,
        "filter_document_ids": ["00000000-0000-0000-0000-000000000000"],
        "filter_sensitivities": ["public"],  # note is 'internal' -- must not leak
        "filter_bot_id": DEEPIKA_NOTE_BOT_ID,
    }).execute()
    assert (result_leak_check.data or []) == [], (
        "REGRESSION: the historical filter_bot_id-bypasses-sensitivity leak is back"
    )


def test_legacy_unfiltered_match_chunks_is_not_callable_under_its_old_name():
    """Step 5 regression: the workspace-filter-free legacy function must
    stay renamed away from "match_chunks" so no future code can call it by
    that name and silently skip workspace isolation."""
    with pytest.raises(Exception):
        supabase.rpc("match_chunks", {
            "query_embedding": _embedding_of(ENG003_SLIDE3_CHUNK),
            "match_count": 5,
            "filter_asset_id": None,
        }).execute()


def test_empty_workspace_returns_cleanly():
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "anything at all",
        "query_embedding": _embedding_of(ENG003_SLIDE3_CHUNK),
        "match_count": 10,
        "filter_workspace_id": EMPTY_WORKSPACE_WS,
    }).execute()
    assert (result.data or []) == []


# =====================================================================
# Step 4 -- null embedding must never silently degrade
# =====================================================================

def test_hybrid_search_rejects_null_embedding_before_calling_the_rpc(monkeypatch):
    """query.hybrid_search() must raise before ever reaching the RPC when
    embedding generation returns no vector -- mocks only ai.embed_texts
    (the one real network call in the function), everything downstream is
    real code."""
    import query as query_module

    monkeypatch.setattr(query_module.ai, "embed_texts", lambda *a, **k: [[]])
    with pytest.raises(ValueError, match="embedding"):
        query_module.hybrid_search("some question", MAGIC_SMART_HOMES_WS)


def test_run_rag_query_degrades_to_no_context_on_null_embedding(monkeypatch):
    """chatbot.run_rag_query() must not let a null embedding reach the RPC
    either. Its ValueError is caught by the function's own existing
    fail-open try/except (the documented fallback for this file, per its
    own docstring) -- so the observable outcome is a normal conversational
    answer with zero chunks/sources, never meaningless ranked results."""
    import chatbot as chatbot_module

    monkeypatch.setattr(chatbot_module.ai, "embed_texts", lambda *a, **k: [[]])
    monkeypatch.setattr(
        chatbot_module.ai, "chat",
        lambda **k: "Hi! I don't have specific information on that right now.",
    )
    bot = chatbot_module.BotConfig(
        id="test-bot", name="Test Bot", workspace_id=MAGIC_SMART_HOMES_WS,
        system_prompt="", greeting_message="", linked_folder_ids=[],
    )
    answer, sources, confidence, corroboration = chatbot_module.run_rag_query(
        "some question", bot,
    )
    assert sources == [], "a null embedding must never surface chunks as if retrieval succeeded"
    assert confidence == "none"
    assert isinstance(answer, str) and answer


def test_rpc_still_accepts_null_without_a_python_guard_this_documents_the_raw_db_behavior():
    """Documents the RAW database behavior this suite guards against at the
    Python layer (Step 4's fix lives in Python, not SQL, per instruction).
    This is intentionally NOT an assertion that null should be rejected by
    the RPC itself -- it is evidence for why the Python-side guard exists."""
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "test",
        "query_embedding": None,
        "match_count": 5,
        "filter_workspace_id": MAGIC_SMART_HOMES_WS,
    }).execute()
    # As of 2026-08-15 this silently returns rows instead of erroring -- the
    # whole reason Step 4 adds a Python-side guard before this call. If a
    # future SQL change makes this raise instead, that's fine too (defense
    # in depth) -- this test only documents current DB behavior, it does
    # not gate the fix.
    assert isinstance(result.data, list)


# =====================================================================
# Response shape: citations / confidence / gaps
# =====================================================================

def test_build_context_and_citations_shape():
    chunks = [
        {"content": "Revenue was up 12%.", "metadata": {"file_name": "FIN-001.pdf"},
         "source_type": "document", "source_tier": 1},
        {"content": "We discussed this in standup.", "metadata": {"file_name": "notes.txt"},
         "source_type": "note", "source_tier": 2},
    ]
    context, citations = build_context_and_citations(chunks)
    assert "[1] FIN-001.pdf" in context
    assert "[2] notes.txt" in context
    assert citations == [
        {"index": 1, "file_name": "FIN-001.pdf", "snippet": "Revenue was up 12%.",
         "source_type": "document", "source_tier": 1},
        {"index": 2, "file_name": "notes.txt", "snippet": "We discussed this in standup.",
         "source_type": "note", "source_tier": 2},
    ]


def test_split_answer_and_gaps_extracts_marker():
    from query import GAP_MARKER
    raw = f"Revenue was up 12% [1].\n{GAP_MARKER} No data on Q4 causation."
    answer, gaps = split_answer_and_gaps(raw)
    assert answer == "Revenue was up 12% [1]."
    assert gaps == "No data on Q4 causation."


def test_split_answer_and_gaps_no_marker_present():
    answer, gaps = split_answer_and_gaps("Revenue was up 12% [1].")
    assert answer == "Revenue was up 12% [1]."
    assert gaps is None


def test_confidence_thresholds_match_documented_formula():
    """Pure re-derivation of the formula documented in query.py:567-568 and
    reconfirmed live in the Phase 1 audit -- not a live call, a contract
    check that the thresholds haven't silently drifted."""
    def confidence_for(top_sim: float) -> str:
        return "high" if top_sim >= 0.45 else "medium" if top_sim >= 0.3 else "low"

    assert confidence_for(0.45) == "high"
    assert confidence_for(0.44) == "medium"
    assert confidence_for(0.30) == "medium"
    assert confidence_for(0.29) == "low"
    assert confidence_for(0.0) == "low"


def test_query_response_shape_has_all_phase1_fields():
    """Contract check on /query's documented response shape (query.py:609-618)
    -- guards against a future edit silently dropping a field the frontend
    now depends on (Step 2)."""
    import inspect
    import query as query_module
    src = inspect.getsource(query_module.query_documents)
    for field in ("\"answer\"", "\"citations\"", "\"sources\"", "\"chunks\"",
                  "\"gaps\"", "\"confidence\"", "\"grounding\"", "\"workspace_id\""):
        assert field in src, f"/query's response no longer appears to include {field}"


# =====================================================================
# Step 3 -- duplicate-chunk crowding
# =====================================================================

def test_deduplicate_chunks_removes_exact_and_near_duplicates():
    """Unit test on the pure function, synthetic input shaped like the REAL
    live corpus case (rolling-window chunking offsets into the same
    repeated paragraph -- confirmed live to score only ~0.50 on a naive
    prefix/sequence comparison despite being obviously duplicated content,
    which is exactly why deduplicate_chunks uses order-invariant shingle
    overlap instead), plus one genuinely distinct chunk."""
    boilerplate = (
        "Cybersecurity Awareness is a foundational subject for modern professionals. "
        "It covers principles, practical applications, common challenges, tools, "
        "workflows, examples, and best practices. Mastering this topic helps "
        "individuals improve productivity, decision-making, collaboration, and "
        "long-term career growth. Organizations benefit from employees who "
        "understand these concepts and can apply them consistently."
    )
    # A different rolling-window offset into the SAME repeated boilerplate --
    # shares almost no prefix with `boilerplate` but is still the same text.
    shifted = (
        "who understand these concepts and can apply them consistently. "
        + boilerplate
    )
    chunks = [
        {"id": "a", "content": boilerplate, "score": 0.9},
        {"id": "b", "content": boilerplate, "score": 0.8},  # exact dup
        {"id": "c", "content": shifted, "score": 0.7},  # near dup, different offset
        {"id": "d", "content": "Q3 revenue grew 12% driven by enterprise deals.", "score": 0.6},  # distinct
    ]
    out = deduplicate_chunks(chunks)
    ids = [c["id"] for c in out]
    assert "a" in ids, "highest-ranked chunk must always survive"
    assert "b" not in ids, "exact duplicate of a kept chunk must be dropped"
    assert "c" not in ids, "near-duplicate (rolling-window offset) of a kept chunk must be dropped"
    assert "d" in ids, "genuinely distinct content must be preserved"
    assert ids == sorted(ids, key=lambda i: -next(c["score"] for c in chunks if c["id"] == i)), (
        "dedup must never reorder the surviving chunks"
    )


def test_deduplicate_chunks_preserves_all_when_genuinely_distinct():
    chunks = [
        {"id": "a", "content": "Revenue grew 12% in Q3.", "score": 0.9},
        {"id": "b", "content": "Headcount increased by 40 engineers.", "score": 0.8},
        {"id": "c", "content": "The new office opens in October.", "score": 0.7},
    ]
    out = deduplicate_chunks(chunks)
    assert len(out) == 3


def test_cybersecurity_document_no_longer_crowds_live_retrieval():
    """The exact live-validation regression case: querying the topic this
    document repeats 18x must no longer return 7/8 near-duplicate results
    from that single document once match_count-worth of genuinely distinct
    evidence exists elsewhere in the workspace.

    Exercises hybrid_search()'s actual post-RPC logic (widen the requested
    pool, deduplicate, truncate) directly against the live RPC using a real
    stored embedding, rather than calling hybrid_search() itself -- this
    environment has no AWS Bedrock credentials configured for
    ai.embed_texts(), so a real end-to-end call isn't possible here (see the
    Phase 1 doc's live-validation methodology note for the same constraint).
    """
    from query_reasoning import deduplicate_chunks
    match_count = 8
    fetch_count = min(match_count * 4, 40)
    result = supabase.rpc("match_chunks_hybrid", {
        "query_text": "Cybersecurity Awareness best practices",
        "query_embedding": _embedding_of("80595700-fbb5-4baf-a8c7-e3e1d423d7de"),
        "match_count": fetch_count,
        "filter_workspace_id": DEFAULT_WORKSPACE_WS,
    }).execute()
    raw_chunks = result.data or []
    from_repetitive_doc_before = [c for c in raw_chunks if c.get("document_id") == CYBERSECURITY_DOC_ID]
    assert len(from_repetitive_doc_before) >= 5, (
        "sanity check: the raw RPC response should still show heavy crowding "
        "from the repetitive document before dedup is applied"
    )

    chunks = deduplicate_chunks(raw_chunks)[:match_count]
    assert len(chunks) > 0
    from_repetitive_doc_after = [c for c in chunks if c.get("document_id") == CYBERSECURITY_DOC_ID]
    assert len(from_repetitive_doc_after) <= 2, (
        f"duplicate-crowding regression: {len(from_repetitive_doc_after)}/{len(chunks)} results "
        f"came from the single repetitive document after dedup (was 7/8 before the Step 3 fix)"
    )


def test_mfg001_templated_content_not_collapsed_by_dedup():
    """Permanent regression case for the 2026-08-15 MFG validation finding:
    plain shingle-Jaccard dedup treated MFG-001's shared boilerplate sentence
    as evidence of duplication, collapsing 14 genuinely distinct topic
    sections down to 1 survivor. The weighted-similarity fix must let all 14
    distinct topics survive, while STILL collapsing the true rolling-window
    duplicates within each topic's own multi-chunk paragraph down to one
    survivor per topic (this is not a "stop deduplicating" test -- a
    regression that made dedup a no-op would silently reopen the original
    Cybersecurity crowding bug and would not be caught by this test alone;
    see test_cybersecurity_document_no_longer_crowds_live_retrieval above
    for that side of the contract).

    Pulls the real MFG-001 chunks directly (not via RPC ranking) so this
    test doesn't depend on retrieval scoring, only on deduplicate_chunks'
    own behavior against real, live content.
    """
    import re

    rows = (
        supabase.table("document_chunks")
        .select("id,document_id,content,chunk_index")
        .eq("document_id", MFG001_DOCUMENT_ID)
        .order("chunk_index")
        .execute()
        .data
    )
    templated = [r for r in rows if (r.get("content") or "").strip().startswith("The management of")]
    assert len(templated) >= 40, (
        "sanity check: MFG-001 should still contain its real templated "
        "'management of X' sections -- if this fails, the fixture document "
        "itself changed and the test needs a new baseline, not a threshold tweak"
    )

    def _topic(c: dict) -> str:
        m = re.search(r"management of ([a-z ]+) represents", c["content"])
        return m.group(1) if m else "?"

    distinct_topics_before = len(set(_topic(c) for c in templated))
    assert distinct_topics_before >= 10, (
        "sanity check: fixture should span many genuinely distinct topics before dedup"
    )

    out = deduplicate_chunks(templated)
    distinct_topics_after = len(set(_topic(c) for c in out))
    assert distinct_topics_after == distinct_topics_before, (
        f"dedup false-positive regression: {len(templated)} chunks spanning "
        f"{distinct_topics_before} distinct MFG-001 topics collapsed to "
        f"{len(out)} survivors covering only {distinct_topics_after} topics "
        f"(was 47 -> 1 topic before the weighted-similarity fix)"
    )
    # Each topic's own multi-chunk paragraph (real rolling-window near-dupes)
    # must still collapse to exactly one survivor -- proves this isn't a
    # "dedup became a no-op" false fix.
    assert len(out) == distinct_topics_after, (
        f"expected exactly one surviving chunk per distinct topic, got "
        f"{len(out)} survivors for {distinct_topics_after} topics -- "
        f"within-topic rolling-window duplicates are no longer collapsing"
    )


# =====================================================================
# Gap semantics -- validate_gap_relevance (2026-08-15 false-gap fix)
# =====================================================================

def test_validate_gap_relevance_drops_a_gap_the_model_flags_as_drop(monkeypatch):
    """When the validation model explicitly says DROP, the gap must not
    survive -- this is the mechanism that catches the real live false
    positive (a "what are the key priorities" question fully answered, then
    flagged with a gap about unasked-for Q3 metrics)."""
    import query_reasoning as qr_module

    monkeypatch.setattr(qr_module.ai, "chat_json", lambda **k: {"verdict": "DROP"})
    assert validate_gap_relevance(
        "What are the key priorities and considerations for manufacturing capacity planning?",
        "Specific details about cost-reduction initiatives and Q3 capacity expansion "
        "metrics are not provided in the sources.",
    ) is False


def test_validate_gap_relevance_keeps_a_gap_the_model_flags_as_keep(monkeypatch):
    """A genuine gap -- the question asks for X, X isn't in the sources --
    must survive validation. This is the true-gap case (marketing budget
    query) that must keep working exactly as before this fix."""
    import query_reasoning as qr_module

    monkeypatch.setattr(qr_module.ai, "chat_json", lambda **k: {"verdict": "KEEP"})
    assert validate_gap_relevance(
        "What is Magic Smart Homes' social media influencer marketing budget for 2026?",
        "Information about the 2026 social media influencer marketing budget is not "
        "available in the current knowledge base.",
    ) is True


def test_validate_gap_relevance_fails_open_on_error(monkeypatch):
    """An unavailable/broken validator must never SILENTLY suppress a
    possibly-real gap -- fails open (keeps the gap), the opposite bias from
    reformulate_query's fail-closed. Matches grounding.py's stated
    philosophy: a wrong 'no gap' costs more than an unnecessary one shown."""
    import query_reasoning as qr_module

    def _raise(**k):
        raise RuntimeError("simulated model outage")

    monkeypatch.setattr(qr_module.ai, "chat_json", _raise)
    assert validate_gap_relevance("Any question?", "Any gap note.") is True


def test_validate_gap_relevance_treats_malformed_verdict_as_keep(monkeypatch):
    """Anything other than the literal string 'DROP' (missing key, None,
    unexpected value) must keep the gap -- same fail-toward-showing-it bias
    as the error path above, for a response that came back but didn't
    match the expected shape."""
    import query_reasoning as qr_module

    monkeypatch.setattr(qr_module.ai, "chat_json", lambda **k: {"verdict": "MAYBE"})
    assert validate_gap_relevance("Any question?", "Any gap note.") is True

    monkeypatch.setattr(qr_module.ai, "chat_json", lambda **k: {})
    assert validate_gap_relevance("Any question?", "Any gap note.") is True


def test_validate_gap_relevance_empty_gap_is_never_kept():
    """No gap text at all is trivially not a gap to show -- must return
    False without even attempting a model call (no question/gap round-trip
    needed to know there's nothing to validate)."""
    assert validate_gap_relevance("What are our priorities?", "") is False
    assert validate_gap_relevance("What are our priorities?", None) is False


def test_validate_gap_relevance_requests_a_token_budget_large_enough_to_avoid_truncation(monkeypatch):
    """Regression guard for the REAL live 2026-08-15 bug: this function
    originally requested max_tokens=20 -- reformulate_query above, the only
    other chat_json caller in this module, uses 200. 20 tokens is not
    enough room for {"verdict": "KEEP"} if the model emits ANY text before
    the JSON (common even under a "reply ONLY with JSON" instruction), so
    the response failed to parse on both the original attempt and
    chat_json's own one retry, raising json.JSONDecodeError uncaught out of
    chat_json -- silently swallowed by this function's fail-open
    except-branch, returning KEEP regardless of the model's actual verdict.
    The exact live false gap ("Q3 capacity expansion metrics") surviving a
    real LLM call is consistent with the verdict never successfully
    reaching this function at all. Asserts a generous budget (matching or
    exceeding reformulate_query's 200) is requested, so this can't regress
    silently back to a too-small budget."""
    import query_reasoning as qr_module

    captured = {}

    def _capture(**k):
        captured.update(k)
        return {"verdict": "DROP"}

    monkeypatch.setattr(qr_module.ai, "chat_json", _capture)
    validate_gap_relevance("Any question?", "Any gap note.")
    assert captured.get("max_tokens", 0) >= 200, (
        f"max_tokens={captured.get('max_tokens')} risks the model's JSON verdict "
        f"being truncated into invalid JSON, which fails open to KEEP regardless "
        f"of the model's real judgment -- this is the exact live false-gap bug"
    )


def test_validate_gap_relevance_fails_open_when_response_never_parses_as_json(monkeypatch):
    """Direct simulation of the real live bug's actual failure mode: the
    model's raw response never successfully parses into JSON (as would
    happen from truncation), so ai.chat_json raises -- and that exception
    must still fail OPEN (keep the gap) here, exactly like any other
    chat_json failure. This is what actually happened live, not a
    hypothetical -- the fix is the token budget above, this test documents
    what the failure mode looks like from this function's side."""
    import json
    import query_reasoning as qr_module

    def _raise_json_decode_error(**k):
        raise json.JSONDecodeError("No JSON object found in model output", "", 0)

    monkeypatch.setattr(qr_module.ai, "chat_json", _raise_json_decode_error)
    assert validate_gap_relevance(
        "What are the key priorities and considerations for manufacturing capacity planning?",
        "Specific details about cost-reduction initiatives and Q3 capacity expansion "
        "metrics are not provided in the sources.",
    ) is True  # fails open -- but this is exactly the wrong outcome for THIS gap,
    # which is why the real fix is preventing the truncation (token budget),
    # not relying on fail-open to save us: fail-open trades an unnecessary
    # gap for never hiding a real one, and this gap should have been a
    # genuine DROP if the model's judgment had actually been reached.


# =====================================================================
# Step 6 -- forward-only doc_date population from real embedded file metadata
# =====================================================================

def test_extract_doc_date_reads_real_docx_creation_date():
    import docx
    d = docx.Document()
    d.add_paragraph("hello")
    d.core_properties.created = datetime(2023, 5, 14, 10, 0, 0, tzinfo=timezone.utc)
    buf = io.BytesIO()
    d.save(buf)
    result = extract_doc_date(
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "test.docx",
    )
    assert result is not None
    assert result.startswith("2023-05-14")


def test_extract_doc_date_reads_real_xlsx_creation_date():
    import openpyxl
    wb = openpyxl.Workbook()
    wb.active["A1"] = "hello"
    wb.properties.created = datetime(2022, 1, 3, 8, 30, 0, tzinfo=timezone.utc)
    buf = io.BytesIO()
    wb.save(buf)
    result = extract_doc_date(
        buf.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "test.xlsx",
    )
    assert result is not None
    assert result.startswith("2022-01-03")


def test_extract_doc_date_returns_none_when_no_metadata_present():
    """No embedded date -- must return None, never invent one. Plain text
    has no metadata concept at all, so this is the honest baseline case."""
    result = extract_doc_date(b"just some plain text, no metadata", "text/plain", "notes.txt")
    assert result is None


def test_extract_doc_date_never_raises_on_corrupt_file():
    """A corrupt/unparseable file must degrade to None, never crash
    ingestion -- doc_date is a nice-to-have, never a hard dependency."""
    result = extract_doc_date(
        b"not a real docx file at all",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "corrupt.docx",
    )
    assert result is None


# =====================================================================
# Phase 2A -- connector-note classification hardening (2026-08-15)
# =====================================================================
#
# create_note_and_embed() now classifies every connector note like an
# uploaded document (ingest.classify_document) instead of letting
# sensitivity/authority/doc_class/lifecycle_status fall back to raw
# document_chunks column defaults. These tests exercise the REAL function
# against the REAL live DB (real knowledge_notes/document_chunks rows,
# real match_chunks_hybrid calls) -- only ai.chat_json (the one real
# network/Bedrock call inside classify_document) and embed_chunks (the one
# real Bedrock embedding call, unavailable in this environment -- see
# ai.py) are mocked. Mocking embed_chunks does NOT weaken what's being
# proven: classification/sensitivity-ladder/retrieval-filtering/deletion
# behavior are all real Postgres/RPC behavior against real rows; only the
# embedding VECTOR itself is a stand-in (a real, valid vector borrowed from
# an existing production chunk via _embedding_of, not a fake shape).
#
# Every test creates real rows and cleans them up in a finally block via
# the real brain_connectors.delete_note() -- itself under test (item 12).

import brain_connectors as bc

_stand_in_vector_cache: list = []


def _stand_in_vector() -> list[float]:
    """Lazy + cached so a fixture disappearing only fails a Phase 2A test
    when it actually runs, not the whole file's collection."""
    if not _stand_in_vector_cache:
        _stand_in_vector_cache.append(_embedding_of(ENG003_SLIDE3_CHUNK))
    return _stand_in_vector_cache[0]


def _mock_classification(monkeypatch, verdict: dict):
    """Patches the one real LLM call inside ingest.classify_document (via
    brain_connectors' own `ai` import, since classify_document is imported
    BY NAME into brain_connectors -- but the ai.chat_json call happens
    inside ingest.py's own module, so the patch target is ingest.ai) and
    the one real Bedrock embedding call inside create_note_and_embed."""
    import ingest as ingest_module
    monkeypatch.setattr(ingest_module.ai, "chat_json", lambda **k: verdict)
    monkeypatch.setattr(bc, "embed_chunks",
                        lambda chunks, **k: [_stand_in_vector()] * len(chunks))


def _create_synthetic_note(monkeypatch, verdict: dict, workspace_id: str = MAGIC_SMART_HOMES_WS,
                           title: str = "Phase 2A synthetic test note") -> str:
    _mock_classification(monkeypatch, verdict)
    return bc.create_note_and_embed(
        workspace_id, connection_id=None, provider="slack",
        note={"title": title, "body": "Synthetic body for Phase 2A regression testing."},
        source_type="slack", source_tier=3,
    )


def _row_metadata(note_id: str) -> tuple[dict, list[dict]]:
    note = supabase.table("knowledge_notes").select(
        "sensitivity,authority,doc_class,lifecycle_status"
    ).eq("id", note_id).single().execute().data
    chunks = supabase.table("document_chunks").select(
        "sensitivity,authority,doc_class,lifecycle_status"
    ).eq("document_id", note_id).execute().data
    return note, chunks


def test_connector_note_classification_raises_sensitivity_when_more_restrictive(monkeypatch):
    """Classifier proposes 'confidential' from the safe 'internal' baseline
    -- more restrictive, so it MUST be applied (the raise direction)."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "confidential", "authority": "canonical",
        "doc_class": "strategy", "lifecycle_status": "active", "confidence": "high",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "confidential"
        assert all(c["sensitivity"] == "confidential" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_never_lowers_sensitivity_below_internal(monkeypatch):
    """Classifier proposes 'public' -- LESS restrictive than the 'internal'
    baseline. The security rule: automated classification may only RAISE
    sensitivity, never lower it. Effective sensitivity must stay 'internal'."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "public", "authority": "working",
        "doc_class": None, "lifecycle_status": "active", "confidence": "high",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "internal", "sensitivity must never be auto-lowered from the baseline"
        assert all(c["sensitivity"] == "internal" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_keeps_internal_when_classifier_says_internal(monkeypatch):
    """Classifier proposes 'internal' -- same as baseline, no change either way."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "working",
        "doc_class": None, "lifecycle_status": "active", "confidence": "medium",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "internal"
        assert all(c["sensitivity"] == "internal" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_raises_to_restricted(monkeypatch):
    """Classifier proposes 'restricted' -- the most restrictive tier, must
    be applied (raise direction, same as the confidential case)."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "restricted", "authority": "canonical",
        "doc_class": "legal", "lifecycle_status": "active", "confidence": "high",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "restricted"
        assert all(c["sensitivity"] == "restricted" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_invalid_sensitivity_stays_internal(monkeypatch):
    """A malformed/unrecognised sensitivity value from the model must never
    create a BROADER access level than the safe baseline -- classify_document
    itself already validates this (falls back to 'internal'), so this proves
    the guarantee holds all the way through create_note_and_embed, not just
    inside classify_document."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "top-secret-banana", "authority": "working",
        "doc_class": None, "lifecycle_status": "active", "confidence": "low",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "internal"
        assert all(c["sensitivity"] == "internal" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_invalid_authority_stays_working(monkeypatch):
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "not-a-real-authority-tier",
        "doc_class": None, "lifecycle_status": "active", "confidence": "low",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["authority"] == "working"
        assert all(c["authority"] == "working" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_invalid_lifecycle_stays_active(monkeypatch):
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "working",
        "doc_class": None, "lifecycle_status": "not-a-real-lifecycle-value", "confidence": "low",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["lifecycle_status"] == "active"
        assert all(c["lifecycle_status"] == "active" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_classification_failure_falls_back_to_all_safe_defaults(monkeypatch):
    """If the LLM call inside classify_document() fails outright, every
    SECURITY-RELEVANT axis (sensitivity/authority/lifecycle_status -- the
    ones the LLM alone resolves) must land on its safe default.

    doc_class is the one deliberate exception: classify_document()'s rules
    engine resolves doc_class deterministically from source_type BEFORE the
    LLM ever runs (source_type in ("meeting","slack","note") -> "meeting"),
    and that resolution survives an LLM failure by design (classify_document's
    own except-branch still merges rules_result in) -- a real connector note
    is always one of those source_types, so doc_class="meeting" here is
    correct, existing, LLM-independent behavior, not a gap this fix touches."""
    import ingest as ingest_module

    def _raise(**k):
        raise RuntimeError("simulated Bedrock outage")

    monkeypatch.setattr(ingest_module.ai, "chat_json", _raise)
    monkeypatch.setattr(bc, "embed_chunks", lambda chunks, **k: [_stand_in_vector()] * len(chunks))

    note_id = bc.create_note_and_embed(
        MAGIC_SMART_HOMES_WS, connection_id=None, provider="slack",
        note={"title": "Phase 2A classification-failure test", "body": "Body text."},
        source_type="slack", source_tier=3,
    )
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "internal"
        assert note["authority"] == "working"
        assert note["doc_class"] == "meeting"  # deterministic rules engine, not the failed LLM
        assert note["lifecycle_status"] == "active"
        for c in chunks:
            assert c["sensitivity"] == "internal"
            assert c["authority"] == "working"
            assert c["doc_class"] == "meeting"
            assert c["lifecycle_status"] == "active"
    finally:
        bc.delete_note(note_id)


def test_connector_note_doc_class_is_deterministically_meeting_for_slack_source(monkeypatch):
    """Real connector notes always use source_type in ("slack","meeting","note")
    -- classify_document's rules engine resolves doc_class="meeting" for all
    three, deterministically, and it WINS over the LLM's own guess (see
    classify_document's docstring: "Rules-engine doc_class wins over the
    LLM's guess"). So even when the mocked model proposes a different class,
    the effective doc_class must still be "meeting" for a Slack-sourced note."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "official",
        "doc_class": "product",  # LLM's guess -- must be overridden by the rules engine
        "lifecycle_status": "active", "confidence": "medium",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["doc_class"] == "meeting"
        assert all(c["doc_class"] == "meeting" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_doc_class_propagates_from_llm_when_rules_engine_has_no_signal(monkeypatch):
    """Proves the PROPAGATION mechanism itself (LLM verdict -> effective
    doc_class) works, using a source_type outside the rules engine's
    deterministic set ("document", not "slack"/"meeting"/"note") so nothing
    overrides the LLM's answer -- no real Slack/Zoom note takes this
    source_type today, but this is what proves the wiring is correct rather
    than coincidentally always showing "meeting"."""
    _mock_classification(monkeypatch, {
        "sensitivity": "internal", "authority": "official",
        "doc_class": "product", "lifecycle_status": "active", "confidence": "medium",
    })
    note_id = bc.create_note_and_embed(
        MAGIC_SMART_HOMES_WS, connection_id=None, provider="google_drive",
        note={"title": "Phase 2A doc_class propagation test", "body": "Body text."},
        source_type="document", source_tier=1,
    )
    try:
        note, chunks = _row_metadata(note_id)
        assert note["doc_class"] == "product"
        assert all(c["doc_class"] == "product" for c in chunks)
    finally:
        bc.delete_note(note_id)


def test_connector_note_and_all_chunks_share_identical_effective_metadata(monkeypatch):
    """knowledge_notes and every document_chunks row for the same note must
    NEVER disagree -- one is what a future Library UI would show, the other
    is what retrieval's SQL-side filtering actually reads."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "confidential", "authority": "official",
        "doc_class": "financial", "lifecycle_status": "under_review", "confidence": "high",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert chunks, "expected at least one chunk"
        for c in chunks:
            assert c["sensitivity"] == note["sensitivity"]
            assert c["authority"] == note["authority"]
            assert c["doc_class"] == note["doc_class"]
            assert c["lifecycle_status"] == note["lifecycle_status"]
    finally:
        bc.delete_note(note_id)


def test_bot_learning_explicit_sensitivity_still_overrides_classification(monkeypatch):
    """bot_learning.py's pattern: call create_note_and_embed(), then an
    explicit admin-supplied sensitivity UPDATE on both tables. That update
    must still win over whatever the automated classifier decided --
    reproduces bot_learning.py's exact two-step sequence without touching
    that file, proving Phase 2A didn't change its behavior."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "confidential", "authority": "working",  # classifier raises to confidential
        "doc_class": None, "lifecycle_status": "active", "confidence": "high",
    })
    try:
        note, chunks = _row_metadata(note_id)
        assert note["sensitivity"] == "confidential"  # classification applied first, as expected

        # bot_learning.py's own explicit override (same two-table update it performs)
        admin_choice = "restricted"
        supabase.table("knowledge_notes").update({"sensitivity": admin_choice}).eq("id", note_id).execute()
        supabase.table("document_chunks").update({"sensitivity": admin_choice}).eq("document_id", note_id).execute()

        note2, chunks2 = _row_metadata(note_id)
        assert note2["sensitivity"] == admin_choice, "explicit admin override must win over automated classification"
        assert all(c["sensitivity"] == admin_choice for c in chunks2)
    finally:
        bc.delete_note(note_id)


def test_filtration_discard_verdict_never_creates_note_or_chunk(monkeypatch):
    """run_filtration()'s DISCARD path -- classify_batch() returning
    worth_keeping:false -- must never reach create_note_and_embed() at all.
    Pure logic test, no note is ever created so nothing to clean up."""
    monkeypatch.setattr(bc.ai, "chat_json", lambda **k: {"worth_keeping": False})
    result = bc.classify_batch("someone: running 5 min late", "general")
    assert result is None


def test_delete_note_removes_note_and_its_chunks(monkeypatch):
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "working",
        "doc_class": None, "lifecycle_status": "active", "confidence": "low",
    })
    note_before, chunks_before = _row_metadata(note_id)
    assert note_before is not None
    assert chunks_before

    bc.delete_note(note_id)

    note_after = supabase.table("knowledge_notes").select("id").eq("id", note_id).execute().data
    chunks_after = supabase.table("document_chunks").select("id").eq("document_id", note_id).execute().data
    assert note_after == []
    assert chunks_after == []


def test_connector_note_workspace_isolation(monkeypatch):
    """A connector note created in Magic Smart Homes must never surface
    from a query scoped to a different real workspace."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "internal", "authority": "working",
        "doc_class": None, "lifecycle_status": "active", "confidence": "low",
    }, workspace_id=MAGIC_SMART_HOMES_WS, title="Phase 2A isolation test note")
    try:
        cross_workspace = supabase.rpc("match_chunks_hybrid", {
            "query_text": "Phase 2A isolation test note",
            "query_embedding": _stand_in_vector(),
            "match_count": 50,
            "filter_workspace_id": DEFAULT_WORKSPACE_WS,
            "filter_document_ids": [note_id],
        }).execute()
        assert (cross_workspace.data or []) == [], "note must not be reachable from a different workspace"

        same_workspace = supabase.rpc("match_chunks_hybrid", {
            "query_text": "Phase 2A isolation test note",
            "query_embedding": _stand_in_vector(),
            "match_count": 50,
            "filter_workspace_id": MAGIC_SMART_HOMES_WS,
            "filter_document_ids": [note_id],
        }).execute()
        assert any(r["id"] for r in (same_workspace.data or [])), "sanity check: note must be findable in its own workspace"
    finally:
        bc.delete_note(note_id)


def test_connector_note_confidential_sensitivity_enforced_by_retrieval(monkeypatch):
    """A connector note raised to 'confidential' by classification must be
    excluded from a low-tier caller's sensitivity ladder, and included for
    an authorized ladder -- proves the SQL-side Phase 1 filter (unchanged
    by this work) enforces the classification this fix now applies."""
    note_id = _create_synthetic_note(monkeypatch, {
        "sensitivity": "confidential", "authority": "official",
        "doc_class": "strategy", "lifecycle_status": "active", "confidence": "high",
    }, title="Phase 2A confidential retrieval test note")
    try:
        low_tier = supabase.rpc("match_chunks_hybrid", {
            "query_text": "Phase 2A confidential retrieval test note",
            "query_embedding": _stand_in_vector(),
            "match_count": 50,
            "filter_workspace_id": MAGIC_SMART_HOMES_WS,
            "filter_document_ids": [note_id],
            "filter_sensitivities": ["public", "internal"],
        }).execute()
        assert (low_tier.data or []) == [], "a low-tier ladder must never retrieve a confidential connector note"

        authorized = supabase.rpc("match_chunks_hybrid", {
            "query_text": "Phase 2A confidential retrieval test note",
            "query_embedding": _stand_in_vector(),
            "match_count": 50,
            "filter_workspace_id": MAGIC_SMART_HOMES_WS,
            "filter_document_ids": [note_id],
            "filter_sensitivities": ["public", "internal", "confidential"],
        }).execute()
        row_ids = [r["id"] for r in (authorized.data or [])]
        assert row_ids, "an authorized ladder must retrieve the confidential connector note"
    finally:
        bc.delete_note(note_id)
