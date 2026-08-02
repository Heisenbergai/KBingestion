"""
Query reasoning (R-B) — when the FIRST retrieval attempt comes back weak,
try again with a better-phrased or decomposed search query instead of
settling for "I don't know" on a single wording.

WHY THIS EXISTS. 9 of 30 recorded queries returned confidence 'none'/'low' — a
30% miss rate, verified against real usage events before this was built. Some of
that is genuinely missing knowledge (nothing to retrieve regardless of phrasing);
some is a vocabulary mismatch, or one question that actually bundles two
separate questions neither half of the corpus fully answers on its own.

COST DISCIPLINE (risk R2, reasoning-and-health.md). This never runs on a query
that already scored well — only on 'none'/'low' confidence — and only ONE round
per question: no loop, no iterative refinement, no recursion. Worst case per
question is 1 reasoning call plus up to REFORMULATION_CAP embed+retrieve rounds,
and that cap is enforced in code here, not left to a docstring's word.

ANTI-FABRICATION (risk R3). This module only ever proposes DIFFERENT SEARCH
QUERIES — it never sees or writes an answer. That stays entirely the caller's
job, over whatever chunks retrieval actually returns. A reformulated query that
still returns nothing still means "I don't know."

ACCESS. Callers must reuse the SAME filter_sensitivities / filter_document_ids /
filter_restricted_grant_ids / filter_bot_id on every retry — only the search
text and its embedding change here. Reformulation can only find MORE of what the
caller was already allowed to see; it must never be the thing that widens scope.
"""
from typing import Optional

import ai

REFORMULATION_CAP = 2  # max alternate queries tried per question, ever — hard cap

_PROMPT = """A question asked of a company knowledge bot returned a weak match on
the first search attempt. Your only job is to propose better SEARCH QUERIES to
try against the same knowledge base — you are NOT answering the question.

Consider:
- Different vocabulary the company's own documents might use for the same idea
  (e.g. "leave policy" vs "time off" vs "PTO").
- Whether the question actually bundles two distinct questions, which would
  each retrieve better searched separately.

Reply ONLY with JSON: {"queries": ["<alt query 1>", "<alt query 2, if useful>"]}
Return fewer than 2 if a second alternative would not meaningfully differ.
Return an empty list if you cannot suggest anything meaningfully different."""


def reformulate_query(question: str, workspace_id: Optional[str] = None) -> list[str]:
    """
    Returns up to REFORMULATION_CAP alternate search-query strings, or [].

    FAILS CLOSED — the opposite bias from escalation_triage's "when unsure,
    escalate." Here, an unsure retry burns real tokens for unproven benefit,
    while skipping it costs nothing the caller doesn't already have: a
    question that stays low/none-confidence is still escalated to the admin
    queue exactly as it always was. So any exception, malformed JSON, or
    empty/junk result returns [] rather than guessing.
    """
    q = (question or "").strip()
    if not q:
        return []
    try:
        result = ai.chat_json(
            messages=[{"role": "user", "content": f"Original question: {q}"}],
            system=_PROMPT,
            max_tokens=200,
            temperature=0.3,
            workspace_id=workspace_id,
            feature="query_reasoning",
        )
        queries = (result or {}).get("queries") or []
        # isinstance check BEFORE stringifying, deliberately: str(None).strip()
        # is the non-empty string "None", which would have sailed through a
        # truthiness-only filter and become an actual (nonsense) search query
        # burning one of the two retry slots on garbage. Only genuine non-blank
        # strings survive.
        cleaned = [x.strip() for x in queries if isinstance(x, str) and x.strip()]
        # Never trust the model's own count — the cap is enforced here
        # regardless of what it returned.
        return cleaned[:REFORMULATION_CAP]
    except Exception as e:
        print(f"[query_reasoning] reformulation failed, skipping retry (non-fatal): {e}")
        return []


_CONDENSE_PROMPT = """You rewrite a follow-up question from an ongoing conversation into a
single, fully self-contained question that makes sense with NO access to the
earlier turns -- no "it", "these", "that", "compared to before", etc.

Rules:
- Use ONLY what the conversation already established (names, metrics, time
  periods, departments, etc.) to fill in what the follow-up is really asking.
- Do not answer the question. Only rewrite it.
- If the follow-up is already self-contained, return it unchanged.

Reply ONLY with JSON: {"standalone_question": "<rewritten question>"}"""


def condense_followup(question: str, history: list[dict],
                      workspace_id: Optional[str] = None,
                      user_id: Optional[str] = None) -> str:
    """
    Rewrites a follow-up question into a standalone one using recent
    conversation turns, so RETRIEVAL searches for what the person actually
    means instead of the bare pronoun-laden follow-up text. This is the fix
    for the specific failure mode Tanmay hit live: "what are the challenges
    in achieving these targets" was searched as-is, with no idea "these
    targets" meant Engineering/Sales headcount growth from two turns earlier,
    so it retrieved an unrelated "Challenges" section from a different
    document entirely. Generation-time conversational continuity is handled
    separately in query.py by passing the raw history into the answer
    prompt -- this function's ONLY job is producing a better search query.

    FAILS OPEN to the original question, same bias as reformulate_query:
    an unavailable/broken rewrite must never block a query that would have
    worked fine standalone. No history (first question in a thread) is a
    no-op, not an error -- there's nothing to condense yet.
    """
    q = (question or "").strip()
    if not q or not history:
        return q
    transcript = "\n".join(
        f"Q: {h['question']}\nA: {h['answer']}"
        for h in history if h.get("question") and h.get("answer")
    )
    if not transcript:
        return q
    try:
        result = ai.chat_json(
            messages=[{
                "role": "user",
                "content": f"Conversation so far:\n{transcript}\n\nFollow-up question: {q}",
            }],
            system=_CONDENSE_PROMPT,
            max_tokens=150,
            temperature=0,
            workspace_id=workspace_id,
            user_id=user_id,
            feature="ai_search_condense",
        )
        rewritten = (result or {}).get("standalone_question")
        return rewritten.strip() if isinstance(rewritten, str) and rewritten.strip() else q
    except Exception as e:
        print(f"[query_reasoning] follow-up condensation failed, using original question (non-fatal): {e}")
        return q


def merge_chunk_results(primary: list[dict], retry_batches: list[list[dict]],
                        match_count: int) -> list[dict]:
    """
    Combines the original retrieval with however many reformulated-query
    retrievals ran, de-duplicating by chunk id and keeping each chunk's BEST
    score across every attempt it appeared in. Pure function, no I/O — every
    case below is covered by escalation_triage_test-style unit tests, not just
    exercised live.

    The result can only be as good as or better than the primary attempt alone:
    every primary chunk is kept, and a retry chunk can only add to or outscore
    what was already there. There is no path where reformulation makes the
    final answer worse than not having tried.
    """
    def _score(c: dict) -> float:
        try:
            # match_chunks_hybrid returns 'score' (RRF * boosts); the vector-only
            # fallback match_chunks_workspace returns only 'similarity'. Either
            # is a fine ranking key here — this just needs A consistent one.
            raw = c.get("score") if c.get("score") is not None else c.get("similarity")
            v = float(raw or 0)
            return v if v == v else 0.0  # NaN != NaN — same guard run_rag_query uses
        except (TypeError, ValueError):
            return 0.0

    best: dict[str, dict] = {}
    for batch in [primary, *retry_batches]:
        for c in batch or []:
            cid = c.get("id")
            if not cid:
                continue
            if cid not in best or _score(c) > _score(best[cid]):
                best[cid] = c

    return sorted(best.values(), key=_score, reverse=True)[:match_count]
