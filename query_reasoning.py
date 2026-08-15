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

# Weighted-Jaccard similarity (0-1) two chunks' word-shingle sets must reach
# to count as a near-duplicate for deduplicate_chunks() below. Order-invariant
# by design: the real case found in the 2026-08-15 live validation (a
# generated training doc with the same paragraph repeated 18x) chunks at
# ROLLING WINDOW boundaries, so two duplicate chunks often start at different
# offsets into the same repeated text -- a prefix/sequence-based comparison
# scored that real pair at only ~0.50 (looked different) while shingle
# overlap correctly scored it 0.95 (verified against the actual live rows).
# 0.6 was chosen with a wide margin below that 0.95 and well above genuinely
# different chunks that merely share a topic (~0.0-0.1 in the same test).
#
# This threshold is unchanged from the original PLAIN-Jaccard version, but
# the similarity itself is now shingle-frequency-WEIGHTED (see
# _shingle_weight below), found necessary by a second real case, MFG-001
# (2026-08-15 MFG validation): a templated "mad-libs" style document where a
# long boilerplate sentence is repeated verbatim across ~14 genuinely
# distinct topic paragraphs with only 2-3 words substituted per topic (e.g.
# "The management of factory layout represents a critical variable..." vs
# "...bom control represents a critical variable..."). Plain Jaccard can't
# tell "shared because it's boilerplate" apart from "shared because it's a
# real duplicate" -- both look like high shingle overlap -- so it collapsed
# 47 real MFG-001 chunks spanning 14 distinct topics down to 1 survivor.
_DEDUP_JACCARD_THRESHOLD = 0.6
_DEDUP_SHINGLE_SIZE = 8  # words per shingle

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


_GAP_VALIDATION_PROMPT = """You check whether a candidate "knowledge gap" note genuinely
describes information the user's question REQUIRES to be answered, or whether it
describes something more specific or tangential than what was actually asked.

Work through this in order, then answer:
1. In a few words, what is the CORE thing the question is asking for?
2. In a few words, what does the gap note claim is missing?
3. Is what's missing (2) actually part of what was asked (1), or is it a MORE
   SPECIFIC, narrower, or tangential detail that goes beyond what was asked?

Decision rule:
- Question asks an open/general question ("What are the X of Y?", e.g. priorities,
  considerations, differences) and the gap describes a specific number, metric, or
  narrower sub-detail that was NOT explicitly requested -> DROP. The general
  question was already answered; the gap is inventing a more specific question
  nobody asked.
- Question asks for a specific/exact fact ("What is the exact <fact>?") and the gap
  says that exact fact isn't available -> KEEP. This is precisely what was asked.
- Question asks for two things ("X, and what is Y?") and the sources cover X but
  not Y -> KEEP a gap about Y specifically, even though X was fully answered.
- If genuinely uncertain, KEEP -- a shown-but-unnecessary gap is a smaller problem
  than a hidden real one.

Example -- DROP:
Question: "What are the key priorities and considerations for manufacturing capacity
planning?"
Gap: "Specific details about cost-reduction initiatives and Q3 capacity expansion
metrics are not provided."
-> question_asks_for: "priorities/considerations for capacity planning (general)"
-> gap_describes: "specific Q3 cost/expansion metrics"
-> gap_is_narrower_than_asked: true -> verdict: DROP

Example -- KEEP:
Question: "What is the 2026 social media influencer marketing budget?"
Gap: "Information about the 2026 social media influencer marketing budget is not
available."
-> question_asks_for: "the 2026 influencer marketing budget"
-> gap_describes: "the 2026 influencer marketing budget"
-> gap_is_narrower_than_asked: false -> verdict: KEEP

Example -- KEEP (partial):
Question: "What are the key priorities for capacity planning, and what is the exact
defect rate percentage for the Magic Hub assembly line?"
Gap: "The exact defect rate percentage for the Magic Hub assembly line is not
provided."
-> question_asks_for: "priorities for capacity planning AND the exact Magic Hub
defect rate"
-> gap_describes: "the exact Magic Hub defect rate"
-> gap_is_narrower_than_asked: false (it's one of the two things explicitly asked)
-> verdict: KEEP

Reply ONLY with JSON, no other text:
{"question_asks_for": "<few words>", "gap_describes": "<few words>",
 "gap_is_narrower_than_asked": true or false, "verdict": "KEEP" or "DROP"}"""


def validate_gap_relevance(question: str, gap: str,
                            workspace_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> bool:
    """
    Returns True if `gap` should be shown to the user (it genuinely describes
    something the question asked for), False if it should be dropped
    (tangential/unrequested detail).

    WHY THIS EXISTS. Live 2026-08-15: even after tightening the answer
    system prompt's own gap instructions, the model still produced gaps like
    "Specific details about cost-reduction initiatives... are not provided"
    on a fully-answered "what are the priorities" question -- real,
    demonstrated LLM self-regulation failure, not a hypothetical. A second,
    narrowly-scoped, cheap validation call (same pattern as
    reformulate_query above) catches what the generation prompt alone
    didn't. ONLY called when the model already produced a non-empty gap --
    zero added cost on the (much more common) no-gap path.

    REAL BUG FOUND in the first version of this function (also live
    2026-08-15, same day): max_tokens was 20 -- reformulate_query above,
    the only other chat_json caller in this module, uses 200. 20 tokens is
    not enough room for {"verdict": "KEEP"} if the model emits ANY text
    before the JSON (common even under a "reply ONLY with JSON"
    instruction), so the response silently failed to parse on BOTH the
    original attempt and chat_json's own one retry, raising
    json.JSONDecodeError uncaught out of chat_json -- which this function's
    fail-open except-branch swallowed, returning KEEP regardless of what
    the model actually judged. The false gap "surviving a real LLM call"
    was consistent with the model's verdict never successfully reaching
    this function at all. Fixed by matching reformulate_query's budget
    (with headroom for this prompt's added reasoning fields) and by logging
    the raw model response so this is directly verifiable, not theoretical,
    on the next live run.

    FAILS OPEN (keeps the gap) on any error, exception, or malformed
    output -- the opposite bias from reformulate_query's fail-closed. This
    matches grounding.py's stated philosophy (a wrong "no gap" that hides a
    real limitation costs more than an unnecessary gap shown), so an
    unavailable validator must never silently suppress a possibly-real gap.
    """
    q = (question or "").strip()
    g = (gap or "").strip()
    if not g:
        return False
    if not q:
        return True
    try:
        result = ai.chat_json(
            messages=[{"role": "user", "content": f"Question: {q}\n\nCandidate gap note: {g}"}],
            system=_GAP_VALIDATION_PROMPT,
            max_tokens=300,
            temperature=0,
            workspace_id=workspace_id,
            user_id=user_id,
            feature="gap_validation",
        )
        verdict = (result or {}).get("verdict")
        keep = verdict != "DROP"
        # Direct evidence for the next live run, not a hypothesis -- prints
        # the full raw model response, not just the final bool, so a repeat
        # of the max_tokens bug (or any other parse issue) is visible
        # immediately instead of silently reappearing as an unexplained KEEP.
        print(f"[gap_validation] question={q!r} gap={g!r} raw_result={result!r} verdict={verdict!r} keep={keep}")
        return keep
    except Exception as e:
        print(f"[query_reasoning] gap relevance validation failed, keeping gap (fail-open, non-fatal): {e}")
        return True


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


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Drops near-duplicate chunks, keeping the highest-ranked (first)
    occurrence of each. Pure function, no I/O.

    WHY THIS EXISTS (found in the 2026-08-15 live Phase 1 validation, not
    hypothetical): a real generated training document had the same
    paragraph repeated 18 times as separate chunks. A live query against
    that exact topic returned 7 of the top 8 results from that single
    document -- only 1 slot carried genuinely different evidence. Highly
    repetitive source documents can silently crowd out real evidence,
    wasting most of the LLM's context window on redundant text and
    inflating apparent corroboration from what is actually one source
    repeated.

    Input is assumed already ranked by match_chunks_hybrid's score (highest
    first) -- this function only ever REMOVES a lower-ranked duplicate of
    something already kept. It never reorders or promotes anything, so it
    cannot change which chunk wins when two are genuinely distinct.

    Matches on exact normalized content first (free, no false positives),
    then WEIGHTED word-shingle Jaccard overlap for near-duplicates --
    deliberately order-invariant, since real near-duplicates in this corpus
    are rolling-window chunking offsets into the same repeated text, not just
    paraphrase rewording. Content-based, not ID-based -- complementary to
    merge_chunk_results()'s id-based dedup across retry batches, not a
    replacement for it.

    THE WEIGHTING (why plain Jaccard wasn't enough -- MFG-001 case): a
    shingle that recurs across MORE THAN HALF of the chunks in this batch is
    treated as shared boilerplate/template structure, not evidence of
    duplication, and its contribution to the similarity score is tapered
    down toward ~0. A shingle held by half the batch or fewer is normal,
    full-weight content -- this is what still lets two genuinely repeated
    real paragraphs (e.g. the same document uploaded 3x) collapse together,
    since that kind of duplication typically doesn't touch the MAJORITY of
    an unrelated, larger candidate pool the way a template skeleton does.
    Computed per dedup call, from this batch alone -- no external corpus,
    no persisted state, still a pure function.
    """
    def _shingles(text: str) -> set[str]:
        words = text.split()
        if len(words) < _DEDUP_SHINGLE_SIZE:
            return {" ".join(words)} if words else set()
        return {
            " ".join(words[i:i + _DEDUP_SHINGLE_SIZE])
            for i in range(len(words) - _DEDUP_SHINGLE_SIZE + 1)
        }

    norms = [" ".join((ch.get("content") or "").split()).lower() for ch in chunks]
    shingle_sets = [_shingles(n) for n in norms]

    n = len(chunks)
    doc_freq: dict[str, int] = {}
    for s in shingle_sets:
        for sh in s:
            doc_freq[sh] = doc_freq.get(sh, 0) + 1

    _weight_cache: dict[str, float] = {}

    def _weight(sh: str) -> float:
        cached = _weight_cache.get(sh)
        if cached is not None:
            return cached
        frac = doc_freq[sh] / n
        # Full weight up to appearing in half the batch; linear taper from
        # 1.0 -> 0.05 (never exactly 0, to avoid a fully-empty union score)
        # as it approaches appearing in every chunk.
        w = 1.0 if frac <= 0.5 else max(0.05, 1.0 - 1.9 * (frac - 0.5))
        _weight_cache[sh] = w
        return w

    def _weighted_similarity(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        union_weight = sum(_weight(s) for s in union)
        if union_weight == 0:
            return 0.0
        inter_weight = sum(_weight(s) for s in (a & b))
        return inter_weight / union_weight

    kept: list[dict] = []
    kept_idx: list[int] = []
    for i, ch in enumerate(chunks):
        norm = norms[i]
        is_dup = any(
            norm == norms[j] or _weighted_similarity(shingle_sets[i], shingle_sets[j]) >= _DEDUP_JACCARD_THRESHOLD
            for j in kept_idx
        )
        if not is_dup:
            kept.append(ch)
            kept_idx.append(i)
    return kept
