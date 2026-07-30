"""
Grounded self-critique (R-C) — is this answer actually tied to what was
retrieved, and does it draw from more than one independent source?

WHY THIS EXISTS. A model asked to "try harder" on a weak retrieval will invent
a plausible-sounding answer. That is strictly worse than "I don't know" for a
company knowledge base — a wrong answer stated confidently costs more than an
honest gap, because nobody double-checks a confident answer. This module's job
is to REJECT overconfident claims, never to embellish a thin one.

ZERO ADDED AI COST, DELIBERATELY. Both functions here are pure and
programmatic — no LLM call, no extra retrieval round. They work off data the
pipeline already produced (the generated answer's own citation markers, and
which documents the cited chunks came from), so grounding runs on every single
answer with no token cost and no added latency.

TWO NAMED LIMITATIONS, not oversights:
  * citation_coverage() checks whether a claim has ANY citation attached — it
    is a coarse proxy for "did the model bother to cite this", not a semantic
    check that chunk [2] actually SUPPORTS that specific sentence. Verifying
    that would need an LLM comparison call.
  * corroboration_level() checks whether an answer's claims are cited to more
    than one DOCUMENT — it does not verify those documents actually AGREE with
    each other, only that they were cited together. Verifying agreement is
    the same class of problem R-A's own docstring named and deferred.
  Both are the same trade R-A made for corroboration: a fast, honest, coarse
  signal now, with the real semantic version deferred to an explicit LLM step
  if the coarse one proves insufficient in practice.
"""
import re

_CITATION_RE = re.compile(r"\[\d+\](?:\[\d+\])*")
_HEADER_RE = re.compile(r"^\s*#{1,6}\s")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_LIST_MARKER_RE = re.compile(r"^\s*([-*]|\d+\.)\s+")

# Below this length a "claim" reads as a label ("Summary:", "Note:") rather
# than an assertion worth citing — excluding it avoids flagging harmless
# structural text as an uncited claim.
_MIN_CLAIM_LEN = 15


def _split_claims(text: str) -> list[str]:
    """
    Splits answer text into individually-auditable units: one per sentence for
    ordinary prose, but a markdown list item or table row is kept whole on its
    own line rather than sentence-split, since a bullet rarely ends in
    "word. word." punctuation the way a sentence does.

    A table's HEADER row (e.g. "| Metric | Value |") is excluded the same as
    its separator row ("|---|---|") — detected by lookahead, since a header
    row is always immediately followed by a separator row. Column labels are
    not a claim any more than a markdown heading is.
    """
    raw_lines = [ln.strip() for ln in (text or "").split("\n")]
    n = len(raw_lines)
    claims: list[str] = []
    for i, line in enumerate(raw_lines):
        if not line or _HEADER_RE.match(line) or _TABLE_SEP_RE.match(line):
            continue  # blank / heading / table divider — never a claim
        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(raw_lines[i + 1]):
            continue  # this IS the header row a separator row follows
        if _LIST_MARKER_RE.match(line) or "|" in line:
            claims.append(line)
            continue
        for piece in re.split(r"(?<=[.!?])\s+", line):
            piece = piece.strip()
            if piece:
                claims.append(piece)
    return claims


def citation_coverage(answer_text: str) -> dict:
    """
    Returns {total_claims, cited_claims, uncited, coverage_ratio}.

    An answer with NO substantive claims at all (a greeting, a one-line "I
    don't know") gets coverage_ratio 1.0 — there is nothing uncited, which is
    the correct reading, not a 0/0 red flag.

    A claim ending in "?" is treated as a question, not an assertion, and
    excluded — a closing rhetorical line like "Would you like more detail?"
    is not something that needs a citation. NAMED LIMITATION: this means a
    genuinely informative sentence phrased as a question ("Did you know
    revenue grew 20% [1]?") is excluded too and never checked for citation —
    an accepted, rare trade for not flagging harmless rhetorical closers as
    fabricated claims.
    """
    claims = _split_claims(answer_text)
    substantive = [
        c for c in claims
        if len(c) >= _MIN_CLAIM_LEN and not c.rstrip().endswith("?")
    ]
    uncited = [c for c in substantive if not _CITATION_RE.search(c)]
    cited_count = len(substantive) - len(uncited)
    coverage_ratio = 1.0 if not substantive else cited_count / len(substantive)
    return {
        "total_claims":   len(substantive),
        "cited_claims":   cited_count,
        "uncited":        uncited,
        "coverage_ratio": round(coverage_ratio, 3),
    }


def corroboration_level(chunks: list[dict]) -> str:
    """
    'none' | 'single_source' | 'multi_source', from how many DISTINCT
    document_ids appear among the given chunks. Callers should pass only the
    chunks that actually fed the answer (the cited ones, where citations
    exist — the retrieved-and-used ones otherwise), not the full unfiltered
    candidate pool, so this reflects what the answer actually draws from.
    """
    doc_ids = {c.get("document_id") for c in (chunks or []) if c.get("document_id")}
    if not doc_ids:
        return "none"
    return "multi_source" if len(doc_ids) >= 2 else "single_source"


def downgrade_for_weak_grounding(confidence: str, coverage_ratio: float) -> str:
    """
    A single-step, MECHANICAL downgrade (high->medium, medium->low) when less
    than half of an answer's claims carry a citation. Never touches 'low' or
    'none' — 'none' specifically means zero chunks retrieved elsewhere in this
    codebase, and overloading it here for a citation-coverage failure would be
    a different kind of "none" wearing the same word. Never a multi-step jump,
    and never invents a HIGHER confidence than retrieval itself reported —
    this function only ever moves in the direction of MORE caution.
    """
    if coverage_ratio >= 0.5:
        return confidence
    if confidence == "high":
        return "medium"
    if confidence == "medium":
        return "low"
    return confidence
