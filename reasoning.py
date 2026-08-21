"""
Phase 7A -- Organizational Reasoning Foundation: an evidence-bounded
analysis layer over the ALREADY-COMPOSED, ALREADY-AUTHORIZED retrieval
context. Phase 6 answered "what does the company know?"; this layer begins
answering "what does that knowledge mean?" -- without inventing anything.

THIS IS NOT A SECOND ANSWER ENGINE. query.py's pipeline is unchanged and
remains the answer engine:

    retrieval (hybrid_search)
      -> evidence composition (graph_retrieval + memory_retrieval merges)
      -> [THIS MODULE: reasoning analysis over that same composed context]
      -> answer validation
      -> final answer

This module never retrieves, never re-queries the vector store, never calls
hybrid_search, and never produces the user-facing answer text. It consumes
the exact `chunks`/`graph_context`/`memory_context` query.py already built
and already gated, and produces a separate, structured ReasoningResult that
a caller MAY surface alongside the answer. query.py/chatbot.py are NOT
modified by this phase -- wiring the result into a user-facing response is
deliberately left to a later sub-phase, so this pass adds analysis without
changing a single production answer (Part 18: reasoning only, no action).

THE FOUR STATES ARE ASSIGNED DETERMINISTICALLY, NEVER SELF-REPORTED BY THE
MODEL. This is the single most important property in this file, and the
explicit STOP condition of Phase 7A. A model asked "is this observed or
inferred?" will happily label its own invention "OBSERVED" -- so the model
is never asked. Instead the model proposes candidate conclusions that cite
claim_ids from a deterministic inventory, and `_classify_state()` assigns
the state purely from verifiable facts about that citation:

  OBSERVED  -- the conclusion's content is grounded (majority content-word
               overlap, the same measure Phase 6H added to
               wiki_generation.py after an adversarial battery proved
               citation-existence alone was insufficient) in the text of
               EXACTLY ONE cited claim that is itself a real, visible
               evidence/graph/memory item. Nothing was combined; it restates
               what a single real source says.
  DERIVED   -- grounded in the union of TWO OR MORE cited claims AND those
               claims are genuinely connected: they share a real
               knowledge_relationships edge, a real memory grounding, or a
               real evidence id. Combination of verified facts -- Part 4's
               first allowed operation.
  INFERRED  -- the model produced content NOT grounded in its own cited
               claims (or cited claims with no real connection). This is a
               real, honestly-labeled model leap. It is never promoted, never
               written anywhere, and callers must present it as model
               reasoning, not company fact.
  UNKNOWN   -- no relevant claim exists in the authorized context at all, or
               the model declined, or validation rejected the output
               entirely.

Because the state is a pure function of (conclusion text, cited claim ids,
real claim inventory, real relationship edges), the same inputs always
produce the same state -- and a fabrication cannot be dressed up as OBSERVED
by asserting it confidently.

SECURITY (Part 14): this module performs NO access control of its own,
because it performs no fetching of its own. Every claim in the inventory
comes from a chunk/graph/memory item the caller was ALREADY authorized to
see by query.py's own `filter_sensitivities` resolution and by
graph_retrieval/memory_retrieval's own per-evidence visibility filtering.
There is no code path here that can reach an unauthorized row -- not by
discipline, but because this module never touches the database at all (it
imports no supabase client and no brain_connectors).

NO WRITES, ANYWHERE (Part 18): this module has no insert/update/delete of
any kind. It cannot promote memory, create relationships, modify the graph,
touch the Wiki, or schedule anything.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import ai

# Reasoning states -- frozen vocabulary, never extended for convenience.
OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
INFERRED = "INFERRED"
UNKNOWN = "UNKNOWN"

# Same measure Phase 6H added to wiki_generation.py after the adversarial
# battery proved "cited a real id" alone lets fabrications through. Kept as
# its own local copy rather than imported: wiki_generation's copy is tuned
# to Wiki prose (its stopword list carries Wiki-page vocabulary like
# "status"/"active"), and coupling two independently-tuned thresholds so a
# future Wiki tweak silently changes reasoning classification would be worse
# than a small, documented duplicate -- matching this codebase's own
# established "small per-file helpers over shared coupling" convention.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "of", "in", "on",
    "for", "and", "or", "to", "with", "by", "from", "at", "also", "which",
    "who", "has", "have", "had", "not", "but", "because", "there", "their",
})
_MAX_NOVEL_CONTENT_RATIO = 0.5

# Part 4/12's forbidden semantics. A conclusion using any of these words
# that are NOT already present in its own cited claims is never allowed to
# reach OBSERVED/DERIVED -- it is forced to INFERRED at best, regardless of
# how well-grounded the rest of its wording is. Word-boundary regexes.
_FORBIDDEN_INFERENCE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bowns\b", r"\bowner\b", r"\bownership\b",
    r"\bemploys\b", r"\bemployee\b", r"\bemployed\b", r"\bemployment\b", r"\bworks for\b",
    r"\bmember of\b", r"\bmembership\b", r"\breports to\b", r"\bmanages\b", r"\bsupervises\b",
    r"\bcaused\b", r"\bcauses\b", r"\bbecause of\b", r"\bled to\b", r"\bresulted in\b",
)]


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _is_grounded(text: str, source_text: str) -> bool:
    """Majority of the conclusion's real content words must appear in the
    combined text of its own cited claims."""
    text_words = _content_words(text)
    if not text_words:
        return False
    novel = text_words - _content_words(source_text)
    return (len(novel) / len(text_words)) <= _MAX_NOVEL_CONTENT_RATIO


# =====================================================================
# Part 6 -- the deterministic claim/conclusion contract. No new table
# (Part 6 explicitly: "Do NOT create a new persistence table yet").
# =====================================================================

@dataclass
class ReasoningClaim:
    """One atomic, already-authorized fact available to reason over. Text is
    real source content, never model output."""
    claim_id: str
    text: str
    source_kind: str                  # 'chunk' | 'graph_relationship' | 'memory'
    evidence_refs: list[str]          # real evidence/source identifiers
    temporal_context: str             # 'current' or the as_of ISO string
    sensitivity: Optional[str] = None
    # Real linkage keys used to decide whether two claims are genuinely
    # connected for DERIVED (never text similarity -- real ids only).
    linkage_keys: set = field(default_factory=set)


@dataclass
class ReasoningConclusion:
    text: str
    state: str                        # OBSERVED | DERIVED | INFERRED | UNKNOWN
    cited_claim_ids: list[str]
    evidence_chain: list[dict]        # conclusion -> claims -> evidence refs
    reason: str                       # why this state was assigned (deterministic explanation)


@dataclass
class ReasoningResult:
    question: str
    workspace_id: str
    temporal_context: str
    overall_state: str
    conclusions: list[ReasoningConclusion] = field(default_factory=list)
    unresolved_contradictions: list[dict] = field(default_factory=list)
    claims_considered: int = 0
    reasoned_by: str = "fallback"     # 'llm' | 'fallback'
    metadata: dict = field(default_factory=dict)


# =====================================================================
# Part 3 -- reasoning context construction. Consumes ONLY already-authorized,
# already-resolved objects. No database access anywhere in this module.
# =====================================================================

def build_claim_inventory(chunks: list[dict], graph_context=None, memory_context=None,
                           as_of: Optional[datetime] = None) -> list[ReasoningClaim]:
    """Deterministic. Every claim's text is real retrieved/graph/memory
    content -- zero LLM involvement, zero fetching. `chunks` are query.py's
    own already-filtered candidates; graph_context/memory_context are
    graph_retrieval's/memory_retrieval's own already-visibility-filtered
    contexts. Ordering is stable (chunks, then graph, then memory, each in
    the order the caller already established) so the inventory -- and
    therefore every claim_id -- is reproducible for identical inputs.

    DOUBLE-COUNT GUARD (found live during this phase's own real-corpus
    benchmark, not theorized): query.py hands this function the chunk list
    AFTER graph_retrieval.merge_graph_context_into_chunks and
    memory_retrieval.merge_memory_context_into_chunks have already injected
    graph/memory candidates INTO it. Those injected candidates carry
    synthetic document_ids ("graph_relationship:<id>", "org_memory:<id>"),
    so passing both the merged chunks AND the same graph_context/
    memory_context produced TWO claims for one underlying fact (observed
    live: the credential policy appeared as both chunk:0 and memory:0).
    That is not merely redundant -- two claims for one fact could satisfy
    _claims_are_connected() and be classified DERIVED, presenting a single
    restated source as if it were a corroborated combination. Chunks whose
    document_id/source_type identifies an already-represented graph/memory
    candidate are therefore skipped here; the richer, properly-typed
    graph/memory claim is kept instead (it carries real evidence_refs and
    linkage_keys, which the chunk-shaped copy does not)."""
    temporal = as_of.isoformat() if as_of else "current"
    claims: list[ReasoningClaim] = []

    for i, ch in enumerate(chunks or []):
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        doc_id = ch.get("document_id") or ""
        if doc_id.startswith("graph_relationship:") or doc_id.startswith("org_memory:"):
            continue
        if (ch.get("source_type") or "") in ("graph_relationship", "org_memory"):
            continue
        claims.append(ReasoningClaim(
            claim_id=f"chunk:{i}", text=content, source_kind="chunk",
            evidence_refs=[f"document:{doc_id}"] if doc_id else [],
            temporal_context=temporal,
            sensitivity=(ch.get("metadata") or {}).get("sensitivity"),
            linkage_keys={f"document:{doc_id}"} if doc_id else set(),
        ))

    if graph_context is not None:
        for j, rel in enumerate(getattr(graph_context, "relationships", []) or []):
            src = rel.source.label or rel.source.object_id
            tgt = rel.target.label or rel.target.object_id
            text = f"{src} {rel.relationship_type.replace('_', ' ')} {tgt}."
            if rel.rationale:
                text += f" {rel.rationale}"
            linkage = {f"relationship:{rel.id}",
                        f"{rel.source.object_type}:{rel.source.object_id}",
                        f"{rel.target.object_type}:{rel.target.object_id}"}
            linkage |= {f"{ev.evidence_type}:{ev.evidence_id}" for ev in rel.evidence}
            claims.append(ReasoningClaim(
                claim_id=f"graph:{j}", text=text, source_kind="graph_relationship",
                evidence_refs=[f"{ev.evidence_type}:{ev.evidence_id}" for ev in rel.evidence],
                temporal_context=temporal, sensitivity=None, linkage_keys=linkage,
            ))

    if memory_context is not None:
        for k, cand in enumerate(getattr(memory_context, "candidates", []) or []):
            refs = [f"{e['evidence_type']}:{e['evidence_id']}" for e in cand.evidence]
            body = " ".join(e.get("reference") or "" for e in cand.evidence).strip()
            text = (f"Durable {cand.memory_type} memory (promoted via {cand.promotion_basis}, "
                    f"status {cand.lifecycle_status}): {body}")
            claims.append(ReasoningClaim(
                claim_id=f"memory:{k}", text=text, source_kind="memory",
                evidence_refs=refs, temporal_context=temporal,
                sensitivity=cand.sensitivity,
                linkage_keys={f"memory:{cand.memory_id}", *refs},
            ))

    return claims


# =====================================================================
# Part 9 -- contradiction detection. Deterministic, conservative, and
# NEVER resolved by mutation (Part 9: "Do not mutate memory or graph in
# this pass"). Only a real, already-recorded supersession signal can
# resolve a conflict; anything else stays explicitly unresolved.
# =====================================================================

def detect_contradictions(claims: list[ReasoningClaim], memory_context=None) -> list[dict]:
    """Surfaces conflicts rather than silently choosing (Part 8/9).

    V1 is deliberately narrow and honest about it: the ONLY contradiction
    signal available without inventing new semantics is the one Phase 6D
    already computes and already verified -- a memory candidate's
    `possibly_superseded` flag, which is set when a real, active
    supersedes/contradicts relationship targets that memory's own grounding
    evidence. That is a REAL recorded conflict, not a textual guess.

    No text-similarity/negation-detection contradiction finder is built
    here: that would be exactly the kind of unsupported inference Part 12
    forbids, and it would produce false conflicts on the real corpus (where
    e.g. two credential policies are complementary, not contradictory).
    Documented as a real V1 limitation in the final report rather than
    papered over with a heuristic that looks smart and is wrong."""
    out: list[dict] = []
    if memory_context is None:
        return out
    for cand in getattr(memory_context, "candidates", []) or []:
        if getattr(cand, "possibly_superseded", False):
            out.append({
                "kind": "possible_supersession",
                "memory_id": cand.memory_id,
                "memory_type": cand.memory_type,
                "lifecycle_status": cand.lifecycle_status,
                "detail": ("A real, active supersedes/contradicts relationship targets this "
                            "memory's grounding evidence. Neither claim is auto-selected; the "
                            "conflict is reported for human judgement."),
                "resolution": "unresolved",
            })
    return out


# =====================================================================
# Part 1 -- DETERMINISTIC STATE ASSIGNMENT. The model never chooses this.
# =====================================================================

def _claims_are_connected(cited: list[ReasoningClaim]) -> bool:
    """Two or more claims count as genuinely connected only when they share
    a REAL identifier -- a relationship id, an endpoint id, an evidence id,
    a memory id, or a document id. Never text similarity, never topical
    closeness (Part 13: no fuzzy cross-source merging in the reasoner)."""
    if len(cited) < 2:
        return False
    for i in range(len(cited)):
        for j in range(i + 1, len(cited)):
            if cited[i].linkage_keys & cited[j].linkage_keys:
                return True
    return False


def _classify_state(text: str, cited: list[ReasoningClaim]) -> tuple[str, str]:
    """Pure function of real, checkable facts -- (state, reason). This is
    the function the STOP condition rests on, so every branch is explicit."""
    if not cited:
        return UNKNOWN, "no claims cited -- nothing in the authorized context supports this"

    combined = " ".join(c.text for c in cited)

    for pattern in _FORBIDDEN_INFERENCE_PATTERNS:
        if pattern.search(text) and not pattern.search(combined):
            return INFERRED, (f"uses forbidden-inference term {pattern.pattern!r} that appears in "
                               f"no cited claim -- ownership/employment/causality is never derived")

    if not _is_grounded(text, combined):
        return INFERRED, "content is not grounded in the cited claims' own text"

    if len(cited) == 1:
        return OBSERVED, f"directly grounded in a single real source ({cited[0].source_kind})"

    if _claims_are_connected(cited):
        kinds = sorted({c.source_kind for c in cited})
        return DERIVED, f"combines {len(cited)} connected claims sharing a real identifier ({', '.join(kinds)})"

    return INFERRED, ("combines multiple claims that share no real relationship, evidence, or "
                       "entity identifier -- connection is not supported by the graph")


# =====================================================================
# Part 4/15 -- the LLM proposes; the verifier decides. Model failure or
# malformed output falls back to deterministic evidence-grounded reasoning.
# =====================================================================

_SYSTEM_PROMPT = """You analyze a company's verified internal knowledge. You are given a question and a numbered list of CLAIMS, each already verified and already authorized for this reader.

Your ONLY job is to propose conclusions that follow from those claims, citing which claims each conclusion rests on. You do NOT decide how trustworthy a conclusion is -- a separate deterministic verifier does that.

You MAY: combine claims that are genuinely related, trace a relationship, summarize, compare current vs historical state, explain why a conclusion follows, or state that the claims are insufficient.

You MUST NOT: invent people, teams, projects, customers, or relationships; assert ownership, employment, reporting lines, or causality unless a claim says so in those words; use outside world knowledge; or state a date not present in a claim.

If the claims do not answer the question, return a single conclusion with an empty claim_ids list and text explaining what is missing. That is a correct, valuable answer -- never pad it with a guess.

Respond with ONLY a JSON object:
{
  "conclusions": [
    {"text": "<one sentence>", "claim_ids": ["<claim_id>", "..."]}
  ]
}
No markdown fences, no text outside the JSON object."""


def _build_user_prompt(question: str, claims: list[ReasoningClaim], temporal_context: str) -> str:
    lines = [f"Question: {question}", f"Temporal context: {temporal_context}", "", "Claims:"]
    for c in claims:
        lines.append(f"[{c.claim_id}] {c.text}")
    return "\n".join(lines)


def _deterministic_conclusions(question: str, claims: list[ReasoningClaim]) -> list[ReasoningConclusion]:
    """Fallback (Part 15) and the zero-LLM baseline: every relevant claim is
    restated verbatim as its own OBSERVED conclusion. Never combines
    anything (combination is precisely the judgement an unavailable model
    couldn't make), so it can never manufacture a DERIVED or INFERRED
    result. Always available, zero model dependency."""
    out = []
    for c in claims:
        out.append(ReasoningConclusion(
            text=c.text, state=OBSERVED, cited_claim_ids=[c.claim_id],
            evidence_chain=[{"claim_id": c.claim_id, "source_kind": c.source_kind,
                              "evidence_refs": c.evidence_refs}],
            reason="deterministic restatement of a single real source (no model involved)",
        ))
    return out


def _evidence_chain_for(cited: list[ReasoningClaim]) -> list[dict]:
    return [{"claim_id": c.claim_id, "source_kind": c.source_kind,
             "evidence_refs": c.evidence_refs} for c in cited]


def _overall_state(conclusions: list[ReasoningConclusion]) -> str:
    """The weakest honest summary: if anything is UNKNOWN-only, say UNKNOWN;
    otherwise report the STRONGEST state actually achieved, since a page of
    OBSERVED facts plus one INFERRED aside is not an inferred answer -- but
    never report a state no conclusion actually earned."""
    states = {c.state for c in conclusions}
    if not states or states == {UNKNOWN}:
        return UNKNOWN
    for state in (OBSERVED, DERIVED, INFERRED):
        if state in states:
            return state
    return UNKNOWN


def reason(question: str, workspace_id: str, chunks: list[dict],
           graph_context=None, memory_context=None, as_of: Optional[datetime] = None,
           user_id: Optional[str] = None, chat_json_fn=ai.chat_json) -> ReasoningResult:
    """The single public entry point. Consumes already-authorized context
    only; never fetches, never writes. `chat_json_fn` is overridable for
    tests, matching wiki_generation.generate_wiki_page's established shape."""
    import time
    t0 = time.perf_counter()
    temporal_context = as_of.isoformat() if as_of else "current"
    claims = build_claim_inventory(chunks, graph_context, memory_context, as_of)
    t_context_ms = (time.perf_counter() - t0) * 1000

    contradictions = detect_contradictions(claims, memory_context)

    if not claims:
        return ReasoningResult(
            question=question, workspace_id=workspace_id, temporal_context=temporal_context,
            overall_state=UNKNOWN, conclusions=[], unresolved_contradictions=contradictions,
            claims_considered=0, reasoned_by="fallback",
            metadata={"reason": "no authorized claims in context", "context_ms": round(t_context_ms, 2),
                       "model_ms": 0.0, "validation_ms": 0.0},
        )

    t1 = time.perf_counter()
    try:
        raw = chat_json_fn(
            messages=[{"role": "user", "content": _build_user_prompt(question, claims, temporal_context)}],
            system=_SYSTEM_PROMPT, max_tokens=900, temperature=0.1,
            workspace_id=workspace_id, user_id=user_id, feature="organizational_reasoning",
        )
    except Exception as e:
        t_model_ms = (time.perf_counter() - t1) * 1000
        conclusions = _deterministic_conclusions(question, claims)
        return ReasoningResult(
            question=question, workspace_id=workspace_id, temporal_context=temporal_context,
            overall_state=_overall_state(conclusions), conclusions=conclusions,
            unresolved_contradictions=contradictions, claims_considered=len(claims),
            reasoned_by="fallback",
            metadata={"reason": f"model_unavailable: {type(e).__name__}: {e}",
                       "context_ms": round(t_context_ms, 2), "model_ms": round(t_model_ms, 2),
                       "validation_ms": 0.0},
        )
    t_model_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    claim_by_id = {c.claim_id: c for c in claims}
    proposed = raw.get("conclusions") if isinstance(raw, dict) else None
    if not isinstance(proposed, list) or not proposed:
        conclusions = _deterministic_conclusions(question, claims)
        t_validation_ms = (time.perf_counter() - t2) * 1000
        return ReasoningResult(
            question=question, workspace_id=workspace_id, temporal_context=temporal_context,
            overall_state=_overall_state(conclusions), conclusions=conclusions,
            unresolved_contradictions=contradictions, claims_considered=len(claims),
            reasoned_by="fallback",
            metadata={"reason": "malformed model output -- no usable conclusions array",
                       "context_ms": round(t_context_ms, 2), "model_ms": round(t_model_ms, 2),
                       "validation_ms": round(t_validation_ms, 2)},
        )

    conclusions: list[ReasoningConclusion] = []
    for item in proposed:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        raw_ids = item.get("claim_ids")
        raw_ids = raw_ids if isinstance(raw_ids, list) else []
        # An id the model invented is simply not a citation -- dropped here,
        # so it can never contribute to an OBSERVED/DERIVED classification.
        cited = [claim_by_id[cid] for cid in raw_ids if cid in claim_by_id]
        state, reason_text = _classify_state(text, cited)
        conclusions.append(ReasoningConclusion(
            text=text.strip(), state=state, cited_claim_ids=[c.claim_id for c in cited],
            evidence_chain=_evidence_chain_for(cited), reason=reason_text,
        ))
    t_validation_ms = (time.perf_counter() - t2) * 1000

    if not conclusions:
        conclusions = _deterministic_conclusions(question, claims)
        return ReasoningResult(
            question=question, workspace_id=workspace_id, temporal_context=temporal_context,
            overall_state=_overall_state(conclusions), conclusions=conclusions,
            unresolved_contradictions=contradictions, claims_considered=len(claims),
            reasoned_by="fallback",
            metadata={"reason": "every proposed conclusion was structurally unusable",
                       "context_ms": round(t_context_ms, 2), "model_ms": round(t_model_ms, 2),
                       "validation_ms": round(t_validation_ms, 2)},
        )

    return ReasoningResult(
        question=question, workspace_id=workspace_id, temporal_context=temporal_context,
        overall_state=_overall_state(conclusions), conclusions=conclusions,
        unresolved_contradictions=contradictions, claims_considered=len(claims),
        reasoned_by="llm",
        metadata={"reason": "ok", "context_ms": round(t_context_ms, 2),
                   "model_ms": round(t_model_ms, 2), "validation_ms": round(t_validation_ms, 2)},
    )


# =====================================================================
# Part 10 -- confidence COMPATIBILITY, not a new confidence system.
# =====================================================================

_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def reasoning_supports_confidence(result: ReasoningResult, base_confidence: str) -> str:
    """Reasoning may only ever LOWER confidence, never raise it -- retrieval
    similarity plus graph/memory signal (the existing Phase 5K.1/6D fold)
    remains the ceiling, exactly as grounding.downgrade_for_weak_grounding
    already behaves. A purely INFERRED or UNKNOWN result means the composed
    evidence did not actually support a conclusion, and saying "high
    confidence" over that would be the precise dishonesty this phase exists
    to prevent. Returns the SAME three-tier vocabulary; no new scale, no
    numeric score, no overloading of `similarity`."""
    if result.overall_state == UNKNOWN:
        return "none"
    if result.overall_state == INFERRED:
        return "low" if _CONFIDENCE_RANK.get(base_confidence, 0) > 1 else base_confidence
    return base_confidence
