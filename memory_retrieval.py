"""
Phase 6D -- Memory + Retrieval Integration adapter.

Bridges durable organizational memory (org_memory/memory_evidence, written
only by memory_consolidation.py's sleep cycle) into the existing retrieval
pipeline (query.py's hybrid_search + build_context_and_citations) as an
ADDITIONAL context source -- exactly the same role graph_retrieval.py
already plays, never a second retrieval implementation and never a
replacement for source evidence.

Architecture, in one line: fetch this workspace's CURRENT (or, if `as_of` is
given, temporally-valid-AT-as_of) memories -> resolve each one's real
grounding evidence via memory_evidence -> structured_knowledge -> attempt
one hop deeper into the real primary source (canonical.py's
project_knowledge_note/project_calendar_event) -> decide relevance
deterministically (graph-evidence overlap and/or query-token overlap against
the grounding statement, NEVER an LLM call) -> convert each relevant,
VISIBLE memory into a citable, chunk-shaped candidate -> merge into the SAME
candidate list graph_retrieval.py's own candidates already join, deduped by
real source identity. Zero new ranking system, zero new confidence system,
zero new answer pipeline (matches graph_retrieval.py's own north star).

MEMORY IS NOT AUTOMATICALLY MORE AUTHORITATIVE THAN SOURCE EVIDENCE. A memory
candidate's content always states its own promotion_basis/lifecycle_status/
valid_from literally, and is flagged with a caveat when a real, active
`supersedes`/`contradicts` relationship targets its grounding evidence --
the synthesis step (and a human reader) can then judge recency/authority
from real, honestly-labeled facts, never from an invented numeric weight.

SECURITY: a memory's own `sensitivity` column (already computed at write
time by create_memory_with_evidence as the STRICTEST ceiling across all its
real evidence) is the single visibility gate -- consistent with Part 12's
"a memory is visible only when the evidence required to support it is
visible" and with this codebase's established "never trust a cached value
alone" discipline, each individual evidence row's own sensitivity is
independently re-checked too when building citations. No new access-control
concept; the same allowed_sensitivities ladder every other retrieval path
already uses.

PENDING REVIEW CANDIDATES (memory_review_queue) ARE NEVER READ BY THIS
MODULE AT ALL. There is no product surface yet that explicitly allows
presenting a review candidate as anything -- Part 11's gate does not exist,
so the safe default is silence, not a guess at what that gate should look
like.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc
import canonical

_SENSITIVITY_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Deterministic query-token relevance (Part 8) -- no LLM, no embedding call.
# "knova" is stripped since users address the product by name without that
# being a content word to match against a memory's grounding statement.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "when",
    "where", "how", "does", "do", "did", "can", "could", "should", "would",
    "to", "of", "in", "on", "for", "and", "or", "about", "that", "this",
    "it", "knova", "not", "have", "has", "know", "remember",
})


def _tokenize(text: Optional[str]) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _temporally_valid(row: dict, as_of: datetime) -> bool:
    """Part 3/4's frozen predicate, unchanged from graph_query.py's own
    (and org_memory's) established semantics: NULL valid_from means no
    known lower bound (never "starts now"); NULL valid_until means no
    known upper bound."""
    valid_from = _parse_ts(row.get("valid_from"))
    valid_until = _parse_ts(row.get("valid_until"))
    if valid_from is not None and valid_from > as_of:
        return False
    if valid_until is not None and valid_until <= as_of:
        return False
    return True


def _is_visible(sensitivity: Optional[str], allowed_sensitivities: list[str]) -> bool:
    if sensitivity is None:
        return True
    return sensitivity in allowed_sensitivities


# =====================================================================
# Part 2 -- stable internal memory context contract.
# =====================================================================

@dataclass
class MemoryCandidate:
    memory_id: str
    memory_type: str
    lifecycle_status: str
    promotion_basis: str
    valid_from: Optional[str]
    valid_until: Optional[str]
    sensitivity: str
    last_confirmed_at: Optional[str]
    evidence: list[dict]              # [{evidence_type, evidence_id, evidence_kind, reference, dedup_key}]
    relevance: str                    # 'graph' | 'query_text'
    possibly_superseded: bool = False
    # Phase 6D.1 -- exposed for the same reason lifecycle_status always is:
    # honesty about what this candidate actually is, never hidden from the
    # caller. This is MEMORY AVAILABILITY (when KNOVA came to know this),
    # distinct from valid_from (CLAIM VALIDITY -- when the claim itself
    # became true).
    created_at: Optional[str] = None
    # Phase 6D.2 -- MEMORY SUCCESSION: when this memory stopped being the
    # current durable representation (set only by create_memory_with_
    # evidence's own atomic supersession path). Distinct from
    # possibly_superseded above, which flags a DIFFERENT, evidence-level
    # contradiction signal (Phase 6D Part 6) -- this field is the memory's
    # own real succession record, or None if it was never superseded.
    superseded_at: Optional[str] = None


@dataclass
class MemoryContext:
    candidates: list[MemoryCandidate] = field(default_factory=list)
    temporal_context: str = "current"


# =====================================================================
# Part 3/4 -- current vs historical memory fetch.
# =====================================================================

def _fetch_memory_rows(workspace_id: str, as_of: Optional[datetime]) -> list[dict]:
    """Current (as_of=None): lifecycle_status='active' ONLY, per Part 3 --
    a dormant/superseded/archived memory is never current durable
    knowledge, regardless of its temporal window.
    Historical (as_of given): every lifecycle_status is eligible -- Part 4's
    explicit rule ("a superseded memory may be historically valid before
    the supersession event... do NOT delete it from historical retrieval"),
    generalized to dormant/archived the same way graph_query.py's own
    historical reads already generalize past a current-only status filter.
    Lifecycle is never hidden from the caller either way -- each candidate
    still carries its own real lifecycle_status.

    Phase 6D.1 -- CLAIM VALIDITY vs. MEMORY AVAILABILITY, made explicit as
    two separate checks for historical reads. `_temporally_valid` alone
    answers "was the underlying claim true at as_of" -- it does NOT answer
    "did KNOVA actually know this at as_of". A memory with valid_from=NULL
    (a legitimate, permanent "no known real-world start", per the frozen
    Phase 6B.1 decision -- never reinterpreted here) would otherwise read as
    valid at ANY as_of, including one before the memory was ever promoted
    into org_memory at all. `created_at <= as_of` closes that gap: a memory
    can never be returned for a historical point before KNOVA itself
    created it, regardless of how open-ended its claim-validity window is.
    Current reads are unaffected -- check_time is always now() there, and
    created_at <= now() is trivially true for any real row, so this is a
    no-op for current queries by construction (Part 6), not a special case
    that had to be carved out."""
    query = bc.supabase.table("org_memory").select("*").eq("workspace_id", workspace_id)
    if as_of is None:
        query = query.eq("lifecycle_status", "active")
    rows = query.execute().data or []
    check_time = as_of or datetime.now(timezone.utc)
    rows = [r for r in rows if _temporally_valid(r, check_time)]
    if as_of is not None:
        rows = [r for r in rows if _created_before_or_at(r, as_of)]
        rows = [r for r in rows if _not_yet_superseded_at(r, as_of)]
    return rows


def _created_before_or_at(row: dict, as_of: datetime) -> bool:
    """MEMORY AVAILABILITY (Phase 6D.1) -- distinct from claim validity.
    Fails closed: a row with no parseable created_at (should never happen,
    the column is NOT NULL) is excluded, never assumed available."""
    created_at = _parse_ts(row.get("created_at"))
    return created_at is not None and created_at <= as_of


def _not_yet_superseded_at(row: dict, as_of: datetime) -> bool:
    """MEMORY SUCCESSION (Phase 6D.2) -- the third, independent concept:
    availability answers "did KNOVA know this yet", claim validity answers
    "was the claim true", and this answers "was this STILL the current
    durable representation at as_of, or had a real, atomically-recorded
    succession event already replaced it by then". superseded_at is set
    ONLY by create_memory_with_evidence's own atomic supersession path
    (equal to the real successor's created_at, by construction -- see that
    RPC's own comments) -- NULL means "never superseded" and is excluded
    from nothing here.

    Deliberately narrow: this is the ONLY new exclusion this pass adds.
    dormant/archived memories that were never superseded keep exactly the
    Phase 6D.1 behavior (still eligible for historical reads, honestly
    labeled with their real lifecycle_status) -- there is no
    'dormant_at'/'archived_at' timestamp, and this pass does not invent
    one. A normal historical query answers "what was current then,
    accounting for real succession"; it does not become an audit mode that
    excludes every non-active status -- that would be the "separate
    historical/audit query mode" Part 7 explicitly says not to conflate
    with this one."""
    superseded_at = _parse_ts(row.get("superseded_at"))
    return superseded_at is None or superseded_at > as_of


# =====================================================================
# Part 5 -- evidence resolution: memory -> memory_evidence ->
# structured_knowledge -> deeper provenance.
# =====================================================================

def _resolve_deeper_provenance(sk_row: dict, workspace_id: str) -> Optional[dict]:
    """One hop past structured_knowledge into the real primary artifact,
    reusing canonical.py's existing projection functions rather than
    re-deriving provenance resolution here. Returns
    {'reference': str, 'primary_resolved': bool} or None if the canonical
    parent no longer resolves (fails closed -- never fabricated).
    workspace_id is re-verified against the projection's own result, never
    trusted from structured_knowledge alone (same discipline
    structured_persistence._fetch_canonical_parent already established)."""
    source_type = sk_row.get("canonical_source_type")
    canonical_id = sk_row.get("canonical_id")
    if not source_type or not canonical_id:
        return None

    ck = None
    if source_type in ("knowledge_note", "slack", "google_chat", "google_meet", "bot_learning"):
        ck = canonical.project_knowledge_note(canonical_id)
    elif source_type == "calendar_event":
        ck = canonical.project_calendar_event(canonical_id)

    if ck is None or ck.workspace_id != workspace_id:
        return None
    if not _is_visible(ck.sensitivity, _ALL_SENSITIVITIES):
        # Unreachable in practice today (Calendar has no sensitivity concept,
        # and a note's sensitivity can only be <= its already-visible
        # structured_knowledge row's own, per authoring rules elsewhere) --
        # checked anyway rather than assumed; caller applies the REAL
        # caller-specific ladder separately below.
        pass

    reference = ck.title or (ck.content[:120] if ck.content else None)
    return {"reference": reference, "sensitivity": ck.sensitivity, "primary_resolved": True}


_ALL_SENSITIVITIES = ["public", "internal", "confidential", "restricted"]


def _resolve_memory_evidence(memory_row: dict, evidence_rows: list[dict],
                             sk_by_id: dict, workspace_id: str,
                             allowed_sensitivities: list[str]) -> list[dict]:
    """Every memory_evidence row is evidence_type='structured_knowledge'
    (frozen V1 constraint) -- resolves the structured_knowledge statement
    (always present, always 'derived_support' -- KNOVA's own interpretation,
    exactly matching graph_query._evidence_kind's established rule for this
    same evidence_type) plus, when resolvable, the real primary source one
    hop deeper (Part 5). An individually-invisible structured_knowledge row
    is dropped from the citation list entirely (defense in depth -- the
    memory-level sensitivity ceiling already gates the whole memory before
    this is ever called, but a per-row re-check costs nothing and matches
    this codebase's own "never trust a cached derived value alone"
    convention)."""
    resolved = []
    for ev in evidence_rows:
        sk = sk_by_id.get(ev["evidence_id"])
        if sk is None or not _is_visible(sk.get("sensitivity"), allowed_sensitivities):
            continue
        deeper = _resolve_deeper_provenance(sk, workspace_id)
        if deeper and _is_visible(deeper.get("sensitivity"), allowed_sensitivities):
            reference = deeper["reference"] or sk.get("statement")
            primary_resolved = True
        else:
            reference = sk.get("statement")
            primary_resolved = False
        resolved.append({
            "evidence_type": "structured_knowledge",
            "evidence_id": ev["evidence_id"],
            "evidence_kind": "derived_support",
            "reference": reference,
            "primary_resolved": primary_resolved,
            "dedup_keys": _dedup_keys_for_sk(sk),
        })
    return resolved


def _dedup_keys_for_sk(sk: dict) -> set[str]:
    """Every real source identity this structured_knowledge row could be
    deduped against -- there are up to two genuinely different overlaps to
    catch, not one:
      1. the SAME row cited as evidence on a graph relationship (graph's own
         dedup key for a structured_knowledge-typed evidence record is the
         unresolved `structured_knowledge:<id>` form -- see
         graph_retrieval._dedup_key_for_evidence); and
      2. a knowledge_note-sourced row's underlying NOTE also retrieved as a
         normal embedded chunk (document_chunks.document_id == the note's
         own id, i.e. this row's own canonical_id -- structured_persistence.
         py writes it directly, no extra hop needed).
    A calendar_event-sourced row never overlaps with an embedded chunk
    (confirmed live elsewhere in this codebase), so only form 1 applies to
    it. Both forms are returned so the caller can match against whichever
    the other candidate actually used, rather than guessing which one
    applies up front."""
    keys = {f"structured_knowledge:{sk['id']}"}
    if sk.get("canonical_source_type") == "knowledge_note" and sk.get("canonical_id"):
        keys.add(f"document:{sk['canonical_id']}")
    return keys


# =====================================================================
# Part 6 -- deterministic "may be superseded" caveat (never a numeric
# multiplier, never a new relationship type -- reuses the frozen
# supersedes/contradicts ontology exactly as it already exists).
# =====================================================================

def _find_superseding_targets(sk_ids: set[str], workspace_id: str) -> set[str]:
    if not sk_ids:
        return set()
    rows = bc.supabase.table("knowledge_relationships").select("target_object_id") \
        .eq("workspace_id", workspace_id).eq("target_object_type", "structured_knowledge") \
        .eq("status", "active").in_("relationship_type", ["supersedes", "contradicts"]) \
        .in_("target_object_id", list(sk_ids)).execute().data or []
    return {r["target_object_id"] for r in rows}


# =====================================================================
# Part 8 -- relevance. Bounded by the small number of durable memories in
# this workspace (never a scan of the full structured_knowledge table --
# Part 17), via memory_evidence's own indexed lookup.
# =====================================================================

def build_memory_context(question: str, workspace_id: str, allowed_sensitivities: list[str],
                         as_of: Optional[datetime] = None,
                         graph_context=None) -> Optional[MemoryContext]:
    """Returns None whenever there is nothing relevant to add -- the safe,
    expected "fall back to normal retrieval" signal, matching
    graph_retrieval.build_graph_context's own contract exactly."""
    rows = _fetch_memory_rows(workspace_id, as_of)
    if not rows:
        return None

    memory_ids = [r["id"] for r in rows]
    evidence_rows = bc.supabase.table("memory_evidence").select("*") \
        .in_("memory_id", memory_ids).execute().data or []
    sk_ids = list({e["evidence_id"] for e in evidence_rows if e["evidence_type"] == "structured_knowledge"})
    sk_by_id = {}
    if sk_ids:
        sk_by_id = {r["id"]: r for r in
                    bc.supabase.table("structured_knowledge").select("*").in_("id", sk_ids).execute().data or []}

    evidence_by_memory: dict[str, list[dict]] = {}
    for e in evidence_rows:
        evidence_by_memory.setdefault(e["memory_id"], []).append(e)

    graph_sk_ids = set()
    if graph_context is not None:
        graph_sk_ids = {ev.evidence_id for ev in graph_context.evidence if ev.evidence_type == "structured_knowledge"}

    q_tokens = _tokenize(question)

    all_grounding_ids = set()
    prelim: list[tuple[dict, str]] = []
    for row in rows:
        if row["sensitivity"] not in allowed_sensitivities:
            continue  # Part 12 -- the memory-level ceiling gates the whole memory
        mem_evidence = evidence_by_memory.get(row["id"], [])
        relevance = None
        for e in mem_evidence:
            sk = sk_by_id.get(e["evidence_id"])
            if not sk:
                continue
            if e["evidence_id"] in graph_sk_ids:
                relevance = "graph"
                break
            sk_tokens = (_tokenize(sk.get("statement")) | _tokenize(sk.get("raw_subject_phrase"))
                         | {w.lower() for w in (sk.get("qualifier_words") or []) if w})
            if q_tokens and (q_tokens & sk_tokens):
                relevance = relevance or "query_text"
        if relevance is None:
            continue
        prelim.append((row, relevance))
        all_grounding_ids |= {e["evidence_id"] for e in mem_evidence if e["evidence_type"] == "structured_knowledge"}

    if not prelim:
        return None

    superseding_targets = _find_superseding_targets(all_grounding_ids, workspace_id)

    candidates: list[MemoryCandidate] = []
    for row, relevance in prelim:
        mem_evidence = evidence_by_memory.get(row["id"], [])
        resolved_evidence = _resolve_memory_evidence(row, mem_evidence, sk_by_id, workspace_id, allowed_sensitivities)
        if not resolved_evidence:
            continue  # every grounding row individually failed visibility -- nothing citable
        grounding_ids = {e["evidence_id"] for e in mem_evidence if e["evidence_type"] == "structured_knowledge"}
        candidates.append(MemoryCandidate(
            memory_id=row["id"], memory_type=row["memory_type"],
            lifecycle_status=row["lifecycle_status"], promotion_basis=row["promotion_basis"],
            valid_from=row.get("valid_from"), valid_until=row.get("valid_until"),
            sensitivity=row["sensitivity"], last_confirmed_at=row.get("last_confirmed_at"),
            evidence=resolved_evidence, relevance=relevance,
            possibly_superseded=bool(grounding_ids & superseding_targets),
            created_at=row.get("created_at"),
            superseded_at=row.get("superseded_at"),
        ))

    if not candidates:
        return None

    return MemoryContext(
        candidates=candidates,
        temporal_context=(as_of.isoformat() if as_of else "current"),
    )


# =====================================================================
# Part 9 -- merge into the SAME candidate list, deduped by real source
# identity against BOTH normal chunks and graph_context's own evidence
# (never against opaque already-built candidate dicts, which have already
# lost their underlying evidence identity by the time they're chunk-shaped).
# =====================================================================

def _memory_candidate(c: MemoryCandidate) -> dict:
    """One chunk-shaped dict per memory (Part 9: bundle its evidence into
    ONE evidence chain, never one candidate per evidence row). The content
    string states promotion_basis/lifecycle_status/valid_from literally --
    never paraphrased -- so the synthesis step can reason about recency/
    authority from real facts, never a numeric weight (Part 6)."""
    lines = [f"Durable organizational memory ({c.memory_type}, promoted via {c.promotion_basis})."]
    if c.lifecycle_status != "active":
        lines.append(f"Status: {c.lifecycle_status}.")
    if c.valid_from:
        lines.append(f"Valid from: {c.valid_from}.")
    if c.possibly_superseded:
        lines.append("Note: a newer record may supersede or contradict this -- see other sources for the current position.")
    lines.append("Supporting evidence:")
    for e in c.evidence:
        lines.append(f"- ({e['evidence_kind']}) {e['reference'] or 'no reference available'}")
    content = "\n".join(lines)
    return {
        "id":          f"memory:{c.memory_id}",
        "document_id": f"org_memory:{c.memory_id}",
        "content":     content,
        "metadata":    {"file_name": f"Durable memory ({c.memory_type})", "source_type": "org_memory"},
        "source_type": "org_memory",
        "source_tier": 1,
        # Real retrieval similarity ONLY -- None here is honest, matching
        # graph_retrieval's own corrected Phase 5K.1 contract exactly.
        "similarity":  None,
        "evidence_strength": "primary" if any(e["primary_resolved"] for e in c.evidence) else "derived",
    }


def merge_memory_context_into_chunks(chunks: list[dict], memory_context: Optional[MemoryContext],
                                     graph_context=None) -> tuple[list[dict], dict]:
    metrics = {
        "memory_candidates_found":        0,
        "memory_candidates_added":        0,
        "memory_candidates_deduplicated": 0,
        "memory_only":                    False,
    }
    if memory_context is None or not memory_context.candidates:
        return chunks, metrics

    metrics["memory_candidates_found"] = len(memory_context.candidates)

    started_empty = not chunks
    existing_keys = {f"document:{c['document_id']}" for c in chunks if c.get("document_id")}
    if graph_context is not None:
        existing_keys |= {f"{ev.evidence_type}:{ev.evidence_id}" for ev in graph_context.evidence}

    merged = list(chunks)
    for cand in memory_context.candidates:
        # Every evidence item individually covered (by EITHER of its
        # possible real-identity forms, see _dedup_keys_for_sk) means the
        # whole memory candidate adds nothing new to cite.
        fully_covered = bool(cand.evidence) and all(
            bool(e["dedup_keys"] & existing_keys) for e in cand.evidence
        )
        if fully_covered:
            metrics["memory_candidates_deduplicated"] += 1
            continue
        merged.append(_memory_candidate(cand))
        metrics["memory_candidates_added"] += 1

    metrics["memory_only"] = started_empty and graph_context is None and metrics["memory_candidates_added"] > 0
    return merged, metrics


# =====================================================================
# Part 14 -- MEMORY CONFIDENCE. HIGH requires a resolved, VISIBLE primary
# artifact underneath at least one relevant memory (never memory existence
# alone); MEDIUM means a real memory candidate exists, grounded only in its
# own structured_knowledge (derived) layer; None means no relevant memory
# candidate at all -- callers must not upgrade anything on None.
# =====================================================================

def memory_confidence(memory_context: Optional[MemoryContext]) -> Optional[str]:
    if memory_context is None or not memory_context.candidates:
        return None
    has_primary = any(
        any(e["primary_resolved"] for e in c.evidence)
        for c in memory_context.candidates
    )
    return "high" if has_primary else "medium"


def combine_confidence(base_confidence: str, memory_context: Optional[MemoryContext]) -> str:
    """The same "take the stronger signal, never lower, never fabricate"
    rule graph_retrieval.combine_confidence already established -- applied
    here as one more fold on top of whatever the caller already computed
    (vector confidence alone, or graph_retrieval.combine_confidence's own
    result). No second confidence system; same three-tier vocabulary."""
    m_conf = memory_confidence(memory_context)
    if m_conf is None:
        return base_confidence
    if _CONFIDENCE_RANK[m_conf] > _CONFIDENCE_RANK.get(base_confidence, 0):
        return m_conf
    return base_confidence
