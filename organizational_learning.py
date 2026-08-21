"""
Phase 7G -- Organizational Learning: longitudinal pattern detection over
real historical evidence.

LEARNING IS NOT FACT (Part 3, the boundary this module exists to hold).
A FACT is "Capacity must be submitted every Monday" -- it came from a source
and lives in structured_knowledge/org_memory. A LEARNING is "this
organization sustains a recurring weekly capacity process across N observed
source events" -- it is DERIVED from history and lives only here, in memory,
for the duration of a call. A LearningSignal is never written into
structured_knowledge as though a source said it, and never inserted into
org_memory. See `propose_for_review()` for the only sanctioned path toward
durability, which goes through the existing review/promotion contract
(Part 12) rather than around it.

TAXONOMY, DEFENDED OR REJECTED (Part 2). Every supported type states its
source evidence, its minimum, and why it is not merely a fact or a Phase 7D
change. Every minimum is justified the same non-arbitrary way: it is the
point at which the observation exceeds what an EXISTING primitive already
says. No numeric threshold is invented for its own sake, and no type uses
embedding similarity, retrieval frequency, graph centrality, or model
confidence as evidence (all explicitly forbidden by Part 5).

  POLICY_EVOLUTION      source: real supersession chains (superseded_at +
                        supersedes_memory_id). minimum: >=2 supersession
                        events (>=3 generations). why not a change: ONE
                        supersession is already fully described by a Phase
                        7D POLICY_CHANGED event; a multi-generation
                        trajectory is the first thing no single change event
                        states.
  PROCESS_TREND         source: a durable process memory whose grounding
                        evidence spans >=2 structured_knowledge rows with
                        DISTINCT captured_at. why not a fact: a single
                        grounding with recurrence_text is the SOURCE saying
                        it recurs -- that is the fact. Repeated independent
                        observation across time is the longitudinal part.
  REPEATED_REVIEW       source: memory_review_queue. minimum: >=2 DISTINCT
                        pending candidates. why not a signal: one pending
                        item is already a Phase 7E REVIEW signal; "repeated"
                        requires more than one by the plain meaning.
  PERSISTENT_UNCERTAINTY source: a single review candidate still pending
                        across a measurable interval. Reports the REAL
                        elapsed interval; no count threshold.
  STABILITY_PATTERN     source: an unsuperseded memory whose last_confirmed_at
                        is later than its created_at -- i.e. the sleep
                        cycle's revalidation independently re-verified its
                        evidence still exists. Reports the REAL observed
                        interval. This is deliberately NOT inferred from
                        silence (Part 9): the interval is covered by actual
                        re-confirmation events, not by an absence of change.
  RELATIONSHIP_PATTERN  source: >=2 real relationships between the SAME
                        ordered entity pair, or one relationship carrying
                        >=2 evidence records at distinct captured_at. Never
                        upgraded to member_of/works_on/owns/manages
                        (Part 10) -- it describes observed interaction only.

REJECTED, with reasons (see REJECTED_LEARNING_TYPES):
  RECURRING_PATTERN     rejected: not separable from PROCESS_TREND under the
                        current schema -- both would read the same
                        recurrence evidence, so shipping both would be one
                        pattern counted twice.
  KNOWLEDGE_GAP_PATTERN rejected AS ORIGINALLY FRAMED. Part 8 forbids
                        defining a gap as "no row exists", and the genuinely
                        repeated-insufficiency inputs it suggests are NOT
                        recorded anywhere: Phase 7A/7B UNKNOWN outcomes and
                        ambiguous resolutions are computed per-request and
                        never persisted, so "repeated UNKNOWN questions" and
                        "recurring ambiguity" have no history to read. A
                        run-count proxy ("N sleep runs failed to resolve it")
                        was considered and REJECTED as dishonest: the
                        consolidation engine is incremental, so later runs
                        never re-examine an existing pending item -- 85 runs
                        is evidence the engine never looked again, not that
                        it repeatedly failed. PERSISTENT_UNCERTAINTY is the
                        narrow, defensible remainder.

LLM ROLE (Part 17): may only explain an already-established deterministic
signal. It cannot discover an event, establish identity, decide a pattern
exists, or create memory. Any failure falls back to the deterministic
explanation.

SECURITY (Part 19): applied BEFORE aggregation -- every memory/evidence read
passes the caller's ladder through memory_retrieval's own visibility check,
so restricted material never enters a pattern, not even as a count.

NO MUTATION (Parts 12/22): no insert/update/delete/upsert/rpc anywhere.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc
import memory_retrieval

POLICY_EVOLUTION = "POLICY_EVOLUTION"
PROCESS_TREND = "PROCESS_TREND"
REPEATED_REVIEW = "REPEATED_REVIEW"
PERSISTENT_UNCERTAINTY = "PERSISTENT_UNCERTAINTY"
STABILITY_PATTERN = "STABILITY_PATTERN"
RELATIONSHIP_PATTERN = "RELATIONSHIP_PATTERN"

# Reused from Phase 7A -- not a new vocabulary.
DERIVED = "DERIVED"
INFERRED = "INFERRED"
UNKNOWN = "UNKNOWN"

REJECTED_LEARNING_TYPES = {
    "RECURRING_PATTERN": ("not separable from PROCESS_TREND under the current schema -- both read the "
                           "same recurrence evidence, so shipping both would count one pattern twice"),
    "KNOWLEDGE_GAP_PATTERN": ("rejected as originally framed: 'no row exists' is explicitly not a gap, "
                               "and repeated-UNKNOWN/ambiguity history is never persisted. A run-count "
                               "proxy was rejected as dishonest because the consolidation engine is "
                               "incremental and never re-examines an existing pending item. "
                               "PERSISTENT_UNCERTAINTY is the defensible remainder."),
}

# Minimums, each justified in the module docstring by the existing primitive
# it must exceed -- never a tuned magic number.
MIN_SUPERSESSIONS_FOR_EVOLUTION = 2      # >=3 generations
MIN_OBSERVATIONS_FOR_PROCESS_TREND = 2   # distinct captured_at
MIN_PENDING_FOR_REPEATED_REVIEW = 2      # distinct candidates
MIN_INTERACTIONS_FOR_RELATIONSHIP = 2    # relationships or distinct-time evidence


@dataclass
class LearningSignal:
    learning_type: str
    workspace_id: str
    subject: dict                      # {kind, id, label}
    reasoning_state: str               # DERIVED | INFERRED | UNKNOWN
    explanation: str
    support_count: int                 # how many REAL observations back it
    observation_window: dict           # {"start": iso|None, "end": iso|None}
    evidence_ids: list = field(default_factory=list)
    memory_ids: list = field(default_factory=list)
    relationship_ids: list = field(default_factory=list)
    affected_entities: list = field(default_factory=list)
    contradicting_evidence: list = field(default_factory=list)
    review_required: bool = False
    explanation_source: str = "deterministic"
    temporal_context: str = "current"


@dataclass
class LearningResult:
    workspace_id: str
    temporal_context: str
    signals: list = field(default_factory=list)
    rejected_types: dict = field(default_factory=lambda: dict(REJECTED_LEARNING_TYPES))
    scanned: dict = field(default_factory=dict)


def _parse(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _visible_memories(workspace_id: str, allowed_sensitivities: list[str],
                       as_of: Optional[datetime]) -> list[dict]:
    """Security before aggregation: the ladder is applied here, so nothing
    restricted can reach any pattern below."""
    rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    return [r for r in rows
            if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]


def _chain_memories(workspace_id: str, allowed_sensitivities: list[str],
                     as_of: Optional[datetime]) -> list[dict]:
    """Supersession chains need a DIFFERENT read than every other detector.

    memory_retrieval._fetch_memory_rows(as_of=None) filters to
    lifecycle_status='active' -- correct for "what is true now", but it
    structurally hides every superseded ancestor. Walking a chain from that
    set could never return more than the surviving generation, so
    POLICY_EVOLUTION fed from it would be a permanently dead detector that
    silently reports "no evolution" no matter how much real history exists.

    A supersession chain is inherently historical, so this read deliberately
    keeps non-active lifecycle rows. It does NOT invent a temporal rule: the
    ladder is still applied, and for a historical as_of the same Phase 6D.1
    MEMORY AVAILABILITY rule is reused verbatim (`_created_before_or_at`),
    so a generation created after as_of is never counted. `superseded_at` is
    intentionally NOT filtered here -- excluding superseded rows is exactly
    what would break chain-walking."""
    rows = bc.supabase.table("org_memory").select("*") \
        .eq("workspace_id", workspace_id).execute().data or []
    rows = [r for r in rows
            if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]
    if as_of is not None:
        rows = [r for r in rows if memory_retrieval._created_before_or_at(r, as_of)]
    return rows


def _grounding(memory_ids: list) -> dict:
    """Only `supports` evidence counts as an observation. The column also
    admits non-supporting stances, and counting one of those as support
    would inflate a pattern with evidence that argues against it."""
    if not memory_ids:
        return {}
    rows = bc.supabase.table("memory_evidence").select("memory_id,evidence_id") \
        .in_("memory_id", list(memory_ids)).eq("evidence_type", "structured_knowledge") \
        .eq("stance", "supports").order("id").execute().data or []
    out: dict = {}
    for r in rows:
        out.setdefault(r["memory_id"], []).append(r["evidence_id"])
    return out


def _sk_rows(sk_ids: list, allowed_sensitivities: list[str]) -> dict:
    if not sk_ids:
        return {}
    rows = bc.supabase.table("structured_knowledge") \
        .select("id,statement,sensitivity,captured_at,recurrence_text,requirement_kind") \
        .in_("id", list(sk_ids)).execute().data or []
    return {r["id"]: r for r in rows
            if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)}


def _label_for(memory_id: str, grounding: dict, sk_by_id: dict) -> Optional[str]:
    for sk_id in grounding.get(memory_id, []):
        sk = sk_by_id.get(sk_id)
        if sk:
            return sk["statement"]
    return None


# =====================================================================
# Individual detectors. Each returns [] when its minimum is not met --
# a sparse corpus produces NO learning, never a weak guess (Part 16 case 7).
# =====================================================================

def _policy_evolution(workspace_id, memories, grounding, sk_by_id) -> list:
    """Walks REAL supersession chains. Requires >=2 supersession events so
    the observation exceeds a single Phase 7D POLICY_CHANGED event."""
    by_id = {m["id"]: m for m in memories}
    successor_of = {m["supersedes_memory_id"]: m for m in memories if m.get("supersedes_memory_id")}
    roots = [m for m in memories
             if m.get("supersedes_memory_id") is None and m["id"] in successor_of]

    out = []
    for root in roots:
        chain, node = [root], root
        while node["id"] in successor_of:
            node = successor_of[node["id"]]
            chain.append(node)
        supersessions = len(chain) - 1
        if supersessions < MIN_SUPERSESSIONS_FOR_EVOLUTION:
            continue
        starts = [_parse(c.get("created_at")) for c in chain if c.get("created_at")]
        out.append(LearningSignal(
            learning_type=POLICY_EVOLUTION, workspace_id=workspace_id,
            subject={"kind": "memory", "id": chain[-1]["id"],
                      "label": _label_for(chain[-1]["id"], grounding, sk_by_id)},
            reasoning_state=DERIVED,
            explanation=(f"This {chain[-1]['memory_type']} has been revised {supersessions} times across "
                          f"{len(chain)} recorded generations. Each step is a real, atomically recorded "
                          f"supersession; no judgement about why it changed is implied."),
            support_count=supersessions,
            observation_window={"start": _iso(min(starts)) if starts else None,
                                 "end": _iso(max(starts)) if starts else None},
            memory_ids=[c["id"] for c in chain],
            evidence_ids=[f"structured_knowledge:{s}" for c in chain for s in grounding.get(c["id"], [])],
        ))
    return out


def _process_trend(workspace_id, memories, grounding, sk_by_id) -> list:
    """A durable process backed by >=2 source observations at DISTINCT
    times -- the single-observation case is the source fact, not learning."""
    out = []
    for m in memories:
        if m.get("memory_type") != "process":
            continue
        sks = [sk_by_id[s] for s in grounding.get(m["id"], []) if s in sk_by_id]
        times = sorted({sk.get("captured_at") for sk in sks if sk.get("captured_at")})
        if len(times) < MIN_OBSERVATIONS_FOR_PROCESS_TREND:
            continue
        recurrence = next((sk.get("recurrence_text") for sk in sks if sk.get("recurrence_text")), None)
        out.append(LearningSignal(
            learning_type=PROCESS_TREND, workspace_id=workspace_id,
            subject={"kind": "memory", "id": m["id"], "label": _label_for(m["id"], grounding, sk_by_id)},
            reasoning_state=DERIVED,
            explanation=(f"This durable process is supported by {len(times)} independent source "
                          f"observations recorded at different times"
                          + (f", with recurrence stated in the source as '{recurrence}'." if recurrence
                             else ".")
                          + " No owner, department, or responsible person is implied."),
            support_count=len(times),
            observation_window={"start": times[0], "end": times[-1]},
            memory_ids=[m["id"]],
            evidence_ids=[f"structured_knowledge:{s}" for s in grounding.get(m["id"], [])],
        ))
    return out


def _stability(workspace_id, memories, grounding, sk_by_id) -> list:
    """Stability from REAL re-confirmation, never from silence: the sleep
    cycle's revalidation independently re-verified the grounding evidence
    still exists, and last_confirmed_at records that. The reported window is
    the actual covered interval, with no duration threshold invented."""
    out = []
    for m in memories:
        created, confirmed = _parse(m.get("created_at")), _parse(m.get("last_confirmed_at"))
        if m.get("superseded_at") or not created or not confirmed or confirmed <= created:
            continue
        out.append(LearningSignal(
            learning_type=STABILITY_PATTERN, workspace_id=workspace_id,
            subject={"kind": "memory", "id": m["id"], "label": _label_for(m["id"], grounding, sk_by_id)},
            reasoning_state=DERIVED,
            explanation=(f"This {m['memory_type']} has not been superseded and its supporting evidence "
                          f"was independently re-confirmed after creation, covering an observed window "
                          f"from {_iso(created)} to {_iso(confirmed)}. Stability is claimed only for that "
                          f"window and only because re-confirmation events exist -- not because nothing "
                          f"was heard."),
            support_count=1,
            observation_window={"start": _iso(created), "end": _iso(confirmed)},
            memory_ids=[m["id"]],
            evidence_ids=[f"structured_knowledge:{s}" for s in grounding.get(m["id"], [])],
        ))
    return out


def _review_patterns(workspace_id, allowed_sensitivities, now: datetime) -> list:
    rows = bc.supabase.table("memory_review_queue").select("*") \
        .eq("workspace_id", workspace_id).order("created_at").execute().data or []
    sk_by_id = _sk_rows([r["structured_knowledge_id"] for r in rows], allowed_sensitivities)
    pending = [r for r in rows
               if r.get("status") == "pending" and r["structured_knowledge_id"] in sk_by_id]
    out = []

    if len(pending) >= MIN_PENDING_FOR_REPEATED_REVIEW:
        starts = [_parse(r["created_at"]) for r in pending if r.get("created_at")]
        out.append(LearningSignal(
            learning_type=REPEATED_REVIEW, workspace_id=workspace_id,
            subject={"kind": "workspace", "id": workspace_id, "label": "review queue"},
            reasoning_state=DERIVED,
            explanation=(f"{len(pending)} distinct items are pending human review. The consolidation "
                          f"engine repeatedly could not decide them automatically. This describes the "
                          f"review backlog, not a conclusion about the knowledge itself."),
            support_count=len(pending),
            observation_window={"start": _iso(min(starts)) if starts else None,
                                 "end": _iso(max(starts)) if starts else None},
            evidence_ids=[f"structured_knowledge:{r['structured_knowledge_id']}" for r in pending],
            review_required=True,
        ))

    for r in pending:
        created = _parse(r.get("created_at"))
        if not created:
            continue
        days = (now - created).total_seconds() / 86400.0
        out.append(LearningSignal(
            learning_type=PERSISTENT_UNCERTAINTY, workspace_id=workspace_id,
            subject={"kind": "structured_knowledge", "id": r["structured_knowledge_id"],
                      "label": sk_by_id[r["structured_knowledge_id"]]["statement"]},
            reasoning_state=DERIVED,
            explanation=(f"This item has remained unresolved for {days:.1f} days since being routed to "
                          f"review. Reported as elapsed time only -- the consolidation engine is "
                          f"incremental and does not re-examine existing pending items, so this is NOT "
                          f"evidence that it repeatedly failed to decide."),
            support_count=1,
            observation_window={"start": _iso(created), "end": _iso(now)},
            evidence_ids=[f"structured_knowledge:{r['structured_knowledge_id']}"],
            review_required=True,
        ))
    return out


def _relationship_patterns(workspace_id, as_of: Optional[datetime]) -> list:
    """Repeated INTERACTION only. Never upgraded to membership, ownership,
    employment, or management (Part 10)."""
    rels = bc.supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", workspace_id).eq("status", "active").execute().data or []
    if as_of is not None:
        rels = [r for r in rels
                if (_parse(r.get("valid_from")) is None or _parse(r["valid_from"]) <= as_of)]
    if not rels:
        return []

    ev_rows = bc.supabase.table("knowledge_relationship_evidence") \
        .select("relationship_id,captured_at").is_("revoked_at", "null") \
        .eq("stance", "supports") \
        .in_("relationship_id", [r["id"] for r in rels]).execute().data or []
    times_by_rel: dict = {}
    for e in ev_rows:
        times_by_rel.setdefault(e["relationship_id"], set()).add(e.get("captured_at"))

    by_pair: dict = {}
    for r in rels:
        key = (r["source_object_id"], r["target_object_id"])
        by_pair.setdefault(key, []).append(r)

    out = []
    for (src, tgt), group in by_pair.items():
        distinct_times = set()
        for r in group:
            distinct_times |= {t for t in times_by_rel.get(r["id"], set()) if t}
        support = max(len(group), len(distinct_times))
        if support < MIN_INTERACTIONS_FOR_RELATIONSHIP:
            continue
        times = sorted(t for t in distinct_times if t)
        out.append(LearningSignal(
            learning_type=RELATIONSHIP_PATTERN, workspace_id=workspace_id,
            subject={"kind": "entity_pair", "id": f"{src}->{tgt}",
                      "label": f"{src} -> {tgt}"},
            reasoning_state=DERIVED,
            explanation=(f"These two entities are connected by {support} recorded interaction "
                          f"observations. This describes observed interaction only -- it does not "
                          f"establish membership, ownership, employment, or management."),
            support_count=support,
            observation_window={"start": times[0] if times else None,
                                 "end": times[-1] if times else None},
            relationship_ids=[r["id"] for r in group],
            affected_entities=[src, tgt],
        ))
    return out


def _contradictions(workspace_id, memories, grounding, sk_by_id) -> list:
    """A real, active `contradicts` relationship targeting a memory's own
    grounding -- reusing the frozen ontology, never a text heuristic. The
    conflict is reported UNKNOWN and routed to review; no winner is picked."""
    sk_ids = {s for m in memories for s in grounding.get(m["id"], [])}
    if not sk_ids:
        return []
    rows = bc.supabase.table("knowledge_relationships").select("id,target_object_id") \
        .eq("workspace_id", workspace_id).eq("status", "active") \
        .eq("target_object_type", "structured_knowledge") \
        .eq("relationship_type", "contradicts") \
        .in_("target_object_id", list(sk_ids)).execute().data or []
    if not rows:
        return []
    contested = {r["target_object_id"] for r in rows}

    out = []
    for m in memories:
        hits = [s for s in grounding.get(m["id"], []) if s in contested]
        if not hits:
            continue
        out.append(LearningSignal(
            learning_type=PERSISTENT_UNCERTAINTY, workspace_id=workspace_id,
            subject={"kind": "memory", "id": m["id"], "label": _label_for(m["id"], grounding, sk_by_id)},
            reasoning_state=UNKNOWN,
            explanation=("A real, active contradicting relationship targets this memory's own grounding "
                          "evidence. The conflict is unresolved; no side is selected and nothing is "
                          "changed."),
            support_count=len(hits),
            observation_window={"start": None, "end": None},
            memory_ids=[m["id"]],
            evidence_ids=[f"structured_knowledge:{s}" for s in hits],
            contradicting_evidence=[r["id"] for r in rows if r["target_object_id"] in hits],
            review_required=True,
        ))
    return out


# =====================================================================
# Public entry point.
# =====================================================================

def detect_learning(workspace_id: str, allowed_sensitivities: list[str],
                     as_of: Optional[datetime] = None, chat_json_fn=None) -> LearningResult:
    """Deterministic longitudinal detection. Bounded reads only: memories,
    their grounding, review queue, relationships and their evidence -- each
    fetched once and batched (Part 20). No corpus rescan, no parallel DB
    access, no graph database."""
    now = as_of or datetime.now(timezone.utc)
    memories = _visible_memories(workspace_id, allowed_sensitivities, as_of)
    chain_rows = _chain_memories(workspace_id, allowed_sensitivities, as_of)
    grounding = _grounding(sorted({m["id"] for m in memories} | {c["id"] for c in chain_rows}))
    sk_by_id = _sk_rows([s for ids in grounding.values() for s in ids], allowed_sensitivities)

    signals: list[LearningSignal] = []
    signals += _policy_evolution(workspace_id, chain_rows, grounding, sk_by_id)
    signals += _process_trend(workspace_id, memories, grounding, sk_by_id)
    signals += _stability(workspace_id, memories, grounding, sk_by_id)
    signals += _review_patterns(workspace_id, allowed_sensitivities, now)
    signals += _relationship_patterns(workspace_id, as_of)
    signals += _contradictions(workspace_id, memories, grounding, sk_by_id)

    temporal = as_of.isoformat() if as_of else "current"
    for s in signals:
        s.temporal_context = temporal
        if chat_json_fn is not None:
            s.explanation, s.explanation_source = _explain(s, chat_json_fn, workspace_id)

    return LearningResult(
        workspace_id=workspace_id, temporal_context=temporal, signals=signals,
        scanned={"memories": len(memories), "grounding_rows": sum(len(v) for v in grounding.values())},
    )


def _explain(signal: LearningSignal, chat_json_fn, workspace_id: str) -> tuple:
    """LLM may only rephrase an already-established signal (Part 17). Any
    failure or malformed output keeps the deterministic text."""
    try:
        raw = chat_json_fn(
            messages=[{"role": "user", "content":
                        f"Rephrase this internal observation more clearly, adding NO new facts:\n\n"
                        f"{signal.explanation}"}],
            system=("You rephrase a verified internal observation about historical patterns. Never add a "
                     "fact, cause, motive, name, or consequence. Never imply ownership, membership, or "
                     'blame. Respond ONLY as {"explanation": "<text>"}.'),
            max_tokens=250, temperature=0.1,
            workspace_id=workspace_id, feature="organizational_learning_explanation",
        )
        text = raw.get("explanation") if isinstance(raw, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip(), "llm"
    except Exception:
        pass
    return signal.explanation, "deterministic"


# =====================================================================
# Part 12 -- the ONLY sanctioned path toward durability. Produces a
# review-shaped PROPOSAL and writes nothing; a human/existing promotion
# path decides. Learning never becomes org_memory directly.
# =====================================================================

def propose_for_review(signal: LearningSignal) -> dict:
    """Returns the proposal a caller could hand to the existing review
    contract. This function performs NO database write of any kind -- the
    evidence/promotion system remains authoritative (Part 12)."""
    return {
        "workspace_id": signal.workspace_id,
        "structured_knowledge_id": None,     # a learning has no source row, by definition
        "proposed_reason": (f"Longitudinal {signal.learning_type} observed with support_count="
                             f"{signal.support_count} over {signal.observation_window}. "
                             f"{signal.explanation}"),
        "reasoning_state": signal.reasoning_state,
        "evidence_ids": list(signal.evidence_ids),
        "requires_human_decision": True,
        "note": ("This is a derived learning, not a sourced fact. It must not be inserted into "
                  "structured_knowledge or org_memory without going through the existing "
                  "review/promotion contract."),
    }


# =====================================================================
# Part 13/14 -- compatibility surfaces. No second attention engine, no
# CompanyState contract change.
# =====================================================================

def learning_for_company_state(result: LearningResult) -> dict:
    """A derived view a future CompanyState/Dashboard may consume without
    modifying either contract."""
    return {
        "emerging_patterns": [s for s in result.signals
                               if s.learning_type in (POLICY_EVOLUTION, PROCESS_TREND)],
        "stable_patterns": [s for s in result.signals if s.learning_type == STABILITY_PATTERN],
        "recurring_review": [s for s in result.signals if s.learning_type == REPEATED_REVIEW],
        "persistent_uncertainty": [s for s in result.signals
                                    if s.learning_type == PERSISTENT_UNCERTAINTY],
        "interaction_patterns": [s for s in result.signals
                                  if s.learning_type == RELATIONSHIP_PATTERN],
    }
