"""
Phase 7E -- Proactive Intelligence: a CONTROLLED ATTENTION layer that turns
already-detected, already-verified organizational change into evidence-bound
signals a human may act on.

THIS IS NOT AN AUTONOMOUS-ACTION SYSTEM. It sends nothing, writes nothing,
and decides nothing on a person's behalf. It produces a ProactiveSignal and
returns it to the caller. There is no Slack, email, push, or task path in
this module -- not disabled, absent (Part 9/20).

THE HARD SEPARATION this phase rests on (its explicit STOP condition):
a RECOMMENDATION must never be presentable as an ORGANIZATIONAL FACT.
That separation is structural here, not stylistic:
  * every FACT a signal carries comes from a Phase 7D ChangeEvent or a
    Phase 7C ImpactPath -- both built exclusively from persisted rows, both
    carrying their own OBSERVED/DERIVED state assigned by deterministic code.
  * every RECOMMENDATION lives in its own field (`recommended_action`),
    always carries `recommendation_state=INFERRED`, and is always emitted
    with `is_hypothesis=True`. A caller cannot render a recommendation
    without also having the flag that says it is one.
  * the LLM may only ever touch `explanation` -- never a state, never an
    audience, never whether a change occurred (Part 15).

AUDIENCE (Part 4), and the honest finding behind it: this codebase has NO
owner, assignee, member, or audience column on ANY knowledge table
(verified live against information_schema -- zero matches across
org_memory, memory_review_queue, knowledge_entities,
knowledge_relationships, structured_knowledge, memory_evidence). The ONLY
verified link between a graph entity and a real human identity is
knowledge_entity_identifiers.identifier_value where identifier_type='email',
which exists today only for Person entities.

Therefore audience is established ONLY for Person entities reachable by a
real impact path, via their real email identifier. A Department entity has
no membership data anywhere in this architecture, so "notify Product" is
NOT derivable -- and this module returns AUDIENCE_NOT_ESTABLISHED rather
than falling back to "everyone in the workspace", which Part 4 forbids and
which would be the exact failure mode that makes proactive systems hated.

SECURITY (Part 12): a signal is generated per-caller-ladder and inherits
workspace, sensitivity, evidence visibility, and temporal visibility from
the ChangeEvent/ImpactPath it was built from -- both of which already
filtered on that same ladder. A change a caller cannot see produces no
event, therefore no signal. An entity in an impact path the caller cannot
resolve never enters the audience.

NO SECOND ENGINE (Part 1): this module detects nothing and traverses
nothing. It consumes ChangeEvent objects from change_detection.py and
ImpactResult objects from impact_analysis.py, both passed in by the caller.
"""
from dataclasses import dataclass, field
from typing import Optional

import brain_connectors as bc
import change_detection as cd

# --- attention vocabulary (Part 3). Deterministic, no numeric score. ---
INFORM = "INFORM"
REVIEW = "REVIEW"
ACTION_RECOMMENDED = "ACTION_RECOMMENDED"
CRITICAL = "CRITICAL"

# --- reasoning vocabulary, reused from Phase 7A unchanged (Part 6) ---
OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
INFERRED = "INFERRED"
UNKNOWN = "UNKNOWN"

AUDIENCE_ESTABLISHED = "audience_established"
AUDIENCE_NOT_ESTABLISHED = "audience not established"

# Change types that represent a durable-memory replacement. Frozen here by
# reference to Phase 7D's own vocabulary rather than restated as strings.
_SUPERSESSION_TYPES = (cd.POLICY_CHANGED, cd.PROCESS_CHANGED, cd.MEMORY_SUPERSEDED)


@dataclass
class AudienceResolution:
    status: str                       # AUDIENCE_ESTABLISHED | AUDIENCE_NOT_ESTABLISHED
    members: list = field(default_factory=list)   # [{entity_id, label, identifier_type, identifier_value}]
    reason: str = ""


@dataclass
class ProactiveSignal:
    signal_id: str                    # deterministic dedup identity (Part 10)
    signal_type: str                  # mirrors the ChangeEvent.change_type
    workspace_id: str
    attention: str                    # INFORM | REVIEW | ACTION_RECOMMENDED | CRITICAL
    subject: dict
    change_event: object              # the real ChangeEvent this came from
    reasoning_state: str              # state of the FACT (OBSERVED/DERIVED/UNKNOWN)
    affected_entities: list = field(default_factory=list)
    affected_memories: list = field(default_factory=list)
    evidence_ids: list = field(default_factory=list)
    audience: AudienceResolution = None
    explanation: str = ""             # factual; deterministic unless a model is supplied
    recommended_action: Optional[str] = None
    recommendation_state: str = INFERRED   # a recommendation is ALWAYS a hypothesis
    is_hypothesis: bool = True
    explanation_source: str = "deterministic"   # 'deterministic' | 'llm'
    expires_at: Optional[str] = None
    expiry_basis: str = ""
    temporal_context: str = "current"


# =====================================================================
# Part 3 -- FROZEN attention rules. Derived only from signals Phase 7D
# already computed from frozen architectural decisions; nothing new is
# scored, weighted, or thresholded here.
# =====================================================================

def classify_attention(event, impact_paths: list) -> str:
    """Exact, ordered rules:

    1. CRITICAL            -- a durable memory was REPLACED and Phase 7D
                              already marked the change `critical` (which it
                              does for supersession, and for any memory
                              change at confidential/restricted sensitivity).
    2. REVIEW              -- knowledge is sitting in the review queue
                              unresolved; a human decision is genuinely
                              outstanding.
    3. ACTION_RECOMMENDED  -- a meaningful change that has at least one REAL,
                              evidence-backed impact path to something else.
                              Someone else is demonstrably affected, so
                              suggesting a look is warranted.
    4. INFORM              -- everything else, including every informational
                              (data-arrival) change.

    No other input is consulted -- not recency, not frequency, not
    centrality, not volume."""
    if event.change_type in _SUPERSESSION_TYPES and event.significance == cd.CRITICAL:
        return CRITICAL
    if event.change_type == cd.REVIEW_REQUIRED:
        return REVIEW
    if event.significance == cd.MEANINGFUL and impact_paths:
        return ACTION_RECOMMENDED
    return INFORM


# =====================================================================
# Part 4 -- audience, from verified identity only.
# =====================================================================

def resolve_audience(workspace_id: str, candidate_entity_ids: list) -> AudienceResolution:
    """Person entities carrying a REAL email identifier are the only
    audience this architecture can establish (see module docstring for the
    live schema finding). Everything else returns AUDIENCE_NOT_ESTABLISHED
    -- never a workspace-wide fallback.

    One bounded, workspace-scoped query per call; no per-signal query
    fan-out (Part 19)."""
    if not candidate_entity_ids:
        return AudienceResolution(status=AUDIENCE_NOT_ESTABLISHED, members=[],
                                   reason="no affected entities were established by evidence")

    ent_rows = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label,status") \
        .eq("workspace_id", workspace_id).in_("id", list(set(candidate_entity_ids))) \
        .execute().data or []
    people = {r["id"]: r for r in ent_rows if r["entity_type"] == "person" and r["status"] == "active"}
    if not people:
        kinds = sorted({r["entity_type"] for r in ent_rows}) or ["unknown"]
        return AudienceResolution(
            status=AUDIENCE_NOT_ESTABLISHED, members=[],
            reason=(f"affected entities are of type {kinds} -- this architecture has no membership, "
                     f"owner, or assignee data for them, so no specific audience is derivable"))

    id_rows = bc.supabase.table("knowledge_entity_identifiers") \
        .select("entity_id,identifier_type,identifier_value") \
        .eq("workspace_id", workspace_id).in_("entity_id", list(people.keys())) \
        .eq("identifier_type", "email").execute().data or []

    members = [{"entity_id": r["entity_id"], "label": people[r["entity_id"]]["canonical_label"],
                "identifier_type": r["identifier_type"], "identifier_value": r["identifier_value"]}
               for r in id_rows if r["entity_id"] in people]
    if not members:
        return AudienceResolution(
            status=AUDIENCE_NOT_ESTABLISHED, members=[],
            reason="affected people carry no verified email identifier")
    return AudienceResolution(status=AUDIENCE_ESTABLISHED, members=members,
                               reason="resolved from verified knowledge_entity_identifiers email records")


# =====================================================================
# Part 7 -- recommendations. Always a hypothesis, never a fact.
# =====================================================================

_RECOMMENDATIONS = {
    cd.POLICY_CHANGED: "Review the replaced policy and confirm the newer version is the one in force.",
    cd.PROCESS_CHANGED: "Review the replaced process and confirm teams are following the newer version.",
    cd.MEMORY_SUPERSEDED: "Review the superseded durable memory and confirm the successor is correct.",
    cd.REVIEW_REQUIRED: "Review this pending item -- the system could not safely decide it automatically.",
    cd.RELATIONSHIP_ADDED: "Check the related Wiki page to confirm the new connection reads correctly.",
    cd.RELATIONSHIP_CHANGED: "Confirm the relationship's new status reflects the real organizational position.",
    cd.MEMORY_PROMOTED: "Confirm this newly durable knowledge is accurate.",
}


def _deterministic_explanation(event, impact_paths: list, audience: AudienceResolution) -> str:
    """Facts only, stated literally from real fields. This is also the
    fallback whenever a model is unavailable or misbehaves (Part 25's
    fallback requirement)."""
    parts = [event.explanation or f"A {event.change_type} change was recorded."]
    if impact_paths:
        labels = [p.target.label for p in impact_paths if p.target.label]
        if labels:
            shown = ", ".join(str(l)[:60] for l in labels[:3])
            parts.append(f"Evidence explicitly connects this to: {shown}.")
    else:
        parts.append("No evidence-backed path connects this change to any other part of the organization.")
    if audience.status == AUDIENCE_NOT_ESTABLISHED:
        parts.append(f"Audience not established -- {audience.reason}.")
    return " ".join(parts)


def _explain_with_model(event, deterministic_text: str, chat_json_fn, workspace_id: str) -> tuple:
    """The LLM's ONLY permitted role (Part 15): rephrasing an
    already-established factual explanation. It is never asked whether a
    change happened, who is affected, or how important it is. Any failure,
    any malformed output, any attempt to return something other than a
    string falls back to the deterministic text -- so a model outage can
    never suppress or distort a signal."""
    try:
        raw = chat_json_fn(
            messages=[{"role": "user", "content":
                        f"Rephrase this internal notice more clearly, adding NO new facts:\n\n{deterministic_text}"}],
            system=("You rephrase an already-verified internal notice. Never add a fact, name, team, date, "
                     'or consequence. Respond ONLY as {"explanation": "<text>"}.'),
            max_tokens=300, temperature=0.1,
            workspace_id=workspace_id, feature="proactive_explanation",
        )
        text = raw.get("explanation") if isinstance(raw, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip(), "llm"
    except Exception:
        pass
    return deterministic_text, "deterministic"


# =====================================================================
# Part 10 -- deterministic dedup identity. Uses only existing event
# identity; no time-window heuristic anywhere.
# =====================================================================

def signal_identity(event) -> str:
    """The SAME real change always yields the SAME signal_id, so repeated
    detection over overlapping windows collapses. A genuinely new change has
    a different subject id and/or a different real occurred_at, so it yields
    a new id. Built only from fields that already exist on the event."""
    subject_id = (event.subject or {}).get("id", "")
    return f"{event.workspace_id}:{event.change_type}:{subject_id}:{event.occurred_at or ''}"


def deduplicate(signals: list) -> list:
    seen, out = set(), []
    for s in signals:
        if s.signal_id in seen:
            continue
        seen.add(s.signal_id)
        out.append(s)
    return out


# =====================================================================
# Part 11 -- expiry. Derived, never persisted, never invented.
# =====================================================================

def _expiry(event) -> tuple:
    """No expires_at is set for any signal, deliberately. A real expiry
    needs either an acknowledgement store (persistence this phase must not
    add) or an arbitrary time window (a heuristic this phase forbids).
    Instead the LIFETIME is derivable at read time from existing state --
    a REVIEW signal is live exactly while its review row is still pending --
    which `is_still_current()` below re-derives without storing anything."""
    if event.change_type == cd.REVIEW_REQUIRED:
        return None, "live while the underlying review row remains pending (re-derived, not stored)"
    return None, "no expiry; historical change events remain true and are filtered by window at read time"


def is_still_current(signal: ProactiveSignal) -> bool:
    """Re-derives liveness from real current state instead of a stored
    expiry. Only REVIEW signals can lapse today (when the review is
    resolved); factual change signals stay true forever."""
    if signal.signal_type != cd.REVIEW_REQUIRED:
        return True
    sk_id = (signal.subject or {}).get("id")
    if not sk_id:
        return True
    rows = bc.supabase.table("memory_review_queue").select("status") \
        .eq("workspace_id", signal.workspace_id).eq("structured_knowledge_id", sk_id) \
        .execute().data or []
    return any(r.get("status") == "pending" for r in rows)


# =====================================================================
# The single public entry point.
# =====================================================================

def build_signals(change_result, workspace_id: str, impact_by_event: Optional[dict] = None,
                   chat_json_fn=None) -> list:
    """ChangeEvents (+ optional per-event ImpactResult) -> ProactiveSignals.

    `impact_by_event` maps signal identity -> ImpactResult, computed by the
    CALLER using Phase 7C unchanged. This module never traverses the graph
    itself, so it can neither widen an impact path nor bypass its security.

    `chat_json_fn` is optional and, when supplied, may only rephrase the
    explanation. Omitted (the default) means a fully deterministic result.
    """
    impact_by_event = impact_by_event or {}
    signals = []

    for event in getattr(change_result, "events", []) or []:
        sid = signal_identity(event)
        impact = impact_by_event.get(sid)
        paths = list(getattr(impact, "paths", []) or []) if impact else []

        # Affected entities: only those the change itself recorded, plus
        # those a REAL impact path reached. Never a guess.
        affected = list(event.affected_entities)
        for p in paths:
            if p.target.kind == "entity" and p.target.object_id not in affected:
                affected.append(p.target.object_id)

        audience = resolve_audience(workspace_id, affected)
        attention = classify_attention(event, paths)

        # FACT state: the change's own state, downgraded to UNKNOWN only
        # when the change itself established nothing.
        fact_state = event.reasoning_state
        if fact_state not in (OBSERVED, DERIVED):
            fact_state = UNKNOWN
        # A fact that reaches another entity via a real 2-hop chain inherits
        # DERIVED from that path -- never upgraded beyond what evidence gave.
        if paths and any(p.reasoning_state == "DERIVED" for p in paths) and fact_state == OBSERVED:
            fact_state = DERIVED

        explanation = _deterministic_explanation(event, paths, audience)
        source = "deterministic"
        if chat_json_fn is not None:
            explanation, source = _explain_with_model(event, explanation, chat_json_fn, workspace_id)

        expires_at, expiry_basis = _expiry(event)

        signals.append(ProactiveSignal(
            signal_id=sid, signal_type=event.change_type, workspace_id=workspace_id,
            attention=attention, subject=dict(event.subject or {}), change_event=event,
            reasoning_state=fact_state, affected_entities=affected,
            affected_memories=list(event.memory_ids), evidence_ids=list(event.evidence_ids),
            audience=audience, explanation=explanation,
            recommended_action=_RECOMMENDATIONS.get(event.change_type),
            recommendation_state=INFERRED, is_hypothesis=True,
            explanation_source=source, expires_at=expires_at, expiry_basis=expiry_basis,
            temporal_context=event.temporal_context,
        ))

    return deduplicate(signals)


# =====================================================================
# Part 16 -- Dashboard compatibility. Views over ONE signal list; Phase 8
# consumes these without a second engine.
# =====================================================================

def summarize_for_dashboard(signals: list) -> dict:
    return {
        "whats_changed": [s for s in signals
                           if s.signal_type in _SUPERSESSION_TYPES + (cd.RELATIONSHIP_CHANGED,)],
        "needs_attention": [s for s in signals if s.attention in (CRITICAL, REVIEW)],
        "should_review": [s for s in signals if s.attention in (REVIEW, ACTION_RECOMMENDED)],
        "whats_important": [s for s in signals if s.attention == CRITICAL],
    }
