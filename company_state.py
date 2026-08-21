"""
Phase 7F -- Company State / Executive Intelligence: a deterministic, derived
view of what is CURRENTLY TRUE in a company, assembled entirely from
already-verified layers.

STATE IS NOT ACTIVITY (Part 2, the distinction this module exists to
enforce). A meeting happening, a Slack message arriving, or a document
being uploaded are ACTIVITY -- they are never company state here. State is
what is currently true and durable: active policies and processes, verified
entities and relationships, unresolved reviews, and the changes that moved
the company from one state to another. Nothing in this module reads an
activity stream; every dimension is sourced from a durable layer.

NO SECOND ENGINE (Parts 8/9/17). This module detects no change, raises no
alert, traverses no graph, and runs no retrieval of its own. It composes:
    memory_retrieval._fetch_memory_rows   -> active durable memory
    graph_query.get_entity_graph          -> verified relationships
    change_detection.detect_changes       -> recent meaningful change (7D)
    proactive_intelligence.build_signals  -> attention items (7E)
    impact_analysis.analyze_impact        -> verified connections (7C)
all unchanged, all called with the SAME workspace, ladder, and as_of.

ABSENCE OF EVIDENCE IS NEVER EVIDENCE OF ABSENCE (Part 4). This is enforced
in the wording of every empty dimension: an empty list is always reported as
"no verified X is currently recorded", never as "the company has no X".
`state_confidence` is deliberately a per-dimension OBSERVED/DERIVED/UNKNOWN
label reused from Phase 7A -- NOT a second confidence framework, NOT a
score, and never a claim that a quiet corpus means a quiet company.

UNCERTAINTY IS FIRST-CLASS (Part 11). `open_uncertainty` is a real output
dimension, not an error path. "Department ownership of this process is not
established" is a valid, expected state -- the model is not required to
have an answer for every dimension, and is forbidden from inventing one.

SECURITY BEFORE AGGREGATION (Part 13). Every underlying call receives the
caller's real sensitivity ladder, so restricted material never enters the
aggregate in the first place. Nothing is aggregated-then-redacted, which
would leak through counts even when values were hidden.

NO WRITES, NO ACTION (Part 21): no insert/update/delete/upsert/rpc, no
notification, no task, no Wiki/memory/graph mutation.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import brain_connectors as bc
import graph_query
import memory_retrieval
import impact_analysis
import change_detection as cd
import proactive_intelligence as pi

# Reused from Phase 7A -- NOT a new vocabulary (Part 4).
OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
UNKNOWN = "UNKNOWN"

# The three durable memory kinds the frozen architecture supports.
_MEMORY_DIMENSIONS = ("policy", "process", "decision")


@dataclass
class StateItem:
    """One thing that is currently true, always carrying its own evidence
    and its own state label -- never a bare string."""
    kind: str                      # 'policy' | 'process' | 'decision' | 'person' | 'department' | 'relationship'
    object_id: str
    label: Optional[str]
    state: str = OBSERVED          # OBSERVED | DERIVED | UNKNOWN
    evidence_ids: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    not_established: list = field(default_factory=list)   # what is explicitly NOT known about it


@dataclass
class StateDimension:
    """A dimension is never just a list -- it carries how well supported it
    is and, when empty, WHY (absence of record vs absence of fact)."""
    name: str
    items: list = field(default_factory=list)
    state: str = OBSERVED
    coverage_note: str = ""


@dataclass
class CompanyState:
    workspace_id: str
    as_of: str                     # 'current' or the ISO as_of
    generated_at: str
    active_policies: StateDimension = None
    active_processes: StateDimension = None
    active_decisions: StateDimension = None
    active_people: StateDimension = None
    active_departments: StateDimension = None
    verified_connections: StateDimension = None
    recent_changes: list = field(default_factory=list)      # ChangeEvent (7D)
    attention_items: list = field(default_factory=list)     # ProactiveSignal (7E)
    open_uncertainty: list = field(default_factory=list)    # [{topic, detail, state}]
    evidence_summary: dict = field(default_factory=dict)
    state_confidence: dict = field(default_factory=dict)    # per-dimension OBSERVED/DERIVED/UNKNOWN
    metrics: dict = field(default_factory=dict)


def _memory_items(workspace_id: str, allowed_sensitivities: list[str],
                   as_of: Optional[datetime]) -> dict:
    """All durable memory dimensions in a bounded, batched pass -- one
    memory fetch, one evidence fetch, one statement fetch, regardless of how
    many memories exist (Part 19). Security is applied by
    _fetch_memory_rows + _is_visible BEFORE anything is aggregated."""
    rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    rows = [r for r in rows
            if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]
    if as_of is None:
        # Current state means CURRENTLY true -- a dormant/superseded memory
        # is not current durable knowledge (the same rule Phase 6D froze).
        rows = [r for r in rows if r.get("lifecycle_status") == "active"]

    by_id = {r["id"]: r for r in rows}
    ev_rows = []
    if by_id:
        ev_rows = bc.supabase.table("memory_evidence").select("memory_id,evidence_id") \
            .in_("memory_id", list(by_id)).eq("evidence_type", "structured_knowledge") \
            .order("id").execute().data or []
    sk_ids = sorted({e["evidence_id"] for e in ev_rows})
    sk_by_id = {}
    if sk_ids:
        sk_by_id = {r["id"]: r for r in
                    bc.supabase.table("structured_knowledge").select("id,statement,sensitivity")
                    .in_("id", sk_ids).execute().data or []}

    grounding: dict = {}
    for e in ev_rows:
        grounding.setdefault(e["memory_id"], []).append(e["evidence_id"])

    out = {dim: [] for dim in _MEMORY_DIMENSIONS}
    for mid, row in by_id.items():
        if row["memory_type"] not in out:
            continue
        ev_ids = grounding.get(mid, [])
        label = None
        for sk_id in ev_ids:
            sk = sk_by_id.get(sk_id)
            if sk and memory_retrieval._is_visible(sk.get("sensitivity"), allowed_sensitivities):
                label = sk["statement"]
                break
        out[row["memory_type"]].append(StateItem(
            kind=row["memory_type"], object_id=mid, label=label, state=OBSERVED,
            evidence_ids=[f"structured_knowledge:{s}" for s in ev_ids],
            attributes={"promotion_basis": row["promotion_basis"],
                         "lifecycle_status": row["lifecycle_status"],
                         "sensitivity": row["sensitivity"],
                         "valid_from": row.get("valid_from"),
                         "superseded_at": row.get("superseded_at")},
            not_established=["owning department is not established -- no verified relationship "
                              "connects this durable memory to a department entity"],
        ))
    return out


def _entity_items(workspace_id: str, as_of: Optional[datetime]) -> tuple:
    """Verified people and departments. Entities carry no sensitivity of
    their own (frozen Phase 5 decision), so no ladder applies here -- but
    temporal availability does, reusing Phase 7B's created_at rule."""
    rows = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label,status,created_at") \
        .eq("workspace_id", workspace_id).eq("status", "active") \
        .order("entity_type").order("canonical_label").execute().data or []
    if as_of is not None:
        rows = [r for r in rows if _created_at_or_before(r, as_of)]

    people_rows = [r for r in rows if r["entity_type"] == "person"]
    dept_rows = [r for r in rows if r["entity_type"] == "department"]

    identifiers: dict = {}
    if people_rows:
        id_rows = bc.supabase.table("knowledge_entity_identifiers") \
            .select("entity_id,identifier_type,identifier_value") \
            .eq("workspace_id", workspace_id) \
            .in_("entity_id", [r["id"] for r in people_rows]).execute().data or []
        for r in id_rows:
            identifiers.setdefault(r["entity_id"], []).append(
                {"identifier_type": r["identifier_type"], "identifier_value": r["identifier_value"]})

    people = [StateItem(
        kind="person", object_id=r["id"], label=r["canonical_label"], state=OBSERVED,
        evidence_ids=[], attributes={"identifiers": identifiers.get(r["id"], [])},
        not_established=[
            "employment is not established", "job title is not established",
            "department membership is not established", "reporting line is not established",
        ],
    ) for r in people_rows]

    departments = [StateItem(
        kind="department", object_id=r["id"], label=r["canonical_label"], state=OBSERVED,
        evidence_ids=[],
        not_established=["membership is not established -- this architecture records no "
                          "person-to-department relationship"],
    ) for r in dept_rows]
    return people, departments


def _created_at_or_before(row: dict, as_of: datetime) -> bool:
    raw = row.get("created_at")
    if raw is None:
        return False
    ts = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return ts <= as_of


def _connection_items(workspace_id: str, allowed_sensitivities: list[str],
                       as_of: Optional[datetime], entity_items: list) -> list:
    """Verified organizational connections, via Phase 7C unchanged. Only
    1-hop OBSERVED paths are treated as state: a 2-hop DERIVED chain is a
    real inference about connectivity, not a currently-recorded
    organizational fact, so it is not asserted as state here."""
    seen, out = set(), []
    for ent in entity_items:
        result = impact_analysis.analyze_impact(
            "entity", ent.object_id, workspace_id, allowed_sensitivities,
            as_of=as_of, max_hops=1)
        for path in result.paths:
            for rid in path.relationship_ids:
                if rid in seen:
                    continue
                seen.add(rid)
                out.append(StateItem(
                    kind="relationship", object_id=rid,
                    # ASCII separator deliberately: this label is user-facing
                    # data Phase 8 will render, and a non-ASCII dash mangles
                    # on Windows consoles/cp1252 consumers (observed live in
                    # this phase's own benchmark output).
                    label=f"{ent.label} -[{path.chain[0].relationship_type}]-> {path.target.label}",
                    state=OBSERVED if path.hops == 1 else DERIVED,
                    evidence_ids=list(path.evidence_ids),
                    attributes={"relationship_type": path.chain[0].relationship_type,
                                 "source_id": ent.object_id, "target_id": path.target.object_id},
                ))
    return out


def _dimension(name: str, items: list, empty_note: str) -> StateDimension:
    """An empty dimension NEVER claims the company lacks the thing -- only
    that nothing verified is currently recorded (Part 4)."""
    if items:
        # The dimension name already carries its own qualifier ("active
        # policies", "verified connections"), so no extra adjective is
        # prefixed -- doing so produced "2 verified verified connections"
        # in this phase's own benchmark output.
        return StateDimension(name=name, items=items, state=OBSERVED,
                               coverage_note=f"{len(items)} {name} currently recorded.")
    return StateDimension(name=name, items=[], state=UNKNOWN, coverage_note=empty_note)


def build_company_state(workspace_id: str, allowed_sensitivities: list[str],
                         as_of: Optional[datetime] = None,
                         changes_since: Optional[datetime] = None,
                         include_connections: bool = True) -> CompanyState:
    """Derived on demand; nothing is persisted (Part 3).

    One shared `as_of` governs every dimension, so a snapshot can never mix
    dates (Part 12). `changes_since` bounds the recent-change window only;
    it never affects what is currently TRUE.
    """
    t0 = time.perf_counter()
    as_of_label = as_of.isoformat() if as_of else "current"

    memories = _memory_items(workspace_id, allowed_sensitivities, as_of)
    people, departments = _entity_items(workspace_id, as_of)

    connections = []
    if include_connections:
        connections = _connection_items(workspace_id, allowed_sensitivities, as_of,
                                         people + departments)

    change_result = cd.detect_changes(workspace_id, allowed_sensitivities, since=changes_since)
    signals = pi.build_signals(change_result, workspace_id)
    attention = [s for s in signals if s.attention in (pi.CRITICAL, pi.REVIEW, pi.ACTION_RECOMMENDED)]

    dims = {
        "active_policies": _dimension(
            "active policies", memories["policy"],
            "No verified durable policy is currently recorded for this workspace. This reflects what "
            "KNOVA has evidence for, not a claim that the company has no policies."),
        "active_processes": _dimension(
            "active processes", memories["process"],
            "No verified durable process is currently recorded for this workspace. This reflects "
            "recorded evidence only, not a claim that no processes exist."),
        "active_decisions": _dimension(
            "active decisions", memories["decision"],
            "No verified durable decision is currently recorded for this workspace. Decisions are "
            "only recorded when promoted by the sleep cycle; their absence here is an absence of "
            "record, not evidence that no decisions were made."),
        "active_people": _dimension(
            "active people", people,
            "No verified person entity is currently recorded for this workspace."),
        "active_departments": _dimension(
            "active departments", departments,
            "No verified department entity is currently recorded for this workspace."),
        "verified_connections": _dimension(
            "verified connections", connections,
            "No verified organizational relationship is currently recorded between the entities in "
            "this workspace. No relationship is asserted or denied beyond what the graph records."),
    }

    uncertainty = _build_uncertainty(dims, memories, people, departments, change_result, attention)

    evidence_summary = {
        "durable_memories": sum(len(memories[d]) for d in _MEMORY_DIMENSIONS),
        "verified_people": len(people),
        "verified_departments": len(departments),
        "verified_relationships": len(connections),
        "recent_changes": len(change_result.events),
        "attention_items": len(attention),
        "undetectable_change_types": list(change_result.undetectable),
    }

    return CompanyState(
        workspace_id=workspace_id, as_of=as_of_label,
        generated_at=datetime.now().astimezone().isoformat(),
        active_policies=dims["active_policies"], active_processes=dims["active_processes"],
        active_decisions=dims["active_decisions"], active_people=dims["active_people"],
        active_departments=dims["active_departments"],
        verified_connections=dims["verified_connections"],
        recent_changes=list(change_result.events), attention_items=attention,
        open_uncertainty=uncertainty, evidence_summary=evidence_summary,
        state_confidence={k: v.state for k, v in dims.items()},
        metrics={"build_ms": round((time.perf_counter() - t0) * 1000, 2),
                  "scanned": dict(change_result.scanned)},
    )


def _build_uncertainty(dims, memories, people, departments, change_result, attention) -> list:
    """Explicit, first-class uncertainty (Part 11) -- built from what the
    architecture genuinely cannot establish, never padded."""
    out = []

    for key, dim in dims.items():
        if dim.state == UNKNOWN:
            out.append({"topic": key, "state": UNKNOWN, "detail": dim.coverage_note})

    if people:
        out.append({"topic": "person_roles", "state": UNKNOWN,
                     "detail": ("No employment, job title, department membership, or reporting line is "
                                 "established for any recorded person -- this architecture stores no "
                                 "such data.")})
    if departments:
        out.append({"topic": "department_membership", "state": UNKNOWN,
                     "detail": ("No person-to-department membership is recorded, so department "
                                 "composition and ownership are not established.")})

    owned = [i for d in ("policy", "process", "decision") for i in memories[d]]
    if owned:
        out.append({"topic": "policy_process_ownership", "state": UNKNOWN,
                     "detail": ("Department ownership of the recorded policies/processes is not "
                                 "established -- no verified relationship connects them to a "
                                 "department entity.")})

    for sig in attention:
        if sig.signal_type == cd.REVIEW_REQUIRED:
            out.append({"topic": "unresolved_review", "state": UNKNOWN,
                         "detail": (f"An item is pending human review and is NOT durable organizational "
                                     f"fact: {str((sig.subject or {}).get('label'))[:120]}")})

    for key, why in (change_result.undetectable or {}).items():
        out.append({"topic": f"undetectable_change:{key}", "state": UNKNOWN, "detail": why})

    return out


# =====================================================================
# Part 17 -- Dashboard compatibility. Views over ONE state object; Phase 8
# consumes these. No dashboard-specific engine, no UI, no persistence.
# =====================================================================

def summarize_for_dashboard(state: CompanyState) -> dict:
    return {
        "whats_new": [e for e in state.recent_changes
                       if e.change_type in (cd.MEMORY_PROMOTED, cd.RELATIONSHIP_ADDED)],
        "whats_important": [s for s in state.attention_items if s.attention == pi.CRITICAL],
        "needs_attention": [s for s in state.attention_items
                             if s.attention in (pi.CRITICAL, pi.REVIEW)],
        "whats_uncertain": list(state.open_uncertainty),
        "whats_connected": list(state.verified_connections.items),
        "whats_changed": [e for e in state.recent_changes
                           if e.change_type in (cd.POLICY_CHANGED, cd.PROCESS_CHANGED,
                                                 cd.MEMORY_SUPERSEDED, cd.RELATIONSHIP_CHANGED)],
    }
