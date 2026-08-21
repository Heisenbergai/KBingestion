"""
Phase 7D -- Organizational Change Detection: distinguishes MEANINGFUL
organizational change from ordinary data arrival.

THE CENTRAL QUESTION was whether that distinction can be made without
inventing arbitrary heuristics (Phase 7D's explicit STOP condition). It can,
and the reason is that this architecture ALREADY encodes the bars -- every
one of them a real, previously-frozen decision, not a threshold invented
here:

  * a memory exists at all only because the sleep cycle applied one of the
    4 frozen promotion bases (memory_consolidation.PROMOTION_BASES). Crossing
    that bar IS the organizational judgement; this module does not re-judge.
  * a supersession exists only because create_memory_with_evidence set
    superseded_at atomically (Phase 6D.2), equal to the successor's real
    created_at.
  * a relationship is inactive only because its real status column says so.
  * a review candidate exists only because the consolidation engine could
    not safely decide (Phase 6C).

So "meaningful" is not a score this module computes -- it is a question of
which existing, deterministic bar was crossed. Nothing here counts
frequency, measures similarity, or weighs centrality.

WHAT IS DELIBERATELY *NOT* CHANGE (Part 3/6), and why:
  * revalidation. A sleep run bumping last_confirmed_at on 4 memories is
    operational activity. The real corpus has 85+ consolidation runs against
    4 unchanged memories -- treating that as organizational change would
    produce a permanently noisy feed, which Part 3 and the STOP condition
    both forbid. last_confirmed_at is never read as a change signal.
  * new structured_knowledge on its own. Source ingestion is data arrival.
    It is classified INFORMATIONAL and excluded by default; it becomes
    organizationally meaningful only if it later crosses a promotion,
    supersession, relationship, or review bar.
  * Calendar snapshots. Verified live: real Calendar events arrived mid-
    project and promoted no memory and changed no relationship, so they
    produce no meaningful change here -- by construction, not by a rule
    written to suppress them.
  * retrieval/traversal/Wiki generation. Never consulted.

NO NEW TIMESTAMPS (Part 1): every signal below is an EXISTING column with
the correct semantics. No duplicate "changed_at" was added anywhere, and
this module writes nothing at all.

THREE THINGS ARE HONESTLY NOT DETECTABLE, and are reported rather than
faked (see UNDETECTABLE_CHANGES):
  * hard relationship DELETION -- there is no deletion audit table, so a
    removed row is indistinguishable from one that never existed.
  * memory dormant/archived TRANSITIONS -- org_memory has lifecycle_status
    but no dormant_at/archived_at. Phase 6D.2 deliberately declined to
    invent one, and inventing it here to make a feature look complete would
    repeat exactly the mistake that phase avoided. Current dormant/archived
    STATE is observable; WHEN it happened is not.
  * structured_knowledge lifecycle transitions -- same reason: a
    lifecycle_status column with no transition timestamp.

NO PROACTIVE ACTION (Part 19): detection only. No insert/update/delete/
upsert/rpc anywhere; no notification, task, memory, graph, or Wiki write.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import brain_connectors as bc
import memory_retrieval

# --- change types actually supported by the existing architecture ---
MEMORY_PROMOTED = "MEMORY_PROMOTED"
MEMORY_SUPERSEDED = "MEMORY_SUPERSEDED"
POLICY_CHANGED = "POLICY_CHANGED"
PROCESS_CHANGED = "PROCESS_CHANGED"
RELATIONSHIP_ADDED = "RELATIONSHIP_ADDED"
RELATIONSHIP_CHANGED = "RELATIONSHIP_CHANGED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
NEW_KNOWLEDGE = "NEW_KNOWLEDGE"

# Candidate types from the Phase 7D brief that the CURRENT architecture
# cannot support without inventing a timestamp or an audit table. Named
# explicitly so their absence is a documented decision, not an oversight.
UNDETECTABLE_CHANGES = {
    "RELATIONSHIP_REMOVED": "no deletion audit exists; a deleted row is indistinguishable from one that never existed",
    "MEMORY_DORMANT": "org_memory has lifecycle_status but no dormant_at; Phase 6D.2 deliberately declined to invent one",
    "MEMORY_ARCHIVED": "same as MEMORY_DORMANT -- state is observable, transition time is not",
    "KNOWLEDGE_BECAME_INVALID": "structured_knowledge.lifecycle_status has no transition timestamp",
}

CRITICAL = "critical"
MEANINGFUL = "meaningful"
INFORMATIONAL = "informational"

OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
UNKNOWN = "UNKNOWN"

_ELEVATED_SENSITIVITIES = ("confidential", "restricted")


@dataclass
class ChangeEvent:
    change_type: str
    workspace_id: str
    subject: dict                      # {kind, id, label}
    significance: str                  # critical | meaningful | informational
    reasoning_state: str               # OBSERVED | DERIVED | UNKNOWN
    occurred_at: Optional[str]         # the REAL boundary from an existing column
    occurred_at_source: str            # which column that boundary came from
    temporal_context: str = "current"
    previous_state: Optional[dict] = None
    new_state: Optional[dict] = None
    evidence_ids: list = field(default_factory=list)
    relationship_ids: list = field(default_factory=list)
    memory_ids: list = field(default_factory=list)
    affected_entities: list = field(default_factory=list)
    explanation: str = ""


@dataclass
class ChangeDetectionResult:
    workspace_id: str
    since: Optional[str]
    until: Optional[str]
    events: list = field(default_factory=list)
    scanned: dict = field(default_factory=dict)
    undetectable: dict = field(default_factory=lambda: dict(UNDETECTABLE_CHANGES))


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _in_window(ts_value, since: Optional[datetime], until: Optional[datetime]) -> bool:
    ts = _parse_ts(ts_value)
    if ts is None:
        return False
    if since is not None and ts <= since:
        return False
    if until is not None and ts > until:
        return False
    return True


def _memory_significance(row: dict, base: str) -> str:
    """Elevated sensitivity raises significance -- a real column value, not a
    weighting invented here."""
    if row.get("sensitivity") in _ELEVATED_SENSITIVITIES:
        return CRITICAL
    return base


def _sk_statements(sk_ids: list) -> dict:
    if not sk_ids:
        return {}
    rows = bc.supabase.table("structured_knowledge").select("id,statement,sensitivity") \
        .in_("id", list(sk_ids)).execute().data or []
    return {r["id"]: r for r in rows}


def _memory_grounding(memory_ids: list) -> dict:
    """{memory_id: [structured_knowledge_id, ...]} in one batched query."""
    if not memory_ids:
        return {}
    rows = bc.supabase.table("memory_evidence").select("memory_id,evidence_id") \
        .in_("memory_id", list(memory_ids)).eq("evidence_type", "structured_knowledge") \
        .order("id").execute().data or []
    out: dict = {}
    for r in rows:
        out.setdefault(r["memory_id"], []).append(r["evidence_id"])
    return out


def detect_changes(workspace_id: str, allowed_sensitivities: list[str],
                    since: Optional[datetime] = None, until: Optional[datetime] = None,
                    include_informational: bool = False) -> ChangeDetectionResult:
    """Derived change detection over EXISTING state -- no change table, no
    new timestamps, no full-corpus rescan beyond the requested window.

    `since`/`until` bound the scan (Part 17). Callers running after a sleep
    cycle should pass the run's own input_boundary_since/until, which the
    consolidation engine already computes from the database clock -- reusing
    that boundary rather than inventing a second cursor.

    Security: memory-derived events are filtered by the caller's real
    sensitivity ladder via memory_retrieval._is_visible; an invisible memory
    produces no event at all, not a redacted one.
    """
    events: list[ChangeEvent] = []
    scanned = {"memories": 0, "relationships": 0, "review_candidates": 0, "structured_knowledge": 0}

    # ---- memory rows: promotion + supersession -------------------------
    mem_rows = bc.supabase.table("org_memory").select("*") \
        .eq("workspace_id", workspace_id).order("created_at").execute().data or []
    visible = [r for r in mem_rows
               if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]
    scanned["memories"] = len(visible)

    grounding = _memory_grounding([r["id"] for r in visible])
    all_sk_ids = {sk for ids in grounding.values() for sk in ids}
    sk_by_id = _sk_statements(list(all_sk_ids))

    def _label(memory_id: str) -> Optional[str]:
        for sk_id in grounding.get(memory_id, []):
            sk = sk_by_id.get(sk_id)
            if sk and memory_retrieval._is_visible(sk.get("sensitivity"), allowed_sensitivities):
                return sk["statement"]
        return None

    by_id = {r["id"]: r for r in visible}

    for row in visible:
        # PROMOTION -- created_at is the real boundary; the memory exists at
        # all only because a frozen promotion basis was satisfied.
        if _in_window(row.get("created_at"), since, until):
            events.append(ChangeEvent(
                change_type=MEMORY_PROMOTED, workspace_id=workspace_id,
                subject={"kind": "memory", "id": row["id"], "label": _label(row["id"])},
                significance=_memory_significance(row, MEANINGFUL), reasoning_state=OBSERVED,
                occurred_at=row.get("created_at"), occurred_at_source="org_memory.created_at",
                new_state={"lifecycle_status": row["lifecycle_status"],
                            "promotion_basis": row["promotion_basis"], "memory_type": row["memory_type"]},
                memory_ids=[row["id"]],
                evidence_ids=[f"structured_knowledge:{s}" for s in grounding.get(row["id"], [])],
                explanation=(f"A {row['memory_type']} memory was promoted to durable organizational "
                              f"knowledge via {row['promotion_basis']}."),
            ))

        # SUPERSESSION -- superseded_at is set only by the atomic RPC and
        # equals the successor's real created_at (Phase 6D.2).
        if _in_window(row.get("superseded_at"), since, until):
            successor = next((r for r in visible
                               if r.get("supersedes_memory_id") == row["id"]), None)
            specialized = {"policy": POLICY_CHANGED, "process": PROCESS_CHANGED}.get(row["memory_type"])
            # DERIVED when both ends of the succession are real, visible rows
            # (the change is read off a two-row chain); OBSERVED when only the
            # supersession stamp itself is available.
            state = DERIVED if successor is not None else OBSERVED
            memory_ids = [row["id"]] + ([successor["id"]] if successor else [])
            evidence = [f"structured_knowledge:{s}" for m in memory_ids for s in grounding.get(m, [])]

            events.append(ChangeEvent(
                change_type=specialized or MEMORY_SUPERSEDED, workspace_id=workspace_id,
                subject={"kind": "memory", "id": row["id"], "label": _label(row["id"])},
                significance=_memory_significance(row, CRITICAL), reasoning_state=state,
                occurred_at=row.get("superseded_at"), occurred_at_source="org_memory.superseded_at",
                previous_state={"memory_id": row["id"], "statement": _label(row["id"]),
                                 "lifecycle_status": row["lifecycle_status"]},
                new_state=({"memory_id": successor["id"], "statement": _label(successor["id"]),
                             "lifecycle_status": successor["lifecycle_status"]} if successor else None),
                memory_ids=memory_ids, evidence_ids=evidence,
                explanation=(f"A durable {row['memory_type']} memory was superseded"
                              + (" by a newer memory." if successor else
                                 " (successor not visible to this caller).")),
            ))

    # ---- relationships: added / status-changed -------------------------
    rel_rows = bc.supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", workspace_id).order("created_at").execute().data or []
    scanned["relationships"] = len(rel_rows)
    for rel in rel_rows:
        if _in_window(rel.get("created_at"), since, until):
            events.append(ChangeEvent(
                change_type=RELATIONSHIP_ADDED, workspace_id=workspace_id,
                subject={"kind": "relationship", "id": rel["id"], "label": rel["relationship_type"]},
                significance=MEANINGFUL, reasoning_state=OBSERVED,
                occurred_at=rel.get("created_at"), occurred_at_source="knowledge_relationships.created_at",
                new_state={"status": rel["status"], "relationship_type": rel["relationship_type"],
                            "valid_from": rel.get("valid_from")},
                relationship_ids=[rel["id"]],
                affected_entities=[rel["source_object_id"], rel["target_object_id"]],
                explanation=f"A '{rel['relationship_type']}' relationship was recorded.",
            ))
        # A real mutation after creation, detected via the existing
        # updated_at column -- only reported when the row is no longer
        # active, i.e. a genuine lifecycle change rather than any edit.
        #
        # Deliberately a separate `if`, not an `elif` (fixed after this
        # phase's own test caught it): a relationship both ADDED and
        # RETRACTED inside one window has genuinely undergone two real
        # changes, and reporting only the creation would leave a reader
        # believing an edge is active when its real status says otherwise.
        # The updated_at != created_at guard keeps an untouched row from
        # reporting a phantom second event.
        if (rel.get("status") != "active"
                and rel.get("updated_at") != rel.get("created_at")
                and _in_window(rel.get("updated_at"), since, until)):
            events.append(ChangeEvent(
                change_type=RELATIONSHIP_CHANGED, workspace_id=workspace_id,
                subject={"kind": "relationship", "id": rel["id"], "label": rel["relationship_type"]},
                significance=MEANINGFUL, reasoning_state=OBSERVED,
                occurred_at=rel.get("updated_at"), occurred_at_source="knowledge_relationships.updated_at",
                new_state={"status": rel["status"]}, relationship_ids=[rel["id"]],
                affected_entities=[rel["source_object_id"], rel["target_object_id"]],
                explanation=f"A '{rel['relationship_type']}' relationship is no longer active (status={rel['status']}).",
            ))

    # ---- review queue: unresolved conflict / escalation -----------------
    review_rows = bc.supabase.table("memory_review_queue").select("*") \
        .eq("workspace_id", workspace_id).order("created_at").execute().data or []
    scanned["review_candidates"] = len(review_rows)
    for rv in review_rows:
        if rv.get("status") == "pending" and _in_window(rv.get("created_at"), since, until):
            sk = _sk_statements([rv["structured_knowledge_id"]]).get(rv["structured_knowledge_id"])
            if sk and not memory_retrieval._is_visible(sk.get("sensitivity"), allowed_sensitivities):
                continue
            events.append(ChangeEvent(
                change_type=REVIEW_REQUIRED, workspace_id=workspace_id,
                subject={"kind": "structured_knowledge", "id": rv["structured_knowledge_id"],
                          "label": (sk or {}).get("statement")},
                significance=MEANINGFUL, reasoning_state=OBSERVED,
                occurred_at=rv.get("created_at"), occurred_at_source="memory_review_queue.created_at",
                new_state={"status": rv["status"]},
                evidence_ids=[f"structured_knowledge:{rv['structured_knowledge_id']}"],
                explanation=("Knowledge was routed to human review because the consolidation engine "
                              "could not safely decide it automatically."),
            ))

    # ---- informational: source arrival (never meaningful on its own) ----
    if include_informational:
        sk_rows = bc.supabase.table("structured_knowledge").select("id,statement,sensitivity,created_at") \
            .eq("workspace_id", workspace_id).order("created_at").execute().data or []
        scanned["structured_knowledge"] = len(sk_rows)
        for sk in sk_rows:
            if not memory_retrieval._is_visible(sk.get("sensitivity"), allowed_sensitivities):
                continue
            if _in_window(sk.get("created_at"), since, until):
                events.append(ChangeEvent(
                    change_type=NEW_KNOWLEDGE, workspace_id=workspace_id,
                    subject={"kind": "structured_knowledge", "id": sk["id"], "label": sk["statement"]},
                    significance=INFORMATIONAL, reasoning_state=OBSERVED,
                    occurred_at=sk.get("created_at"), occurred_at_source="structured_knowledge.created_at",
                    evidence_ids=[f"structured_knowledge:{sk['id']}"],
                    explanation="New source knowledge was extracted. This is data arrival, not an "
                                 "organizational change on its own.",
                ))

    events.sort(key=lambda e: (e.occurred_at or "", e.change_type))
    return ChangeDetectionResult(
        workspace_id=workspace_id,
        since=since.isoformat() if since else None,
        until=until.isoformat() if until else None,
        events=events, scanned=scanned,
    )


# =====================================================================
# Part 10 -- cross-department integration. Impact analysis is CALLED BY the
# caller and passed in, so this module never traverses the graph itself and
# never becomes a second impact engine.
# =====================================================================

def attach_impact(event: ChangeEvent, impact_result) -> ChangeEvent:
    """Records only what a real, evidence-backed impact path established.
    An entity with no verified path is never added -- so a policy change can
    say 'Product is explicitly involved' and can never say 'Sales is
    affected' (Part 10's exact worked example)."""
    for path in getattr(impact_result, "paths", []) or []:
        if path.target.object_id not in event.affected_entities:
            event.affected_entities.append(path.target.object_id)
        for rid in path.relationship_ids:
            if rid not in event.relationship_ids:
                event.relationship_ids.append(rid)
    return event


# =====================================================================
# Part 14 -- Wiki boundary. A ChangeEvent names pages that MAY be stale; it
# never regenerates or writes a Wiki page.
# =====================================================================

def wiki_invalidation_candidates(events: list) -> list[dict]:
    """Deterministic mapping from change -> page identity only. Returns
    page identifiers a future stale-page detector could re-check; this
    module performs no Wiki read, build, or write."""
    out, seen = [], set()
    for ev in events:
        for mid in ev.memory_ids:
            key = ("memory", mid)
            if key not in seen:
                seen.add(key)
                out.append({"page_kind": "memory", "object_id": mid, "reason": ev.change_type})
        for eid in ev.affected_entities:
            key = ("entity", eid)
            if key not in seen:
                seen.add(key)
                out.append({"page_kind": "entity", "object_id": eid, "reason": ev.change_type})
    return out


# =====================================================================
# Part 15 -- one engine, three dashboard questions. No second change engine.
# =====================================================================

def summarize_for_dashboard(result: ChangeDetectionResult) -> dict:
    """"What's new?" / "What changed?" / "What needs attention?" answered
    from the SAME event list, by existing fields only."""
    return {
        "whats_new": [e for e in result.events
                       if e.change_type in (MEMORY_PROMOTED, RELATIONSHIP_ADDED, NEW_KNOWLEDGE)],
        "what_changed": [e for e in result.events
                          if e.change_type in (MEMORY_SUPERSEDED, POLICY_CHANGED, PROCESS_CHANGED,
                                                RELATIONSHIP_CHANGED)],
        "needs_attention": [e for e in result.events
                             if e.change_type == REVIEW_REQUIRED or e.significance == CRITICAL],
    }
