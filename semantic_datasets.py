"""
Phase 8B -- the semantic dataset registry: the ONLY vocabulary through which
Brain data may reach a dashboard.

WHY A REGISTRY AND NOT A QUERY BUILDER. The browser lives in the App DB
(Supabase + RLS); every fact in this file lives in the Brain DB, which the
browser holds no credentials for and never will. So the API in
dashboard_brain_api.py is the sole bridge, and this registry is the security
and semantic boundary underneath it: a caller names a dataset, fields, a
filter, a group_by and an aggregation, and NOTHING they send is ever
interpolated into a query. Every name is looked up in the tables below and
rejected if absent. There is no free-text SQL, no expression language, and
no aggregation outside the four this file names.

SECURITY BEFORE AGGREGATION (Part 8) is the rule the whole module is shaped
around. Every resolver fetches rows already filtered by the caller's real
sensitivity ceiling, and only then are they counted. A restricted memory is
absent from a low-clearance caller's count -- it is never counted and then
hidden, because "3 (1 hidden)" discloses exactly what the ladder exists to
conceal. This is the same rule Phase 7G applied inside the Brain, carried
outward to the product edge.

THE DATASET LAYER IS AN ADAPTER (Part 6). It re-implements no intelligence.
changes -> change_detection.detect_changes; attention ->
proactive_intelligence.build_signals; company_state ->
company_state.build_company_state; learning ->
organizational_learning.detect_learning; graph reads -> graph_query;
memory reads -> memory_retrieval's own visibility and availability rules.
Where this file does its own SQL it is a JOIN (memory -> grounding
statement, entity -> identifier count), never a judgement.

NO PROJECTS (Part 19). Phase 8A established that no project entity, due
date, milestone, or progress field exists anywhere in either database. A
`projects` dataset is therefore absent from this registry rather than
present-and-empty: `decisions` is empty because nothing has been promoted
yet, which is a fact about the data; `projects` would be empty because the
domain does not exist, which is a fact about the schema. Conflating those
two would let a dashboard imply the company has no projects.

NO LLM (Part 15). Resolvers surface the deterministic explanation fields the
Phase 7 modules already computed. Nothing here calls a model.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import brain_connectors as bc
import memory_retrieval
import graph_query
import change_detection
import proactive_intelligence
import company_state as company_state_mod
import organizational_learning

# ── temporal meanings (Part 4) ────────────────────────────────────────────
# A date field MUST declare which of the Brain's three independent temporal
# concepts it is. Phase 6D.1/6D.2 established them as genuinely separate;
# exposing a generic "date" would let a dashboard group availability and
# claim-validity into one axis and be silently wrong.
TEMPORAL_NONE = "none"
TEMPORAL_AVAILABILITY = "availability"        # created_at -- when KNOVA knew it
TEMPORAL_CLAIM_VALIDITY = "claim_validity"    # valid_from/valid_until -- when the claim held
TEMPORAL_SUCCESSION = "succession"            # superseded_at -- when it was replaced
TEMPORAL_OBSERVATION = "observation"          # captured_at/occurred_at -- when observed

TEMPORAL_MEANINGS = frozenset({
    TEMPORAL_NONE, TEMPORAL_AVAILABILITY, TEMPORAL_CLAIM_VALIDITY,
    TEMPORAL_SUCCESSION, TEMPORAL_OBSERVATION,
})

# ── temporal modes ────────────────────────────────────────────────────────
MODE_CURRENT = "current"
MODE_AS_OF = "as_of"
MODE_WINDOW = "window"

# ── aggregations (Part 5) ─────────────────────────────────────────────────
# No SUM/AVG. There is no quantity in the Brain to sum: summing sensitivity
# levels or averaging promotion bases is meaningless, and offering it would
# invite confident nonsense. count/count_distinct/min/max are the only
# operations that mean something over this data.
AGG_COUNT = "count"
AGG_COUNT_DISTINCT = "count_distinct"
AGG_MIN = "min"
AGG_MAX = "max"
ALLOWED_AGGREGATIONS = frozenset({AGG_COUNT, AGG_COUNT_DISTINCT, AGG_MIN, AGG_MAX})

FILTER_OPERATORS = frozenset({"eq", "neq", "in", "gte", "lte", "contains"})
# Phase 8E. A bucket is a way of READING one temporal field, never a way of
# mixing several: the field the caller groups by already declares which of the
# Brain's temporal concepts it expresses, so bucketing cannot blur
# availability with claim-validity the way a generic "date" axis would.
GROUP_BUCKETS = frozenset({"day", "week", "month", "quarter", "year"})

BUCKET_LABEL = {
    "day": "day", "week": "week", "month": "month",
    "quarter": "quarter", "year": "year",
}

# Percentages are only meaningful over a COUNT. A "percentage" of a minimum
# date is not a quantity, so the registry refuses it rather than rendering a
# number nobody can interpret (Part 6).
PERCENTABLE_AGGREGATIONS = frozenset({AGG_COUNT, AGG_COUNT_DISTINCT})

MAX_TOP_N = 100

# ── change markers (Part 14) ──────────────────────────────────────────────
# Each marker maps onto a REAL change_type. `+30 days` and `AT RISK` are
# deliberately absent: both would require a project/risk model that Phase 8A
# proved does not exist, and a marker with no backing event is decoration.
CHANGE_MARKERS = {
    "MEMORY_PROMOTED": "NEW",
    "NEW_KNOWLEDGE": "NEW",
    "POLICY_CHANGED": "UPDATED",
    "PROCESS_CHANGED": "UPDATED",
    "MEMORY_SUPERSEDED": "SUPERSEDED",
    "REVIEW_REQUIRED": "REVIEW",
    "RELATIONSHIP_ADDED": "LINKED",
    "RELATIONSHIP_CHANGED": "LINKED",
}
VALID_MARKERS = frozenset(set(CHANGE_MARKERS.values()) | {"CRITICAL"})


@dataclass(frozen=True)
class SemanticField:
    key: str
    label: str
    datatype: str                       # string | text | number | date | enum | id
    temporal_meaning: str = TEMPORAL_NONE
    sensitivity_gated: bool = False
    filterable: bool = True
    groupable: bool = False
    aggregatable: bool = False
    allowed_aggregations: tuple = ()
    evidence_path: Optional[str] = None
    drilldown: Optional[str] = None


@dataclass(frozen=True)
class SemanticDataset:
    key: str
    label: str
    description: str
    fields: tuple
    default_visualization: str
    temporal_modes: tuple
    drilldown_target: Optional[str]
    security_note: str
    resolver: Callable
    empty_reason: Optional[str] = None      # honest empty-state text (Part 11)
    not_established: tuple = ()             # what this dataset CANNOT answer

    def field(self, key: str) -> Optional[SemanticField]:
        return next((f for f in self.fields if f.key == key), None)


@dataclass
class SemanticRow:
    """One semantic row. `values` holds ONLY registry-declared field keys --
    a raw database row never leaves this module (Part 12)."""
    values: dict
    object_kind: str
    object_id: Optional[str] = None
    evidence_ids: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    explanation: Optional[str] = None
    reasoning_state: Optional[str] = None
    not_established: list = field(default_factory=list)


@dataclass
class DatasetQueryResult:
    dataset: str
    fields: list
    rows: list
    row_count: int
    temporal_context: str
    temporal_mode: str
    generated_at: str
    aggregation: Optional[dict] = None
    evidence_available: bool = False
    drilldown_target: Optional[str] = None
    not_established: list = field(default_factory=list)
    empty_reason: Optional[str] = None
    notes: list = field(default_factory=list)


class DatasetError(ValueError):
    """Registry rejection. The router turns this into a 400 -- never a 500,
    and never a message that reveals whether an id exists."""


def _parse(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


# =====================================================================
# Shared joins. These are LOOKUPS, not intelligence -- they resolve a
# label/statement for display and apply the same visibility rule
# everything else does.
# =====================================================================

def _memory_statements(memory_ids: list, allowed: list[str]) -> dict:
    """memory -> its grounding statement, for display only.

    Only `supports` evidence is followed, and each structured_knowledge row
    passes the caller's ladder before its text can become a label -- a
    restricted claim must not leak as a memory's display name even when the
    memory itself is visible."""
    if not memory_ids:
        return {}
    ev = bc.supabase.table("memory_evidence").select("memory_id,evidence_id") \
        .in_("memory_id", list(memory_ids)).eq("evidence_type", "structured_knowledge") \
        .eq("stance", "supports").order("id").execute().data or []
    sk_ids = sorted({e["evidence_id"] for e in ev})
    if not sk_ids:
        return {}
    sk = bc.supabase.table("structured_knowledge").select("id,statement,sensitivity") \
        .in_("id", sk_ids).execute().data or []
    visible = {r["id"]: r["statement"] for r in sk
               if memory_retrieval._is_visible(r.get("sensitivity"), allowed)}
    out: dict = {}
    for e in ev:
        if e["memory_id"] not in out and e["evidence_id"] in visible:
            out[e["memory_id"]] = visible[e["evidence_id"]]
    return out


def _memory_evidence_ids(memory_ids: list) -> dict:
    if not memory_ids:
        return {}
    rows = bc.supabase.table("memory_evidence").select("memory_id,evidence_id,evidence_type") \
        .in_("memory_id", list(memory_ids)).eq("stance", "supports").execute().data or []
    out: dict = {}
    for r in rows:
        out.setdefault(r["memory_id"], []).append(f"{r['evidence_type']}:{r['evidence_id']}")
    return out


def _entities(workspace_id: str, entity_type: str, as_of: Optional[datetime]) -> list[dict]:
    """Entity availability reuses Phase 7B's established rule verbatim:
    knowledge_entities has no valid_from/valid_until, only created_at, so
    availability at as_of is created_at <= as_of. No new temporal rule."""
    rows = bc.supabase.table("knowledge_entities").select("*") \
        .eq("workspace_id", workspace_id).eq("entity_type", entity_type) \
        .eq("status", "active").execute().data or []
    if as_of is not None:
        rows = [r for r in rows
                if (_parse(r.get("created_at")) or datetime.max.replace(tzinfo=timezone.utc)) <= as_of]
    return rows


def _relationship_counts(workspace_id: str, entity_ids: list) -> dict:
    if not entity_ids:
        return {}
    rows = bc.supabase.table("knowledge_relationships").select("source_object_id,target_object_id") \
        .eq("workspace_id", workspace_id).eq("status", "active").execute().data or []
    counts = {e: 0 for e in entity_ids}
    for r in rows:
        for k in (r.get("source_object_id"), r.get("target_object_id")):
            if k in counts:
                counts[k] += 1
    return counts


# =====================================================================
# Resolvers. Each returns list[SemanticRow] ALREADY filtered by the
# caller's sensitivity ceiling.
# =====================================================================

def _resolve_memories(memory_type: Optional[str]):
    def resolver(ctx) -> list:
        rows = memory_retrieval._fetch_memory_rows(ctx.workspace_id, ctx.as_of)
        rows = [r for r in rows
                if memory_retrieval._is_visible(r.get("sensitivity"), ctx.allowed)]
        if memory_type:
            rows = [r for r in rows if r.get("memory_type") == memory_type]
        ids = [r["id"] for r in rows]
        statements = _memory_statements(ids, ctx.allowed)
        evidence = _memory_evidence_ids(ids)
        out = []
        for r in rows:
            ev = evidence.get(r["id"], [])
            out.append(SemanticRow(
                values={
                    "memory_id": r["id"],
                    "statement": statements.get(r["id"]),
                    "memory_type": r.get("memory_type"),
                    "sensitivity": r.get("sensitivity"),
                    "promotion_basis": r.get("promotion_basis"),
                    "lifecycle_status": r.get("lifecycle_status"),
                    "created_at": r.get("created_at"),
                    "valid_from": r.get("valid_from"),
                    "valid_until": r.get("valid_until"),
                    "superseded_at": r.get("superseded_at"),
                    "last_confirmed_at": r.get("last_confirmed_at"),
                    "evidence_count": len(ev),
                },
                object_kind="memory", object_id=r["id"], evidence_ids=ev,
                not_established=([] if statements.get(r["id"]) else
                                  ["Statement not shown: its grounding evidence is outside "
                                   "your access level."]),
            ))
        return out
    return resolver


def _resolve_departments(ctx) -> list:
    rows = _entities(ctx.workspace_id, "department", ctx.as_of)
    counts = _relationship_counts(ctx.workspace_id, [r["id"] for r in rows])
    app_names = ctx.app_departments or {}
    out = []
    for r in rows:
        app_id = r.get("external_ref_id") if r.get("external_ref_type") == "department_id" else None
        app = app_names.get(app_id) if app_id else None
        ne = []
        if app_id and app is None:
            ne.append("Workspace department record not resolved (cross-database lookup "
                      "unavailable for this request); Brain-side data shown.")
        out.append(SemanticRow(
            values={
                "entity_id": r["id"], "label": r.get("canonical_label"),
                "status": r.get("status"), "created_at": r.get("created_at"),
                "app_department_id": app_id,
                "app_department_name": (app or {}).get("name"),
                "parent_department_name": (app or {}).get("parent_name"),
                "relationship_count": counts.get(r["id"], 0),
            },
            object_kind="entity", object_id=r["id"], not_established=ne,
        ))
    return out


def _resolve_people(ctx) -> list:
    """Deliberately exposes NO department field (Part 10). member_departments
    is empty across the entire database, and person entities carry no
    external_ref -- so a person's department is genuinely unknown, and a
    column of nulls would read as "unassigned" rather than "unknown"."""
    rows = _entities(ctx.workspace_id, "person", ctx.as_of)
    ids = [r["id"] for r in rows]
    counts = _relationship_counts(ctx.workspace_id, ids)
    idents = bc.supabase.table("knowledge_entity_identifiers") \
        .select("entity_id,identifier_type").eq("workspace_id", ctx.workspace_id) \
        .execute().data if ids else []
    ident_counts: dict = {}
    for i in (idents or []):
        if i.get("identifier_type") == "email":
            ident_counts[i["entity_id"]] = ident_counts.get(i["entity_id"], 0) + 1
    return [SemanticRow(
        values={"entity_id": r["id"], "label": r.get("canonical_label"),
                "status": r.get("status"), "created_at": r.get("created_at"),
                "email_identifier_count": ident_counts.get(r["id"], 0),
                "relationship_count": counts.get(r["id"], 0)},
        object_kind="entity", object_id=r["id"],
        not_established=["Department membership is not established for any person: "
                          "no member-department records exist."],
    ) for r in rows]


def _resolve_meetings(ctx) -> list:
    rows = _entities(ctx.workspace_id, "meeting", ctx.as_of)
    return [SemanticRow(
        values={"entity_id": r["id"], "label": r.get("canonical_label"),
                "status": r.get("status"), "created_at": r.get("created_at")},
        object_kind="entity", object_id=r["id"],
    ) for r in rows]


def _resolve_relationships(ctx) -> list:
    """Visibility is delegated to graph_query.get_relationship, which returns
    None when the caller cannot see a relationship's evidence. Re-deriving
    that rule here would be a second copy of a security check -- so this
    calls the audited one per id instead. That is N+1 by construction and
    accepted deliberately: correctness of the visibility rule outranks a
    round-trip on a 3-row graph, and the cost is reported honestly in the
    performance section rather than hidden."""
    ids = bc.supabase.table("knowledge_relationships").select("id") \
        .eq("workspace_id", ctx.workspace_id).eq("status", "active").execute().data or []
    out = []
    for row in ids:
        rel = graph_query.get_relationship(row["id"], ctx.workspace_id, ctx.allowed)
        if rel is None:
            continue
        if ctx.as_of is not None:
            vf = _parse(rel.valid_from)
            if vf and vf > ctx.as_of:
                continue
        out.append(SemanticRow(
            values={
                "relationship_id": rel.id, "relationship_type": rel.relationship_type,
                "status": rel.status, "source_label": rel.source.label,
                "target_label": rel.target.label, "valid_from": rel.valid_from,
                "valid_until": rel.valid_until, "rationale": rel.rationale,
                "evidence_count": len(rel.evidence),
            },
            object_kind="relationship", object_id=rel.id,
            evidence_ids=[f"{e.evidence_type}:{e.evidence_id}" for e in rel.evidence],
        ))
    return out


def _resolve_changes(ctx) -> list:
    result = change_detection.detect_changes(
        ctx.workspace_id, ctx.allowed, since=ctx.since, until=ctx.until,
        include_informational=True)
    out = []
    for e in result.events:
        markers = []
        m = CHANGE_MARKERS.get(e.change_type)
        if m:
            markers.append(m)
        if e.significance == "critical":
            markers.append("CRITICAL")
        out.append(SemanticRow(
            values={
                "change_type": e.change_type, "significance": e.significance,
                "subject_label": (e.subject or {}).get("label"),
                "subject_kind": (e.subject or {}).get("kind"),
                "occurred_at": e.occurred_at, "occurred_at_source": e.occurred_at_source,
                "reasoning_state": e.reasoning_state,
                "evidence_count": len(e.evidence_ids or []),
            },
            object_kind=(e.subject or {}).get("kind") or "change",
            object_id=(e.subject or {}).get("id"),
            evidence_ids=list(e.evidence_ids or []), markers=markers,
            explanation=e.explanation, reasoning_state=e.reasoning_state,
        ))
    return out


def _resolve_attention(ctx) -> list:
    changes = change_detection.detect_changes(
        ctx.workspace_id, ctx.allowed, since=ctx.since, until=ctx.until,
        include_informational=True)
    # chat_json_fn omitted on purpose: no LLM in a dataset resolver (Part 15).
    signals = proactive_intelligence.build_signals(changes, ctx.workspace_id)
    out = []
    for s in signals:
        markers = ["CRITICAL"] if s.attention == "CRITICAL" else []
        audience = getattr(s.audience, "status", None) if s.audience else None
        out.append(SemanticRow(
            values={
                "signal_id": s.signal_id, "signal_type": s.signal_type,
                "attention": s.attention,
                "subject_label": (s.subject or {}).get("label"),
                "reasoning_state": s.reasoning_state,
                "is_hypothesis": s.is_hypothesis,
                "recommendation_state": s.recommendation_state,
                "audience_status": audience,
                "expires_at": s.expires_at,
                "evidence_count": len(s.evidence_ids or []),
            },
            object_kind="signal", object_id=s.signal_id,
            evidence_ids=list(s.evidence_ids or []), markers=markers,
            explanation=s.explanation, reasoning_state=s.reasoning_state,
            not_established=([] if audience == "audience_established"
                              else ["Audience not established for this signal."]),
        ))
    return out


def _resolve_calendar(ctx) -> list:
    """Calendar events the CALLER personally organized or was invited to.

    WHY PARTICIPATION IS THE AUTHORIZATION HERE, AND NOT THE SENSITIVITY
    LADDER. `calendar_event_snapshots` has no `sensitivity` column, and neither
    does `knowledge_entities`, so there is nothing to derive a level from --
    verified, not assumed. The obvious workaround, classifying all calendar
    data as `internal`, is worse than it looks: `internal` is inside the
    ceiling of EVERY authenticated member, so it would hand every meeting
    title and attendee list in the workspace to everyone. Meeting titles are
    routinely among the most sensitive text a company holds.

    Participation is a rule the data can actually support and a person can
    actually check: you see a meeting if you organized it or were invited to
    it. It cannot over-disclose, because the invite list is the company's own
    statement of who was meant to know.

    `caller_emails` is resolved server-side from the verified token. It is
    never a request parameter, so a client cannot ask to be someone else. When
    identity cannot be established the answer is NO ROWS -- failing closed,
    because the alternative on an unidentified caller is disclosing everyone's
    calendar.
    """
    emails = {e.strip().lower() for e in (ctx.caller_emails or []) if isinstance(e, str) and e.strip()}
    if not emails:
        return []

    rows = bc.supabase.table("calendar_event_snapshots").select(
        "id, title, start_time, end_time, organizer, attendees, meeting_url, "
        "recurrence_rule, external_event_id, conference_id, captured_at"
    ).eq("workspace_id", ctx.workspace_id).order("start_time", desc=True).execute().data or []

    # A meeting that KNOVA also holds as a graph ENTITY is one it may know
    # real, evidence-backed things about -- who organized it, who attended.
    # The link is the connector's own identifier (the conference id or the
    # provider's event id), which is why it is trustworthy: it is the same
    # value both sides were built from, not a title match.
    #
    # Resolved in ONE batch rather than per event. Events with no linked
    # entity simply carry None, and the UI says so rather than implying
    # KNOVA knows more than it does.
    idents = bc.supabase.table("knowledge_entity_identifiers").select(
        "entity_id, identifier_type, identifier_value"
    ).eq("workspace_id", ctx.workspace_id) \
        .in_("identifier_type", ["conference_id", "external_event_id"]).execute().data or []
    entity_by_key = {str(i["identifier_value"]): i["entity_id"]
                     for i in idents if i.get("identifier_value")}

    # One snapshot per (event, state) — an event edited three times has three
    # rows. Keep only the most recently captured state per event, so the
    # calendar shows what is true now rather than every revision.
    latest = {}
    for r in rows:
        key = r.get("external_event_id") or r["id"]
        prev = latest.get(key)
        if prev is None or str(r.get("captured_at") or "") > str(prev.get("captured_at") or ""):
            latest[key] = r

    out = []
    for r in latest.values():
        organizer = (r.get("organizer") or "").strip().lower()
        attendee_emails = set()
        for a in (r.get("attendees") or []):
            if isinstance(a, dict):
                em = a.get("email")
                if isinstance(em, str):
                    attendee_emails.add(em.strip().lower())
            elif isinstance(a, str):
                attendee_emails.add(a.strip().lower())

        if organizer not in emails and not (attendee_emails & emails):
            continue

        start = r.get("start_time")
        linked_entity = (entity_by_key.get(str(r.get("conference_id")))
                          or entity_by_key.get(str(r.get("external_event_id"))))

        out.append(SemanticRow(
            values={
                "event_id": r["id"],
                "title": r.get("title"),
                "start_time": start,
                "end_time": r.get("end_time"),
                "organizer": r.get("organizer"),
                # A COUNT, not the list. The widget needs to say "4 people";
                # publishing every address would disclose the full invite list
                # to anyone who happens to be on it, which is more than
                # participation justifies.
                "attendee_count": len(attendee_emails),
                "is_organizer": organizer in emails,
                "meeting_url": r.get("meeting_url"),
                "is_recurring": bool(r.get("recurrence_rule")),
                # Present only when this meeting also exists as a graph
                # entity. That is what lets the UI offer the real evidence
                # chain (who organized, who attended) instead of asserting
                # KNOVA knows nothing -- and its ABSENCE is equally honest,
                # because most meetings genuinely have no captured knowledge.
                "linked_entity_id": linked_entity,
                "has_knowledge": bool(linked_entity),
            },
            # When the meeting IS in the graph, the row drills into the
            # ENTITY through the existing Phase 8D experience -- no second
            # meeting-intelligence system. When it is not, no target is
            # claimed rather than offering a link that dead-ends.
            object_kind="entity" if linked_entity else "calendar_event",
            object_id=linked_entity or r["id"],
        ))
    return out


def _resolve_company_state(ctx) -> list:
    st = company_state_mod.build_company_state(ctx.workspace_id, ctx.allowed, as_of=ctx.as_of)
    out = []
    for name in ("active_policies", "active_processes", "active_decisions",
                  "active_people", "active_departments", "verified_connections"):
        dim = getattr(st, name)
        for item in dim.items:
            out.append(SemanticRow(
                values={
                    "dimension": name, "dimension_state": dim.state,
                    "item_kind": item.kind, "item_label": item.label,
                    "item_state": item.state,
                    "evidence_count": len(item.evidence_ids or []),
                },
                object_kind=item.kind, object_id=item.object_id,
                evidence_ids=list(item.evidence_ids or []),
                reasoning_state=item.state,
                not_established=list(item.not_established or []),
            ))
    return out


def _resolve_learning(ctx) -> list:
    res = organizational_learning.detect_learning(ctx.workspace_id, ctx.allowed, as_of=ctx.as_of)
    return [SemanticRow(
        values={
            "learning_type": s.learning_type,
            "subject_label": (s.subject or {}).get("label"),
            "reasoning_state": s.reasoning_state, "support_count": s.support_count,
            "window_start": (s.observation_window or {}).get("start"),
            "window_end": (s.observation_window or {}).get("end"),
            "review_required": s.review_required,
            "evidence_count": len(s.evidence_ids or []),
        },
        object_kind=(s.subject or {}).get("kind") or "learning",
        object_id=(s.subject or {}).get("id"),
        evidence_ids=list(s.evidence_ids or []),
        markers=(["REVIEW"] if s.review_required else []),
        explanation=s.explanation, reasoning_state=s.reasoning_state,
    ) for s in res.signals]


def _resolve_evidence(ctx) -> list:
    rows = bc.supabase.table("structured_knowledge").select(
        "id,statement,sensitivity,authority,source_tier,captured_at,provider,"
        "canonical_source_type,requirement_kind,recurrence_text,lifecycle_status") \
        .eq("workspace_id", ctx.workspace_id).execute().data or []
    rows = [r for r in rows
            if memory_retrieval._is_visible(r.get("sensitivity"), ctx.allowed)]
    if ctx.as_of is not None:
        rows = [r for r in rows
                if (_parse(r.get("captured_at")) or datetime.max.replace(tzinfo=timezone.utc))
                <= ctx.as_of]
    return [SemanticRow(
        values={k: r.get(k) for k in
                ("id", "statement", "sensitivity", "authority", "source_tier",
                 "captured_at", "provider", "canonical_source_type",
                 "requirement_kind", "recurrence_text")} | {"evidence_id": r["id"]},
        object_kind="structured_knowledge", object_id=r["id"],
        evidence_ids=[f"structured_knowledge:{r['id']}"],
    ) for r in rows]


# =====================================================================
# Field sets.
# =====================================================================

_MEMORY_FIELDS = (
    SemanticField("memory_id", "Memory ID", "id", groupable=False,
                  aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT),
                  evidence_path="org_memory.id", drilldown="memory"),
    SemanticField("statement", "Statement", "text", sensitivity_gated=True,
                  evidence_path="memory_evidence -> structured_knowledge.statement"),
    SemanticField("memory_type", "Type", "enum", groupable=True, aggregatable=True,
                  allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("sensitivity", "Sensitivity", "enum", sensitivity_gated=True,
                  groupable=True, aggregatable=True,
                  allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("promotion_basis", "Promotion basis", "enum", groupable=True,
                  aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("lifecycle_status", "Lifecycle", "enum", groupable=True,
                  aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("created_at", "Became known", "date", TEMPORAL_AVAILABILITY,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
    SemanticField("valid_from", "Claim valid from", "date", TEMPORAL_CLAIM_VALIDITY,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
    SemanticField("valid_until", "Claim valid until", "date", TEMPORAL_CLAIM_VALIDITY,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
    SemanticField("superseded_at", "Superseded", "date", TEMPORAL_SUCCESSION,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
    SemanticField("last_confirmed_at", "Last re-confirmed", "date", TEMPORAL_OBSERVATION,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
    SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                  allowed_aggregations=(AGG_MIN, AGG_MAX)),
)

_ENTITY_BASE = (
    SemanticField("entity_id", "Entity ID", "id", aggregatable=True,
                  allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT), drilldown="entity"),
    SemanticField("label", "Name", "string", groupable=True, aggregatable=True,
                  allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("status", "Status", "enum", groupable=True, aggregatable=True,
                  allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    SemanticField("created_at", "First seen", "date", TEMPORAL_AVAILABILITY,
                  groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
)


def _memory_dataset(key, label, description, memory_type, empty_reason=None):
    return SemanticDataset(
        key=key, label=label, description=description, fields=_MEMORY_FIELDS,
        default_visualization="table", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
        drilldown_target="memory",
        security_note=("org_memory rows are filtered by the caller's sensitivity ceiling "
                        "before any count is taken; grounding statements are filtered "
                        "independently."),
        resolver=_resolve_memories(memory_type), empty_reason=empty_reason,
    )


DATASETS: dict = {}


def _register(ds: SemanticDataset):
    DATASETS[ds.key] = ds


_register(_memory_dataset(
    "policies", "Policies", "Durable policy memories with full temporal history.", "policy"))
_register(_memory_dataset(
    "processes", "Processes", "Durable recurring-process memories.", "process"))
_register(_memory_dataset(
    "decisions", "Decisions", "Durable decision memories.", "decision",
    empty_reason="No decisions have been promoted to durable memory."))
_register(_memory_dataset(
    "memories", "Memories", "All durable organizational memory, any type.", None))

_register(SemanticDataset(
    key="departments", label="Departments",
    description="Department entities observed in company knowledge, joined to the "
                 "workspace's own department records where possible.",
    fields=_ENTITY_BASE + (
        SemanticField("app_department_id", "Workspace department ID", "id",
                      evidence_path="knowledge_entities.external_ref_id -> departments.id"),
        SemanticField("app_department_name", "Workspace department", "string",
                      groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("parent_department_name", "Parent department", "string",
                      groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("relationship_count", "Connections", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="bar", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="entity",
    security_note="Cross-database department names are resolved server-side using the "
                   "caller's own token, so App-DB RLS decides. No service key is used.",
    resolver=_resolve_departments,
    not_established=("Only departments that appear in company knowledge have Brain "
                      "entities; other workspace departments are not represented here.",),
))

_register(SemanticDataset(
    key="people", label="People",
    description="Person entities observed in company knowledge.",
    fields=_ENTITY_BASE + (
        SemanticField("email_identifier_count", "Email identifiers", "number",
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("relationship_count", "Connections", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="table", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="entity",
    security_note="Entity rows are workspace-scoped; identifiers are never returned as "
                   "values, only counted.",
    resolver=_resolve_people,
    not_established=("Department membership is not established for any person: no "
                      "member-department records exist.",
                      "Headcount is not available: person entities are those observed in "
                      "knowledge, not the workspace member list.",),
))

_register(SemanticDataset(
    key="meetings", label="Meetings",
    description="Meeting entities observed in company knowledge.",
    fields=_ENTITY_BASE, default_visualization="table",
    temporal_modes=(MODE_CURRENT, MODE_AS_OF), drilldown_target="entity",
    security_note="Workspace-scoped entity read.",
    resolver=_resolve_meetings,
))

_register(SemanticDataset(
    key="relationships", label="Relationships",
    description="Verified relationships between entities, with evidence counts.",
    fields=(
        SemanticField("relationship_id", "Relationship ID", "id", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT),
                      drilldown="relationship"),
        SemanticField("relationship_type", "Type", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("status", "Status", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("source_label", "From", "string", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("target_label", "To", "string", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("valid_from", "Valid from", "date", TEMPORAL_CLAIM_VALIDITY,
                      groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("valid_until", "Valid until", "date", TEMPORAL_CLAIM_VALIDITY,
                      groupable=True, aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("rationale", "Rationale", "text"),
        SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="bar", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="relationship",
    security_note="Each relationship's visibility is decided by graph_query, which hides "
                   "any relationship whose evidence the caller cannot see.",
    resolver=_resolve_relationships,
))

_register(SemanticDataset(
    key="changes", label="Changes",
    description="Meaningful organizational changes detected over a time window.",
    fields=(
        SemanticField("change_type", "Change", "enum", TEMPORAL_NONE, groupable=True,
                      aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("significance", "Significance", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("subject_label", "Subject", "string", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("subject_kind", "Subject kind", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("occurred_at", "Occurred", "date", TEMPORAL_OBSERVATION, groupable=True,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("occurred_at_source", "Timestamp source", "enum", groupable=True),
        SemanticField("reasoning_state", "Certainty", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="bar", temporal_modes=(MODE_CURRENT, MODE_WINDOW),
    drilldown_target="memory",
    security_note="detect_changes filters by sensitivity before an event is ever emitted.",
    resolver=_resolve_changes,
    not_established=tuple(
        f"{k}: {v}" for k, v in change_detection.UNDETECTABLE_CHANGES.items()),
))

_register(SemanticDataset(
    key="attention", label="Attention",
    description="Proactive signals derived from detected change.",
    fields=(
        SemanticField("signal_id", "Signal ID", "id", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("signal_type", "Signal", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("attention", "Attention level", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("subject_label", "Subject", "string", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("reasoning_state", "Certainty", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("is_hypothesis", "Is hypothesis", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("recommendation_state", "Recommendation", "enum", groupable=True),
        SemanticField("audience_status", "Audience", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("expires_at", "Expires", "date", TEMPORAL_OBSERVATION,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="table", temporal_modes=(MODE_CURRENT, MODE_WINDOW),
    drilldown_target="memory",
    security_note="Signals derive from already-filtered change events; audience is never "
                   "guessed.",
    resolver=_resolve_attention,
))

_register(SemanticDataset(
    key="company_state", label="Company state",
    description="Derived executive state across six dimensions.",
    fields=(
        SemanticField("dimension", "Dimension", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("dimension_state", "Dimension certainty", "enum", groupable=True),
        SemanticField("item_kind", "Item kind", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("item_label", "Item", "string", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("item_state", "Certainty", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="table", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="memory",
    security_note="build_company_state applies the ladder per dimension before assembly.",
    resolver=_resolve_company_state,
))

_register(SemanticDataset(
    key="calendar", label="My calendar",
    description="Meetings you organized or were invited to.",
    fields=(
        SemanticField("event_id", "Event ID", "id", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("title", "Title", "string", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("start_time", "Starts", "date", TEMPORAL_OBSERVATION,
                      groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("end_time", "Ends", "date", TEMPORAL_OBSERVATION,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("organizer", "Organizer", "string", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("attendee_count", "Attendees", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("is_organizer", "You organized", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("meeting_url", "Meeting link", "string"),
        SemanticField("is_recurring", "Recurring", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("linked_entity_id", "Knowledge graph entity", "id"),
        SemanticField("has_knowledge", "KNOVA holds knowledge", "enum",
                      groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
    ),
    default_visualization="table", temporal_modes=(MODE_CURRENT,),
    # Drills into the meeting ENTITY where one exists, reusing the Phase 8D
    # experience. Rows with no linked entity carry no drillable id, so the
    # link is offered only where it genuinely leads somewhere.
    drilldown_target="entity",
    security_note=(
        "Authorized by PARTICIPATION, not by sensitivity level: you see only "
        "meetings you organized or were invited to. calendar_event_snapshots "
        "carries no sensitivity column, and classifying it 'internal' would "
        "expose every meeting title in the workspace to every member."),
    resolver=_resolve_calendar,
    not_established=(
        "Meeting outcomes: a calendar event records that a meeting was "
        "scheduled, never what was decided in it. Minutes, decisions and "
        "action items appear only where they were separately captured and "
        "promoted to memory.",
    ),
))

_register(SemanticDataset(
    key="learning", label="Organizational learning",
    description="Longitudinal patterns detected over real history.",
    fields=(
        SemanticField("learning_type", "Pattern", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("subject_label", "Subject", "string", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("reasoning_state", "Certainty", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("support_count", "Observations", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("window_start", "Observed from", "date", TEMPORAL_OBSERVATION,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("window_end", "Observed to", "date", TEMPORAL_OBSERVATION,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("review_required", "Needs review", "enum", groupable=True,
                      aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("evidence_count", "Evidence items", "number", aggregatable=True,
                      allowed_aggregations=(AGG_MIN, AGG_MAX)),
    ),
    default_visualization="table", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="memory",
    security_note="detect_learning applies the ladder before any pattern is aggregated.",
    resolver=_resolve_learning,
    not_established=tuple(f"{k}: {v}" for k, v in
                           organizational_learning.REJECTED_LEARNING_TYPES.items()),
))

_register(SemanticDataset(
    key="evidence", label="Evidence",
    description="Structured knowledge claims extracted from company sources.",
    fields=(
        SemanticField("evidence_id", "Evidence ID", "id", aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT),
                      drilldown="structured_knowledge"),
        SemanticField("statement", "Statement", "text", sensitivity_gated=True),
        SemanticField("sensitivity", "Sensitivity", "enum", sensitivity_gated=True,
                      groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("authority", "Authority", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("source_tier", "Source tier", "number", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_MIN, AGG_MAX)),
        SemanticField("provider", "Provider", "enum", groupable=True, aggregatable=True,
                      allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("canonical_source_type", "Source type", "enum", groupable=True,
                      aggregatable=True, allowed_aggregations=(AGG_COUNT, AGG_COUNT_DISTINCT)),
        SemanticField("captured_at", "Captured", "date", TEMPORAL_OBSERVATION, groupable=True,
                      aggregatable=True, allowed_aggregations=(AGG_MIN, AGG_MAX)),
        SemanticField("requirement_kind", "Requirement kind", "enum", groupable=True),
        SemanticField("recurrence_text", "Recurrence", "string"),
    ),
    default_visualization="bar", temporal_modes=(MODE_CURRENT, MODE_AS_OF),
    drilldown_target="structured_knowledge",
    security_note="Claims are filtered by the caller's ceiling before counting.",
    resolver=_resolve_evidence,
))

# Guard: `projects` must never appear (Part 19). Asserted at import so a
# future careless addition fails loudly rather than shipping a fabricated
# domain into the product.
assert "projects" not in DATASETS, "Projects are not modelled; see Phase 8A section 4."


@dataclass
class ResolveContext:
    workspace_id: str
    allowed: list
    as_of: Optional[datetime] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    app_departments: Optional[dict] = None
    # The CALLER'S OWN email addresses, resolved server-side from the verified
    # token (never sent by the client). Used by the calendar dataset, where
    # authorization is PARTICIPATION rather than a sensitivity level -- see
    # _resolve_calendar. None means "could not establish who is asking", which
    # that resolver treats as "show nothing".
    caller_emails: Optional[list] = None


# =====================================================================
# Validation + execution.
# =====================================================================

def list_datasets() -> list:
    """Registry description for a builder UI. Contains no workspace data, so
    it is safe for any authenticated caller."""
    out = []
    for ds in DATASETS.values():
        out.append({
            "key": ds.key, "label": ds.label, "description": ds.description,
            "default_visualization": ds.default_visualization,
            "temporal_modes": list(ds.temporal_modes),
            "drilldown_target": ds.drilldown_target,
            "security_note": ds.security_note,
            "not_established": list(ds.not_established),
            "fields": [{
                "key": f.key, "label": f.label, "datatype": f.datatype,
                "temporal_meaning": f.temporal_meaning,
                "sensitivity_gated": f.sensitivity_gated, "filterable": f.filterable,
                "groupable": f.groupable, "aggregatable": f.aggregatable,
                "allowed_aggregations": list(f.allowed_aggregations),
                "evidence_path": f.evidence_path, "drilldown": f.drilldown,
            } for f in ds.fields],
        })
    return out


def get_dataset(key: str) -> SemanticDataset:
    ds = DATASETS.get(key)
    if ds is None:
        raise DatasetError(f"Unknown dataset: {key!r}.")
    return ds


def _validate_filters(ds: SemanticDataset, filters: list) -> list:
    out = []
    for f in filters or []:
        if not isinstance(f, dict):
            raise DatasetError("Each filter must be an object.")
        fkey, op = f.get("field"), f.get("op", "eq")
        sf = ds.field(fkey)
        if sf is None:
            raise DatasetError(f"Unknown field {fkey!r} for dataset {ds.key!r}.")
        if not sf.filterable:
            raise DatasetError(f"Field {fkey!r} is not filterable.")
        if op not in FILTER_OPERATORS:
            raise DatasetError(f"Unsupported filter operator {op!r}.")
        out.append({"field": fkey, "op": op, "value": f.get("value")})
    return out


def _apply_filters(rows: list, filters: list) -> list:
    def keep(row):
        for f in filters:
            v, want, op = row.values.get(f["field"]), f["value"], f["op"]
            if op == "eq" and v != want:
                return False
            if op == "neq" and v == want:
                return False
            if op == "in" and v not in (want or []):
                return False
            if op == "gte" and not (v is not None and str(v) >= str(want)):
                return False
            if op == "lte" and not (v is not None and str(v) <= str(want)):
                return False
            if op == "contains" and (v is None or str(want).lower() not in str(v).lower()):
                return False
        return True
    return [r for r in rows if keep(r)]


def _bucket(value, bucket: str):
    """Reduces one timestamp to its bucket key.

    ISO-8601 prefixes give day/month/year for free; week and quarter are
    derived from the real date rather than guessed from the string, because
    "2026-08" tells you nothing about which week a day falls in. A value that
    will not parse is bucketed as None (rendered "Not set") rather than being
    silently dropped -- a row with no date is a real row."""
    if value is None:
        return None
    s = str(value)
    if bucket == "day":
        return s[:10]
    if bucket == "month":
        return s[:7]
    if bucket == "year":
        return s[:4]
    dt = _parse(s)
    if dt is None:
        return None
    if bucket == "week":
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if bucket == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    return s


def _validate_group_field(ds: SemanticDataset, key: str, bucket, role: str):
    """One rule for the primary group and the series, so a second dimension
    can never be looser than the first."""
    f = ds.field(key)
    if f is None:
        raise DatasetError(f"Unknown field {key!r} for dataset {ds.key!r}.")
    if not f.groupable:
        raise DatasetError(f"Field {key!r} is not groupable, so it cannot be used as {role}.")
    if f.datatype == "date":
        if bucket not in GROUP_BUCKETS:
            raise DatasetError(
                f"Grouping by the date field {key!r} requires group_bucket "
                f"({', '.join(sorted(GROUP_BUCKETS))}). Its temporal meaning is "
                f"{f.temporal_meaning!r}.")
    elif bucket is not None:
        raise DatasetError("group_bucket applies only to date fields.")
    return f


def _aggregate(rows: list, ds: SemanticDataset, group_by: Optional[str],
               aggregation: Optional[str], value_field: Optional[str],
               group_bucket: Optional[str], series_by: Optional[str] = None,
               series_bucket: Optional[str] = None, top_n: Optional[int] = None,
               top_direction: str = "top", percent: bool = False) -> Optional[dict]:
    """Runs over ALREADY-VISIBLE rows only. There is no path in this module
    where a row the caller cannot see reaches this function, which is what
    makes every count -- and now every percentage and every rank -- safe to
    return (Part 8/16). Widening the analysis did not widen the data: the
    same filtered row list feeds all of it."""
    if not aggregation:
        return None
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise DatasetError(f"Unsupported aggregation {aggregation!r}.")

    gf = _validate_group_field(ds, group_by, group_bucket, "a grouping") if group_by else None
    sf = None
    if series_by:
        if not group_by:
            raise DatasetError("series_by requires group_by.")
        if series_by == group_by:
            raise DatasetError("series_by must differ from group_by.")
        sf = _validate_group_field(ds, series_by, series_bucket, "a series")

    if aggregation in (AGG_MIN, AGG_MAX):
        if not value_field:
            raise DatasetError(f"{aggregation} requires value_field.")
    if value_field:
        vf = ds.field(value_field)
        if vf is None:
            raise DatasetError(f"Unknown field {value_field!r}.")
        if not vf.aggregatable or aggregation not in vf.allowed_aggregations:
            raise DatasetError(
                f"Aggregation {aggregation!r} is not permitted on {value_field!r}. "
                f"Allowed: {list(vf.allowed_aggregations)}.")

    if percent and aggregation not in PERCENTABLE_AGGREGATIONS:
        raise DatasetError(
            f"percent is only meaningful for {sorted(PERCENTABLE_AGGREGATIONS)} -- "
            f"a percentage of a {aggregation} is not a quantity.")
    if top_n is not None:
        if not group_by:
            raise DatasetError("top_n requires group_by.")
        if top_n < 1 or top_n > MAX_TOP_N:
            raise DatasetError(f"top_n must be between 1 and {MAX_TOP_N}.")
        if top_direction not in ("top", "bottom"):
            raise DatasetError("top_direction must be 'top' or 'bottom'.")

    target = value_field or group_by

    def key_of(r, fkey, fmeta, bucket):
        v = r.values.get(fkey)
        return _bucket(v, bucket) if fmeta.datatype == "date" else v

    groups: dict = {}
    for r in rows:
        gk = key_of(r, group_by, gf, group_bucket) if group_by else "__all__"
        sk = key_of(r, series_by, sf, series_bucket) if series_by else None
        groups.setdefault((gk, sk), []).append(r)

    def measure(members):
        if aggregation == AGG_COUNT:
            return len(members)
        if aggregation == AGG_COUNT_DISTINCT:
            return len({str(m.values.get(target)) for m in members
                        if m.values.get(target) is not None})
        vals = [m.values.get(value_field) for m in members
                if m.values.get(value_field) is not None]
        if not vals:
            return None
        return min(vals) if aggregation == AGG_MIN else max(vals)

    buckets = [{"group": gk, "series": sk, "value": measure(members),
                "row_count": len(members)}
               for (gk, sk), members in groups.items()]

    # An UNGROUPED count over zero rows must still produce a bucket. Without
    # this a KPI comparing two periods cannot distinguish "the period had no
    # events" (a real answer, value 0) from "no result came back", and the
    # caller would be left inferring one from an empty list.
    if not buckets and not group_by and aggregation in (AGG_COUNT, AGG_COUNT_DISTINCT):
        buckets = [{"group": "__all__", "series": None, "value": 0, "row_count": 0}]

    # Top-N ranks the PRIMARY group by its combined measure, then keeps every
    # series belonging to a surviving group -- so a ranked multi-series chart
    # keeps its series intact instead of losing part of a bar.
    if top_n is not None:
        totals: dict = {}
        for b in buckets:
            v = b["value"]
            totals[b["group"]] = totals.get(b["group"], 0) + (v if isinstance(v, (int, float)) else 0)
        ordered = sorted(totals.items(), key=lambda kv: (kv[1], str(kv[0])),
                         reverse=(top_direction == "top"))
        keep = {g for g, _ in ordered[:top_n]}
        buckets = [b for b in buckets if b["group"] in keep]

    # A percentage is a share of the VISIBLE total -- the same rows that were
    # counted. It is never a share of some larger unfiltered population,
    # which would disclose the size of what the caller cannot see.
    total = None
    if percent:
        total = sum(b["value"] for b in buckets if isinstance(b["value"], (int, float)))
        for b in buckets:
            b["percent"] = (round(b["value"] / total * 100, 2)
                            if total and isinstance(b["value"], (int, float)) else None)

    buckets.sort(key=lambda b: (b["group"] is None, str(b["group"]),
                                b["series"] is None, str(b["series"])))
    return {
        "aggregation": aggregation, "group_by": group_by,
        "group_bucket": group_bucket, "series_by": series_by,
        "series_bucket": series_bucket, "value_field": value_field,
        "top_n": top_n, "top_direction": top_direction if top_n else None,
        "percent": percent, "percent_basis": total,
        "series_values": sorted({str(b["series"]) for b in buckets if b["series"] is not None}),
        "buckets": buckets,
    }



def run_query(dataset: str, workspace_id: str, allowed_sensitivities: list,
              fields: Optional[list] = None, filters: Optional[list] = None,
              group_by: Optional[str] = None, aggregation: Optional[str] = None,
              value_field: Optional[str] = None, group_bucket: Optional[str] = None,
              series_by: Optional[str] = None, series_bucket: Optional[str] = None,
              top_n: Optional[int] = None, top_direction: str = "top",
              percent: bool = False,
              temporal_mode: str = MODE_CURRENT, as_of: Optional[datetime] = None,
              window_days: Optional[int] = None, window_offset_days: Optional[int] = None,
              app_departments: Optional[dict] = None,
              caller_emails: Optional[list] = None) -> DatasetQueryResult:
    """The single entry point. Everything the caller supplied has already
    been checked against the registry by the time a resolver runs, and the
    caller's sensitivity ceiling was derived server-side -- it is never a
    parameter a client can influence."""
    ds = get_dataset(dataset)

    if temporal_mode not in ds.temporal_modes:
        raise DatasetError(
            f"Dataset {dataset!r} does not support temporal mode {temporal_mode!r}. "
            f"Supported: {list(ds.temporal_modes)}.")
    if temporal_mode == MODE_AS_OF and as_of is None:
        raise DatasetError("temporal_mode 'as_of' requires as_of.")
    if temporal_mode != MODE_AS_OF and as_of is not None:
        raise DatasetError("as_of is only valid with temporal_mode 'as_of'.")
    if temporal_mode == MODE_WINDOW and not window_days:
        raise DatasetError("temporal_mode 'window' requires window_days.")
    if window_days is not None and (window_days < 1 or window_days > 365):
        raise DatasetError("window_days must be between 1 and 365.")

    if fields:
        for k in fields:
            if ds.field(k) is None:
                raise DatasetError(f"Unknown field {k!r} for dataset {dataset!r}.")
    validated_filters = _validate_filters(ds, filters or [])

    now = datetime.now(timezone.utc)
    since = until = None
    if MODE_WINDOW in ds.temporal_modes:
        days = window_days or 30
        # window_offset_days shifts the whole window back by N days, which is
        # what makes a KPI comparison REAL: "the 30 days before the last 30"
        # is a second genuine query over the same visibility rules, not a
        # percentage invented to make a tile look livelier (Part 7).
        offset = window_offset_days or 0
        if offset < 0 or offset > 3650:
            raise DatasetError("window_offset_days must be between 0 and 3650.")
        until = now - timedelta(days=offset)
        since = until - timedelta(days=days)

    ctx = ResolveContext(workspace_id=workspace_id, allowed=list(allowed_sensitivities),
                          as_of=as_of, since=since, until=until,
                          app_departments=app_departments,
                          caller_emails=caller_emails)
    rows = ds.resolver(ctx)
    rows = _apply_filters(rows, validated_filters)

    agg = _aggregate(rows, ds, group_by, aggregation, value_field, group_bucket,
                     series_by=series_by, series_bucket=series_bucket,
                     top_n=top_n, top_direction=top_direction, percent=percent)

    selected = list(fields) if fields else [f.key for f in ds.fields]
    projected = []
    for r in rows:
        projected.append({
            "values": {k: r.values.get(k) for k in selected},
            "object_kind": r.object_kind, "object_id": r.object_id,
            "evidence_ids": r.evidence_ids, "markers": r.markers,
            "explanation": r.explanation, "reasoning_state": r.reasoning_state,
            "not_established": r.not_established,
        })

    temporal_context = (as_of.isoformat() if as_of else
                         (f"window:{_iso(since)}..{_iso(until)}" if since else "current"))

    return DatasetQueryResult(
        dataset=ds.key,
        fields=[{"key": f.key, "label": f.label, "datatype": f.datatype,
                  "temporal_meaning": f.temporal_meaning} for f in ds.fields
                 if f.key in selected],
        rows=projected, row_count=len(projected),
        temporal_context=temporal_context, temporal_mode=temporal_mode,
        generated_at=now.isoformat(), aggregation=agg,
        evidence_available=any(r["evidence_ids"] for r in projected),
        drilldown_target=ds.drilldown_target,
        not_established=list(ds.not_established),
        empty_reason=(ds.empty_reason if not projected else None),
        notes=([] if projected or ds.empty_reason else
               ["No rows matched. This means no data was found, not that none exists."]),
    )
