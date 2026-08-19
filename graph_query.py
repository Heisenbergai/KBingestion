"""
Phase 5D -- read-only Graph Query layer over the Phase 5 knowledge graph
(knowledge_entities, knowledge_relationships, knowledge_relationship_evidence).

This module ONLY reads. It never infers, creates, or modifies a relationship
-- if no row exists for a given entity/primitive, the correct answer is an
empty result, never a guessed one (see get_entity_graph's and
get_structured_knowledge_graph's own docstrings on this point specifically).

Visibility contract (frozen, Phase 5 architecture): a relationship is visible
to a caller when at least one of its non-revoked, stance='supports' evidence
records is individually visible to that caller -- evaluated per evidence
record, never as a group minimum/maximum over all evidence. There is no
relationship-level sensitivity cache; visibility is always computed at query
time from the evidence chain, matching this codebase's own repeated
anti-drift convention (see F-13/F-40 in Project context/14_qa_risk_register.md
and 15_flags_and_open_items.md). A relationship with zero visible evidence
is treated as though it doesn't exist for that caller -- it is omitted
entirely, not returned with an empty evidence list.

Sensitivity resolution mirrors query.py's/chatbot.py's own local
_resolve_allowed_sensitivities() copy exactly, kept here as its own small
per-file copy rather than a cross-module import -- matches this codebase's
own stated convention (see query.py's docstring on "small per-file helpers
over shared coupling").
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc

_SENSITIVITY_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

_ENTITY_RELATIONSHIP_TYPES = frozenset({"entity", "structured_knowledge"})


def resolve_allowed_sensitivities(role: Optional[str], is_super_admin: bool) -> list[str]:
    """Identical ladder to query.py/chatbot.py's own copies."""
    if is_super_admin or role == "owner":
        return ["public", "internal", "confidential", "restricted"]
    if role == "admin":
        return ["public", "internal", "confidential"]
    return ["public", "internal"]


def _is_visible(sensitivity: Optional[str], allowed_sensitivities: list[str]) -> bool:
    """NULL sensitivity means "nothing to cap against", not "most
    permissive" -- same reading structured_persistence._fetch_canonical_parent
    already established for Calendar's own null sensitivity. Applies equally
    to evidence types with no sensitivity column at all (external_reference,
    calendar_event_snapshot): no signal means no restriction, not a hidden
    default."""
    if sensitivity is None:
        return True
    return sensitivity in allowed_sensitivities


# =====================================================================
# Data contract (Part 11) -- stable, machine-readable, no JSONB blobs.
# =====================================================================

@dataclass
class GraphEvidence:
    evidence_kind: str          # 'primary_source' | 'derived_support'
    evidence_type: str          # knowledge_note_source | external_reference | calendar_event_snapshot | structured_knowledge
    evidence_id: str
    stance: str                 # supports | contradicts
    source_reference: Optional[str]   # permalink / statement text / meeting url, resolved from the real source row
    captured_at: Optional[str]
    # Phase 5G Part 8 -- orthogonal to evidence_kind (which asks WHERE the
    # evidence originates: real artifact vs. KNOVA's own interpretation).
    # evidence_role asks WHAT the evidence actually PROVES:
    #   'identity'     -- this evidence is what establishes the entity's own
    #                      existence/identity (e.g. the calendar snapshot a
    #                      Meeting's external_event_id identifier resolves
    #                      to -- the entity IS anchored by this observation).
    #   'activity'     -- this evidence shows participation/association, not
    #                      identity. A Person's email appearing as organizer
    #                      or attendee on a Calendar snapshot proves they were
    #                      AT that meeting, not that the email belongs to
    #                      them -- that proof happens elsewhere (today: only
    #                      via the one-time app-DB auth.users/provider_id
    #                      cross-reference performed at construction time,
    #                      which is NOT itself stored as queryable evidence
    #                      anywhere in this graph -- a real, named gap, not
    #                      silently papered over).
    #   'relationship' -- evidence attached to a knowledge_relationship claim
    #                      between two endpoints; neither identity nor
    #                      activity of a single entity.
    evidence_role: str = "relationship"


@dataclass
class GraphEndpoint:
    object_type: str            # entity | structured_knowledge
    object_id: str
    label: Optional[str]        # canonical_label for an entity, statement for a structured_knowledge row


@dataclass
class GraphRelationship:
    id: str
    relationship_type: str
    status: str
    confidence: Optional[float]
    valid_from: str
    valid_until: Optional[str]
    rationale: Optional[str]
    source: GraphEndpoint
    target: GraphEndpoint
    evidence: list[GraphEvidence] = field(default_factory=list)   # VISIBLE evidence only


@dataclass
class GraphEntity:
    id: str
    entity_type: str
    canonical_label: str
    status: str
    identifiers: list[dict] = field(default_factory=list)
    inbound_relationships: list[GraphRelationship] = field(default_factory=list)
    outbound_relationships: list[GraphRelationship] = field(default_factory=list)


# =====================================================================
# Evidence resolution -- one real lookup per evidence_type, never cached,
# never trusted from the caller.
# =====================================================================

def _resolve_evidence_source(evidence_type: str, evidence_id: str) -> Optional[dict]:
    """Returns {'sensitivity': str|None, 'reference': str|None} from the
    REAL current source row, or None if the row no longer exists (evidence
    tables are insert-only in this codebase, so this should only happen if
    something upstream was deleted out-of-band -- treated as invisible,
    fail-closed, not an error)."""
    if evidence_type == "structured_knowledge":
        rows = bc.supabase.table("structured_knowledge") \
            .select("sensitivity,statement").eq("id", evidence_id).execute().data
        if not rows:
            return None
        return {"sensitivity": rows[0]["sensitivity"], "reference": rows[0]["statement"]}

    if evidence_type == "knowledge_note_source":
        rows = bc.supabase.table("knowledge_note_sources") \
            .select("note_id,source_ref").eq("id", evidence_id).execute().data
        if not rows:
            return None
        note_id, source_ref = rows[0]["note_id"], rows[0]["source_ref"]
        notes = bc.supabase.table("knowledge_notes") \
            .select("sensitivity,source_ref").eq("id", note_id).execute().data
        sensitivity = notes[0]["sensitivity"] if notes else None
        reference = source_ref or (notes[0]["source_ref"] if notes else None)
        return {"sensitivity": sensitivity, "reference": reference}

    if evidence_type == "external_reference":
        # No sensitivity column on external_references (confirmed, established
        # earlier this project) -- no signal, not a hidden restriction.
        rows = bc.supabase.table("external_references") \
            .select("external_file_id,linked_object_type").eq("id", evidence_id).execute().data
        if not rows:
            return None
        return {"sensitivity": None, "reference": rows[0].get("external_file_id")}

    if evidence_type == "calendar_event_snapshot":
        # Calendar carries no sensitivity concept anywhere in this codebase,
        # by design -- same reasoning as structured_knowledge's own Calendar
        # nullability precedent.
        rows = bc.supabase.table("calendar_event_snapshots") \
            .select("title,meeting_url").eq("id", evidence_id).execute().data
        if not rows:
            return None
        return {"sensitivity": None, "reference": rows[0].get("meeting_url") or rows[0].get("title")}

    return None


def _evidence_kind(evidence_type: str) -> str:
    """derived_support = an interpretation KNOVA already made (a structured
    primitive). primary_source = the original organizational artifact. This
    distinction is exactly Decision 6's -- never flattened into one generic
    citation."""
    return "derived_support" if evidence_type == "structured_knowledge" else "primary_source"


def _build_visible_evidence(relationship_id: str, allowed_sensitivities: list[str]) -> list[GraphEvidence]:
    """Fetches every non-revoked evidence row for a relationship, resolves
    each one's real source, and returns ONLY the ones this caller may see --
    an invisible record is omitted entirely, never shown as a stub. Ordered
    deterministically by evidence id.

    Single-relationship form, kept unchanged for get_relationship()/
    explain_relationship() (low-volume, one-off lookups). The multi-
    relationship traversal path (_fetch_relationships_for_endpoint) uses
    _build_visible_evidence_batch below instead -- see that function's own
    docstring for why."""
    rows = bc.supabase.table("knowledge_relationship_evidence") \
        .select("*").eq("relationship_id", relationship_id) \
        .is_("revoked_at", "null").order("id").execute().data or []

    visible = []
    for row in rows:
        source = _resolve_evidence_source(row["evidence_type"], row["evidence_id"])
        if source is None:
            continue
        if not _is_visible(source["sensitivity"], allowed_sensitivities):
            continue
        visible.append(GraphEvidence(
            evidence_kind=_evidence_kind(row["evidence_type"]),
            evidence_type=row["evidence_type"],
            evidence_id=row["evidence_id"],
            stance=row["stance"],
            source_reference=source["reference"],
            captured_at=row["captured_at"],
            evidence_role="relationship",
        ))
    return visible


def _resolve_evidence_sources_batch(ev_rows: list[dict]) -> dict:
    """Phase 6H.1 performance pass -- batched form of _resolve_evidence_source,
    one query per real evidence_type PRESENT across all of `ev_rows` (never
    one query per evidence row). Same per-type field selection and the same
    two-hop knowledge_note_source->knowledge_notes resolution, just fetched
    for every id of that type at once via .in_(). Returns
    {(evidence_type, evidence_id): {'sensitivity', 'reference'} | absent} --
    a missing key means "row no longer exists", exactly matching the
    single-lookup function's own None-on-missing behavior (checked with
    .get(key) by the caller, never assumed present)."""
    ids_by_type: dict[str, set] = {}
    for row in ev_rows:
        ids_by_type.setdefault(row["evidence_type"], set()).add(row["evidence_id"])

    out: dict = {}

    sk_ids = ids_by_type.get("structured_knowledge")
    if sk_ids:
        rows = bc.supabase.table("structured_knowledge").select("id,sensitivity,statement") \
            .in_("id", list(sk_ids)).execute().data or []
        for r in rows:
            out[("structured_knowledge", r["id"])] = {"sensitivity": r["sensitivity"], "reference": r["statement"]}

    note_source_ids = ids_by_type.get("knowledge_note_source")
    if note_source_ids:
        ns_rows = bc.supabase.table("knowledge_note_sources").select("id,note_id,source_ref") \
            .in_("id", list(note_source_ids)).execute().data or []
        note_ids = {r["note_id"] for r in ns_rows}
        notes_by_id = {}
        if note_ids:
            notes = bc.supabase.table("knowledge_notes").select("id,sensitivity,source_ref") \
                .in_("id", list(note_ids)).execute().data or []
            notes_by_id = {n["id"]: n for n in notes}
        for r in ns_rows:
            note = notes_by_id.get(r["note_id"])
            sensitivity = note["sensitivity"] if note else None
            reference = r["source_ref"] or (note["source_ref"] if note else None)
            out[("knowledge_note_source", r["id"])] = {"sensitivity": sensitivity, "reference": reference}

    ext_ref_ids = ids_by_type.get("external_reference")
    if ext_ref_ids:
        rows = bc.supabase.table("external_references").select("id,external_file_id,linked_object_type") \
            .in_("id", list(ext_ref_ids)).execute().data or []
        for r in rows:
            out[("external_reference", r["id"])] = {"sensitivity": None, "reference": r.get("external_file_id")}

    cal_ids = ids_by_type.get("calendar_event_snapshot")
    if cal_ids:
        rows = bc.supabase.table("calendar_event_snapshots").select("id,title,meeting_url") \
            .in_("id", list(cal_ids)).execute().data or []
        for r in rows:
            out[("calendar_event_snapshot", r["id"])] = {"sensitivity": None, "reference": r.get("meeting_url") or r.get("title")}

    return out


def _build_visible_evidence_batch(relationship_ids: list[str], allowed_sensitivities: list[str]) -> dict:
    """Phase 6H.1 performance pass -- batched form of _build_visible_evidence
    for the multi-relationship traversal path ONLY (_fetch_relationships_for_
    endpoint). Fetches every relationship's non-revoked evidence rows in ONE
    query (.in_("relationship_id", ...)) instead of one query per
    relationship, then resolves all real sources via
    _resolve_evidence_sources_batch instead of one query per evidence row.
    Same visibility filter, same per-relationship id ordering (the
    underlying query is still ordered by evidence id) -- output is
    byte-identical to calling _build_visible_evidence once per id in
    `relationship_ids`, just fetched more cheaply. Returns
    {relationship_id: [GraphEvidence, ...]}; a relationship_id with no
    visible evidence is simply absent (caller uses .get(id, [])), matching
    the single-relationship function's own empty-list result."""
    if not relationship_ids:
        return {}
    ev_rows = bc.supabase.table("knowledge_relationship_evidence").select("*") \
        .in_("relationship_id", relationship_ids).is_("revoked_at", "null") \
        .order("id").execute().data or []
    if not ev_rows:
        return {}

    sources = _resolve_evidence_sources_batch(ev_rows)

    by_rel: dict = {}
    for row in ev_rows:
        source = sources.get((row["evidence_type"], row["evidence_id"]))
        if source is None:
            continue
        if not _is_visible(source["sensitivity"], allowed_sensitivities):
            continue
        by_rel.setdefault(row["relationship_id"], []).append(GraphEvidence(
            evidence_kind=_evidence_kind(row["evidence_type"]),
            evidence_type=row["evidence_type"],
            evidence_id=row["evidence_id"],
            stance=row["stance"],
            source_reference=source["reference"],
            captured_at=row["captured_at"],
            evidence_role="relationship",
        ))
    return by_rel


# =====================================================================
# Endpoint label resolution -- one hop out, for the depth-2 traversal view.
# =====================================================================

def _resolve_endpoint_label(object_type: str, object_id: str) -> Optional[str]:
    """Single-endpoint form, kept unchanged for get_relationship() (a
    one-off, low-volume lookup). The multi-relationship traversal path uses
    _resolve_endpoint_labels_batch below instead."""
    if object_type == "entity":
        rows = bc.supabase.table("knowledge_entities").select("canonical_label").eq("id", object_id).execute().data
        return rows[0]["canonical_label"] if rows else None
    if object_type == "structured_knowledge":
        rows = bc.supabase.table("structured_knowledge").select("statement").eq("id", object_id).execute().data
        return rows[0]["statement"] if rows else None
    return None


def _resolve_endpoint_labels_batch(rows: list[dict]) -> dict:
    """Phase 6H.1 performance pass -- batched form of _resolve_endpoint_label
    for the multi-relationship traversal path ONLY. `rows` are real
    knowledge_relationships rows; collects every distinct (object_type,
    object_id) referenced as EITHER a source or a target across all of them,
    then resolves all entity labels in one query and all structured_
    knowledge statements in one query (at most 2 queries total, regardless
    of how many relationships or endpoints are involved) instead of one
    query per endpoint per relationship. Returns
    {(object_type, object_id): label|None} -- a caller does
    .get((object_type, object_id)), which returns None for a missing key,
    matching _resolve_endpoint_label's own None-if-not-found behavior
    exactly (never KeyError, never assumed present)."""
    entity_ids, sk_ids = set(), set()
    for row in rows:
        for otype, oid in ((row["source_object_type"], row["source_object_id"]),
                           (row["target_object_type"], row["target_object_id"])):
            if otype == "entity":
                entity_ids.add(oid)
            elif otype == "structured_knowledge":
                sk_ids.add(oid)

    labels: dict = {}
    if entity_ids:
        rows_ = bc.supabase.table("knowledge_entities").select("id,canonical_label") \
            .in_("id", list(entity_ids)).execute().data or []
        for r in rows_:
            labels[("entity", r["id"])] = r["canonical_label"]
    if sk_ids:
        rows_ = bc.supabase.table("structured_knowledge").select("id,statement") \
            .in_("id", list(sk_ids)).execute().data or []
        for r in rows_:
            labels[("structured_knowledge", r["id"])] = r["statement"]
    return labels


# =====================================================================
# Relationship read (Part 2)
# =====================================================================

def get_relationship(relationship_id: str, workspace_id: str,
                     allowed_sensitivities: list[str]) -> Optional[GraphRelationship]:
    """One relationship by id. Returns None if it doesn't exist, belongs to
    a different workspace (fail closed, never a partial/leaked read), or has
    zero evidence visible to this caller -- a relationship with no visible
    justification is treated as not existing for this caller, matching the
    frozen visibility contract exactly."""
    rows = bc.supabase.table("knowledge_relationships").select("*") \
        .eq("id", relationship_id).eq("workspace_id", workspace_id).execute().data
    if not rows:
        return None
    row = rows[0]

    evidence = _build_visible_evidence(row["id"], allowed_sensitivities)
    if not evidence:
        return None

    return GraphRelationship(
        id=row["id"],
        relationship_type=row["relationship_type"],
        status=row["status"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        rationale=row["rationale"],
        source=GraphEndpoint(row["source_object_type"], row["source_object_id"],
                             _resolve_endpoint_label(row["source_object_type"], row["source_object_id"])),
        target=GraphEndpoint(row["target_object_type"], row["target_object_id"],
                             _resolve_endpoint_label(row["target_object_type"], row["target_object_id"])),
        evidence=evidence,
    )


# =====================================================================
# Temporal filter (Part 5) -- one predicate, shared by current/historical.
# =====================================================================

def _temporally_valid(row: dict, as_of: datetime) -> bool:
    valid_from = _parse_ts(row["valid_from"])
    valid_until = _parse_ts(row["valid_until"]) if row["valid_until"] else None
    if valid_from is not None and valid_from > as_of:
        return False
    if valid_until is not None and valid_until <= as_of:
        return False
    return True


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# =====================================================================
# Entity read + depth-2 traversal (Parts 1 & 6)
# =====================================================================

def get_entity_graph(entity_id: str, workspace_id: str, allowed_sensitivities: list[str],
                     as_of: Optional[datetime] = None) -> Optional[GraphEntity]:
    """"Give me this entity and its relationships" -- entity identity,
    identifiers, and every VISIBLE inbound/outbound relationship valid at
    `as_of` (defaults to now(), i.e. "current"). Depth-2 by construction:
    each returned GraphRelationship's source/target already carry a
    one-hop-out label, matching Part 6's shallow-traversal scope exactly --
    no recursion, no graph database.

    Returns None only if the entity itself doesn't exist in this workspace.
    An entity with zero relationships (e.g. Operations, Meeting -- see Part
    7/8) is a REAL, valid result: the entity itself, with empty
    inbound/outbound lists. This function never infers a relationship that
    isn't a real row in knowledge_relationships.

    Phase 5K Part 3 -- status semantics: `as_of` OMITTED (None) means "give
    me what's CURRENTLY true" -- only status='active' relationships are
    eligible, regardless of their temporal window. `as_of` EXPLICITLY
    supplied by the caller means "what was true AT that point" -- a
    superseded/contradicted/retracted relationship may still be returned if
    it was temporally valid at that instant, because status describes what
    KNOVA believes NOW about a relationship's correctness, not whether it
    was ever true. This is exactly the historical/as-of contract: status
    filtering only applies to the implicit-"now" read."""
    is_historical = as_of is not None
    as_of = as_of or datetime.now(timezone.utc)

    # Phase 6H.1 performance pass -- a thread-pooled parallel version of
    # these 4 independent fetches was implemented and empirically tested
    # here (5 concurrent trials against this real database returned
    # correct, uncorrupted results with no cross-contamination). It was
    # REVERTED after a real, reproducible httpcore.ReadError ("WinError
    # 10035: a non-blocking socket operation could not be completed
    # immediately") surfaced under sustained repeated load -- a Windows-
    # specific HTTP/2 connection-pool race the smaller isolated test didn't
    # trigger. This is exactly the "cannot be achieved safely without a
    # larger architectural redesign" case this phase's own instructions
    # anticipated: do not invent dangerous optimizations. Sequential calls
    # only, matching the pre-6H.1 behavior exactly -- see the Phase 6H.1
    # report's Performance section for the full writeup and numbers.
    rows = bc.supabase.table("knowledge_entities").select("*") \
        .eq("id", entity_id).eq("workspace_id", workspace_id).execute().data
    if not rows:
        return None
    entity_row = rows[0]

    identifiers = bc.supabase.table("knowledge_entity_identifiers") \
        .select("identifier_type,identifier_value,connection_id") \
        .eq("entity_id", entity_id).order("identifier_type").execute().data or []

    inbound = _fetch_relationships_for_endpoint("entity", entity_id, workspace_id, allowed_sensitivities, as_of, as_target=True, include_non_active=is_historical)
    outbound = _fetch_relationships_for_endpoint("entity", entity_id, workspace_id, allowed_sensitivities, as_of, as_target=False, include_non_active=is_historical)

    return GraphEntity(
        id=entity_row["id"],
        entity_type=entity_row["entity_type"],
        canonical_label=entity_row["canonical_label"],
        status=entity_row["status"],
        identifiers=identifiers,
        inbound_relationships=inbound,
        outbound_relationships=outbound,
    )


def get_structured_knowledge_graph(structured_knowledge_id: str, workspace_id: str,
                                   allowed_sensitivities: list[str],
                                   as_of: Optional[datetime] = None) -> Optional[dict]:
    """Same shape as get_entity_graph, entered from the structured_knowledge
    side instead -- "structured_knowledge -> relationship -> entity" (Part
    6's other worked example). Returns {'id', 'statement', 'outbound_relationships'}
    since a structured_knowledge row is never itself a relationship target
    in the current frozen ontology (only entities are), so only outbound is
    meaningful here -- checked as both, for correctness, not assumed.

    Phase 5K Part 3: same status semantics as get_entity_graph -- see that
    function's docstring."""
    is_historical = as_of is not None
    as_of = as_of or datetime.now(timezone.utc)

    rows = bc.supabase.table("structured_knowledge").select("id,statement") \
        .eq("id", structured_knowledge_id).eq("workspace_id", workspace_id).execute().data
    if not rows:
        return None

    outbound = _fetch_relationships_for_endpoint("structured_knowledge", structured_knowledge_id,
                                                  workspace_id, allowed_sensitivities, as_of, as_target=False, include_non_active=is_historical)
    inbound = _fetch_relationships_for_endpoint("structured_knowledge", structured_knowledge_id,
                                                 workspace_id, allowed_sensitivities, as_of, as_target=True, include_non_active=is_historical)

    return {
        "id": rows[0]["id"],
        "statement": rows[0]["statement"],
        "outbound_relationships": outbound,
        "inbound_relationships": inbound,
    }


def _fetch_relationships_for_endpoint(object_type: str, object_id: str, workspace_id: str,
                                      allowed_sensitivities: list[str], as_of: datetime,
                                      as_target: bool, include_non_active: bool = False) -> list[GraphRelationship]:
    """Pure read -- selects real rows only, applies the status predicate
    (Phase 5K Part 3), the temporal predicate, and the evidence-visibility
    filter, never constructs a relationship that isn't a real persisted row
    (Part 7's no-fabrication rule).

    include_non_active=False (the default, used for every "current" read):
    only status='active' rows are eligible at all -- a superseded/
    contradicted/retracted relationship is excluded regardless of whether
    its valid_from/valid_until would otherwise cover `as_of`. This is the
    smallest correct fix, applied in this one shared function so every
    consumer (get_entity_graph, get_structured_knowledge_graph, and
    anything built on top of either) benefits automatically -- no consumer
    needs its own status check.

    include_non_active=True (only ever set by an explicit historical/as-of
    caller): status is not filtered here at all -- the temporal predicate
    alone decides visibility, matching "historical reads may retrieve
    non-active relationships when they were valid at the requested time".

    Phase 6H.1 performance pass: endpoint-label resolution and evidence
    resolution are now BATCHED across every candidate relationship (one
    query each, at most, instead of one query per relationship per
    endpoint/evidence-row) -- see _resolve_endpoint_labels_batch and
    _build_visible_evidence_batch. The status/temporal filter runs FIRST,
    before either batch fetch, so a relationship that would be excluded
    anyway never contributes its endpoints/evidence to what gets batched --
    same selectivity as before, just resolved for the survivors all at
    once. Output is byte-identical to the original per-relationship loop;
    only the number of real network calls changed."""
    col_type = "target_object_type" if as_target else "source_object_type"
    col_id = "target_object_id" if as_target else "source_object_id"

    rows = bc.supabase.table("knowledge_relationships").select("*") \
        .eq(col_type, object_type).eq(col_id, object_id) \
        .eq("workspace_id", workspace_id) \
        .order("valid_from").order("id").execute().data or []

    candidate_rows = [row for row in rows
                       if (include_non_active or row["status"] == "active") and _temporally_valid(row, as_of)]
    if not candidate_rows:
        return []

    evidence_by_rel = _build_visible_evidence_batch([r["id"] for r in candidate_rows], allowed_sensitivities)
    label_by_endpoint = _resolve_endpoint_labels_batch(candidate_rows)

    result = []
    for row in candidate_rows:
        evidence = evidence_by_rel.get(row["id"], [])
        if not evidence:
            continue
        result.append(GraphRelationship(
            id=row["id"],
            relationship_type=row["relationship_type"],
            status=row["status"],
            confidence=row["confidence"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            rationale=row["rationale"],
            source=GraphEndpoint(row["source_object_type"], row["source_object_id"],
                                 label_by_endpoint.get((row["source_object_type"], row["source_object_id"]))),
            target=GraphEndpoint(row["target_object_type"], row["target_object_id"],
                                 label_by_endpoint.get((row["target_object_type"], row["target_object_id"]))),
            evidence=evidence,
        ))
    return result


# =====================================================================
# Evidence explanation (Part 3) -- "why does KNOVA believe this exists?"
# =====================================================================

def explain_relationship(relationship_id: str, workspace_id: str,
                         allowed_sensitivities: list[str]) -> Optional[list[GraphEvidence]]:
    """Thin, explicit wrapper over get_relationship for the "why" question
    specifically -- returns just the visible evidence chain, each entry
    already tagged primary_source vs derived_support, each already resolved
    to its real source_reference (permalink or statement text). Returns None
    if the relationship itself isn't visible at all (same fail-closed rule
    as get_relationship)."""
    rel = get_relationship(relationship_id, workspace_id, allowed_sensitivities)
    return rel.evidence if rel else None


# =====================================================================
# Entity-level primary evidence (Phase 5E, Part 10) -- "why does this
# ENTITY exist?", independent of any relationship.
# =====================================================================

def get_entity_primary_evidence(entity_id: str, workspace_id: str) -> list[GraphEvidence]:
    """Resolves an entity's own primary source evidence via its real
    identifiers (knowledge_entity_identifiers) -- not via any relationship,
    and without a new field on knowledge_entities. Today this has a real
    mapping only for Meeting entities: identifier_type='external_event_id'
    resolves to the matching calendar_event_snapshots. This generalizes to
    any future identifier-anchored evidence type without schema change --
    it's exactly what the identifier mechanism was designed to support
    (Phase 5's own Meeting Identity contract: "resolve identity through
    knowledge_entity_identifiers").

    No allowed_sensitivities parameter: Calendar carries no sensitivity
    concept anywhere in this codebase (by design, same reasoning as
    structured_knowledge's own Calendar nullability), so there is nothing
    to gate here -- adding an unused parameter for symmetry alone would be
    exactly the premature complexity this codebase's own conventions warn
    against.

    Phase 5F extension: identifier_type='email' resolves to any real
    calendar_event_snapshot in this workspace where that email is the
    organizer or appears in the attendees array -- the same real-evidence
    pattern as Meeting's external_event_id resolution, now proving out for
    Person. Being listed as an attendee (any response_status) is real
    evidence of association with the meeting regardless of whether they
    ever RSVP'd -- stance='supports' reflects "this evidence supports the
    entity's association with this meeting," not the attendee's own RSVP
    state, which are deliberately not conflated.

    Phase 5G Part 8 -- identity vs. activity evidence, made explicit rather
    than left implicit in prose: the external_event_id/conference_id branch
    below is evidence_role='identity' (the Meeting entity's very existence
    is anchored by this exact snapshot observation -- there is no other
    proof of the Meeting's identity anywhere). The email branch is
    evidence_role='activity' (a Calendar snapshot proves attendance/
    organizing, never that the email belongs to this Person -- that proof,
    today, happens once, out-of-band, via the app-DB auth.users/provider_id
    cross-reference performed at Person-construction time, and is NOT
    itself stored as a queryable evidence row anywhere in this graph). This
    function does not invent a new provenance table to close that gap
    (Part 8 explicitly says not to unless absolutely required); it only
    labels what it already returns honestly, and callers must not read
    'activity' evidence as identity proof. Identity evidence is returned
    before activity evidence so a caller taking evidence[0] as "the" reason
    an entity exists gets the strongest available claim first.
    """
    identifiers = bc.supabase.table("knowledge_entity_identifiers") \
        .select("*").eq("entity_id", entity_id).eq("workspace_id", workspace_id).execute().data or []

    evidence: list[GraphEvidence] = []
    for ident in identifiers:
        if ident["identifier_type"] == "external_event_id" and ident.get("connection_id"):
            snapshots = bc.supabase.table("calendar_event_snapshots").select("*") \
                .eq("workspace_id", workspace_id).eq("connection_id", ident["connection_id"]) \
                .eq("external_event_id", ident["identifier_value"]) \
                .order("created_at").execute().data or []
            for snap in snapshots:
                evidence.append(GraphEvidence(
                    evidence_kind="primary_source",
                    evidence_type="calendar_event_snapshot",
                    evidence_id=snap["id"],
                    stance="supports",
                    source_reference=snap.get("meeting_url") or snap.get("title"),
                    captured_at=snap.get("captured_at"),
                    evidence_role="identity",
                ))

        elif ident["identifier_type"] == "email":
            email = (ident["identifier_value"] or "").strip().lower()
            all_snapshots = bc.supabase.table("calendar_event_snapshots").select("*") \
                .eq("workspace_id", workspace_id).order("created_at").execute().data or []
            for snap in all_snapshots:
                is_organizer = (snap.get("organizer") or "").strip().lower() == email
                is_attendee = any(
                    (a.get("email") or "").strip().lower() == email
                    for a in (snap.get("attendees") or [])
                )
                if is_organizer or is_attendee:
                    evidence.append(GraphEvidence(
                        evidence_kind="primary_source",
                        evidence_type="calendar_event_snapshot",
                        evidence_id=snap["id"],
                        stance="supports",
                        source_reference=snap.get("meeting_url") or snap.get("title"),
                        captured_at=snap.get("captured_at"),
                        evidence_role="activity",
                    ))

    # Identity evidence first, activity evidence after -- a stable sort
    # (Python's sort is stable, so within each role the original per-
    # identifier, then per-snapshot creation order is preserved).
    role_rank = {"identity": 0, "activity": 1, "relationship": 2}
    evidence.sort(key=lambda e: role_rank.get(e.evidence_role, 99))
    return evidence
