"""
Phase 6E -- Company Wiki Foundation: a deterministic, evidence-traceable
projection layer over the frozen SOURCE -> CANONICAL -> STRUCTURED KNOWLEDGE
-> GRAPH+MEMORY architecture (Phase 5 graph + Phase 6 memory).

This module is NOT a second source of truth and NOT a second retrieval
system. It builds a WikiPageModel purely by composing existing, already-
verified read paths:
  - graph_query.get_entity_graph / get_structured_knowledge_graph /
    get_entity_primary_evidence for every entity fact (Person/Department/
    Meeting pages, and the graph side of cross-linking).
  - memory_retrieval._fetch_memory_rows / _resolve_memory_evidence /
    _is_visible for every durable-memory fact (Policy/Process/Decision
    pages, and the memory side of cross-linking).

Deliberate exception to this codebase's usual "small per-file copy over
cross-module coupling" convention (see graph_query.py's own module
docstring): this module imports graph_query and memory_retrieval as modules
and calls a handful of their PRIVATE (underscore) helpers directly, instead
of re-copying temporal/visibility logic a third time. That convention exists
for small, cheap-to-duplicate ladder functions; `_fetch_memory_rows`,
`_is_visible`, and `_resolve_memory_evidence` are not that -- they ARE the
temporal-availability/claim-validity/succession and evidence-visibility
CONTRACTS Phase 6D/6D.1/6D.2 spent three sub-phases getting exactly right.
Re-deriving them here would be precisely the "second retrieval system"
Phase 6E's own instructions forbid. Staying byte-identical by calling the
real functions is the correct reading of "never duplicate," not a shortcut.

PAGE TYPES (V1, frozen to what the graph ontology actually supports today):
  Entity-backed:  person, department, meeting       (graph_query)
  Memory-backed:  policy, process, decision          (memory_retrieval; the
                   memory_type column already maps 1:1 onto these three, no
                   new categorization invented)
Project/Product/Customer/Topic/Team pages are explicitly NOT built here --
knowledge_entities' CHECK constraint on entity_type happens to also permit
the literal strings 'policy'/'process' (see test_phase5_schema.py's
"all five frozen types" test), but zero real construction code path in this
codebase has ever produced such a row (confirmed live: entity_type in
production is only ever 'person'/'department'/'meeting'). Policy/Process/
Decision are therefore built exclusively from org_memory, never treated as
graph entities -- this is the user's own explicit architecture decision for
this phase, not a gap this module papers over.

WHAT A PAGE IS: a WikiPageModel is generated ON DEMAND from current database
state every time it's requested. There is no Wiki source-of-truth table this
phase, and no page ever stores prose -- every section item is a plain dict of
real column values or real, already-resolved evidence references. Two calls
against unchanged underlying state must produce the same content_hash (Part
15) -- this is verified by construction (every upstream fetch this module
depends on is already deterministically ordered, or is made so here) rather
than by caching.

LINKING (Part 8/9): a WikiLink is only ever created from a REAL row in
knowledge_relationships. A link never comes from keyword co-occurrence,
semantic similarity, or an LLM guess. Two directions are handled, sharing one
function (_build_links):
  entity page  -> other entity            (existing entity<->entity edges)
  entity page  -> policy/process/decision (an inbound edge whose SOURCE is a
                   structured_knowledge row that happens to be one of the
                   CURRENT/as_of-visible memories' own real grounding)
  memory page  -> entity                   (the reverse of the above, walked
                   from the memory's own grounding structured_knowledge rows
                   via graph_query.get_structured_knowledge_graph)
A structured_knowledge counterpart that does NOT ground any real, currently-
in-scope memory produces NO link -- the underlying relationship still shows
as a plain fact (Relationships section) with its real counterpart label, but
it is never promoted into a fabricated page-to-page cross-reference.

SECURITY (Part 11) -- one new re-check, found during this phase's audit:
graph_query._resolve_endpoint_label() resolves a structured_knowledge
endpoint's `statement` text UNCONDITIONALLY, with no sensitivity check --
correct for graph_query's own contract (the relationship itself is already
gated on its own visible evidence; the endpoint label is a convenience for
depth-2 display). Wiki surfaces that same label as real page content
(a Relationships-section item, or a WikiLink's `label`), so a structured_
knowledge counterpart whose OWN sensitivity exceeds the caller's ceiling
would otherwise leak its statement text through Wiki even though the
relationship's evidence was legitimately visible. This module closes that
gap at ITS OWN new consumption point (_filter_visible_relationships) rather
than modifying graph_query.py's frozen contract -- same "never trust a
derived value alone, re-check sensitivity at each new consumption point"
discipline this codebase already applies elsewhere (e.g. memory_retrieval.
_resolve_memory_evidence's per-row re-check). A relationship with an
invisible structured_knowledge counterpart is dropped entirely, matching
graph_query's own established "omitted entirely, never a stub" rule for
invisible evidence. Verified against live data to be a no-op today (every
real structured_knowledge row in this workspace is 'internal') -- a latent
gap closed before it was ever exploitable, not a reaction to a live leak.

SECURITY (Phase 6G addendum) -- a second gap found during Phase 6G's own
mandated fresh re-audit of this file: _sk_to_memory_context() (used by both
page builders to resolve memory-side link targets) built its memory context
from memory_retrieval._fetch_memory_rows() alone, which does NOT filter by
sensitivity -- that filtering happens later, per-caller, only inside
_build_memory_page's own _is_visible() check on the CURRENT page's own
memory row. A relationship whose structured_knowledge counterpart grounds a
DIFFERENT memory above the caller's ceiling would still produce a real
WikiLink to it, one that then resolved to None the moment that caller
actually opened it -- a followable-looking "hidden page" existence leak.
Proven live with a synthetic restricted memory before the fix, closed by
threading allowed_sensitivities into _sk_to_memory_context and filtering
there, the one shared place both builders get this context from.

Everything else inherits its security posture unchanged: entity identity
itself carries no sensitivity (frozen graph_query/Phase 5 decision, unchanged
here); memory pages are gated by the memory's own sensitivity ceiling
(memory_retrieval._is_visible) plus a per-evidence-row re-check inside
_resolve_memory_evidence; PENDING REVIEW CANDIDATES (memory_review_queue)
are never read by this module, structurally -- every memory page is built
exclusively from memory_retrieval._fetch_memory_rows, which reads org_memory
only.
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc
import graph_query
import memory_retrieval

_ENTITY_PAGE_TYPES = ("person", "department", "meeting")
_MEMORY_PAGE_TYPES = ("policy", "process", "decision")


# =====================================================================
# Part 5 -- WikiPageModel contract.
# =====================================================================

@dataclass
class WikiLink:
    target_page_type: str             # one of _ENTITY_PAGE_TYPES | _MEMORY_PAGE_TYPES
    target_id: str                    # the target's real entity id or memory id
    label: str                        # real canonical_label, or real structured_knowledge statement text (re-checked for visibility -- see module docstring)
    relationship_id: str              # the REAL knowledge_relationships.id this link is traced back to -- never synthesized
    relationship_type: str
    rationale: Optional[str]          # the real knowledge_relationships.rationale -- reused verbatim, never paraphrased (Part 7)


@dataclass
class WikiSection:
    section_type: str                 # 'identity' | 'relationships' | 'evidence'
    title: str
    items: list = field(default_factory=list)   # plain dicts of real values only -- never prose, never LLM output


@dataclass
class WikiPageModel:
    page_id: str                      # f"{page_type}:{canonical_entity_id or memory_id}" -- deterministic, stable
    workspace_id: str
    page_type: str
    canonical_entity_id: Optional[str]
    memory_id: Optional[str]
    title: str
    sections: list[WikiSection]
    generated_at: str                 # wall-clock time of THIS build -- excluded from content_hash
    content_hash: str                 # sha256 over every OTHER field, canonical JSON -- Part 15 determinism
    evidence: list[dict]              # the page's whole flattened, deduped citation list
    links: list[WikiLink]
    temporal_context: str             # 'current' or the as_of ISO string


# =====================================================================
# Cross-linking support -- shared by every page builder, entity or memory
# side alike. See module docstring for the two directions this covers.
# =====================================================================

def _sk_to_memory_context(workspace_id: str, as_of: Optional[datetime], allowed_sensitivities: list[str]) -> tuple[dict, dict]:
    """(structured_knowledge_id -> memory_id, memory_id -> memory row), built
    from the EXACT SAME temporally-scoped memory set memory_retrieval itself
    would return for this as_of/workspace -- reused directly, not re-derived.
    Deterministic tie-break (documented, not just assumed) if a structured_
    knowledge row ever grounds more than one memory: first by memory_evidence
    row id -- never happens in the current real corpus, but the rule is
    explicit rather than left to accidental query-result order.

    SECURITY (found during Phase 6G's fresh-audit pass, fixed here):
    memory_retrieval._fetch_memory_rows() does NOT filter by sensitivity --
    that check happens later, per-caller, in _build_memory_page's own
    _is_visible() call. This function used to skip that check entirely,
    which meant a relationship whose structured_knowledge counterpart
    grounds a memory ABOVE the caller's sensitivity ceiling would still
    produce a real, followable-looking WikiLink to that memory -- a link
    that then resolved to None the moment the caller actually opened it.
    Proven live with a synthetic restricted memory before this fix: a
    'public'/'internal'-only caller's Department page listed a link to a
    'restricted' policy page that immediately 404'd for that same caller --
    exactly the "hidden page" existence leak Phase 6G's Part 15 prohibits by
    name. Filtering here, at the one shared place both _build_entity_page
    and _build_memory_page get their memory-linking context from, closes it
    for every caller of this function at once."""
    rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    rows = [r for r in rows if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]
    memory_rows_by_id = {r["id"]: r for r in rows}
    if not rows:
        return {}, {}
    ev_rows = bc.supabase.table("memory_evidence").select("memory_id,evidence_id,evidence_type") \
        .in_("memory_id", list(memory_rows_by_id.keys())).eq("evidence_type", "structured_knowledge") \
        .order("id").execute().data or []
    sk_to_memory: dict = {}
    for ev in ev_rows:
        sk_to_memory.setdefault(ev["evidence_id"], ev["memory_id"])
    return sk_to_memory, memory_rows_by_id


def _sk_sensitivity_map(sk_ids: set[str]) -> dict:
    if not sk_ids:
        return {}
    rows = bc.supabase.table("structured_knowledge").select("id,sensitivity") \
        .in_("id", list(sk_ids)).execute().data or []
    return {r["id"]: r["sensitivity"] for r in rows}


def _filter_visible_relationships(relationships: list, allowed_sensitivities: list[str]) -> list:
    """See module docstring's Security section. Drops a relationship
    entirely (never a redacted stub) when either endpoint is a structured_
    knowledge row whose OWN sensitivity is not visible to this caller."""
    sk_ids = {ep.object_id for rel in relationships for ep in (rel.source, rel.target)
              if ep.object_type == "structured_knowledge"}
    sens = _sk_sensitivity_map(sk_ids)
    out = []
    for rel in relationships:
        visible = True
        for ep in (rel.source, rel.target):
            if ep.object_type == "structured_knowledge" and not memory_retrieval._is_visible(sens.get(ep.object_id), allowed_sensitivities):
                visible = False
                break
        if visible:
            out.append(rel)
    return out


def _resolve_link_target(endpoint, entity_types: dict, sk_to_memory: dict, memory_rows_by_id: dict) -> Optional[tuple]:
    """(target_page_type, target_id) for a real graph endpoint, or None when
    it doesn't resolve to a real Wiki page today -- an entity_type the Wiki
    doesn't cover, or a structured_knowledge row that isn't a real, in-scope
    memory's grounding. Never a guess."""
    if endpoint.object_type == "entity":
        etype = entity_types.get(endpoint.object_id)
        return (etype, endpoint.object_id) if etype in _ENTITY_PAGE_TYPES else None
    if endpoint.object_type == "structured_knowledge":
        memory_id = sk_to_memory.get(endpoint.object_id)
        if memory_id is None:
            return None
        mem = memory_rows_by_id.get(memory_id)
        return (mem["memory_type"], memory_id) if mem else None
    return None


def _build_links(relationships: list, own_object_type: str, own_ids: set, workspace_id: str,
                  sk_to_memory: dict, memory_rows_by_id: dict) -> list[WikiLink]:
    """One shared implementation for both directions (entity page or memory
    page as the 'own' side) -- the counterpart is whichever endpoint is NOT
    this page's own object_type/id set."""
    pairs = []
    for rel in relationships:
        is_source_own = rel.source.object_type == own_object_type and rel.source.object_id in own_ids
        counterpart = rel.target if is_source_own else rel.source
        pairs.append((rel, counterpart))

    entity_ids = {cp.object_id for _, cp in pairs if cp.object_type == "entity"}
    entity_types = {}
    if entity_ids:
        rows = bc.supabase.table("knowledge_entities").select("id,entity_type") \
            .in_("id", list(entity_ids)).eq("workspace_id", workspace_id).execute().data or []
        entity_types = {r["id"]: r["entity_type"] for r in rows}

    links, seen = [], set()
    for rel, cp in pairs:
        target = _resolve_link_target(cp, entity_types, sk_to_memory, memory_rows_by_id)
        if target is None:
            continue
        key = (target[0], target[1], rel.id)
        if key in seen:
            continue
        seen.add(key)
        links.append(WikiLink(target_page_type=target[0], target_id=target[1], label=cp.label,
                               relationship_id=rel.id, relationship_type=rel.relationship_type,
                               rationale=rel.rationale))
    return links


def _relationship_item(rel, own_object_type: str, own_ids: set) -> dict:
    is_outbound = rel.source.object_type == own_object_type and rel.source.object_id in own_ids
    counterpart = rel.target if is_outbound else rel.source
    return {
        "relationship_id": rel.id, "relationship_type": rel.relationship_type,
        "direction": "outbound" if is_outbound else "inbound",
        "status": rel.status,
        "counterpart_type": counterpart.object_type, "counterpart_id": counterpart.object_id,
        "counterpart_label": counterpart.label,
        "valid_from": rel.valid_from, "valid_until": rel.valid_until,
        "confidence": rel.confidence, "rationale": rel.rationale,
    }


def _dedupe_evidence_items(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = (it["evidence_type"], it["evidence_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _finalize_page(page_type: str, canonical_entity_id: Optional[str], memory_id: Optional[str],
                    workspace_id: str, title: str, sections: list[WikiSection], evidence: list[dict],
                    links: list[WikiLink], as_of: Optional[datetime]) -> WikiPageModel:
    temporal_context = as_of.isoformat() if as_of else "current"
    page_id = f"{page_type}:{canonical_entity_id or memory_id}"
    structural = {
        "page_id": page_id, "page_type": page_type,
        "canonical_entity_id": canonical_entity_id, "memory_id": memory_id, "title": title,
        "sections": [{"section_type": s.section_type, "title": s.title, "items": s.items} for s in sections],
        "evidence": evidence,
        "links": [{"target_page_type": l.target_page_type, "target_id": l.target_id, "label": l.label,
                    "relationship_id": l.relationship_id, "relationship_type": l.relationship_type,
                    "rationale": l.rationale} for l in links],
        "temporal_context": temporal_context,
    }
    content_hash = hashlib.sha256(json.dumps(structural, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return WikiPageModel(
        page_id=page_id, workspace_id=workspace_id, page_type=page_type,
        canonical_entity_id=canonical_entity_id, memory_id=memory_id, title=title,
        sections=sections, generated_at=datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash, evidence=evidence, links=links, temporal_context=temporal_context,
    )


# =====================================================================
# Entity-backed pages -- Person / Department / Meeting.
# =====================================================================

def _build_entity_page(entity_id: str, expected_entity_type: str, workspace_id: str,
                        allowed_sensitivities: list[str], as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    graph = graph_query.get_entity_graph(entity_id, workspace_id, allowed_sensitivities, as_of)
    if graph is None:
        return None
    if graph.entity_type != expected_entity_type:
        # A real entity, but the wrong page type asked for it (e.g. a
        # department id passed to build_person_page). Never render an entity
        # under a page type it isn't.
        return None

    relationships = _filter_visible_relationships(graph.inbound_relationships + graph.outbound_relationships, allowed_sensitivities)
    own_ids = {entity_id}

    identity_item = {"id": graph.id, "entity_type": graph.entity_type,
                      "canonical_label": graph.canonical_label, "status": graph.status}
    if graph.identifiers:
        identity_item["identifiers"] = sorted(
            ({"identifier_type": i["identifier_type"], "identifier_value": i["identifier_value"]}
             for i in graph.identifiers),
            key=lambda d: (d["identifier_type"], d["identifier_value"] or ""),
        )

    relationship_items = [_relationship_item(rel, "entity", own_ids) for rel in relationships]

    evidence_items = [
        {"evidence_kind": e.evidence_kind, "evidence_type": e.evidence_type, "evidence_id": e.evidence_id,
         "stance": e.stance, "reference": e.source_reference, "captured_at": e.captured_at,
         "evidence_role": e.evidence_role}
        for e in graph_query.get_entity_primary_evidence(entity_id, workspace_id)
    ]
    for rel in relationships:
        evidence_items.extend(
            {"evidence_kind": e.evidence_kind, "evidence_type": e.evidence_type, "evidence_id": e.evidence_id,
             "stance": e.stance, "reference": e.source_reference, "captured_at": e.captured_at,
             "evidence_role": e.evidence_role}
            for e in rel.evidence
        )
    evidence_items = _dedupe_evidence_items(evidence_items)

    sk_to_memory, memory_rows_by_id = _sk_to_memory_context(workspace_id, as_of, allowed_sensitivities)
    links = _build_links(relationships, "entity", own_ids, workspace_id, sk_to_memory, memory_rows_by_id)

    sections = [
        WikiSection("identity", "Identity", [identity_item]),
        WikiSection("relationships", "Relationships", relationship_items),
        WikiSection("evidence", "Evidence", evidence_items),
    ]
    return _finalize_page(expected_entity_type, entity_id, None, workspace_id, graph.canonical_label,
                           sections, evidence_items, links, as_of)


def build_person_page(entity_id: str, workspace_id: str, allowed_sensitivities: list[str],
                       as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_entity_page(entity_id, "person", workspace_id, allowed_sensitivities, as_of)


def build_department_page(entity_id: str, workspace_id: str, allowed_sensitivities: list[str],
                           as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_entity_page(entity_id, "department", workspace_id, allowed_sensitivities, as_of)


def build_meeting_page(entity_id: str, workspace_id: str, allowed_sensitivities: list[str],
                        as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_entity_page(entity_id, "meeting", workspace_id, allowed_sensitivities, as_of)


# =====================================================================
# Memory-backed pages -- Policy / Process / Decision. Pure projections over
# org_memory + memory_evidence + structured_knowledge + graph -- never a new
# entity, never inserted into knowledge_entities.
# =====================================================================

def _build_memory_page(memory_id: str, expected_memory_type: str, workspace_id: str,
                        allowed_sensitivities: list[str], as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    row = next((r for r in rows if r["id"] == memory_id), None)
    if row is None or row["memory_type"] != expected_memory_type:
        return None
    if not memory_retrieval._is_visible(row.get("sensitivity"), allowed_sensitivities):
        return None

    ev_rows = bc.supabase.table("memory_evidence").select("*") \
        .eq("memory_id", memory_id).order("id").execute().data or []
    sk_ids = sorted({e["evidence_id"] for e in ev_rows if e["evidence_type"] == "structured_knowledge"})
    sk_by_id = {}
    if sk_ids:
        sk_by_id = {r["id"]: r for r in
                    bc.supabase.table("structured_knowledge").select("*").in_("id", sk_ids).execute().data or []}

    resolved = memory_retrieval._resolve_memory_evidence(row, ev_rows, sk_by_id, workspace_id, allowed_sensitivities)
    if not resolved:
        # Every grounding row individually failed visibility -- the memory
        # passed its own sensitivity ceiling but nothing under it is
        # actually citable to this caller. Fail the whole page closed,
        # matching build_memory_context's identical rule for this exact case.
        return None

    identity_item = {
        "id": row["id"], "memory_type": row["memory_type"], "lifecycle_status": row["lifecycle_status"],
        "promotion_basis": row["promotion_basis"], "sensitivity": row["sensitivity"],
        "valid_from": row.get("valid_from"), "valid_until": row.get("valid_until"),
        "created_at": row.get("created_at"), "last_confirmed_at": row.get("last_confirmed_at"),
        "superseded_at": row.get("superseded_at"),
    }

    # memory_evidence carries no stance column (unlike knowledge_relationship_
    # evidence) -- "supports" is not read from a row, it is a construction-
    # time invariant of create_memory_with_evidence (a memory is only ever
    # grounded BY the evidence that justifies it), the same kind of derived-
    # not-stored fact graph_query._evidence_kind() already computes for
    # structured_knowledge evidence. evidence_role='memory_grounding' is a
    # new value in THIS module's own combined evidence vocabulary (Wiki
    # merges graph-sourced and memory-sourced evidence into one list; the two
    # upstream modules use non-overlapping evidence_role vocabularies) -- a
    # display-layer tag, not a schema or contract change to either module.
    evidence_items = _dedupe_evidence_items([
        {"evidence_kind": e["evidence_kind"], "evidence_type": e["evidence_type"], "evidence_id": e["evidence_id"],
         "stance": "supports", "reference": e["reference"], "captured_at": None, "evidence_role": "memory_grounding"}
        for e in resolved
    ])

    grounding_sk_ids = set(sk_ids)
    relationships, seen_rel_ids = [], set()
    for sk_id in sk_ids:  # sorted -- deterministic traversal order
        sk_graph = graph_query.get_structured_knowledge_graph(sk_id, workspace_id, allowed_sensitivities, as_of)
        if sk_graph is None:
            continue
        for rel in sk_graph["outbound_relationships"] + sk_graph["inbound_relationships"]:
            if rel.id in seen_rel_ids:
                continue
            seen_rel_ids.add(rel.id)
            relationships.append(rel)
    relationships = _filter_visible_relationships(relationships, allowed_sensitivities)

    relationship_items = [_relationship_item(rel, "structured_knowledge", grounding_sk_ids) for rel in relationships]

    sk_to_memory, memory_rows_by_id = _sk_to_memory_context(workspace_id, as_of, allowed_sensitivities)
    links = _build_links(relationships, "structured_knowledge", grounding_sk_ids, workspace_id, sk_to_memory, memory_rows_by_id)

    sections = [
        WikiSection("identity", "Identity", [identity_item]),
        WikiSection("relationships", "Relationships", relationship_items),
        WikiSection("evidence", "Evidence", evidence_items),
    ]

    # Title reuses a real statement verbatim -- never a new authoritative
    # text field. Prefers THIS memory's own specific grounding statement
    # (sk.statement) over resolved[0]["reference"], which for a note-sourced
    # grounding is the deeper PARENT NOTE's title (_resolve_deeper_provenance
    # prefers that for citation display) -- found live during Phase 6E's own
    # proof pass: two distinct real policies grounded in the same note
    # otherwise both title as the note's own name (e.g. "Credential Change
    # Policy" for both "must be logged" and "must not be shared in Slack").
    # first_sk is guaranteed present and visible: resolved[0] only exists
    # because _resolve_memory_evidence already confirmed it.
    first_sk = sk_by_id.get(resolved[0]["evidence_id"])
    title = (first_sk.get("statement") if first_sk else None) or resolved[0]["reference"] \
        or f"{expected_memory_type.capitalize()} memory {memory_id}"

    return _finalize_page(expected_memory_type, None, memory_id, workspace_id, title,
                           sections, evidence_items, links, as_of)


def build_policy_page(memory_id: str, workspace_id: str, allowed_sensitivities: list[str],
                       as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_memory_page(memory_id, "policy", workspace_id, allowed_sensitivities, as_of)


def build_process_page(memory_id: str, workspace_id: str, allowed_sensitivities: list[str],
                        as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_memory_page(memory_id, "process", workspace_id, allowed_sensitivities, as_of)


def build_decision_page(memory_id: str, workspace_id: str, allowed_sensitivities: list[str],
                         as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    return _build_memory_page(memory_id, "decision", workspace_id, allowed_sensitivities, as_of)


# =====================================================================
# Dispatch + a thin, read-only index. Part 16 (UI integration) only needs a
# decision this phase, not a route -- these two functions are the entire
# "minimal integration point" a future API layer would call.
# =====================================================================

PAGE_BUILDERS = {
    "person": build_person_page,
    "department": build_department_page,
    "meeting": build_meeting_page,
    "policy": build_policy_page,
    "process": build_process_page,
    "decision": build_decision_page,
}


def build_page(page_type: str, object_id: str, workspace_id: str, allowed_sensitivities: list[str],
               as_of: Optional[datetime] = None) -> Optional[WikiPageModel]:
    """Single dispatch entry point. Returns None for an unrecognized
    page_type rather than raising -- matches every other 'not found' path in
    this module (an unresolvable id is a normal, expected outcome, not an
    error)."""
    builder = PAGE_BUILDERS.get(page_type)
    if builder is None:
        return None
    return builder(object_id, workspace_id, allowed_sensitivities, as_of)


def list_available_pages(workspace_id: str, allowed_sensitivities: list[str],
                          as_of: Optional[datetime] = None) -> list[dict]:
    """A thin, cheap index -- [{page_type, object_id, title}] for every real
    entity/memory row this workspace currently has THAT THIS CALLER MAY SEE.

    SECURITY (found during Phase 6H's mandated re-audit, fixed here): this
    function's own signature has taken allowed_sensitivities since Phase 6E,
    but the memory branch never actually applied it -- a memory above the
    caller's ceiling still listed (with title=None, but a real, resolvable
    object_id and page_type), which is exactly the kind of existence leak
    Phase 6G's Part 15 names ("must not appear as... a 'hidden page'
    placeholder"). Harmless as an internal detail in Phase 6E (nothing
    consumed this list yet), but Phase 6G's /wiki/pages endpoint now exposes
    it directly to real callers as a browsable index, so listing a page a
    click would immediately 404 on is a real leak, not a theoretical one --
    proven live with a synthetic restricted memory before this fix. Filtered
    the same way _build_memory_page gates its OWN memory row
    (memory_retrieval._is_visible against the memory's top-level
    sensitivity). This is still only the TOP-LEVEL ceiling, same as
    everywhere else in this module -- a listed memory can still legitimately
    fail build_page's per-evidence-row re-check if every individual
    grounding row happens to be invisible even though the memory's own
    ceiling passed; that finer-grained case was always true and remains
    honestly caveated, not newly introduced.

    Entities always list (entities carry no sensitivity of their own,
    frozen graph_query/Phase 5 decision) -- nothing to filter there. Memory
    rows list with title=None rather than duplicating build_page's
    evidence-resolution cost/logic just to compute a title here."""
    entities = bc.supabase.table("knowledge_entities").select("id,entity_type,canonical_label") \
        .eq("workspace_id", workspace_id).in_("entity_type", list(_ENTITY_PAGE_TYPES)) \
        .order("entity_type").order("canonical_label").execute().data or []
    memory_rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    memory_rows = [r for r in memory_rows if memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]

    out = [{"page_type": e["entity_type"], "object_id": e["id"], "title": e["canonical_label"]} for e in entities]
    for r in sorted(memory_rows, key=lambda r: (r["memory_type"], r["id"])):
        if r["memory_type"] in _MEMORY_PAGE_TYPES:
            out.append({"page_type": r["memory_type"], "object_id": r["id"], "title": None})
    return out
