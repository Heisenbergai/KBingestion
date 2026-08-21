"""
Phase 8D -- the universal object detail behind the dashboard's drill-down.

ONE DRILL-DOWN SYSTEM. This is not a second one: `/dashboard/drilldown` is
still the only endpoint, and this module is the per-object-type resolver it
delegates to. Factoring it out keeps the router thin and puts every "what
does a Policy/Person/Meeting/Change/Learning detail contain" decision in one
readable place, instead of a growing if/elif in an HTTP handler.

THE TRUST CHAIN THIS EXISTS TO SERVE:

    visualization -> object -> state/change -> reasoning -> evidence -> source

Every section below is a real link in that chain. Nothing is assembled that
the data does not support, and every "we don't know" is returned explicitly
rather than omitted -- an absent section reads as "nothing to say", which is
a different and often wrong claim.

WHAT IS DELIBERATELY NOT INFERRED (Parts 8/9/10/18):
  * a person's employer, manager, job title, department, or team;
  * a department's members, owner, or headcount;
  * employment or team membership from meeting attendance;
  * any "related" object that is not a real graph edge, a real evidence
    link, a real memory grounding, or a bounded impact path.
Semantic similarity is never a relationship. Where the data is silent, the
response says "not established" and the UI renders that verbatim.

NO LLM (Part 24). Every explanation here was computed deterministically by
Phase 7 and is passed through unchanged. AI-generated explanation is 8G.
"""
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc
import memory_retrieval
import graph_query
import change_detection
import impact_analysis
import organizational_learning
import semantic_datasets as sd


class DetailNotFound(Exception):
    """Raised for "does not exist", "other workspace", AND "you may not see
    it" alike. The caller turns all three into one identical 404, so a
    probing client can never distinguish a real hidden object from a
    fabricated id (Part 19)."""


def _parse(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# =====================================================================
# Evidence + provenance.
# =====================================================================

def _evidence_chain(sk_ids: list, allowed: list[str]) -> tuple[list, list]:
    """structured_knowledge -> knowledge_note -> knowledge_note_source ->
    the original Slack/Chat/Drive/Calendar reference.

    Returns (evidence, unresolved_notes). Each evidence item carries as much
    of the chain as REALLY resolves; when the note or its source row is
    missing the item still appears with `source_resolved: False` rather than
    a fabricated link, and the caller surfaces that as an explicit
    limitation (Part 14).
    """
    if not sk_ids:
        return [], []
    rows = bc.supabase.table("structured_knowledge").select(
        "id,statement,sensitivity,captured_at,provider,canonical_source_type,"
        "canonical_id,authority,source_tier,requirement_kind,recurrence_text") \
        .in_("id", list(sk_ids)).execute().data or []
    visible = [r for r in rows
               if memory_retrieval._is_visible(r.get("sensitivity"), allowed)]

    note_ids = sorted({r["canonical_id"] for r in visible
                       if r.get("canonical_source_type") == "knowledge_note" and r.get("canonical_id")})
    notes = {}
    sources = {}
    if note_ids:
        for n in (bc.supabase.table("knowledge_notes")
                  .select("id,title,provider,source_type,source_ref,occurred_at,authority,sensitivity")
                  .in_("id", note_ids).execute().data or []):
            # The note carries its own sensitivity; a visible claim whose
            # note is restricted must not expose the note's title or link.
            if memory_retrieval._is_visible(n.get("sensitivity"), allowed):
                notes[n["id"]] = n
        for s in (bc.supabase.table("knowledge_note_sources")
                  .select("note_id,provider,source_type,source_ref,occurred_at,channel_id")
                  .in_("note_id", note_ids).execute().data or []):
            sources.setdefault(s["note_id"], s)

    evidence, unresolved = [], []
    for r in visible:
        note = notes.get(r.get("canonical_id"))
        src = sources.get(r.get("canonical_id")) if note else None
        item = {
            "evidence_id": r["id"],
            "evidence_type": "structured_knowledge",
            "statement": r.get("statement"),
            "captured_at": r.get("captured_at"),
            "provider": r.get("provider"),
            "authority": r.get("authority"),
            "source_tier": r.get("source_tier"),
            "source_type": r.get("canonical_source_type"),
            "note_title": (note or {}).get("title"),
            "source_reference": (src or {}).get("source_ref"),
            "source_observed_at": (src or {}).get("occurred_at"),
            "source_resolved": bool(src),
        }
        evidence.append(item)
        if not src:
            unresolved.append(r["id"])
    return evidence, unresolved


def _memory_grounding(memory_ids: list) -> list:
    if not memory_ids:
        return []
    rows = bc.supabase.table("memory_evidence").select("evidence_id,evidence_type") \
        .in_("memory_id", list(memory_ids)).eq("stance", "supports").execute().data or []
    return [r["evidence_id"] for r in rows if r["evidence_type"] == "structured_knowledge"]


# =====================================================================
# Changes + impact, both reusing Phase 7 as-is.
# =====================================================================

def _changes_for(workspace_id: str, allowed: list[str], object_id: str) -> list:
    try:
        det = change_detection.detect_changes(workspace_id, allowed, include_informational=True)
    except Exception as e:
        print(f"DETAIL: change lookup failed: {e}")
        return []
    out = []
    for e in det.events:
        subj = e.subject or {}
        if subj.get("id") != object_id and object_id not in (e.memory_ids or []):
            continue
        marker = sd.CHANGE_MARKERS.get(e.change_type)
        out.append({
            "change_type": e.change_type,
            "significance": e.significance,
            "occurred_at": e.occurred_at,
            "occurred_at_source": e.occurred_at_source,
            "reasoning_state": e.reasoning_state,
            # Deterministic, from Phase 7D. Never generated here.
            "explanation": e.explanation or "Reason not established.",
            "previous_state": e.previous_state,
            "new_state": e.new_state,
            "affected_entities": list(e.affected_entities or []),
            "evidence_ids": list(e.evidence_ids or []),
            "markers": ([marker] if marker else [])
                       + (["CRITICAL"] if e.significance == "critical" else []),
        })
    return out


def _impact(origin_kind: str, object_id: str, workspace_id: str, allowed: list[str],
            as_of: Optional[datetime], max_hops: int) -> tuple[list, list]:
    """Phase 7C's engine, unchanged. Bounded to 1 hop by default and never
    more than 2 -- there is no recursive crawl here (Part 5)."""
    try:
        res = impact_analysis.analyze_impact(origin_kind, object_id, workspace_id, allowed,
                                             as_of=as_of, max_hops=max_hops)
    except Exception as e:
        print(f"DETAIL: impact lookup failed: {e}")
        return [], ["Impact could not be established for this request."]
    # ImpactPath.hops is the hop COUNT (an int); `chain` holds the ImpactHop
    # objects. Read from the real dataclass rather than assumed.
    affected = [{
        "kind": p.target.kind,
        "object_id": p.target.object_id,
        "label": p.target.label,
        "reasoning_state": p.reasoning_state,
        "explanation": p.explanation,
        "hops": p.hops,
        "relationship_types": [h.relationship_type for h in (p.chain or [])],
        "evidence_ids": list(p.evidence_ids or []),
    } for p in res.paths]
    return affected, list(res.not_established or [])


# =====================================================================
# Per-type resolvers.
# =====================================================================

def _memory_detail(object_id, workspace_id, allowed, as_of, max_hops) -> dict:
    rows = [r for r in memory_retrieval._fetch_memory_rows(workspace_id, as_of)
            if r["id"] == object_id
            and memory_retrieval._is_visible(r.get("sensitivity"), allowed)]
    if not rows:
        raise DetailNotFound()
    m = rows[0]
    sk_ids = _memory_grounding([m["id"]])
    evidence, unresolved = _evidence_chain(sk_ids, allowed)
    label = evidence[0]["statement"] if evidence else f"{m.get('memory_type')} memory"

    not_established = []
    if not evidence:
        not_established.append(
            "The evidence supporting this is outside your access level.")
    if unresolved:
        not_established.append(
            f"{len(unresolved)} evidence item(s) could not be traced to an original source.")
    if m.get("superseded_at") is None:
        pass  # not being superseded is a fact, not an unknown

    affected, impact_unknown = _impact("memory", m["id"], workspace_id, allowed, as_of, max_hops)
    not_established.extend(impact_unknown)

    return {
        "header": {
            "kind": "memory",
            "type_label": (m.get("memory_type") or "memory").title(),
            "id": m["id"],
            "label": label,
            "status": m.get("lifecycle_status"),
            "superseded": bool(m.get("superseded_at")),
        },
        "attributes": {
            "memory_type": m.get("memory_type"),
            "promotion_basis": m.get("promotion_basis"),
            "lifecycle_status": m.get("lifecycle_status"),
            "sensitivity": m.get("sensitivity"),
            "created_at": m.get("created_at"),
            "valid_from": m.get("valid_from"),
            "valid_until": m.get("valid_until"),
            "superseded_at": m.get("superseded_at"),
            "last_confirmed_at": m.get("last_confirmed_at"),
        },
        "changes": _changes_for(workspace_id, allowed, m["id"]),
        "evidence": evidence,
        "affected": affected,
        "connections": [],
        "not_established": not_established,
    }


def _entity_detail(object_id, workspace_id, allowed, as_of, max_hops) -> dict:
    ent = graph_query.get_entity_graph(object_id, workspace_id, allowed, as_of)
    if ent is None:
        raise DetailNotFound()

    rels = list(ent.outbound_relationships) + list(ent.inbound_relationships)
    connections = [{
        "relationship_id": r.id,
        "relationship_type": r.relationship_type,
        "direction": "outbound" if r in ent.outbound_relationships else "inbound",
        "other_kind": (r.target if r in ent.outbound_relationships else r.source).object_type,
        "other_id": (r.target if r in ent.outbound_relationships else r.source).object_id,
        "other_label": (r.target if r in ent.outbound_relationships else r.source).label,
        "valid_from": r.valid_from,
        "valid_until": r.valid_until,
        "rationale": r.rationale,
        "evidence_count": len(r.evidence),
    } for r in rels]

    sk_ids = [e.evidence_id for r in rels for e in r.evidence
              if e.evidence_type == "structured_knowledge"]
    primary = graph_query.get_entity_primary_evidence(ent.id, workspace_id)
    sk_ids += [e.evidence_id for e in primary if e.evidence_type == "structured_knowledge"]
    evidence, unresolved = _evidence_chain(sorted(set(sk_ids)), allowed)

    attributes = {"entity_type": ent.entity_type, "status": ent.status}
    not_established = []

    if ent.entity_type == "person":
        # Part 8: an email identifier is the ONLY verified identity fact the
        # system holds about a person. Everything an org chart would show is
        # genuinely unknown, and each is named so the UI shows an explicit
        # "not established" rather than an empty field that reads as "none".
        emails = [i for i in ent.identifiers if i.get("identifier_type") == "email"]
        attributes["verified_email_identifiers"] = len(emails)
        not_established += [
            "Job title is not established.",
            "Department membership is not established.",
            "Reporting line is not established.",
            "Employment relationship is not established — this person appears in "
            "company knowledge, which is not the same as being a member of this workspace.",
        ]
    elif ent.entity_type == "department":
        # Part 9: relationships are real; membership/ownership/headcount are not.
        not_established += [
            "Department membership is not established — no member records exist.",
            "Headcount is not available.",
            "Department ownership is not established.",
        ]
        app_id = None
        raw = bc.supabase.table("knowledge_entities") \
            .select("external_ref_type,external_ref_id").eq("id", ent.id).execute().data or []
        if raw and raw[0].get("external_ref_type") == "department_id":
            app_id = raw[0].get("external_ref_id")
        attributes["workspace_department_id"] = app_id
    elif ent.entity_type == "meeting":
        snap = _meeting_snapshot(ent, workspace_id)
        if snap:
            attributes.update(snap)
        else:
            not_established.append("No calendar record is available for this meeting.")
        # Part 10: attendance is attendance. It is not employment or team
        # membership, and nothing here may be read as either.
        not_established.append(
            "Attendance does not establish employment, team membership, or reporting lines.")

    affected, impact_unknown = _impact("entity", ent.id, workspace_id, allowed, as_of, max_hops)
    not_established.extend(impact_unknown)
    if unresolved:
        not_established.append(
            f"{len(unresolved)} evidence item(s) could not be traced to an original source.")
    if not connections:
        not_established.append("No verified relationships are recorded for this entity.")

    return {
        "header": {
            "kind": "entity",
            "type_label": ent.entity_type.title(),
            "id": ent.id,
            "label": ent.canonical_label,
            "status": ent.status,
            "superseded": False,
        },
        "attributes": attributes,
        "changes": _changes_for(workspace_id, allowed, ent.id),
        "evidence": evidence,
        "affected": affected,
        "connections": connections,
        "not_established": not_established,
    }


def _meeting_snapshot(ent, workspace_id: str) -> Optional[dict]:
    """Calendar evidence for a meeting entity, matched on its own recorded
    external event identifier -- never on a title guess."""
    ids = bc.supabase.table("knowledge_entity_identifiers") \
        .select("identifier_type,identifier_value").eq("entity_id", ent.id).execute().data or []
    ext = next((i["identifier_value"] for i in ids
                if i.get("identifier_type") == "external_event_id"), None)
    if not ext:
        return None
    rows = bc.supabase.table("calendar_event_snapshots").select(
        "title,start_time,end_time,organizer,attendees,meeting_url,captured_at") \
        .eq("workspace_id", workspace_id).eq("external_event_id", ext) \
        .order("captured_at", desc=True).limit(1).execute().data or []
    if not rows:
        return None
    s = rows[0]
    attendees = s.get("attendees")
    return {
        "meeting_title": s.get("title"),
        "start_time": s.get("start_time"),
        "end_time": s.get("end_time"),
        "organizer": s.get("organizer"),
        "attendee_count": len(attendees) if isinstance(attendees, list) else None,
        "meeting_url": s.get("meeting_url"),
        "calendar_captured_at": s.get("captured_at"),
    }


def _relationship_detail(object_id, workspace_id, allowed, as_of, max_hops) -> dict:
    rel = graph_query.get_relationship(object_id, workspace_id, allowed)
    if rel is None:
        raise DetailNotFound()
    sk_ids = [e.evidence_id for e in rel.evidence if e.evidence_type == "structured_knowledge"]
    evidence, unresolved = _evidence_chain(sk_ids, allowed)

    not_established = []
    if not rel.rationale:
        not_established.append("No rationale was recorded for this relationship.")
    if unresolved:
        not_established.append(
            f"{len(unresolved)} evidence item(s) could not be traced to an original source.")
    if rel.valid_until is None:
        pass  # open-ended validity is a fact, not an unknown

    return {
        "header": {
            "kind": "relationship",
            "type_label": "Relationship",
            "id": rel.id,
            # ASCII arrow deliberately (same reason Phase 7F switched away from
            # an em-dash): this string reaches server logs on a cp1252 console,
            # where a U+2192 becomes an unreadable replacement character.
            "label": f"{rel.source.label} -[{rel.relationship_type}]-> {rel.target.label}",
            "status": rel.status,
            "superseded": False,
        },
        "attributes": {
            "source": rel.source.label,
            "relationship_type": rel.relationship_type,
            "target": rel.target.label,
            "status": rel.status,
            "valid_from": rel.valid_from,
            "valid_until": rel.valid_until,
            "rationale": rel.rationale,
        },
        "changes": _changes_for(workspace_id, allowed, rel.id),
        "evidence": evidence,
        "affected": [],
        "connections": [
            {"relationship_id": rel.id, "relationship_type": rel.relationship_type,
             "direction": "source", "other_kind": rel.source.object_type,
             "other_id": rel.source.object_id, "other_label": rel.source.label,
             "valid_from": rel.valid_from, "valid_until": rel.valid_until,
             "rationale": rel.rationale, "evidence_count": len(rel.evidence)},
            {"relationship_id": rel.id, "relationship_type": rel.relationship_type,
             "direction": "target", "other_kind": rel.target.object_type,
             "other_id": rel.target.object_id, "other_label": rel.target.label,
             "valid_from": rel.valid_from, "valid_until": rel.valid_until,
             "rationale": rel.rationale, "evidence_count": len(rel.evidence)},
        ],
        "not_established": not_established,
    }


def _structured_knowledge_detail(object_id, workspace_id, allowed, as_of, max_hops) -> dict:
    """SECURITY (Phase 8D finding): the sensitivity of the claim is checked
    HERE, before anything is read out of graph_query.

    graph_query.get_structured_knowledge_graph selects the row by id +
    workspace and applies `allowed_sensitivities` only to the RELATIONSHIPS
    around it -- so it will happily return a restricted statement to a
    low-clearance caller. Its other two callers (wiki_projection,
    impact_analysis) both filter the claim upstream before calling, so they
    are unaffected; this endpoint is the one that accepts an object id
    straight from a client, which is exactly the case Part 19 forbids
    trusting. Gating here closes it without changing a Phase 5 primitive
    that two audited callers depend on.
    """
    rows = bc.supabase.table("structured_knowledge") \
        .select("id,statement,sensitivity,captured_at,provider,canonical_source_type,"
                 "canonical_id,authority,source_tier,requirement_kind,recurrence_text") \
        .eq("id", object_id).eq("workspace_id", workspace_id).execute().data or []
    if not rows or not memory_retrieval._is_visible(rows[0].get("sensitivity"), allowed):
        raise DetailNotFound()
    r = rows[0]

    evidence, unresolved = _evidence_chain([r["id"]], allowed)
    g = graph_query.get_structured_knowledge_graph(object_id, workspace_id, allowed, as_of)
    connections = []
    for rel in ((g or {}).get("outbound_relationships") or []) + \
               ((g or {}).get("inbound_relationships") or []):
        connections.append({
            "relationship_id": rel.id, "relationship_type": rel.relationship_type,
            "direction": "outbound", "other_kind": rel.target.object_type,
            "other_id": rel.target.object_id, "other_label": rel.target.label,
            "valid_from": rel.valid_from, "valid_until": rel.valid_until,
            "rationale": rel.rationale, "evidence_count": len(rel.evidence),
        })

    not_established = []
    if unresolved:
        not_established.append("This claim could not be traced to an original source.")
    if not connections:
        not_established.append("No verified relationships reference this claim.")

    return {
        "header": {
            "kind": "structured_knowledge", "type_label": "Evidence", "id": r["id"],
            "label": r.get("statement") or "Evidence", "status": None, "superseded": False,
        },
        "attributes": {
            "provider": r.get("provider"), "source_type": r.get("canonical_source_type"),
            "authority": r.get("authority"), "source_tier": r.get("source_tier"),
            "sensitivity": r.get("sensitivity"), "captured_at": r.get("captured_at"),
            "requirement_kind": r.get("requirement_kind"),
            "recurrence_text": r.get("recurrence_text"),
        },
        "changes": [], "evidence": evidence, "affected": [],
        "connections": connections, "not_established": not_established,
    }


def _learning_detail(object_id, workspace_id, allowed, as_of, max_hops) -> dict:
    """A LearningSignal has no database row -- it is derived per call. It is
    located by re-deriving the current signals and matching the subject,
    which keeps this consistent with the widget that linked here."""
    res = organizational_learning.detect_learning(workspace_id, allowed, as_of=as_of)
    sig = next((s for s in res.signals if (s.subject or {}).get("id") == object_id), None)
    if sig is None:
        raise DetailNotFound()

    evidence, unresolved = _evidence_chain(
        [e.split(":", 1)[1] for e in (sig.evidence_ids or [])
         if e.startswith("structured_knowledge:")], allowed)

    not_established = [
        "This is a derived pattern, not a fact stated by any source.",
    ]
    if sig.reasoning_state == "INFERRED":
        not_established.append("This is a hypothesis, not an established conclusion.")
    if sig.contradicting_evidence:
        not_established.append(
            "Contradicting evidence exists for this pattern; it is unresolved.")
    if unresolved:
        not_established.append(
            f"{len(unresolved)} evidence item(s) could not be traced to an original source.")

    return {
        "header": {
            "kind": "learning", "type_label": "Learning",
            "id": object_id,
            "label": (sig.subject or {}).get("label") or sig.learning_type,
            "status": sig.reasoning_state, "superseded": False,
        },
        "attributes": {
            "learning_type": sig.learning_type,
            "reasoning_state": sig.reasoning_state,
            "support_count": sig.support_count,
            "observed_from": (sig.observation_window or {}).get("start"),
            "observed_to": (sig.observation_window or {}).get("end"),
            "review_required": sig.review_required,
            "contradicting_evidence": len(sig.contradicting_evidence or []),
        },
        "changes": [],
        "evidence": evidence,
        "affected": [{"kind": "entity", "object_id": e, "label": e,
                       "reasoning_state": sig.reasoning_state, "explanation": None,
                       "hops": 0, "relationship_types": [], "evidence_ids": []}
                      for e in (sig.affected_entities or [])],
        "connections": [],
        "explanation": sig.explanation,
        "not_established": not_established,
    }


_RESOLVERS = {
    "memory": _memory_detail,
    "entity": _entity_detail,
    "relationship": _relationship_detail,
    "structured_knowledge": _structured_knowledge_detail,
    "learning": _learning_detail,
}

SUPPORTED_OBJECT_KINDS = tuple(sorted(_RESOLVERS))


def build_detail(object_kind: str, object_id: str, workspace_id: str,
                 allowed_sensitivities: list, as_of: Optional[datetime] = None,
                 max_hops: int = 1) -> dict:
    """The universal object detail. `allowed_sensitivities` is derived by the
    router from the verified token on every call -- it is never a client
    input, and no authorization from a previous request is reused."""
    resolver = _RESOLVERS.get(object_kind)
    if resolver is None:
        raise DetailNotFound()

    detail = resolver(object_id, workspace_id, list(allowed_sensitivities), as_of, max_hops)

    # Universal disclosures every object carries.
    detail.setdefault("explanation", None)
    detail["not_established"] = list(detail.get("not_established") or [])
    detail["undetectable_changes"] = [
        f"{k}: {v}" for k, v in change_detection.UNDETECTABLE_CHANGES.items()
    ]
    if not detail["changes"]:
        detail["not_established"].append(
            "No recorded change for this object in the detected window.")

    detail["temporal_context"] = as_of.isoformat() if as_of else "current"
    detail["generated_at"] = datetime.now(timezone.utc).isoformat()
    detail["max_hops"] = max_hops

    # Part 17: the grounded context an "Ask KNOVA about this" affordance
    # hands to the EXISTING query/reasoning path. It carries identifiers and
    # the caller's temporal frame only -- no answer, no prompt, and nothing
    # that would let a downstream caller skip re-authorization.
    detail["ask_context"] = {
        "workspace_id": workspace_id,
        "object_kind": object_kind,
        "object_id": object_id,
        "label": detail["header"]["label"],
        "temporal_context": detail["temporal_context"],
        "evidence_ids": [e["evidence_id"] for e in detail.get("evidence", [])],
        "suggested_question": f"What should I know about {detail['header']['label']}?",
    }
    return detail
