"""
Phase 7B -- Reasoning-Aware Query Understanding: a small, deterministic
reference resolver that turns natural-language references into the REAL
entities/memories they denote -- or honestly refuses to.

BETTER RESOLUTION, NOT MORE INFERENCE. This module answers exactly one
question: "which already-existing thing is the user talking about?" It never
answers "what is true about it" -- that remains reasoning.py's job, and the
OBSERVED/DERIVED/INFERRED/UNKNOWN states are still assigned there, never
here (Part 10).

WHY THIS EXISTS: Phase 7A measured that "Who organized the meeting?" returns
UNKNOWN while "Who organized Knova Test Meeting 1?" reasons correctly. The
cause is not a reasoning weakness -- graph_retrieval.resolve_entity_mentions
is a deterministic exact/alias matcher, and no entity is literally LABELLED
"meeting", so nothing resolves and the composed context is empty. This
module closes that gap for DEFINITE references ("the meeting", "that
policy") without loosening anything.

NO LLM, ANYWHERE (Part 2). No fuzzy/edit-distance matching, no embedding
similarity, no "pick the top-ranked candidate". Every resolution is either
an exact real-identifier match or a definite-reference match where EXACTLY
ONE candidate of the right type exists in the caller's authorized,
temporally-valid scope. Anything else is AMBIGUOUS or UNRESOLVED, and both
of those must stop the pipeline from inventing a referent (Part 3).

THREE OUTCOMES, NEVER TWO:
    RESOLVED    exactly one candidate is supported
    AMBIGUOUS   two or more candidates are plausible -- never silently pick
    UNRESOLVED  no candidate is supported

SECURITY (Part 6): every candidate lookup is workspace-scoped IN THE QUERY
itself -- this module never fetches globally and filters afterward. Entities
carry no sensitivity of their own (frozen Phase 5 decision, unchanged here);
memory candidates are filtered through memory_retrieval's own
_fetch_memory_rows + _is_visible, so an invisible memory behaves exactly as
if it does not exist. No second authorization layer is created.

TEMPORAL (Part 5): reuses the existing, already-verified semantics rather
than inventing new ones. For memories, memory_retrieval._fetch_memory_rows
already enforces all three Phase 6D/6D.1/6D.2 concepts (claim validity,
memory availability, memory succession) -- it is called directly, unchanged.
For entities, knowledge_entities has no valid_from/valid_until (confirmed
live against the real schema); its only temporal column is created_at, so
entity availability at `as_of` is `created_at <= as_of` -- exactly the
MEMORY AVAILABILITY rule Phase 6D.1 already established for org_memory,
reused rather than reinvented. No historical timestamp is ever fabricated.

NOT SOURCE-TYPE SPECIFIC (Part 16): the noun->type mapping below is driven
entirely by the frozen ENTITY and MEMORY type vocabularies, never by where
knowledge came from. There is no Slack/Chat/Calendar/Gmail branch anywhere
in this file, so future email ingestion needs no change here -- an email-
sourced meeting or policy resolves through the identical path.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import brain_connectors as bc
import graph_retrieval
import memory_retrieval

RESOLVED = "RESOLVED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"

# Frozen vocabularies -- deliberately mirrored from the real schema, never
# extended to make a query resolve (Part 4).
_ENTITY_TYPES = ("person", "department", "meeting")
_MEMORY_TYPES = ("policy", "process", "decision")

# Definite-reference head nouns -> the ONE kind of thing they may denote.
# A noun maps to entity types OR memory types, never across the two in a way
# that would let "the meeting" reach a Department (Part 4's hard rule).
# Plurals/synonyms are listed explicitly rather than stemmed: an explicit
# list cannot accidentally match a word it was never meant to.
_NOUN_TO_ENTITY_TYPE = {
    "meeting": "meeting",
    "meetings": "meeting",
    "person": "person",
    "people": "person",
    "department": "department",
    "departments": "department",
    "team": "department",     # the real ontology models teams AS departments
    "teams": "department",
}
_NOUN_TO_MEMORY_TYPE = {
    "policy": "policy",
    "policies": "policy",
    "process": "process",
    "processes": "process",
    "decision": "decision",
    "decisions": "decision",
}

# A definite reference is a determiner + one of the head nouns above. "a
# meeting"/"any meeting" are deliberately NOT definite -- they don't assert
# a specific referent, so resolving them would be a guess.
_DEFINITE_DETERMINERS = ("the", "that", "this", "those", "these")
_DEFINITE_RE = re.compile(
    r"\b(?:" + "|".join(_DEFINITE_DETERMINERS) + r")\s+(" +
    "|".join(sorted(set(_NOUN_TO_ENTITY_TYPE) | set(_NOUN_TO_MEMORY_TYPE), key=len, reverse=True)) +
    r")\b", re.IGNORECASE)


@dataclass
class ResolvedReference:
    """One real thing a phrase denotes. `kind` is 'entity' or 'memory' --
    the same two shapes the rest of the Phase 6/7 stack already speaks."""
    phrase: str
    kind: str
    object_id: str
    object_type: str          # entity_type or memory_type
    label: str
    match_basis: str          # 'exact_label' | 'alias' | 'identifier' | 'definite_unique' | 'prior_context'


@dataclass
class ResolutionResult:
    status: str                                   # RESOLVED | AMBIGUOUS | UNRESOLVED
    references: list = field(default_factory=list)      # ResolvedReference
    ambiguous: list = field(default_factory=list)       # [{phrase, object_type, candidate_ids, candidate_labels}]
    unresolved_phrases: list = field(default_factory=list)
    temporal_context: str = "current"
    candidates_considered: int = 0


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _entity_available_at(row: dict, as_of: Optional[datetime]) -> bool:
    """ENTITY AVAILABILITY. knowledge_entities has no valid_from/valid_until
    (verified live against the real schema), so the only honest temporal
    question is "did KNOVA know this entity yet" -- created_at <= as_of.
    Identical in spirit and in code shape to Phase 6D.1's memory-availability
    rule, deliberately reused rather than reinvented. Fails closed: a row
    with no parseable created_at is treated as unavailable, never assumed."""
    if as_of is None:
        return True
    created_at = _parse_ts(row.get("created_at"))
    return created_at is not None and created_at <= as_of


def _fetch_entity_candidates(workspace_id: str, entity_type: str,
                              as_of: Optional[datetime]) -> list[dict]:
    """ONE bounded, workspace-scoped, type-scoped query -- never a global
    search followed by filtering (Part 6), and never unbounded per noun
    phrase (Part 12). Only status='active' entities are candidates: a
    retired entity is not what a user means by "the meeting"."""
    rows = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label,status,created_at") \
        .eq("workspace_id", workspace_id).eq("entity_type", entity_type) \
        .eq("status", "active").order("canonical_label").execute().data or []
    return [r for r in rows if _entity_available_at(r, as_of)]


def _fetch_memory_candidates(workspace_id: str, memory_type: str,
                              allowed_sensitivities: list[str],
                              as_of: Optional[datetime]) -> list[dict]:
    """Delegates entirely to memory_retrieval._fetch_memory_rows, which
    already enforces every Phase 6D/6D.1/6D.2 temporal rule, then applies the
    same visibility check _build_memory_page uses. No new temporal or
    security logic is written here."""
    rows = memory_retrieval._fetch_memory_rows(workspace_id, as_of)
    return [r for r in rows
            if r.get("memory_type") == memory_type
            and memory_retrieval._is_visible(r.get("sensitivity"), allowed_sensitivities)]


def _memory_labels(rows: list[dict], allowed_sensitivities: list[str]) -> dict:
    """{memory_id: label} for ALL rows in exactly two queries, never two per
    row. A memory's human label is its real grounding statement -- the same
    choice wiki_projection already makes for memory page titles -- falling
    back to a typed placeholder rather than inventing prose.

    Batched after this phase's own benchmark measured the per-row version at
    ~2.9s for the real 3-policy ambiguous case (2 sequential round-trips per
    candidate at ~300ms each). Same batching discipline as Phase 6H.1's
    graph_query pass; output is identical, only the number of round-trips
    changed. Deterministic tie-break preserved: the lowest memory_evidence
    row id per memory, matching the per-row version's .order('id').limit(1)."""
    if not rows:
        return {}
    memory_ids = [r["id"] for r in rows]
    ev_rows = bc.supabase.table("memory_evidence").select("id,memory_id,evidence_id") \
        .in_("memory_id", memory_ids).eq("evidence_type", "structured_knowledge") \
        .order("id").execute().data or []
    first_ev_by_memory: dict = {}
    for ev in ev_rows:
        first_ev_by_memory.setdefault(ev["memory_id"], ev["evidence_id"])

    sk_by_id: dict = {}
    sk_ids = list({v for v in first_ev_by_memory.values()})
    if sk_ids:
        sk_rows = bc.supabase.table("structured_knowledge").select("id,statement,sensitivity") \
            .in_("id", sk_ids).execute().data or []
        sk_by_id = {r["id"]: r for r in sk_rows}

    labels: dict = {}
    for row in rows:
        sk = sk_by_id.get(first_ev_by_memory.get(row["id"]))
        if sk and memory_retrieval._is_visible(sk.get("sensitivity"), allowed_sensitivities):
            labels[row["id"]] = sk["statement"]
        else:
            labels[row["id"]] = f"{row['memory_type']} memory {row['id']}"
    return labels


# =====================================================================
# Part 2A/2B -- exact references and real identifiers.
# =====================================================================

def _resolve_exact_entities(question: str, workspace_id: str,
                             as_of: Optional[datetime]) -> list[ResolvedReference]:
    """Reuses graph_retrieval.resolve_entity_mentions COMPLETELY UNCHANGED
    (exact canonical_label/alias word-boundary matching, workspace-scoped,
    ambiguity-dropping) -- this module adds no second exact matcher. Its
    results are then re-checked for temporal availability, which it does not
    itself apply."""
    matched = graph_retrieval.resolve_entity_mentions(question, workspace_id) or []
    if not matched:
        return []
    ids = [m["id"] for m in matched]
    rows = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label,status,created_at") \
        .eq("workspace_id", workspace_id).in_("id", ids).execute().data or []
    by_id = {r["id"]: r for r in rows if _entity_available_at(r, as_of) and r["status"] == "active"}

    # An alias hit and a canonical hit for the SAME entity are one reference,
    # not two -- dedup by real id.
    out, seen = [], set()
    for m in matched:
        row = by_id.get(m["id"])
        if row is None or row["id"] in seen:
            continue
        seen.add(row["id"])
        basis = "exact_label" if m["canonical_label"].lower() in question.lower() else "alias"
        out.append(ResolvedReference(
            phrase=m["canonical_label"], kind="entity", object_id=row["id"],
            object_type=row["entity_type"], label=row["canonical_label"], match_basis=basis,
        ))
    return out


def _resolve_identifiers(question: str, workspace_id: str,
                          as_of: Optional[datetime]) -> list[ResolvedReference]:
    """Exact match against real knowledge_entity_identifiers values (emails,
    conference ids, external event ids). Workspace-scoped in the query.
    Only fires when the identifier appears literally in the question -- an
    identifier is precise by construction, so an exact hit is unambiguous."""
    rows = bc.supabase.table("knowledge_entity_identifiers") \
        .select("entity_id,identifier_type,identifier_value") \
        .eq("workspace_id", workspace_id).execute().data or []
    q_lower = question.lower()
    hits = [r for r in rows if r["identifier_value"] and r["identifier_value"].lower() in q_lower]
    if not hits:
        return []
    ent_rows = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label,status,created_at") \
        .eq("workspace_id", workspace_id).in_("id", [h["entity_id"] for h in hits]).execute().data or []
    by_id = {r["id"]: r for r in ent_rows if _entity_available_at(r, as_of) and r["status"] == "active"}

    out, seen = [], set()
    for h in hits:
        row = by_id.get(h["entity_id"])
        if row is None or row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append(ResolvedReference(
            phrase=h["identifier_value"], kind="entity", object_id=row["id"],
            object_type=row["entity_type"], label=row["canonical_label"], match_basis="identifier",
        ))
    return out


# =====================================================================
# Part 2C/3 -- definite references, with the ambiguity rule.
# =====================================================================

def _definite_phrases(question: str) -> list[tuple[str, str]]:
    """[(full_phrase, head_noun)] for every definite reference found."""
    return [(m.group(0), m.group(1).lower()) for m in _DEFINITE_RE.finditer(question or "")]


def resolve_references(question: str, workspace_id: str, allowed_sensitivities: list[str],
                        as_of: Optional[datetime] = None,
                        prior_references: Optional[list] = None) -> ResolutionResult:
    """The single public entry point. Deterministic; no LLM; no fuzzy match.

    `prior_references` (Part 9) is the minimal, in-memory continuity hook:
    a caller may pass the ResolvedReference list from the IMMEDIATELY
    preceding turn. It is never persisted, never written to any table, and
    never treated as organizational memory -- it is only consulted when a
    definite reference would otherwise be AMBIGUOUS or UNRESOLVED, and only
    when the prior turn established exactly one thing of that same type."""
    temporal_context = as_of.isoformat() if as_of else "current"
    references: list[ResolvedReference] = []
    ambiguous: list[dict] = []
    unresolved: list[str] = []
    candidates_considered = 0

    references.extend(_resolve_exact_entities(question, workspace_id, as_of))
    for ref in _resolve_identifiers(question, workspace_id, as_of):
        if not any(r.object_id == ref.object_id for r in references):
            references.append(ref)
    already_resolved_types = {r.object_type for r in references}

    for phrase, noun in _definite_phrases(question):
        entity_type = _NOUN_TO_ENTITY_TYPE.get(noun)
        memory_type = _NOUN_TO_MEMORY_TYPE.get(noun)

        # If an exact reference of the same type is already present, the
        # definite phrase is almost certainly referring to it ("Tell me about
        # Knova Test Meeting 1... who organized the meeting?" in one query).
        # Not a guess: the referent is present in the very same question.
        if entity_type and entity_type in already_resolved_types:
            continue

        if entity_type:
            cands = _fetch_entity_candidates(workspace_id, entity_type, as_of)
            candidates_considered += len(cands)
            if len(cands) == 1:
                row = cands[0]
                references.append(ResolvedReference(
                    phrase=phrase, kind="entity", object_id=row["id"],
                    object_type=row["entity_type"], label=row["canonical_label"],
                    match_basis="definite_unique",
                ))
                continue
            if len(cands) > 1:
                prior = _prior_of_type(prior_references, "entity", entity_type)
                if prior is not None and any(c["id"] == prior.object_id for c in cands):
                    references.append(ResolvedReference(
                        phrase=phrase, kind="entity", object_id=prior.object_id,
                        object_type=prior.object_type, label=prior.label,
                        match_basis="prior_context",
                    ))
                    continue
                ambiguous.append({
                    "phrase": phrase, "object_type": entity_type,
                    "candidate_ids": [c["id"] for c in cands],
                    "candidate_labels": [c["canonical_label"] for c in cands],
                })
                continue
            # zero candidates -- fall through to unresolved below
            prior = _prior_of_type(prior_references, "entity", entity_type)
            if prior is not None:
                references.append(ResolvedReference(
                    phrase=phrase, kind="entity", object_id=prior.object_id,
                    object_type=prior.object_type, label=prior.label, match_basis="prior_context",
                ))
            else:
                unresolved.append(phrase)
            continue

        if memory_type:
            cands = _fetch_memory_candidates(workspace_id, memory_type, allowed_sensitivities, as_of)
            candidates_considered += len(cands)
            if len(cands) == 1:
                row = cands[0]
                references.append(ResolvedReference(
                    phrase=phrase, kind="memory", object_id=row["id"],
                    object_type=row["memory_type"],
                    label=_memory_labels([row], allowed_sensitivities)[row["id"]],
                    match_basis="definite_unique",
                ))
                continue
            if len(cands) > 1:
                prior = _prior_of_type(prior_references, "memory", memory_type)
                if prior is not None and any(c["id"] == prior.object_id for c in cands):
                    references.append(ResolvedReference(
                        phrase=phrase, kind="memory", object_id=prior.object_id,
                        object_type=prior.object_type, label=prior.label, match_basis="prior_context",
                    ))
                    continue
                labels = _memory_labels(cands, allowed_sensitivities)
                ambiguous.append({
                    "phrase": phrase, "object_type": memory_type,
                    "candidate_ids": [c["id"] for c in cands],
                    "candidate_labels": [labels[c["id"]] for c in cands],
                })
                continue
            prior = _prior_of_type(prior_references, "memory", memory_type)
            if prior is not None:
                references.append(ResolvedReference(
                    phrase=phrase, kind="memory", object_id=prior.object_id,
                    object_type=prior.object_type, label=prior.label, match_basis="prior_context",
                ))
            else:
                unresolved.append(phrase)

    if references:
        status = RESOLVED
    elif ambiguous:
        status = AMBIGUOUS
    else:
        status = UNRESOLVED

    # A question can legitimately resolve one reference and be ambiguous
    # about another ("who organized the meeting and what is the policy?").
    # AMBIGUOUS wins in that case: reporting RESOLVED would hide a real
    # ambiguity the caller must not paper over.
    if references and ambiguous:
        status = AMBIGUOUS

    return ResolutionResult(
        status=status, references=references, ambiguous=ambiguous,
        unresolved_phrases=unresolved, temporal_context=temporal_context,
        candidates_considered=candidates_considered,
    )


def _prior_of_type(prior_references, kind: str, object_type: str):
    """Part 9's minimal continuity: usable ONLY when the prior turn
    established EXACTLY ONE thing of this kind+type. Two prior meetings are
    just as ambiguous as two current ones."""
    if not prior_references:
        return None
    matches = [r for r in prior_references if r.kind == kind and r.object_type == object_type]
    return matches[0] if len(matches) == 1 else None


# =====================================================================
# Part 7/10 -- handing resolution to the EXISTING pipeline. This produces a
# better search string for the already-existing retrieval path; it never
# retrieves, never reasons, and never answers.
# =====================================================================

def rewrite_question_with_references(question: str, result: ResolutionResult) -> str:
    """Replaces each resolved DEFINITE phrase with the real label, so the
    existing graph_retrieval.resolve_entity_mentions (an exact matcher) can
    find it -- the same mechanism query.py's own condense_followup already
    uses to make a question searchable. Exact/alias/identifier references
    are left untouched: they already match literally.

    This is the ONLY integration surface this module offers. It deliberately
    returns a STRING for the existing pipeline rather than pre-fetching
    context itself, so there is no second retrieval engine and no shortcut
    that bypasses reasoning.py (Part 10)."""
    if result.status == AMBIGUOUS:
        # Never silently pick one. The caller keeps the original question,
        # retrieval finds nothing for the vague phrase, and reasoning
        # correctly reports UNKNOWN -- which is the desired safe outcome.
        return question
    rewritten = question
    for ref in result.references:
        if ref.match_basis in ("definite_unique", "prior_context") and ref.phrase in rewritten:
            rewritten = rewritten.replace(ref.phrase, ref.label)
    return rewritten
