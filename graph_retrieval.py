"""
Phase 5J -- Graph + Retrieval Integration adapter.

Bridges the read-only Knowledge Graph (graph_query.py) into the existing
retrieval pipeline (query.py's hybrid_search + build_context_and_citations)
as an ADDITIONAL context source, never a replacement. The graph is never the
final source authority -- every graph-derived candidate this module produces
still carries a real, resolvable citation back to its underlying evidence
(a calendar snapshot, a structured_knowledge row, a note source), exactly
like a normal retrieved chunk.

Architecture, in one line: detect (cheap, deterministic) -> resolve entities
(deterministic, exact/alias match, never LLM) -> query the existing depth-2
graph read layer (graph_query.get_entity_graph, UNMODIFIED) -> convert each
relationship's VISIBLE evidence into a citable, chunk-shaped candidate ->
merge into the SAME candidate list hybrid_search() already produces, deduped
against it by real source identity -- so the exact same citation-numbering
code in query.py (build_context_and_citations) and the exact same generation
prompt handle the merged result with ZERO changes to either. No second
ranking system, no second answer pipeline (Part 13).

SECURITY: this module adds no new access-control logic of its own. Every
security property (workspace isolation, sensitivity ceiling, restricted
grants, temporal validity) is INHERITED from graph_query.py's existing,
already-tested contract -- resolve_entity_mentions is workspace-scoped by
construction, and get_entity_graph()'s own evidence-visibility filtering
(per-evidence-record, never a group minimum) is reused completely unchanged.
This module never re-implements or bypasses that filtering; it only consumes
the already-filtered result.

PHASE 5K.1 UPDATE (confidence, corrected): a graph candidate's `similarity`
field is None, always -- `similarity` means real vector/keyword retrieval
similarity everywhere else in this codebase, and a graph candidate has no
semantic embedding, so it cannot honestly carry one. (Phase 5K briefly
assigned synthetic 1.0/0.35 values here to fix the "graph-only answers
always report low confidence" gap -- that observable fix was correct, but
representing it AS similarity was a semantic layering violation, corrected
this pass.) The same fix now happens honestly via two dedicated functions,
graph_confidence() and combine_confidence() (below, near
merge_graph_context_into_chunks) -- a real, separately-labeled graph signal,
combined with query.py's/chatbot.py's own unchanged vector-similarity
confidence, never disguised as something it isn't. Still not a second
confidence system: combine_confidence() produces the same three-value
high/medium/low/none vocabulary the rest of this codebase already uses, it
just has two honest inputs instead of one overloaded one.

PHASE 5K.1 UPDATE (as_of): build_graph_context's as_of parameter (added in
Phase 5J) is now reachable from chatbot.py's run_rag_query() too, not just
query.py's /query -- see chatbot.py's own as_of docstring for the exact
propagation path.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import brain_connectors as bc
import graph_query as gq

# =====================================================================
# Part 2 -- graph-relevant query detection. Deterministic, cheap: one
# regex search over a fixed indicator list taken directly from the spec's
# own RELATIONAL examples. No LLM call, no per-query cost beyond a single
# compiled-regex search over the (already-condensed) question string.
# =====================================================================

_RELATIONAL_INDICATORS = (
    "who", "who owns", "who attended", "who organized", "who organised",
    "who is responsible", "responsible for",
    "depends on", "dependency", "dependencies",
    "requires approval", "require approval", "approval from", "approval",
    # ^ "approval" bare, not just the fixed phrases above -- found live via
    # the Phase 5J Part 10 benchmark: "What requires Product approval?" and
    # "Why does KNOVA say Product approval is required?" both put the
    # approving party BETWEEN "requires"/"required" and "approval", so the
    # adjacent-phrase patterns alone missed them. "approval" inherently
    # names a relationship between an approver and the thing needing it, so
    # it is a safe standalone indicator in this domain -- unlike a generic
    # word like "required" alone, which would be too broad/noisy to add.
    "connected to", "related to", "relationship", "relationships",
    "owns", "attended", "organized", "organised",
    "works on", "work on", "working on",
)
_RELATIONAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _RELATIONAL_INDICATORS) + r")\b",
    re.IGNORECASE,
)


def is_graph_relevant(question: str) -> bool:
    """True if the question contains at least one deterministic relational
    indicator. Deliberately conservative: NON-graph cases (exact document
    lookup, exact quote, broad semantic lookup with no relational
    requirement) simply match nothing here and fall through unchanged --
    there is no negative pattern list, the absence of a positive match IS
    the non-graph decision."""
    if not question:
        return False
    return bool(_RELATIONAL_PATTERN.search(question))


# =====================================================================
# Part 3 -- deterministic entity resolution. Exact, case-insensitive,
# word-boundary match against THIS WORKSPACE's real canonical_label/alias
# values only. Never fuzzy, never LLM-assisted -- a mention that resolves
# to more than one distinct entity, or to none, is left unresolved.
# =====================================================================

def _load_workspace_entities(workspace_id: str) -> list[dict]:
    entities = bc.supabase.table("knowledge_entities") \
        .select("id,entity_type,canonical_label") \
        .eq("workspace_id", workspace_id).execute().data or []
    aliases = bc.supabase.table("knowledge_entity_aliases") \
        .select("entity_id,alias_text") \
        .eq("workspace_id", workspace_id).execute().data or []

    alias_by_entity: dict[str, list[str]] = {}
    for a in aliases:
        alias_by_entity.setdefault(a["entity_id"], []).append(a["alias_text"])

    for e in entities:
        e["_mention_texts"] = [e["canonical_label"], *alias_by_entity.get(e["id"], [])]
    return entities


def resolve_entity_mentions(question: str, workspace_id: str) -> list[dict]:
    """Returns the list of real entity rows (id/entity_type/canonical_label)
    this question deterministically references, or [] if nothing resolves
    or a mention is ambiguous. [] is a valid, expected, SAFE result -- the
    caller must treat it as "fall back to normal retrieval", never as an
    error to retry or guess through (Part 3's hard requirement)."""
    entities = _load_workspace_entities(workspace_id)
    q_lower = question.lower()

    matches: list[tuple[dict, str]] = []
    for e in entities:
        for text in e["_mention_texts"]:
            if not text:
                continue
            if re.search(r"\b" + re.escape(text.lower()) + r"\b", q_lower):
                matches.append((e, text))
                break  # one matched mention text per entity is enough

    if not matches:
        return []

    # Ambiguity: the SAME literal mention text matching more than one
    # DISTINCT entity id. Those mentions are dropped entirely -- never
    # guessed through.
    by_text: dict[str, set] = {}
    for e, text in matches:
        by_text.setdefault(text.lower(), set()).add(e["id"])
    ambiguous = {t for t, ids in by_text.items() if len(ids) > 1}
    matches = [(e, t) for e, t in matches if t.lower() not in ambiguous]
    if not matches:
        return []

    # Nesting: a shorter matched text that is a strict substring of another
    # matched (longer) text is subsumed by it -- e.g. "John" is dropped in
    # favor of "John Snow" when both literally match. Longest match wins,
    # never a fuzzy/semantic judgment.
    all_texts = [t for _, t in matches]
    kept: list[dict] = []
    seen_ids: set = set()
    for e, t in sorted(matches, key=lambda m: -len(m[1])):
        if e["id"] in seen_ids:
            continue
        subsumed = any(
            t.lower() != other.lower() and t.lower() in other.lower()
            for other in all_texts
        )
        if subsumed:
            continue
        kept.append(e)
        seen_ids.add(e["id"])
    return kept


# =====================================================================
# Part 13 -- stable internal graph context contract.
# =====================================================================

@dataclass
class GraphContext:
    matched_entities: list[gq.GraphEntity] = field(default_factory=list)
    relationships: list[gq.GraphRelationship] = field(default_factory=list)
    evidence: list[gq.GraphEvidence] = field(default_factory=list)
    traversal_depth: int = 2
    temporal_context: str = "current"


def build_graph_context(question: str, workspace_id: str, allowed_sensitivities: list[str],
                        as_of: Optional[datetime] = None) -> Optional[GraphContext]:
    """Orchestrates Part 2 -> Part 3 -> Part 4 (depth-2 traversal, reusing
    graph_query.get_entity_graph completely unchanged -- that function is
    ALREADY depth-2 by construction, so no new traversal logic exists here
    at all). Returns None whenever graph expansion should not happen:
    non-graph-relevant question, no entity resolved, or every resolved
    entity has zero VISIBLE relationships. None is the correct, safe
    "nothing to add" signal -- callers proceed with normal retrieval alone."""
    if not is_graph_relevant(question):
        return None

    resolved = resolve_entity_mentions(question, workspace_id)
    if not resolved:
        return None

    entities: list[gq.GraphEntity] = []
    rel_by_id: dict[str, gq.GraphRelationship] = {}
    for e in resolved:
        ge = gq.get_entity_graph(e["id"], workspace_id, allowed_sensitivities, as_of=as_of)
        if ge is None:
            continue
        entities.append(ge)
        for r in (*ge.inbound_relationships, *ge.outbound_relationships):
            rel_by_id[r.id] = r  # dedup by relationship id across resolved entities

    if not entities:
        return None

    relationships = list(rel_by_id.values())

    evidence_by_key: dict[tuple, gq.GraphEvidence] = {}
    for r in relationships:
        for ev in r.evidence:
            evidence_by_key[(ev.evidence_type, ev.evidence_id)] = ev

    return GraphContext(
        matched_entities=entities,
        relationships=relationships,
        evidence=list(evidence_by_key.values()),
        traversal_depth=2,
        temporal_context=(as_of.isoformat() if as_of else "current"),
    )


# =====================================================================
# Part 5 -- merging graph evidence into the SAME candidate list normal
# retrieval already produces, deduplicated by real source identity.
# =====================================================================

def _resolve_note_id_for_dedup(evidence_id: str) -> Optional[str]:
    """knowledge_note_source evidence overlaps with the embedded-chunk id
    space (document_chunks.document_id == the note's own id -- confirmed
    live against this workspace's real corpus before writing this module).
    structured_knowledge and calendar_event_snapshot are NEVER embedded
    (confirmed live, zero document_chunks rows reference either), so they
    never collide with a normal chunk and need no such resolution."""
    rows = bc.supabase.table("knowledge_note_sources").select("note_id") \
        .eq("id", evidence_id).execute().data
    return rows[0]["note_id"] if rows else None


def _dedup_key_for_evidence(ev: gq.GraphEvidence) -> str:
    if ev.evidence_type == "knowledge_note_source":
        note_id = _resolve_note_id_for_dedup(ev.evidence_id)
        if note_id:
            return f"document:{note_id}"
    return f"{ev.evidence_type}:{ev.evidence_id}"


def _evidence_strength(rel: gq.GraphRelationship) -> str:
    """'primary' | 'derived' -- Part 9's hard requirement: PRIMARY SOURCE
    (Calendar snapshot, Slack/Chat source) and DERIVED SUPPORT
    (a structured_knowledge primitive -- KNOVA's own interpretation) must
    never collapse into one meaning. 'primary' if ANY of the relationship's
    visible evidence is evidence_kind='primary_source'; 'derived' only when
    EVERY visible evidence record is 'derived_support'. Categorical, not a
    float -- there is no partial credit between the two meanings."""
    if any(ev.evidence_kind == "primary_source" for ev in rel.evidence):
        return "primary"
    return "derived"


def _relationship_candidate(rel: gq.GraphRelationship) -> dict:
    """One chunk-SHAPED dict per relationship (not per evidence record) --
    Part 5's own example ("relationship + its structured primitive + its
    Calendar/Chat evidence should become ONE evidence chain, not three
    unrelated hits") is satisfied by bundling every one of the
    relationship's VISIBLE evidence records into a single candidate's
    content/citation, never by emitting one candidate per evidence row.

    The content string is built ENTIRELY from real, already-resolved
    GraphRelationship/GraphEvidence fields -- relationship_type is stated
    literally (never paraphrased into invented English like "owns" or
    "manages"), and every evidence line quotes the real source_reference
    exactly as graph_query.py resolved it. No LLM involved in constructing
    this (Part 17's determinism requirement).

    Phase 5K.1 Part 2 -- CORRECTED: `similarity` is left None, same as any
    other non-semantically-scored candidate. `similarity` means real
    vector/keyword retrieval similarity ONLY, everywhere else in this
    codebase -- Phase 5K's `1.0`/`0.35` synthetic values were a semantic
    layering violation (a graph candidate has no semantic embedding at all,
    so it cannot honestly have a similarity score), even though the
    OBSERVABLE behavior they produced (graph-only answers no longer forced
    to "low") was correct. That observable behavior is now produced
    correctly instead, via evidence_strength here plus the dedicated
    graph_confidence()/combine_confidence() functions below -- a real graph
    signal, kept in its own field, never disguised as something it isn't."""
    label = f"{rel.source.label} → {rel.relationship_type} → {rel.target.label}"
    evidence_lines = [
        f"- ({ev.evidence_kind}) {ev.source_reference or 'no reference available'}"
        for ev in rel.evidence
    ] or ["- no visible evidence"]
    content = label + "\n\nEvidence:\n" + "\n".join(evidence_lines)
    return {
        "id":                 f"graph:{rel.id}",
        "document_id":        f"graph_relationship:{rel.id}",
        "content":            content,
        "metadata":           {"file_name": label, "source_type": "graph_relationship"},
        "source_type":        "graph_relationship",
        "source_tier":        1,
        # Real retrieval similarity ONLY -- None here is honest, not a gap.
        "similarity":         None,
        # Graph-specific field, kept separate from similarity/ranking on
        # purpose (Part 4: no field collision between semantic score and
        # graph evidence strength). `graph_only` is deliberately NOT
        # duplicated onto each candidate -- it describes the overall
        # retrieval mix (did any real chunk exist alongside graph
        # candidates), not a property of one candidate, so it stays exactly
        # where Phase 5K already put it: merge_graph_context_into_chunks's
        # metrics dict.
        "evidence_strength":  _evidence_strength(rel),
    }


def merge_graph_context_into_chunks(chunks: list[dict],
                                    graph_context: Optional[GraphContext]) -> tuple[list[dict], dict]:
    """Appends one candidate per graph relationship to `chunks`, skipping
    any relationship whose ENTIRE visible evidence set is already covered
    by a chunk already present (recognized via the SAME real source
    identity, never by text similarity). Never removes or reorders an
    existing chunk -- graph expansion only ever adds, matching the
    north star's "additional layer, not a replacement" framing.

    Returns (merged_chunks, metrics) -- metrics is the Part 15 diagnostic
    dict, printed by the caller (query.py) using this codebase's existing
    print-based timing convention (see chatbot.py's [chatbot][timing]
    lines), not a new DB table."""
    metrics = {
        "graph_queries_invoked":        0,
        "graph_entities_resolved":      0,
        "graph_relationships_found":    0,
        "graph_candidates_added":       0,
        "graph_candidates_deduplicated": 0,
        # Phase 5K Part 11 -- distinguishes a query answered PURELY from
        # graph context (no vector/keyword candidate at all going in) from
        # one where graph merely supplements normal retrieval. Computed,
        # never guessed: True only once at least one graph candidate is
        # actually added below AND the caller's own `chunks` started empty.
        "graph_only":                   False,
    }
    if graph_context is None:
        return chunks, metrics

    metrics["graph_queries_invoked"] = 1
    metrics["graph_entities_resolved"] = len(graph_context.matched_entities)
    metrics["graph_relationships_found"] = len(graph_context.relationships)

    if not graph_context.relationships:
        return chunks, metrics

    started_empty = not chunks
    existing_keys = {f"document:{c['document_id']}" for c in chunks if c.get("document_id")}

    merged = list(chunks)
    for rel in graph_context.relationships:
        if not rel.evidence:
            continue
        rel_keys = {_dedup_key_for_evidence(ev) for ev in rel.evidence}
        if rel_keys and rel_keys <= existing_keys:
            # every real source this relationship cites is already present
            # as a retrieved chunk -- nothing new to add.
            metrics["graph_candidates_deduplicated"] += 1
            continue
        merged.append(_relationship_candidate(rel))
        metrics["graph_candidates_added"] += 1

    metrics["graph_only"] = started_empty and metrics["graph_candidates_added"] > 0
    return merged, metrics


# =====================================================================
# Phase 5K.1 Part 3 -- the confidence contract, corrected. Three distinct
# concepts, per the North Star, never overloaded into one number:
#
#   VECTOR CONFIDENCE  -- query.py's/chatbot.py's OWN existing computation,
#                         `top_sim`-thresholded from real chunk similarity.
#                         Completely unchanged by this module. Since a graph
#                         candidate's `similarity` is now honestly None, it
#                         can no longer influence this number at all (Part 4:
#                         no field collision, no artificial domination).
#   GRAPH CONFIDENCE   -- graph_confidence() below. Computed ENTIRELY from
#                         graph-native signals (relationship presence +
#                         evidence_strength), never from a borrowed/fake
#                         similarity number.
#   ANSWER CONFIDENCE  -- combine_confidence() below. The single value
#                         query.py/chatbot.py actually report, folding the
#                         two together. Everything downstream of this
#                         (the "if not answer.strip(): confidence='none'"
#                         override, grounding.downgrade_for_weak_grounding)
#                         is untouched -- this only changes what feeds INTO
#                         that existing pipeline, not the pipeline itself.
# =====================================================================

_CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def graph_confidence(graph_context: Optional[GraphContext]) -> Optional[str]:
    """The graph's OWN confidence signal -- 'high' | 'medium' | None.

    'high': at least one relationship in the context is backed by at least
    one PRIMARY SOURCE evidence record (a real organizational artifact --
    Calendar snapshot, chat/note source, external reference -- not KNOVA's
    own interpretation). Deterministic entity identity (already required to
    reach build_graph_context at all) + an active, temporally-valid
    relationship (already required to reach graph_context.relationships at
    all, per the Phase 5K status/temporal filter) + primary evidence is the
    strongest claim this system can make -- Part 3's explicit HIGH case.

    'medium': the graph contributed relationships, but NONE of them carry
    primary-source evidence -- every one is backed only by DERIVED SUPPORT
    (a structured_knowledge row). Part 9's explicit rule: a derived
    primitive is never treated as equivalent to primary source evidence, so
    this never reaches 'high'.

    None: the graph contributed nothing at all (no context, or a context
    with zero relationships -- e.g. ambiguous entity resolution, or a
    real entity with no CURRENT relationships). Callers must not upgrade
    anything when this is None -- fall back to vector confidence entirely,
    per Part 3's "for weak/ambiguous graph context... fall back to existing
    retrieval confidence."

    Deliberately NOT '1.0'/'0.35' or any other float: Part 2 forbids
    representing this as fake similarity, and a bounded 3-value contract
    (high/medium/none) is the smallest one that still lets combine_confidence
    make a real decision without inventing a numeric scale nothing else in
    this codebase's confidence system uses."""
    if graph_context is None or not graph_context.relationships:
        return None
    has_primary = any(
        ev.evidence_kind == "primary_source"
        for rel in graph_context.relationships for ev in rel.evidence
    )
    return "high" if has_primary else "medium"


def combine_confidence(vector_confidence: str, graph_context: Optional[GraphContext]) -> str:
    """ANSWER CONFIDENCE (Part 3) -- the one value query.py/chatbot.py
    should report as their base confidence, before the existing "no answer
    text" override and grounding.downgrade_for_weak_grounding (both stay
    exactly as they were, applied by the caller afterward, unchanged).

    Rule: take the STRONGER of vector_confidence (computed by the caller
    exactly as before, from real chunk similarity only) and graph_confidence
    (computed above, from graph-native signals only). Never the other
    direction -- this function only ever raises confidence, and only when
    the graph genuinely, deterministically earned it; it never lowers
    vector_confidence, and never fabricates a value neither signal actually
    supports. This is the direct answer to Part 4's mixed-retrieval question:
    a strong vector result plus a real graph result simply keeps whichever
    of the two is already stronger -- there is no multiplication, no
    weighted average, no second ranking engine, just "the best-supported
    claim wins," computed from two clearly separate, honestly-labeled
    signals."""
    g_conf = graph_confidence(graph_context)
    if g_conf is None:
        return vector_confidence
    if _CONFIDENCE_RANK[g_conf] > _CONFIDENCE_RANK.get(vector_confidence, 0):
        return g_conf
    return vector_confidence
