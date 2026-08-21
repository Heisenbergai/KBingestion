"""
Phase 7C -- Cross-Department Intelligence: evidence-bounded organizational
impact analysis.

Answers "when something is known in one part of the organization, what other
parts are EXPLICITLY affected by it?" -- by tracing real, evidence-backed
relationship chains, never by finding things that merely sound related.

FROZEN RULE (Part 2): SEMANTIC SIMILARITY IS NOT ORGANIZATIONAL IMPACT.
There is no embedding call, no keyword overlap, no frequency count, and no
centrality heuristic anywhere in this module. An impact path exists only
when every hop in it is a real row in knowledge_relationships whose evidence
is visible to the caller. If the ontology cannot express a connection, the
honest answer is that no path exists -- absence of an edge is a valid
result, not a gap to paper over (Part 18).

NO SECOND GRAPH ENGINE (Part 1): traversal is composed entirely from
graph_query.get_entity_graph / get_structured_knowledge_graph, both used
COMPLETELY UNCHANGED. Those functions already return, per node, every
VISIBLE inbound/outbound relationship valid at `as_of`, each with its
resolved endpoint labels and its real evidence -- which is exactly one hop.
This module only walks that existing one-hop primitive at most twice and
records what it found. It writes nothing, creates no edge, and adds no
relationship type.

BOUNDED (Part 5/16): max_hops is 1 or 2, never more. Hop 2 issues one
get_entity_graph call per distinct visible hop-1 counterpart -- bounded by
the real graph's tiny fan-out. Deliberately NOT parallelized: Phase 6H.1
implemented and then reverted a thread-pool optimization here after a real,
reproducible Windows HTTP/2 socket failure under sustained load. Batching
where possible, sequential otherwise, is the lesson that stuck.

CLASSIFICATION (Part 6), assigned deterministically by path shape -- never
by a model, and never self-reported:
  OBSERVED  -- ONE real relationship directly connects origin and target.
               The evidence states the connection itself.
  DERIVED   -- TWO real relationships chained through a real shared
               intermediate node. Nothing is inferred: both hops are
               persisted rows, and the intermediate is a real entity/
               structured_knowledge id present in both.
  UNKNOWN   -- no such chain exists within the bound. Returned explicitly
               for a named target so the caller can say "the evidence does
               not establish this" rather than staying silent (Part 7's
               worked example).
INFERRED is never produced by traversal. This module cannot manufacture a
hypothesis, because every path it emits is made of persisted rows. When a
caller explicitly asks "what MIGHT be affected", the honest output is the
established paths plus explicit not_established entries -- which is exactly
the Part 7 "Good" example, and structurally cannot become the "Bad" one.

RELATIONSHIP SEMANTICS ARE NEVER RESTATED (Part 3/13): a path records the
real relationship_type verbatim and reuses the real rationale text. This
module never translates 'requires_approval_from' into 'owns', never turns
'attended' into 'works in', and never converts an edge between A and B into
a claim about A's authority over B. The explanation it builds is a literal
recitation of what the rows say.

SECURITY (Part 11): inherited, never re-implemented. graph_query already
drops any relationship with zero caller-visible evidence, treating it as
non-existent for that caller. A path that would require an invisible
relationship therefore never forms -- the caller sees the best conclusion
their visible evidence supports, with no placeholder, no count, and no hint
that something was withheld.

NO WRITES (Part 19): no insert/update/delete/upsert/rpc anywhere.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import graph_query

OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
UNKNOWN = "UNKNOWN"

_VALID_HOPS = (1, 2)


@dataclass
class ImpactNode:
    kind: str                 # 'entity' | 'structured_knowledge'
    object_id: str
    label: Optional[str]
    object_type: Optional[str] = None      # entity_type when known


@dataclass
class ImpactHop:
    relationship_id: str
    relationship_type: str
    direction: str            # 'outbound' | 'inbound' relative to the node it was walked FROM
    from_node: ImpactNode
    to_node: ImpactNode
    rationale: Optional[str]
    evidence_ids: list = field(default_factory=list)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


@dataclass
class ImpactPath:
    source: ImpactNode
    target: ImpactNode
    hops: int
    relationship_ids: list = field(default_factory=list)
    evidence_ids: list = field(default_factory=list)
    temporal_context: str = "current"
    reasoning_state: str = OBSERVED
    explanation: str = ""
    chain: list = field(default_factory=list)          # ImpactHop, in order


@dataclass
class ImpactResult:
    origin: ImpactNode
    paths: list = field(default_factory=list)          # ImpactPath
    not_established: list = field(default_factory=list)  # [{target_label, reason, reasoning_state}]
    temporal_context: str = "current"
    max_hops: int = 2
    relationships_examined: int = 0
    graph_queries: int = 0


def _node_from_endpoint(ep, entity_types: Optional[dict] = None) -> ImpactNode:
    return ImpactNode(
        kind=ep.object_type, object_id=ep.object_id, label=ep.label,
        object_type=(entity_types or {}).get(ep.object_id),
    )


def _load_node_graph(node: ImpactNode, workspace_id: str, allowed_sensitivities: list[str],
                      as_of: Optional[datetime]):
    """One hop, via the EXISTING primitive. Returns (relationships, self_node)
    or (None, None) when the node isn't visible/doesn't exist -- which is
    indistinguishable, by design, from it not existing at all."""
    if node.kind == "entity":
        graph = graph_query.get_entity_graph(node.object_id, workspace_id, allowed_sensitivities, as_of)
        if graph is None:
            return None, None
        rels = list(graph.inbound_relationships) + list(graph.outbound_relationships)
        return rels, ImpactNode("entity", graph.id, graph.canonical_label, graph.entity_type)
    if node.kind == "structured_knowledge":
        sk = graph_query.get_structured_knowledge_graph(node.object_id, workspace_id,
                                                         allowed_sensitivities, as_of)
        if sk is None:
            return None, None
        rels = list(sk["outbound_relationships"]) + list(sk["inbound_relationships"])
        return rels, ImpactNode("structured_knowledge", sk["id"], sk["statement"], None)
    return None, None


def _hop_from_relationship(rel, from_node: ImpactNode) -> ImpactHop:
    is_outbound = (rel.source.object_type == from_node.kind and
                    rel.source.object_id == from_node.object_id)
    counterpart = rel.target if is_outbound else rel.source
    return ImpactHop(
        relationship_id=rel.id, relationship_type=rel.relationship_type,
        direction="outbound" if is_outbound else "inbound",
        from_node=from_node, to_node=_node_from_endpoint(counterpart),
        rationale=rel.rationale,
        evidence_ids=[f"{ev.evidence_type}:{ev.evidence_id}" for ev in rel.evidence],
        valid_from=rel.valid_from, valid_until=rel.valid_until,
    )


def _phrase(relationship_type: str, direction: str) -> str:
    """Literal recitation of the real relationship_type -- never a synonym
    that adds authority the edge does not carry. Inbound is rendered as the
    passive form of the SAME verb, which changes grammar only, never
    semantics (Part 13's hard line: 'attended' must never become
    'works in', 'requires_approval_from' must never become 'owns')."""
    verb = relationship_type.replace("_", " ")
    return verb if direction == "outbound" else f"is the target of '{verb}' from"


def _explain(chain: list) -> str:
    parts = []
    for hop in chain:
        src = hop.from_node.label or hop.from_node.object_id
        tgt = hop.to_node.label or hop.to_node.object_id
        if hop.direction == "outbound":
            parts.append(f"{src} --{hop.relationship_type}--> {tgt}")
        else:
            parts.append(f"{tgt} --{hop.relationship_type}--> {src}")
    return " ; then ".join(parts)


def analyze_impact(origin_kind: str, origin_id: str, workspace_id: str,
                    allowed_sensitivities: list[str], as_of: Optional[datetime] = None,
                    max_hops: int = 2, candidate_targets: Optional[list] = None) -> ImpactResult:
    """The single public entry point.

    `candidate_targets` is an optional list of {'kind','object_id','label'}
    the caller explicitly asked about (e.g. "is Sales affected?"). Any such
    target with no real path is returned in `not_established` with state
    UNKNOWN -- producing Part 7's Good example ("evidence explicitly links
    Product...; not enough evidence to conclude Sales is affected") instead
    of silence, and never the Bad one.
    """
    if max_hops not in _VALID_HOPS:
        raise ValueError(f"max_hops must be one of {_VALID_HOPS} -- Phase 7C's bounded-traversal rule")

    temporal_context = as_of.isoformat() if as_of else "current"
    origin = ImpactNode(origin_kind, origin_id, None)
    rels, resolved_origin = _load_node_graph(origin, workspace_id, allowed_sensitivities, as_of)
    graph_queries = 1
    if rels is None:
        return ImpactResult(origin=origin, paths=[], not_established=[],
                             temporal_context=temporal_context, max_hops=max_hops,
                             relationships_examined=0, graph_queries=graph_queries)
    origin = resolved_origin

    paths: list[ImpactPath] = []
    relationships_examined = len(rels)
    origin_key = (origin.kind, origin.object_id)
    seen_targets = {origin_key}

    hop1: list[ImpactHop] = []
    for rel in rels:
        hop = _hop_from_relationship(rel, origin)
        hop1.append(hop)
        target_key = (hop.to_node.kind, hop.to_node.object_id)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        paths.append(ImpactPath(
            source=origin, target=hop.to_node, hops=1,
            relationship_ids=[hop.relationship_id], evidence_ids=list(hop.evidence_ids),
            temporal_context=temporal_context, reasoning_state=OBSERVED,
            explanation=_explain([hop]), chain=[hop],
        ))

    if max_hops == 2:
        for first in hop1:
            mid = first.to_node
            mid_rels, resolved_mid = _load_node_graph(mid, workspace_id, allowed_sensitivities, as_of)
            graph_queries += 1
            if mid_rels is None:
                continue
            mid = resolved_mid
            relationships_examined += len(mid_rels)
            for rel2 in mid_rels:
                if rel2.id == first.relationship_id:
                    continue          # the same edge walked back -- not a second hop
                second = _hop_from_relationship(rel2, mid)
                target_key = (second.to_node.kind, second.to_node.object_id)
                if target_key in seen_targets:
                    continue          # already reachable at 1 hop, or is the origin
                seen_targets.add(target_key)
                chain = [first, second]
                paths.append(ImpactPath(
                    source=origin, target=second.to_node, hops=2,
                    relationship_ids=[first.relationship_id, second.relationship_id],
                    evidence_ids=list(first.evidence_ids) + list(second.evidence_ids),
                    temporal_context=temporal_context, reasoning_state=DERIVED,
                    explanation=_explain(chain), chain=chain,
                ))

    not_established = []
    if candidate_targets:
        reachable = {(p.target.kind, p.target.object_id) for p in paths}
        reachable_labels = {(p.target.label or "").lower() for p in paths}
        for cand in candidate_targets:
            key = (cand.get("kind"), cand.get("object_id"))
            label = (cand.get("label") or "").strip()
            if key in reachable or (label and label.lower() in reachable_labels):
                continue
            not_established.append({
                "target_label": label or cand.get("object_id"),
                "reasoning_state": UNKNOWN,
                "reason": (f"No evidence-backed relationship path of {max_hops} hop(s) or fewer connects "
                            f"{origin.label or origin.object_id} to this target in the visible graph."),
            })

    return ImpactResult(
        origin=origin, paths=paths, not_established=not_established,
        temporal_context=temporal_context, max_hops=max_hops,
        relationships_examined=relationships_examined, graph_queries=graph_queries,
    )


# =====================================================================
# Part 15 -- optional reasoning context. Impact analysis is NOT mandatory
# for every question and never becomes an answer engine: this converts
# established paths into the SAME chunk-shape the existing pipeline already
# merges, so reasoning.py classifies them exactly like any other claim.
# =====================================================================

def impact_paths_as_claim_rows(result: ImpactResult) -> list[dict]:
    """Chunk-shaped dicts for the existing merge/claim-inventory path. Uses
    the same honest conventions graph_retrieval already established:
    similarity is None (an impact path has no semantic embedding), and the
    content states the real relationship chain literally.

    Deliberately returns rows rather than calling reasoning.py itself --
    the reasoner stays the sole authority on OBSERVED/DERIVED/INFERRED/
    UNKNOWN, and this module never assigns a claim's final state (Part 15)."""
    rows = []
    for p in result.paths:
        rows.append({
            "id": f"impact:{'-'.join(p.relationship_ids)}",
            "document_id": f"impact_path:{'-'.join(p.relationship_ids)}",
            "content": (f"Evidence-backed organizational path ({p.hops} hop"
                         f"{'s' if p.hops > 1 else ''}): {p.explanation}"),
            "metadata": {"file_name": f"Impact path ({p.hops} hop)", "source_type": "impact_path"},
            "source_type": "impact_path",
            "source_tier": 1,
            "similarity": None,
        })
    return rows
