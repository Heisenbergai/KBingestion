"""
Phase 6G -- Company Wiki Knowledge Network: bounded graph navigation
(outbound links, backlinks, 1-2 hop neighborhood) over Phase 6E's
WikiPageModel / WikiLink. The graph is the navigation backbone -- this
module adds NO new relationship storage, NO new query beyond calling
wiki_projection.build_page (already fully audited, security-correct, and
temporally-correct) once per node actually visited.

PART 2 DECISION (contract sufficiency), reached after a fresh re-read of
wiki_projection.py, not assumed from the Phase 6E report: the existing
WikiLink (target_page_type, target_id, label, relationship_id,
relationship_type, rationale) is SUFFICIENT for V1 navigation and is reused
completely UNCHANGED. Backlinks require knowing which DIRECTION a link runs
(is this page the relationship's source or target?) -- WikiLink itself
carries no direction field, but it doesn't need one: every WikiLink's
`relationship_id` already matches exactly one item in that SAME page's own
Relationships section (WikiSection section_type='relationships'), and that
item already carries `direction` ('outbound'/'inbound'). This is
STRUCTURALLY guaranteed, not incidental -- wiki_projection._build_entity_page
and _build_memory_page both build their `links` and `relationship_items`
from the exact same filtered `relationships` list in the same function call
(see wiki_projection.py lines building `relationship_items` and `links`
back to back from the identical `relationships` variable). Cross-referencing
by `relationship_id` (_link_direction, below) recovers direction with zero
new fields, zero new storage, and zero new queries -- the correct reading of
"if sufficient, reuse it unchanged," not a shortcut.

SECURITY (found during this phase's mandated fresh audit, fixed in
wiki_projection.py, not here): _sk_to_memory_context() previously built its
memory-linking context without applying the caller's sensitivity ladder,
so a WikiLink could point at a memory the caller could not actually open --
a "hidden page" existence leak. See wiki_projection.py's own module
docstring (Phase 6G addendum) for the full writeup; this module inherits the
fix automatically since it only ever consumes already-built WikiPageModels.

Every navigation function in this module is a pure composition over
wiki_projection.build_page -- it never imports brain_connectors, never
queries Supabase directly, and never invents an edge. A 2-hop neighbor is
independently, freshly re-gated by the SAME build_page() call any direct
navigation to it would use; an invisible or nonexistent 2-hop neighbor is
silently skipped, never shown as a placeholder (Part 15).

BOUNDED TRAVERSAL (Part 8): hops is 1 (default) or 2, never more. 2-hop
expansion calls build_page() once per already-visible 1-hop neighbor --
bounded by that neighbor count, which the real graph keeps tiny (at most a
handful of links off any real page today). This is intentionally NOT
batched into a single bulk query: doing so would mean reaching into
wiki_projection.py's internals to add a new bulk-fetch code path, which
Phase 6G's own "do not redesign WikiPageModel" instruction rules out.
Documented explicitly in the final report's Performance section as a real,
deliberate, bounded N+1 shape -- correct and safe at current real scale,
not free at a much larger hypothetical scale, and not pre-optimized against
that hypothetical (Part 22: "optimize for correctness... do not prematurely
introduce graph databases").

NOT BUILT (deliberate V1 scope, explained in the final report rather than
silently omitted):
  - Server-side breadcrumbs / hierarchy: Part 13 explicitly warns against
    implying a hierarchy that doesn't exist ("Product -> Meeting does not
    mean Meeting is inside Product"). A breadcrumb trail is fundamentally
    client-side navigation-HISTORY state (what the user actually clicked),
    not something this module can honestly compute server-side. No
    parent/child field exists anywhere in WikiNavigationContext, so the
    frontend cannot accidentally render a fake hierarchy even if it tried.
  - Arbitrary path-finding between two pages (the Part 11 North Star
    example chain): the bounded 1-2 hop neighborhood IS the V1 mechanism
    for exploring cross-department connections -- a user follows real edges
    hop by hop. No recursive/unbounded graph search is built; if two pages
    aren't connected within 2 hops, the Wiki has nothing dishonest to show
    and shows nothing, rather than searching further or inventing a link.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import wiki_projection
from wiki_projection import WikiPageModel, WikiLink

_VALID_HOPS = (1, 2)


@dataclass
class NeighborRef:
    page_type: str
    object_id: str
    label: str
    relationship_type: str
    relationship_id: str
    rationale: Optional[str]
    direction: str          # 'outbound' | 'inbound' -- relative to the page this ref was found FROM


@dataclass
class WikiNavigationContext:
    current_page: dict                          # {page_type, object_id, title} -- an identity stub, not the full page
    outbound_links: list[NeighborRef] = field(default_factory=list)
    inbound_links: list[NeighborRef] = field(default_factory=list)     # backlinks
    related_pages_2hop: list[NeighborRef] = field(default_factory=list)  # only populated when hops=2; excludes anything already at 1-hop
    evidence_links: list[dict] = field(default_factory=list)           # pass-through of page.evidence -- Part 5's "C"
    temporal_context: str = "current"
    traversal_depth: int = 1


def _link_direction(page: WikiPageModel, link: WikiLink) -> str:
    """See module docstring's Part 2 decision. Falls back to 'outbound' only
    as an explicit, safe default -- structurally unreachable given every
    real WikiLink is built from the same relationships list as the
    Relationships section, but a silent KeyError would be worse than a
    documented, impossible-in-practice default."""
    rel_section = next((s for s in page.sections if s.section_type == "relationships"), None)
    if rel_section:
        for item in rel_section.items:
            if item["relationship_id"] == link.relationship_id:
                return item["direction"]
    return "outbound"


def _to_neighbor_ref(link: WikiLink, direction: str) -> NeighborRef:
    return NeighborRef(page_type=link.target_page_type, object_id=link.target_id, label=link.label,
                        relationship_type=link.relationship_type, relationship_id=link.relationship_id,
                        rationale=link.rationale, direction=direction)


def get_navigation_context(page: WikiPageModel, workspace_id: str, allowed_sensitivities: list[str],
                            as_of: Optional[datetime] = None, hops: int = 1) -> WikiNavigationContext:
    """The single entry point this phase adds. `page` must already be a
    real, built WikiPageModel (from wiki_projection.build_page) -- this
    function never builds the origin page itself, only its neighborhood.
    `as_of` should be the SAME value used to build `page` -- passing a
    different one would silently desynchronize the neighborhood's temporal
    frame from the page's own, so this is the caller's responsibility, not
    re-derived here (mirrors wiki_projection's own "the caller already
    decided as_of" pattern)."""
    if hops not in _VALID_HOPS:
        raise ValueError(f"hops must be one of {_VALID_HOPS} -- Phase 6G's own bounded-traversal requirement")

    outbound: list[NeighborRef] = []
    inbound: list[NeighborRef] = []
    for link in page.links:
        direction = _link_direction(page, link)
        (outbound if direction == "outbound" else inbound).append(_to_neighbor_ref(link, direction))

    related_2hop: list[NeighborRef] = []
    if hops == 2:
        origin_id = page.canonical_entity_id or page.memory_id
        seen = {(page.page_type, origin_id)} | {(n.page_type, n.object_id) for n in outbound + inbound}
        for n in outbound + inbound:
            hop2_page = wiki_projection.build_page(n.page_type, n.object_id, workspace_id, allowed_sensitivities, as_of)
            if hop2_page is None:
                continue  # invisible or since-deleted -- never shown as a stub (Part 15)
            for link2 in hop2_page.links:
                key = (link2.target_page_type, link2.target_id)
                if key in seen:
                    continue
                seen.add(key)
                related_2hop.append(_to_neighbor_ref(link2, _link_direction(hop2_page, link2)))

    origin_id = page.canonical_entity_id or page.memory_id
    return WikiNavigationContext(
        current_page={"page_type": page.page_type, "object_id": origin_id, "title": page.title},
        outbound_links=outbound, inbound_links=inbound, related_pages_2hop=related_2hop,
        evidence_links=list(page.evidence), temporal_context=page.temporal_context, traversal_depth=hops,
    )
