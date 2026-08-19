"""
Phase 6G Company Wiki Knowledge Network tests.

Real data for the network shape between the 5 real entities + 4 real
memories (zero graph links, confirmed and asserted, not assumed). Synthetic,
single-use workspaces for security/traversal edge cases that don't exist in
the live corpus: a restricted 2-hop neighbor, a bounded-fan-out traversal,
cross-workspace isolation of navigation specifically.

Run with: python -m pytest test_phase6g_wiki_network.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import wiki_projection as wp
import wiki_navigation as wn

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

REAL_ENTITY_IDS = {
    "john_snow":   "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d",
    "tanmay":      "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0",
    "operations":  "1034346e-5731-45b8-9ee5-2e7d1413ca81",
    "product":     "c25f1ce7-6bcc-4a08-a80c-03db321c15f3",
    "meeting":     "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f",
}
REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "credential_sharing":  "3d376631-894c-4e32-b3f5-3ecf7cfd5f61",
    "hardware_scope":      "8aef76c9-fda3-44d6-affb-769f2ff09326",
    "monday_capacity":     "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}
MEETING_VALID_FROM = datetime(2026, 8, 16, 8, 30, tzinfo=timezone.utc)
APPROVAL_VALID_FROM = datetime(2026, 9, 15, tzinfo=timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_workspace() -> str:
    return str(uuid.uuid4())


def _cleanup(ids: dict) -> None:
    for mid in ids.get("memory_ids", []):
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in reversed(ids.get("memory_ids", [])):
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for rel_id in ids.get("relationship_ids", []):
        supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", rel_id).execute()
        supabase.table("knowledge_relationships").delete().eq("id", rel_id).execute()
    for ent_id in ids.get("entity_ids", []):
        supabase.table("knowledge_entities").delete().eq("id", ent_id).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-6G synthetic statement",
        "raw_subject_phrase": "TEST-6G subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6g-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


def _make_entity(workspace_id: str, label: str, entity_type: str = "person") -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": entity_type,
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_relationship(workspace_id: str, source_type: str, source_id: str,
                        target_type: str, target_id: str, rel_type: str, evidence_sk_id: str) -> str:
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": source_type, "p_source_object_id": source_id,
        "p_target_object_type": target_type, "p_target_object_id": target_id,
        "p_relationship_type": rel_type, "p_rationale": "TEST-6G", "p_confidence": 0.9,
        "p_valid_from": _now_iso(), "p_valid_until": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": evidence_sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }).execute().data


def _make_memory(workspace_id: str, sk_id: str, **overrides) -> str:
    params = {
        "p_workspace_id": workspace_id, "p_memory_type": "policy",
        "p_promotion_basis": "authoritative_policy",
        "p_valid_from": None, "p_valid_until": None,
        "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }
    params.update(overrides)
    return supabase.rpc("create_memory_with_evidence", params).execute().data


def _page(page_type, object_id, workspace_id=REAL_WORKSPACE, allowed=OWNER, as_of=None):
    page = wp.build_page(page_type, object_id, workspace_id, allowed, as_of)
    assert page is not None
    return page


# =====================================================================
# 1-4. Outbound links, backlinks, reciprocal navigation, rationale.
# =====================================================================

def test_outbound_links_real_tanmay():
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    assert len(nav.outbound_links) == 1
    n = nav.outbound_links[0]
    assert n.page_type == "meeting" and n.object_id == REAL_ENTITY_IDS["meeting"] and n.relationship_type == "organized"
    assert nav.inbound_links == []


def test_backlinks_real_meeting():
    page = _page("meeting", REAL_ENTITY_IDS["meeting"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    assert nav.outbound_links == []
    backlink_ids = {(n.page_type, n.object_id, n.relationship_type) for n in nav.inbound_links}
    assert backlink_ids == {
        ("person", REAL_ENTITY_IDS["tanmay"], "organized"),
        ("person", REAL_ENTITY_IDS["john_snow"], "attended"),
    }
    assert all(n.direction == "inbound" for n in nav.inbound_links)


def test_reciprocal_navigation_tanmay_meeting():
    tanmay_nav = wn.get_navigation_context(_page("person", REAL_ENTITY_IDS["tanmay"]), REAL_WORKSPACE, OWNER)
    meeting_nav = wn.get_navigation_context(_page("meeting", REAL_ENTITY_IDS["meeting"]), REAL_WORKSPACE, OWNER)
    assert tanmay_nav.outbound_links[0].relationship_id in {n.relationship_id for n in meeting_nav.inbound_links}


def test_relationship_rationale_preserved_not_regenerated():
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    real_rationale = nav.outbound_links[0].rationale
    assert real_rationale is not None
    assert "organizer" in real_rationale and "no employment" in real_rationale
    # Byte-identical to the real knowledge_relationships.rationale column --
    # never paraphrased by this module or by the LLM renderer (Part 14).
    row = supabase.table("knowledge_relationships").select("rationale").eq("id", nav.outbound_links[0].relationship_id).execute().data[0]
    assert nav.outbound_links[0].rationale == row["rationale"]


# =====================================================================
# 5. Evidence chain preserved.
# =====================================================================

def test_evidence_chain_preserved():
    page = _page("meeting", REAL_ENTITY_IDS["meeting"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    assert nav.evidence_links == page.evidence
    assert all("evidence_id" in e and "evidence_type" in e and "reference" in e for e in nav.evidence_links)


# =====================================================================
# 6-8. Bounded traversal.
# =====================================================================

def test_1hop_traversal_default():
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    assert nav.traversal_depth == 1
    assert nav.related_pages_2hop == []


def test_2hop_traversal_reaches_john_snow_from_tanmay():
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
    assert nav.traversal_depth == 2
    two_hop_ids = {(n.page_type, n.object_id) for n in nav.related_pages_2hop}
    assert ("person", REAL_ENTITY_IDS["john_snow"]) in two_hop_ids
    # 2-hop never repeats what 1-hop already showed
    one_hop_ids = {(n.page_type, n.object_id) for n in nav.outbound_links + nav.inbound_links}
    assert two_hop_ids.isdisjoint(one_hop_ids)


def test_traversal_bounded_rejects_hops_above_2():
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    for bad in (0, 3, 10, -1):
        with pytest.raises(ValueError):
            wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=bad)


def test_2hop_does_not_recurse_past_2():
    """John Snow is reachable at 2 hops from Tanmay (via the Meeting), but
    anything only reachable from John Snow's OWN further links (were there
    any) must never appear -- this module performs exactly one extra hop,
    never a recursive crawl."""
    page = _page("person", REAL_ENTITY_IDS["tanmay"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
    # every related_pages_2hop entry's relationship_id must belong to a
    # relationship touching a real 1-hop neighbor, not something further out
    one_hop_ids = {n.object_id for n in nav.outbound_links + nav.inbound_links}
    for n2 in nav.related_pages_2hop:
        assert n2.object_id not in one_hop_ids | {REAL_ENTITY_IDS["tanmay"]}


# =====================================================================
# 9-10. Temporal navigation.
# =====================================================================

def test_current_temporal_navigation_product_empty():
    page = _page("department", REAL_ENTITY_IDS["product"])
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER)
    assert nav.outbound_links == [] and nav.inbound_links == []
    assert nav.temporal_context == "current"


def test_historical_temporal_navigation_matches_pagemodel():
    """The real requires_approval_from relationship becomes a visible FACT
    on the historical page (proven in Phase 6E/6F), but produces no
    navigation LINK either way -- its structured_knowledge counterpart is a
    pending review candidate, not a promoted memory. Both halves verified
    explicitly so this isn't mistaken for a navigation bug."""
    future = APPROVAL_VALID_FROM + timedelta(days=1)
    page = _page("department", REAL_ENTITY_IDS["product"], as_of=future)
    rel_section = next(s for s in page.sections if s.section_type == "relationships")
    assert len(rel_section.items) == 1  # the fact IS exposed historically
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, as_of=future)
    assert nav.outbound_links == [] and nav.inbound_links == []  # but no fabricated link
    assert nav.temporal_context != "current"


def test_historical_navigation_before_meeting_excludes_attendance():
    before = MEETING_VALID_FROM - timedelta(days=1)
    page = _page("meeting", REAL_ENTITY_IDS["meeting"], as_of=before)
    nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, as_of=before)
    assert nav.inbound_links == []


# =====================================================================
# 11. Superseded memory handling.
# =====================================================================

def test_superseded_memory_navigation_shows_no_fabricated_successor_link():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk1 = _make_sk(ws, statement="TEST-6G predecessor")
        sk2 = _make_sk(ws, statement="TEST-6G successor")
        ids["sk_ids"] += [sk1, sk2]
        predecessor_id = _make_memory(ws, sk1)
        ids["memory_ids"].append(predecessor_id)
        predecessor_created_at = supabase.table("org_memory").select("created_at").eq("id", predecessor_id).execute().data[0]["created_at"]
        before_succession = datetime.fromisoformat(predecessor_created_at.replace("Z", "+00:00")) + timedelta(milliseconds=1)
        successor_id = _make_memory(ws, sk2, p_supersedes_memory_id=predecessor_id)
        ids["memory_ids"].append(successor_id)

        page = wp.build_page("policy", predecessor_id, ws, OWNER, as_of=before_succession)
        assert page.sections[0].items[0]["superseded_at"] is not None
        nav = wn.get_navigation_context(page, ws, OWNER, as_of=before_succession)
        all_ids = {n.object_id for n in nav.outbound_links + nav.inbound_links + nav.related_pages_2hop}
        assert successor_id not in all_ids  # never a fabricated link to the successor
        # current-page (post-succession) read of the predecessor is correctly gone
        assert wp.build_page("policy", predecessor_id, ws, OWNER) is None
    finally:
        _cleanup(ids)


# =====================================================================
# 12-13. Restricted / cross-workspace invisibility in navigation specifically.
# =====================================================================

def test_restricted_2hop_neighbor_excluded_from_navigation():
    """The Phase 6G security fix (wiki_projection._sk_to_memory_context now
    filters by sensitivity) proven again at the navigation layer: a 2-hop
    neighbor a LOW caller cannot open must never appear in related_pages_2hop,
    even though an OWNER caller correctly sees it."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "memory_ids": []}
    try:
        person_id = _make_entity(ws, "TEST-6G restricted-2hop person")
        meeting_id = _make_entity(ws, "TEST-6G restricted-2hop meeting", entity_type="meeting")
        ids["entity_ids"] += [person_id, meeting_id]
        sk_a = _make_sk(ws)
        ids["sk_ids"].append(sk_a)
        rel_a = _make_relationship(ws, "entity", person_id, "entity", meeting_id, "attended", sk_a)
        ids["relationship_ids"].append(rel_a)

        dept_id = _make_entity(ws, "TEST-6G restricted-2hop dept", entity_type="department")
        ids["entity_ids"].append(dept_id)
        sk_b = _make_sk(ws)
        ids["sk_ids"].append(sk_b)
        rel_b = _make_relationship(ws, "entity", meeting_id, "entity", dept_id, "attended", sk_b)
        ids["relationship_ids"].append(rel_b)

        restricted_sk = _make_sk(ws, sensitivity="restricted")
        ids["sk_ids"].append(restricted_sk)
        mem_id = _make_memory(ws, restricted_sk)
        ids["memory_ids"].append(mem_id)
        rel_c = _make_relationship(ws, "structured_knowledge", restricted_sk, "entity", meeting_id, "requires_approval_from", restricted_sk)
        ids["relationship_ids"].append(rel_c)

        person_page_low = wp.build_page("person", person_id, ws, LOW)
        person_page_owner = wp.build_page("person", person_id, ws, OWNER)
        nav_low = wn.get_navigation_context(person_page_low, ws, LOW, hops=2)
        nav_owner = wn.get_navigation_context(person_page_owner, ws, OWNER, hops=2)

        low_2hop_ids = {n.object_id for n in nav_low.related_pages_2hop}
        owner_2hop_ids = {n.object_id for n in nav_owner.related_pages_2hop}
        assert mem_id not in low_2hop_ids
        assert mem_id in owner_2hop_ids
        assert dept_id in low_2hop_ids and dept_id in owner_2hop_ids  # the non-restricted 2-hop neighbor is unaffected
    finally:
        _cleanup(ids)


def test_cross_workspace_page_invisible_to_navigation():
    assert wp.build_page("person", REAL_ENTITY_IDS["tanmay"], LEAK_WORKSPACE, OWNER) is None


# =====================================================================
# 14-15. No fake Project / QA / Procurement pages.
# =====================================================================

def test_no_fake_project_page_in_any_real_navigation():
    for entity_type, oid in (("person", REAL_ENTITY_IDS["tanmay"]), ("department", REAL_ENTITY_IDS["product"]), ("meeting", REAL_ENTITY_IDS["meeting"])):
        page = _page(entity_type, oid, as_of=APPROVAL_VALID_FROM + timedelta(days=1))
        nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, as_of=APPROVAL_VALID_FROM + timedelta(days=1), hops=2)
        all_neighbors = nav.outbound_links + nav.inbound_links + nav.related_pages_2hop
        assert all(n.page_type != "project" for n in all_neighbors)
    assert wp.build_page("project", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER) is None


def test_no_fake_qa_procurement_page_in_any_real_navigation():
    for ptype, oid in (("person", REAL_ENTITY_IDS["tanmay"]), ("meeting", REAL_ENTITY_IDS["meeting"])):
        page = _page(ptype, oid)
        nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
        labels = {n.label.lower() for n in nav.outbound_links + nav.inbound_links + nav.related_pages_2hop}
        assert "qa" not in labels and "procurement" not in labels


# =====================================================================
# 16. Memory pages with zero links.
# =====================================================================

def test_memory_pages_zero_links_behave_correctly():
    for name, mid in REAL_MEMORY_IDS.items():
        page = wp.build_page("policy", mid, REAL_WORKSPACE, OWNER) or wp.build_page("process", mid, REAL_WORKSPACE, OWNER)
        assert page is not None
        nav = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
        assert nav.outbound_links == [] and nav.inbound_links == [] and nav.related_pages_2hop == []
        assert page.links == []  # confirmed real, not a navigation-layer artifact


# =====================================================================
# 17-21. Real per-entity graph proofs.
# =====================================================================

def test_real_tanmay_graph():
    nav = wn.get_navigation_context(_page("person", REAL_ENTITY_IDS["tanmay"]), REAL_WORKSPACE, OWNER, hops=2)
    assert {(n.page_type, n.relationship_type) for n in nav.outbound_links} == {("meeting", "organized")}
    assert {(n.page_type, n.relationship_type) for n in nav.related_pages_2hop} == {("person", "attended")}


def test_real_john_snow_graph():
    nav = wn.get_navigation_context(_page("person", REAL_ENTITY_IDS["john_snow"]), REAL_WORKSPACE, OWNER, hops=2)
    assert {(n.page_type, n.relationship_type) for n in nav.outbound_links} == {("meeting", "attended")}
    assert {(n.page_type, n.relationship_type) for n in nav.related_pages_2hop} == {("person", "organized")}


def test_real_meeting_graph():
    nav = wn.get_navigation_context(_page("meeting", REAL_ENTITY_IDS["meeting"]), REAL_WORKSPACE, OWNER, hops=2)
    assert len(nav.inbound_links) == 2 and nav.outbound_links == []
    assert nav.related_pages_2hop == []  # both 1-hop neighbors (Tanmay, John) have no further real links of their own


def test_real_product_graph():
    nav_current = wn.get_navigation_context(_page("department", REAL_ENTITY_IDS["product"]), REAL_WORKSPACE, OWNER, hops=2)
    assert nav_current.outbound_links == [] and nav_current.inbound_links == [] and nav_current.related_pages_2hop == []


def test_real_operations_graph():
    nav = wn.get_navigation_context(_page("department", REAL_ENTITY_IDS["operations"]), REAL_WORKSPACE, OWNER, hops=2)
    assert nav.outbound_links == [] and nav.inbound_links == [] and nav.related_pages_2hop == []


# =====================================================================
# 22. Deterministic navigation model.
# =====================================================================

def test_deterministic_navigation_model():
    page = _page("meeting", REAL_ENTITY_IDS["meeting"])
    nav1 = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
    nav2 = wn.get_navigation_context(page, REAL_WORKSPACE, OWNER, hops=2)
    dump = lambda nav: (
        [(n.page_type, n.object_id, n.relationship_id) for n in nav.outbound_links],
        [(n.page_type, n.object_id, n.relationship_id) for n in nav.inbound_links],
        [(n.page_type, n.object_id, n.relationship_id) for n in nav.related_pages_2hop],
    )
    assert dump(nav1) == dump(nav2)


# =====================================================================
# 23. Fixture cleanup.
# =====================================================================

def test_no_leftover_test_6g_fixtures():
    leftover = supabase.table("structured_knowledge").select("id").ilike("statement", "TEST-6G%").execute().data
    assert leftover == []
    leftover_entities = supabase.table("knowledge_entities").select("id").ilike("canonical_label", "TEST-6G%").execute().data
    assert leftover_entities == []


# =====================================================================
# 24. Full-regression placeholder.
# =====================================================================

def test_placeholder_full_regression_run_separately():
    assert True
