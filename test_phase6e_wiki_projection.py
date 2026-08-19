"""
Phase 6E Company Wiki Foundation tests.

Real data for the 9 real pages (5 entities + 4 memories) and their real
cross-linking behavior (including the real pending review candidate and the
real future-dated relationship). Synthetic, single-use workspaces for
security/determinism/linking edge cases that don't exist in the live corpus
(a restricted-sensitivity structured_knowledge counterpart, a structured_
knowledge row grounding both a relationship AND a memory, a controlled
before/after hash-change comparison).

Run with: python -m pytest test_phase6e_wiki_projection.py -v
"""
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import memory_retrieval as mr
import wiki_projection as wp

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)  # ['public', 'internal']

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
SK_Q4_LAUNCH_APPROVAL = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"  # real pending review candidate; also a real relationship source
MEETING_VALID_FROM = datetime(2026, 8, 16, 8, 30, tzinfo=timezone.utc)
APPROVAL_VALID_FROM = datetime(2026, 9, 15, tzinfo=timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


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
        "primitive_type": "fact", "statement": "TEST-6E synthetic statement",
        "raw_subject_phrase": "TEST-6E subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6e-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


def _make_entity(workspace_id: str, label: str, entity_type: str = "person") -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": entity_type,
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_relationship(workspace_id: str, source_type: str, source_id: str,
                        target_type: str, target_id: str, rel_type: str, evidence_sk_id: str,
                        rationale: str = "TEST-6E") -> str:
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": source_type, "p_source_object_id": source_id,
        "p_target_object_type": target_type, "p_target_object_id": target_id,
        "p_relationship_type": rel_type, "p_rationale": rationale, "p_confidence": 0.9,
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


# =====================================================================
# 1-6. Entity pages -- real data.
# =====================================================================

def test_person_page_real_john_snow():
    page = wp.build_person_page(REAL_ENTITY_IDS["john_snow"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert page.page_type == "person" and page.title == "John Snow"
    identity = page.sections[0].items[0]
    assert identity["identifiers"] == [{"identifier_type": "email", "identifier_value": "kingjohnsnow0@gmail.com"}]
    rel_types = {r["relationship_type"] for r in page.sections[1].items}
    assert rel_types == {"attended"}
    assert any(e["evidence_role"] == "activity" for e in page.sections[2].items)


def test_person_page_real_tanmay_two_identifiers_two_evidence():
    page = wp.build_person_page(REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    assert page is not None
    identity = page.sections[0].items[0]
    assert len(identity["identifiers"]) == 2
    assert len(page.evidence) == 2
    assert page.sections[1].items[0]["relationship_type"] == "organized"


def test_department_page_real_operations_empty_is_valid_not_none():
    page = wp.build_department_page(REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert page.sections[1].items == [] and page.sections[2].items == []
    assert page.links == []


def test_department_page_real_product_current_excludes_future_relationship():
    page = wp.build_department_page(REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert page.sections[1].items == []  # requires_approval_from starts 2026-09-15, not yet current


def test_department_page_real_product_asof_future_includes_relationship_but_no_link():
    future = APPROVAL_VALID_FROM + timedelta(days=1)
    page = wp.build_department_page(REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER, as_of=future)
    assert page is not None
    assert len(page.sections[1].items) == 1
    item = page.sections[1].items[0]
    assert item["relationship_type"] == "requires_approval_from"
    assert item["counterpart_type"] == "structured_knowledge"
    assert "QA" in item["rationale"]  # honest quotation of real source text, not a fabricated QA entity
    # The grounding sk (SK_Q4_LAUNCH_APPROVAL) is a real PENDING REVIEW
    # CANDIDATE, not a promoted memory -- the raw relationship fact shows,
    # but it must never become a page-to-page Wiki link.
    assert page.links == []


def test_meeting_page_real_both_attendance_edges_and_identity_evidence():
    page = wp.build_meeting_page(REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER)
    assert page is not None
    rel_types = {(r["relationship_type"], r["counterpart_id"]) for r in page.sections[1].items}
    assert rel_types == {("organized", REAL_ENTITY_IDS["tanmay"]), ("attended", REAL_ENTITY_IDS["john_snow"])}
    assert page.sections[2].items[0]["evidence_role"] == "identity"
    link_targets = {(l.target_page_type, l.target_id) for l in page.links}
    assert link_targets == {("person", REAL_ENTITY_IDS["tanmay"]), ("person", REAL_ENTITY_IDS["john_snow"])}


# =====================================================================
# 7-12. Memory pages -- real data.
# =====================================================================

def test_policy_page_real_credential_logging():
    page = wp.build_policy_page(REAL_MEMORY_IDS["credential_logging"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert page.page_type == "policy"
    assert "recorded in the security log" in page.title
    assert page.sections[0].items[0]["promotion_basis"] == "authoritative_policy"


def test_policy_page_real_credential_sharing():
    page = wp.build_policy_page(REAL_MEMORY_IDS["credential_sharing"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert "Slack" in page.title


def test_policy_page_real_hardware_scope():
    page = wp.build_policy_page(REAL_MEMORY_IDS["hardware_scope"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert "hardware categories" in page.title


def test_process_page_real_monday_capacity():
    page = wp.build_process_page(REAL_MEMORY_IDS["monday_capacity"], REAL_WORKSPACE, OWNER)
    assert page is not None
    assert page.page_type == "process"
    assert page.sections[0].items[0]["promotion_basis"] == "recurring_durable_process"


def test_memory_page_titles_distinct_not_note_title_collision():
    """Real finding from this phase's own proof pass: credential_logging and
    credential_sharing are both grounded in the SAME source note. Titling
    from the deeper note-level reference (as citation display does) makes
    both pages show the identical, non-distinguishing note title. Fixed to
    prefer each memory's own specific grounding statement."""
    p1 = wp.build_policy_page(REAL_MEMORY_IDS["credential_logging"], REAL_WORKSPACE, OWNER)
    p2 = wp.build_policy_page(REAL_MEMORY_IDS["credential_sharing"], REAL_WORKSPACE, OWNER)
    assert p1.title != p2.title


def test_decision_page_no_real_decision_memory_exists():
    """Documented live gap, not a bug: zero memory_type='decision' rows
    exist in the real corpus today (0 decisions have ever been promoted).
    build_decision_page is exercised structurally elsewhere (wrong-type and
    synthetic tests); this asserts the honest current-state absence."""
    rows = supabase.table("org_memory").select("id").eq("workspace_id", REAL_WORKSPACE).eq("memory_type", "decision").execute().data
    assert rows == []


# =====================================================================
# 13-16. Dispatch / wrong type / nonexistent.
# =====================================================================

def test_build_page_unknown_page_type_returns_none():
    assert wp.build_page("project", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER) is None


def test_wrong_page_type_for_real_entity_returns_none():
    assert wp.build_person_page(REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER) is None
    assert wp.build_meeting_page(REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER) is None


def test_wrong_page_type_for_real_memory_returns_none():
    assert wp.build_decision_page(REAL_MEMORY_IDS["credential_logging"], REAL_WORKSPACE, OWNER) is None
    assert wp.build_process_page(REAL_MEMORY_IDS["hardware_scope"], REAL_WORKSPACE, OWNER) is None


def test_build_page_nonexistent_id_returns_none():
    fake_id = str(uuid.uuid4())
    assert wp.build_page("person", fake_id, REAL_WORKSPACE, OWNER) is None
    assert wp.build_page("policy", fake_id, REAL_WORKSPACE, OWNER) is None


# =====================================================================
# 17-22. Negative / anti-hallucination checks (Part 14).
# =====================================================================

def test_qa_and_procurement_never_become_entities_or_links():
    """QA/Procurement have no knowledge_entities row (confirmed live) --
    structurally impossible for either to appear as a page or a link
    target anywhere in this module's output."""
    rows = supabase.table("knowledge_entities").select("canonical_label").eq("workspace_id", REAL_WORKSPACE).execute().data
    labels_lower = {r["canonical_label"].lower() for r in rows}
    assert "qa" not in labels_lower and "procurement" not in labels_lower

    future = APPROVAL_VALID_FROM + timedelta(days=1)
    page = wp.build_department_page(REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER, as_of=future)
    assert all(l.target_page_type != "project" for l in page.links)
    assert all(l.label.lower() not in ("qa", "procurement") for l in page.links)


def test_no_ownership_or_employment_relationship_type_in_frozen_ontology():
    rows = supabase.table("knowledge_relationships").select("relationship_type").eq("workspace_id", REAL_WORKSPACE).execute().data
    types = {r["relationship_type"] for r in rows}
    assert types.isdisjoint({"owns", "works_on", "employs", "employee_of", "member_of", "manages"})


def test_q4_launch_not_a_project_entity_anywhere():
    rows = supabase.table("knowledge_entities").select("id").eq("workspace_id", REAL_WORKSPACE).eq("entity_type", "project").execute().data
    assert rows == []
    assert wp.PAGE_BUILDERS.get("project") is None


def test_pending_review_candidate_never_appears_as_memory_page():
    review_rows = supabase.table("memory_review_queue").select("structured_knowledge_id,status") \
        .eq("workspace_id", REAL_WORKSPACE).eq("status", "pending").execute().data
    assert any(r["structured_knowledge_id"] == SK_Q4_LAUNCH_APPROVAL for r in review_rows)  # real, still pending
    for mtype in ("policy", "process", "decision"):
        assert wp.build_page(mtype, SK_Q4_LAUNCH_APPROVAL, REAL_WORKSPACE, OWNER) is None
    listed_ids = {p["object_id"] for p in wp.list_available_pages(REAL_WORKSPACE, OWNER)}
    assert SK_Q4_LAUNCH_APPROVAL not in listed_ids


def test_calendar_snapshot_never_appears_as_memory_page_evidence():
    for mid in REAL_MEMORY_IDS.values():
        page = wp.build_page("policy", mid, REAL_WORKSPACE, OWNER) or wp.build_page("process", mid, REAL_WORKSPACE, OWNER)
        if page is None:
            continue
        assert all(e["evidence_type"] != "calendar_event_snapshot" for e in page.evidence)


def test_entity_identity_and_memory_identity_never_cross_contaminate_fields():
    """Part 6 -- do not show unsupported fields. An entity Identity item
    never carries memory-only concepts; a memory Identity item never
    carries entity-only concepts."""
    person = wp.build_person_page(REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    memory = wp.build_policy_page(REAL_MEMORY_IDS["credential_logging"], REAL_WORKSPACE, OWNER)
    person_identity_keys = set(person.sections[0].items[0].keys())
    memory_identity_keys = set(memory.sections[0].items[0].keys())
    assert person_identity_keys.isdisjoint({"promotion_basis", "lifecycle_status", "superseded_at"})
    assert memory_identity_keys.isdisjoint({"identifiers", "entity_type"})


# =====================================================================
# 23-26. Security.
# =====================================================================

def test_restricted_structured_knowledge_counterpart_filtered_by_sensitivity():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": []}
    try:
        person_id = _make_entity(ws, "TEST-6E restricted-link person")
        ids["entity_ids"].append(person_id)
        restricted_sk = _make_sk(ws, sensitivity="restricted", statement="TEST-6E restricted statement")
        ids["sk_ids"].append(restricted_sk)
        low_sens_evidence_sk = _make_sk(ws, sensitivity="internal", statement="TEST-6E low-sensitivity evidence")
        ids["sk_ids"].append(low_sens_evidence_sk)
        rel_id = _make_relationship(ws, "structured_knowledge", restricted_sk, "entity", person_id,
                                     "requires_approval_from", low_sens_evidence_sk)
        ids["relationship_ids"].append(rel_id)

        low_page = wp.build_page("person", person_id, ws, LOW)
        owner_page = wp.build_page("person", person_id, ws, OWNER)
        assert low_page is not None and low_page.sections[1].items == []
        assert owner_page is not None and len(owner_page.sections[1].items) == 1
    finally:
        _cleanup(ids)


def test_memory_below_sensitivity_ceiling_returns_none_page():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted")
        ids["sk_ids"].append(sk_id)
        mem_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(mem_id)
        # sensitivity is computed by the RPC as the strictest ceiling across
        # evidence -- a restricted-only grounding yields a restricted memory.
        row = supabase.table("org_memory").select("sensitivity").eq("id", mem_id).execute().data[0]
        assert row["sensitivity"] == "restricted"
        assert wp.build_page("policy", mem_id, ws, LOW) is None
        assert wp.build_page("policy", mem_id, ws, OWNER) is not None
    finally:
        _cleanup(ids)


def test_list_available_pages_excludes_restricted_memory_for_low_sensitivity_caller():
    """Phase 6H regression: list_available_pages's signature has taken
    allowed_sensitivities since this file was written, but the memory branch
    never actually applied it -- a restricted memory still listed (with
    title=None, but a real, resolvable object_id/page_type) even though
    build_page correctly returned None for it. Harmless while nothing
    consumed the list; a real leak once Phase 6G's /wiki/pages endpoint
    exposed it directly as a browsable index (a low-sensitivity caller would
    see a real page id that immediately 404s on click -- the exact "hidden
    page placeholder" Part 15 prohibits). Fixed in wiki_projection.py by
    filtering the memory branch through memory_retrieval._is_visible, the
    same check _build_memory_page already applies to its own row."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted")
        ids["sk_ids"].append(sk_id)
        mem_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(mem_id)

        listed_low = wp.list_available_pages(ws, LOW)
        listed_owner = wp.list_available_pages(ws, OWNER)
        assert mem_id not in {p["object_id"] for p in listed_low}
        assert mem_id in {p["object_id"] for p in listed_owner}
    finally:
        _cleanup(ids)


def test_workspace_isolation_entity_page_real_id_under_leak_workspace():
    assert wp.build_page("person", REAL_ENTITY_IDS["tanmay"], LEAK_WORKSPACE, OWNER) is None


def test_workspace_isolation_memory_page_real_id_under_leak_workspace():
    assert wp.build_page("policy", REAL_MEMORY_IDS["credential_logging"], LEAK_WORKSPACE, OWNER) is None


# =====================================================================
# 27-29. Determinism (Part 15).
# =====================================================================

def test_content_hash_stable_across_rebuilds_real_data():
    for ptype, eid in [("person", REAL_ENTITY_IDS["tanmay"]), ("meeting", REAL_ENTITY_IDS["meeting"])]:
        h1 = wp.build_page(ptype, eid, REAL_WORKSPACE, OWNER).content_hash
        h2 = wp.build_page(ptype, eid, REAL_WORKSPACE, OWNER).content_hash
        assert h1 == h2
    for mid in REAL_MEMORY_IDS.values():
        page1 = wp.build_policy_page(mid, REAL_WORKSPACE, OWNER) or wp.build_process_page(mid, REAL_WORKSPACE, OWNER)
        page2 = wp.build_policy_page(mid, REAL_WORKSPACE, OWNER) or wp.build_process_page(mid, REAL_WORKSPACE, OWNER)
        assert page1.content_hash == page2.content_hash


def test_generated_at_excluded_from_content_hash():
    page1 = wp.build_department_page(REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)
    time.sleep(1.1)
    page2 = wp.build_department_page(REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)
    assert page1.generated_at != page2.generated_at
    assert page1.content_hash == page2.content_hash


def test_content_hash_changes_when_relationship_added():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": []}
    try:
        person_id = _make_entity(ws, "TEST-6E hash-change person")
        other_id = _make_entity(ws, "TEST-6E hash-change meeting", entity_type="meeting")
        ids["entity_ids"] += [person_id, other_id]
        before = wp.build_person_page(person_id, ws, OWNER)

        sk_id = _make_sk(ws)
        ids["sk_ids"].append(sk_id)
        rel_id = _make_relationship(ws, "entity", person_id, "entity", other_id, "attended", sk_id)
        ids["relationship_ids"].append(rel_id)
        after = wp.build_person_page(person_id, ws, OWNER)

        assert before.content_hash != after.content_hash
    finally:
        _cleanup(ids)


# =====================================================================
# 30-32. Linking (Part 8/9).
# =====================================================================

def test_entity_to_entity_link_real_tanmay_to_meeting():
    page = wp.build_person_page(REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    assert len(page.links) == 1
    link = page.links[0]
    assert link.target_page_type == "meeting" and link.target_id == REAL_ENTITY_IDS["meeting"]
    assert link.relationship_id == "07826687-1d83-4e2c-9376-a15d022ea911"
    assert link.rationale and "organizer" in link.rationale


def test_entity_to_memory_link_and_back_synthetic():
    """A structured_knowledge row that grounds BOTH a real relationship AND
    a real memory produces a link in both directions -- entity page links to
    the memory page, and the memory page links back to the entity."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "memory_ids": []}
    try:
        dept_id = _make_entity(ws, "TEST-6E linked department", entity_type="department")
        ids["entity_ids"].append(dept_id)
        sk_id = _make_sk(ws, statement="TEST-6E shared grounding statement")
        ids["sk_ids"].append(sk_id)
        rel_id = _make_relationship(ws, "structured_knowledge", sk_id, "entity", dept_id, "requires_approval_from", sk_id)
        ids["relationship_ids"].append(rel_id)
        mem_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(mem_id)

        dept_page = wp.build_department_page(dept_id, ws, OWNER)
        mem_page = wp.build_policy_page(mem_id, ws, OWNER)

        assert len(dept_page.links) == 1 and dept_page.links[0].target_id == mem_id
        assert len(mem_page.links) == 1 and mem_page.links[0].target_id == dept_id
    finally:
        _cleanup(ids)


def test_relationship_without_grounded_memory_shows_fact_but_no_link_real():
    future = APPROVAL_VALID_FROM + timedelta(days=1)
    page = wp.build_department_page(REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER, as_of=future)
    assert len(page.sections[1].items) == 1  # the fact is shown
    assert page.links == []                  # but not promoted into a page link -- its sk is only a review candidate


# =====================================================================
# 33-34. Temporal (Part 10).
# =====================================================================

def test_historical_entity_page_before_meeting_excludes_attendance():
    before = MEETING_VALID_FROM - timedelta(days=1)
    page = wp.build_meeting_page(REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER, as_of=before)
    assert page is not None
    assert page.sections[1].items == []


def test_memory_page_respects_superseded_at_historical():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk1 = _make_sk(ws, statement="TEST-6E predecessor statement")
        sk2 = _make_sk(ws, statement="TEST-6E successor statement")
        ids["sk_ids"] += [sk1, sk2]
        predecessor_id = _make_memory(ws, sk1)
        ids["memory_ids"].append(predecessor_id)
        # Real DB-assigned timestamp, never local wall-clock -- this
        # codebase's own repeatedly-learned lesson (Phase 6C's consolidation_
        # clock() fix, 6D.1/6D.2's as_of-boundary test rewrites): local _now()
        # compared against a server-clock column is a real, measured clock-
        # skew hazard, not a hypothetical one. before_succession is derived
        # from the predecessor's own real created_at instead.
        predecessor_created_at = supabase.table("org_memory").select("created_at").eq("id", predecessor_id).execute().data[0]["created_at"]
        before_succession = datetime.fromisoformat(predecessor_created_at.replace("Z", "+00:00")) + timedelta(milliseconds=1)
        successor_id = _make_memory(ws, sk2, p_supersedes_memory_id=predecessor_id)
        ids["memory_ids"].append(successor_id)
        succeeded_at = supabase.table("org_memory").select("superseded_at").eq("id", predecessor_id).execute().data[0]["superseded_at"]
        assert succeeded_at is not None
        after_succession = datetime.fromisoformat(succeeded_at.replace("Z", "+00:00")) + timedelta(milliseconds=1)

        assert wp.build_policy_page(predecessor_id, ws, OWNER, as_of=before_succession) is not None
        assert wp.build_policy_page(predecessor_id, ws, OWNER, as_of=after_succession) is None
    finally:
        _cleanup(ids)


# =====================================================================
# 35-36. list_available_pages index.
# =====================================================================

def test_list_available_pages_real_includes_all_9():
    listed = {(p["page_type"], p["object_id"]) for p in wp.list_available_pages(REAL_WORKSPACE, OWNER)}
    expected = {("person", REAL_ENTITY_IDS["john_snow"]), ("person", REAL_ENTITY_IDS["tanmay"]),
                ("department", REAL_ENTITY_IDS["operations"]), ("department", REAL_ENTITY_IDS["product"]),
                ("meeting", REAL_ENTITY_IDS["meeting"])} | {("policy", mid) for mid in
                (REAL_MEMORY_IDS["credential_logging"], REAL_MEMORY_IDS["credential_sharing"], REAL_MEMORY_IDS["hardware_scope"])} \
               | {("process", REAL_MEMORY_IDS["monday_capacity"])}
    assert expected <= listed


def test_list_available_pages_isolated_by_workspace():
    leak_listed_ids = {p["object_id"] for p in wp.list_available_pages(LEAK_WORKSPACE, OWNER)}
    assert leak_listed_ids.isdisjoint(REAL_ENTITY_IDS.values())
    assert leak_listed_ids.isdisjoint(REAL_MEMORY_IDS.values())


# =====================================================================
# 37. Full-regression placeholder (Phase 6C/6D/6D.1/6D.2 convention).
# =====================================================================

def test_placeholder_full_regression_run_separately():
    """This suite is written to run standalone or as part of the full
    sequential regression across every test_phase*.py file -- no fixture
    here depends on suite ordering (every synthetic case uses its own fresh
    workspace; every real-data case only reads)."""
    assert True
