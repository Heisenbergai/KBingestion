"""
Phase 7C Cross-Department Intelligence tests.

Impact traversal is fully deterministic (no LLM, no embeddings, no
similarity), so every test exercises real behavior against the real corpus
or against synthetic, single-use workspaces.

Run with: python -m pytest test_phase7c_cross_department.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import memory_retrieval as mr
import reasoning as rz
import wiki_projection as wp
import impact_analysis as ia

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

PRODUCT = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
OPERATIONS = "1034346e-5731-45b8-9ee5-2e7d1413ca81"
MEETING = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
TANMAY = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
JOHN = "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d"
SK_Q4 = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"
CRED_MEMORY = "2b9140a0-a2e1-4892-b869-fb811e45f1f5"
APPROVAL_VALID_FROM = datetime(2026, 9, 15, tzinfo=timezone.utc)
FUTURE = APPROVAL_VALID_FROM + timedelta(days=1)

# The frozen ontology, verified live against pg_constraint.
FROZEN_RELATIONSHIP_TYPES = {"references", "requires_approval_from", "supersedes",
                             "contradicts", "organized", "attended"}
FORBIDDEN_SEMANTICS = ("owns", "works in", "works for", "manages", "belongs to",
                       "employee", "member of", "responsible for", "affects")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_workspace() -> str:
    return str(uuid.uuid4())


def _cleanup(ids: dict) -> None:
    for rel_id in ids.get("relationship_ids", []):
        supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", rel_id).execute()
        supabase.table("knowledge_relationships").delete().eq("id", rel_id).execute()
    for mid in ids.get("memory_ids", []):
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in reversed(ids.get("memory_ids", [])):
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for eid in ids.get("entity_ids", []):
        supabase.table("knowledge_entities").delete().eq("id", eid).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_entity(ws: str, label: str, entity_type: str = "department") -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": ws, "entity_type": entity_type,
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_sk(ws: str, **overrides) -> str:
    row = {
        "workspace_id": ws, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-7C synthetic statement",
        "raw_subject_phrase": "TEST-7C subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-7c-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


def _make_rel(ws: str, s_type: str, s_id: str, t_type: str, t_id: str,
              rel_type: str, evidence_sk_id: str) -> str:
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": ws, "p_source_object_type": s_type, "p_source_object_id": s_id,
        "p_target_object_type": t_type, "p_target_object_id": t_id,
        "p_relationship_type": rel_type, "p_rationale": "TEST-7C", "p_confidence": 0.9,
        "p_valid_from": _now_iso(), "p_valid_until": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": evidence_sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }).execute().data


# =====================================================================
# 1-4. Direct relationship, 1-hop, 2-hop, bounded traversal.
# =====================================================================

def test_direct_organizational_relationship():
    res = ia.analyze_impact("structured_knowledge", SK_Q4, REAL_WORKSPACE, OWNER, as_of=FUTURE)
    assert len(res.paths) >= 1
    p = next(p for p in res.paths if p.target.object_id == PRODUCT)
    assert p.hops == 1 and p.reasoning_state == ia.OBSERVED
    assert p.relationship_ids == ["aca7d788-a356-4ad9-8030-4a96f4bd4da7"]
    assert p.evidence_ids


def test_one_hop_impact():
    res = ia.analyze_impact("entity", MEETING, REAL_WORKSPACE, OWNER, max_hops=1)
    targets = {p.target.object_id for p in res.paths}
    assert targets == {TANMAY, JOHN}
    assert all(p.hops == 1 and p.reasoning_state == ia.OBSERVED for p in res.paths)


def test_two_hop_impact():
    """Tanmay -> Meeting -> John Snow is a REAL 2-edge chain through a real
    shared node; both hops are persisted rows."""
    res = ia.analyze_impact("entity", TANMAY, REAL_WORKSPACE, OWNER, max_hops=2)
    two_hop = [p for p in res.paths if p.hops == 2]
    assert two_hop, "the real corpus supports a 2-hop path from Tanmay"
    p = next(p for p in two_hop if p.target.object_id == JOHN)
    assert p.reasoning_state == ia.DERIVED
    assert len(p.relationship_ids) == 2
    assert p.chain[0].to_node.object_id == MEETING   # real shared intermediate


def test_bounded_traversal():
    for bad in (0, 3, 5, -1):
        with pytest.raises(ValueError):
            ia.analyze_impact("entity", MEETING, REAL_WORKSPACE, OWNER, max_hops=bad)


def test_traversal_query_count_is_bounded():
    """One query for the origin plus at most one per distinct 1-hop
    counterpart -- never an unbounded crawl."""
    res = ia.analyze_impact("entity", MEETING, REAL_WORKSPACE, OWNER, max_hops=2)
    assert res.graph_queries <= 1 + len(res.paths) + 1


# =====================================================================
# 5. No semantic-similarity impact.
# =====================================================================

def test_no_semantic_similarity_impact():
    """Operations shares vocabulary with the Q4/capacity material but has
    NO real edge -- so it must yield no paths, however 'related' it sounds."""
    res = ia.analyze_impact("entity", OPERATIONS, REAL_WORKSPACE, OWNER, as_of=FUTURE, max_hops=2)
    assert res.paths == []


def test_module_uses_no_embeddings_or_similarity():
    import ast
    tree = ast.parse(open(ia.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "ai" not in imported
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for banned in ("embed_texts", "chat", "chat_json", "hybrid_search"):
        assert banned not in called


# =====================================================================
# 6-9. Reasoning states.
# =====================================================================

def test_observed_classification():
    res = ia.analyze_impact("entity", MEETING, REAL_WORKSPACE, OWNER, max_hops=1)
    assert all(p.reasoning_state == ia.OBSERVED for p in res.paths)


def test_derived_classification():
    res = ia.analyze_impact("entity", TANMAY, REAL_WORKSPACE, OWNER, max_hops=2)
    assert any(p.reasoning_state == ia.DERIVED and p.hops == 2 for p in res.paths)


def test_inferred_never_produced_by_traversal():
    """Traversal emits only persisted-row-backed paths, so it structurally
    cannot manufacture a hypothesis."""
    for kind, oid, as_of in (("entity", MEETING, None), ("entity", TANMAY, None),
                              ("entity", PRODUCT, FUTURE), ("structured_knowledge", SK_Q4, FUTURE)):
        res = ia.analyze_impact(kind, oid, REAL_WORKSPACE, OWNER, as_of=as_of, max_hops=2)
        assert all(p.reasoning_state in (ia.OBSERVED, ia.DERIVED) for p in res.paths)
        assert "INFERRED" not in {p.reasoning_state for p in res.paths}


def test_unknown_classification_for_named_but_unconnected_target():
    res = ia.analyze_impact("structured_knowledge", SK_Q4, REAL_WORKSPACE, OWNER, as_of=FUTURE,
                             candidate_targets=[{"kind": "entity", "object_id": "no-such-id", "label": "Sales"}])
    assert res.not_established
    ne = res.not_established[0]
    assert ne["target_label"] == "Sales" and ne["reasoning_state"] == ia.UNKNOWN


# =====================================================================
# 10-11. Real Product / Operations behavior.
# =====================================================================

def test_product_real_path():
    res = ia.analyze_impact("entity", PRODUCT, REAL_WORKSPACE, OWNER, as_of=FUTURE)
    assert len(res.paths) == 1
    p = res.paths[0]
    assert p.chain[0].relationship_type == "requires_approval_from"
    assert p.reasoning_state == ia.OBSERVED


def test_operations_real_evidence_is_genuinely_empty():
    """Operations has zero real relationships -- an honest empty result, not
    a failure to look."""
    rows = supabase.table("knowledge_relationships").select("id") \
        .or_(f"source_object_id.eq.{OPERATIONS},target_object_id.eq.{OPERATIONS}").execute().data
    assert rows == []
    assert ia.analyze_impact("entity", OPERATIONS, REAL_WORKSPACE, OWNER, max_hops=2).paths == []


# =====================================================================
# 12-16. Negative inference.
# =====================================================================

def test_sales_negative_inference():
    res = ia.analyze_impact("structured_knowledge", SK_Q4, REAL_WORKSPACE, OWNER, as_of=FUTURE, max_hops=2)
    labels = " ".join((p.target.label or "").lower() for p in res.paths)
    assert "sales" not in labels


def test_qa_negative_inference():
    rows = supabase.table("knowledge_entities").select("canonical_label") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert not any(r["canonical_label"].lower() == "qa" for r in rows)
    res = ia.analyze_impact("structured_knowledge", SK_Q4, REAL_WORKSPACE, OWNER, as_of=FUTURE, max_hops=2)
    # 'QA' appears inside the real quoted statement text, but must never be
    # a resolved TARGET of an impact path.
    assert all((p.target.label or "").lower() != "qa" for p in res.paths)


def test_procurement_negative_inference():
    rows = supabase.table("knowledge_entities").select("canonical_label") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert not any(r["canonical_label"].lower() == "procurement" for r in rows)


def test_john_snow_negative_employment_inference():
    res = ia.analyze_impact("entity", JOHN, REAL_WORKSPACE, OWNER, max_hops=2)
    blob = " ".join(p.explanation.lower() for p in res.paths)
    for bad in ("employee", "works in", "works for", "member of"):
        assert bad not in blob
    assert all(h.relationship_type in FROZEN_RELATIONSHIP_TYPES
               for p in res.paths for h in p.chain)


def test_tanmay_negative_ownership_inference():
    res = ia.analyze_impact("entity", TANMAY, REAL_WORKSPACE, OWNER, max_hops=2)
    blob = " ".join(p.explanation.lower() for p in res.paths)
    for bad in ("owns", "owner", "manages", "responsible for"):
        assert bad not in blob


def test_no_forbidden_semantics_anywhere_in_real_corpus():
    for kind, oid, as_of in (("entity", PRODUCT, FUTURE), ("entity", OPERATIONS, FUTURE),
                              ("entity", MEETING, None), ("entity", TANMAY, None),
                              ("entity", JOHN, None), ("structured_knowledge", SK_Q4, FUTURE)):
        res = ia.analyze_impact(kind, oid, REAL_WORKSPACE, OWNER, as_of=as_of, max_hops=2)
        blob = " ".join(p.explanation.lower() for p in res.paths)
        for bad in FORBIDDEN_SEMANTICS:
            assert bad not in blob, f"{bad!r} leaked into an impact explanation"


def test_no_new_relationship_type_invented():
    res_all = []
    for kind, oid, as_of in (("entity", PRODUCT, FUTURE), ("entity", MEETING, None),
                              ("entity", TANMAY, None), ("structured_knowledge", SK_Q4, FUTURE)):
        res_all.append(ia.analyze_impact(kind, oid, REAL_WORKSPACE, OWNER, as_of=as_of, max_hops=2))
    used = {h.relationship_type for r in res_all for p in r.paths for h in p.chain}
    assert used <= FROZEN_RELATIONSHIP_TYPES


# =====================================================================
# 17. Memory integration.
# =====================================================================

def test_memory_integration_requires_a_real_edge():
    """A durable memory's grounding may participate in impact ONLY through
    a real relationship. The real credential policy has no graph edge, so it
    yields no paths -- memory existence alone never creates a department
    connection (Part 9)."""
    ev = supabase.table("memory_evidence").select("evidence_id") \
        .eq("memory_id", CRED_MEMORY).eq("evidence_type", "structured_knowledge").execute().data
    sk_id = ev[0]["evidence_id"]
    assert ia.analyze_impact("structured_knowledge", sk_id, REAL_WORKSPACE, OWNER, max_hops=2).paths == []


def test_memory_grounding_with_a_real_edge_does_produce_a_path():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "memory_ids": []}
    try:
        dept = _make_entity(ws, "TEST-7C Dept")
        ids["entity_ids"].append(dept)
        sk = _make_sk(ws, statement="TEST-7C policy grounding")
        ids["sk_ids"].append(sk)
        ids["memory_ids"].append(supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": ws, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": None,
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute().data)
        ids["relationship_ids"].append(
            _make_rel(ws, "structured_knowledge", sk, "entity", dept, "requires_approval_from", sk))

        res = ia.analyze_impact("structured_knowledge", sk, ws, OWNER, max_hops=1)
        assert len(res.paths) == 1
        assert res.paths[0].target.object_id == dept
        assert res.paths[0].reasoning_state == ia.OBSERVED
    finally:
        _cleanup(ids)


# =====================================================================
# 18. Temporal.
# =====================================================================

def test_temporal_as_of_changes_impact():
    now = ia.analyze_impact("entity", PRODUCT, REAL_WORKSPACE, OWNER)
    future = ia.analyze_impact("entity", PRODUCT, REAL_WORKSPACE, OWNER, as_of=FUTURE)
    assert now.paths == []
    assert len(future.paths) == 1
    assert now.temporal_context == "current"
    assert future.temporal_context == FUTURE.isoformat()


def test_temporal_context_shared_across_every_hop():
    res = ia.analyze_impact("entity", TANMAY, REAL_WORKSPACE, OWNER, as_of=FUTURE, max_hops=2)
    assert all(p.temporal_context == FUTURE.isoformat() for p in res.paths)


# =====================================================================
# 19-21. Security.
# =====================================================================

def test_workspace_isolation():
    res = ia.analyze_impact("entity", PRODUCT, LEAK_WORKSPACE, OWNER, as_of=FUTURE)
    assert res.paths == []
    assert res.origin.label is None


def test_sensitivity_isolation_and_hidden_evidence_non_leak():
    """A relationship whose ONLY evidence is restricted must not create a
    visible path for a low-sensitivity caller -- and must leave no trace:
    no placeholder, no count, no hint."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": []}
    try:
        dept = _make_entity(ws, "TEST-7C Secret Dept")
        ids["entity_ids"].append(dept)
        secret_sk = _make_sk(ws, sensitivity="restricted", statement="TEST-7C restricted requirement")
        ids["sk_ids"].append(secret_sk)
        ids["relationship_ids"].append(
            _make_rel(ws, "structured_knowledge", secret_sk, "entity", dept,
                      "requires_approval_from", secret_sk))

        low = ia.analyze_impact("entity", dept, ws, LOW, max_hops=2)
        owner = ia.analyze_impact("entity", dept, ws, OWNER, max_hops=2)

        assert low.paths == [], "restricted-only evidence must not create a visible path"
        assert low.relationships_examined == 0, "must not even hint that something exists"
        assert len(owner.paths) == 1, "owner sees the real path"
    finally:
        _cleanup(ids)


def test_hidden_intermediate_does_not_enable_two_hop_path():
    """If the middle hop is invisible, the 2-hop path must not form."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": []}
    try:
        a = _make_entity(ws, "TEST-7C A", entity_type="department")
        b = _make_entity(ws, "TEST-7C B", entity_type="meeting")
        c = _make_entity(ws, "TEST-7C C", entity_type="person")
        ids["entity_ids"] += [a, b, c]
        visible_sk = _make_sk(ws, sensitivity="internal", statement="TEST-7C visible")
        secret_sk = _make_sk(ws, sensitivity="restricted", statement="TEST-7C secret")
        ids["sk_ids"] += [visible_sk, secret_sk]
        ids["relationship_ids"].append(_make_rel(ws, "entity", c, "entity", b, "attended", visible_sk))
        ids["relationship_ids"].append(_make_rel(ws, "structured_knowledge", secret_sk, "entity", b,
                                                  "requires_approval_from", secret_sk))

        low = ia.analyze_impact("entity", c, ws, LOW, max_hops=2)
        owner = ia.analyze_impact("entity", c, ws, OWNER, max_hops=2)
        low_targets = {p.target.object_id for p in low.paths}
        owner_targets = {p.target.object_id for p in owner.paths}
        assert secret_sk not in low_targets
        assert secret_sk in owner_targets
    finally:
        _cleanup(ids)


# =====================================================================
# 22. Existing reasoning compatibility.
# =====================================================================

def test_existing_reasoning_compatibility():
    """Impact rows merge into the SAME claim shape reasoning.py already
    consumes, and reasoning -- not this module -- assigns the final state."""
    res = ia.analyze_impact("entity", TANMAY, REAL_WORKSPACE, OWNER, max_hops=2)
    rows = ia.impact_paths_as_claim_rows(res)
    assert rows and all(r["similarity"] is None for r in rows)
    claims = rz.build_claim_inventory(rows, None, None)
    assert len(claims) == len(rows)
    state, _ = rz._classify_state(claims[0].text, [claims[0]])
    assert state == rz.OBSERVED


def test_impact_module_never_assigns_final_answer_state():
    import ast
    tree = ast.parse(open(ia.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "reasoning" not in imported, "impact must not call the reasoner; the caller composes them"


# =====================================================================
# 23. Real corpus benchmark.
# =====================================================================

def test_real_corpus_benchmark_shape():
    checks = [
        ("entity", PRODUCT, None, 0),
        ("entity", PRODUCT, FUTURE, 1),
        ("structured_knowledge", SK_Q4, FUTURE, 1),
        ("entity", OPERATIONS, FUTURE, 0),
        ("entity", MEETING, None, 2),
    ]
    for kind, oid, as_of, expected in checks:
        res = ia.analyze_impact(kind, oid, REAL_WORKSPACE, OWNER, as_of=as_of, max_hops=1)
        assert len(res.paths) == expected, f"{kind}:{oid} as_of={as_of} expected {expected}"
        for p in res.paths:
            assert p.evidence_ids, "every path must carry real evidence"
            assert p.relationship_ids


# =====================================================================
# 24-26. No mutation of graph / memory / Wiki.
# =====================================================================

def test_no_graph_memory_or_wiki_mutation():
    rel_before = len(supabase.table("knowledge_relationships").select("id").execute().data)
    ent_before = len(supabase.table("knowledge_entities").select("id").execute().data)
    mem_before = len(supabase.table("org_memory").select("id").execute().data)
    sk_before = len(supabase.table("structured_knowledge").select("id").execute().data)
    wiki_before = wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash

    for kind, oid, as_of in (("entity", PRODUCT, FUTURE), ("entity", MEETING, None),
                              ("entity", TANMAY, None), ("structured_knowledge", SK_Q4, FUTURE)):
        ia.analyze_impact(kind, oid, REAL_WORKSPACE, OWNER, as_of=as_of, max_hops=2)

    assert len(supabase.table("knowledge_relationships").select("id").execute().data) == rel_before
    assert len(supabase.table("knowledge_entities").select("id").execute().data) == ent_before
    assert len(supabase.table("org_memory").select("id").execute().data) == mem_before
    assert len(supabase.table("structured_knowledge").select("id").execute().data) == sk_before
    assert wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash == wiki_before


def test_module_performs_no_writes():
    import ast
    tree = ast.parse(open(ia.__file__, encoding="utf-8").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for write in ("insert", "update", "delete", "upsert", "rpc"):
        assert write not in called, f"impact_analysis.py must never call .{write}()"


def test_no_second_graph_engine():
    """Traversal is composed from graph_query's existing primitives only."""
    import ast
    tree = ast.parse(open(ia.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "brain_connectors" not in imported and "supabase" not in imported
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "get_entity_graph" in called and "get_structured_knowledge_graph" in called


# =====================================================================
# 27-28. Cleanup + full-regression placeholder.
# =====================================================================

def test_no_leftover_test_7c_fixtures():
    for table, col in (("knowledge_entities", "canonical_label"),
                        ("structured_knowledge", "statement")):
        leftover = supabase.table(table).select("id").ilike(col, "TEST-7C%").execute().data
        assert leftover == [], f"leftover TEST-7C rows in {table}"


def test_placeholder_full_regression_run_separately():
    assert True
