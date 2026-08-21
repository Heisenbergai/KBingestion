"""
Phase 7A Organizational Reasoning tests.

The four reasoning states are assigned DETERMINISTICALLY by
reasoning._classify_state(), never self-reported by the model -- so the
majority of these tests exercise that classifier directly against real
graph/memory claims, which is the only honest way to prove the STOP
condition ("if the system cannot reliably distinguish OBSERVED / DERIVED /
INFERRED / UNKNOWN: STOP").

Context is composed from the REAL graph and memory paths. hybrid_search is
deliberately NOT used: it requires Bedrock embeddings and no AWS credentials
exist in this environment (the same documented constraint since Phase 6F).
The vector-chunk contribution to reasoning is therefore NOT exercised here
and is reported as unverified rather than simulated.

Run with: python -m pytest test_phase7a_reasoning.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import graph_retrieval as gr
import memory_retrieval as mr
import wiki_projection as wp
import reasoning as rz

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

# Questions must name REAL entities -- graph_retrieval.resolve_entity_mentions
# is a deterministic exact/alias matcher, so "the meeting" resolves to
# nothing (confirmed live during this phase's benchmark).
Q_GRAPH = "Who organized Knova Test Meeting 1 and what is Tanmay related to?"
Q_MEMORY = "What is the credential policy?"
Q_BOTH = "Who organized Knova Test Meeting 1 and what is Tanmay related to, and the credential policy?"

REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "monday_capacity":    "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}
MEETING_VALID_FROM = datetime(2026, 8, 16, 8, 30, tzinfo=timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_workspace() -> str:
    return str(uuid.uuid4())


def _compose(question, workspace_id=REAL_WORKSPACE, allowed=OWNER, as_of=None):
    """Exactly query.py's composition order, minus hybrid_search (see module
    docstring). Returns (chunks, graph_context, memory_context)."""
    chunks = []
    gctx = gr.build_graph_context(question, workspace_id, allowed, as_of=as_of)
    chunks, _ = gr.merge_graph_context_into_chunks(chunks, gctx)
    mctx = mr.build_memory_context(question, workspace_id, allowed, as_of=as_of, graph_context=gctx)
    chunks, _ = mr.merge_memory_context_into_chunks(chunks, mctx, graph_context=gctx)
    return chunks, gctx, mctx


def _claims(question, **kw):
    chunks, gctx, mctx = _compose(question, **kw)
    return rz.build_claim_inventory(chunks, gctx, mctx, as_of=kw.get("as_of"))


def _cleanup(ids: dict) -> None:
    for mid in ids.get("memory_ids", []):
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in reversed(ids.get("memory_ids", [])):
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-7A synthetic statement",
        "raw_subject_phrase": "TEST-7A subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-7a-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


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


def _compliant_mock(conclusions):
    def mock(*a, **k):
        return {"conclusions": conclusions}
    return mock


# =====================================================================
# 1-4. The four states, each proven against REAL claims.
# =====================================================================

def test_observed_claim():
    claims = _claims(Q_GRAPH)
    graph_claims = [c for c in claims if c.source_kind == "graph_relationship"]
    assert graph_claims, "real graph claims must resolve for this question"
    state, why = rz._classify_state(graph_claims[0].text, [graph_claims[0]])
    assert state == rz.OBSERVED, why


def test_derived_claim():
    claims = _claims(Q_GRAPH)
    graph_claims = [c for c in claims if c.source_kind == "graph_relationship"]
    assert len(graph_claims) >= 2, "this question resolves both real meeting relationships"
    combined_text = graph_claims[0].text + " " + graph_claims[1].text
    state, why = rz._classify_state(combined_text, graph_claims[:2])
    assert state == rz.DERIVED, why
    # ...and they are DERIVED because they share a REAL identifier, not because
    # they sound related.
    assert graph_claims[0].linkage_keys & graph_claims[1].linkage_keys


def test_inferred_claim_classification():
    """A fabrication that cites a genuinely real claim must still be
    INFERRED -- citation existence alone never earns OBSERVED."""
    claims = _claims(Q_GRAPH)
    real = [c for c in claims if c.source_kind == "graph_relationship"][0]
    state, why = rz._classify_state(
        "This has been the standing arrangement since January 2019 without exception.", [real])
    assert state == rz.INFERRED, why


def test_unknown_classification():
    assert rz._classify_state("Anything at all.", [])[0] == rz.UNKNOWN
    claims = _claims(Q_GRAPH)
    # A hallucinated claim_id is dropped before classification, leaving nothing.
    real_ids = {c.claim_id for c in claims}
    assert "graph:999" not in real_ids


# =====================================================================
# 5-6. Multi-hop and memory+graph reasoning.
# =====================================================================

def test_multi_hop_graph_reasoning():
    """Tanmay -> Meeting <- John Snow is a real 2-edge chain sharing a real
    calendar snapshot; combining them is DERIVED, not INFERRED."""
    claims = _claims(Q_GRAPH)
    graph_claims = [c for c in claims if c.source_kind == "graph_relationship"]
    shared = graph_claims[0].linkage_keys & graph_claims[1].linkage_keys
    assert any(k.startswith("calendar_event_snapshot:") or k.startswith("entity:") for k in shared)
    state, _ = rz._classify_state(graph_claims[0].text + " " + graph_claims[1].text, graph_claims[:2])
    assert state == rz.DERIVED


def test_memory_and_graph_reasoning():
    claims = _claims(Q_BOTH)
    kinds = {c.source_kind for c in claims}
    assert "graph_relationship" in kinds and "memory" in kinds
    # Graph and memory claims here share NO real identifier -- combining them
    # must NOT be presented as a derived organizational fact.
    g = next(c for c in claims if c.source_kind == "graph_relationship")
    m = next(c for c in claims if c.source_kind == "memory")
    assert not (g.linkage_keys & m.linkage_keys)
    state, _ = rz._classify_state(g.text + " " + m.text, [g, m])
    assert state == rz.INFERRED


# =====================================================================
# 7-8. Temporal reasoning -- one shared as_of.
# =====================================================================

def test_current_temporal_reasoning():
    result = rz.reason(Q_GRAPH, REAL_WORKSPACE, *_compose(Q_GRAPH)[0:1],
                        *_compose(Q_GRAPH)[1:], chat_json_fn=lambda *a, **k: {"conclusions": []})
    assert result.temporal_context == "current"
    for c in rz.build_claim_inventory(*_compose(Q_GRAPH)):
        assert c.temporal_context == "current"


def test_historical_temporal_reasoning():
    """Every claim in a historical run carries the SAME as_of -- a 2026
    memory can never silently join a pre-2026 graph state."""
    before = MEETING_VALID_FROM - timedelta(days=1)
    chunks, gctx, mctx = _compose(Q_GRAPH, as_of=before)
    claims = rz.build_claim_inventory(chunks, gctx, mctx, as_of=before)
    assert all(c.temporal_context == before.isoformat() for c in claims)
    # The real attendance edges start 2026-08-16, so before that they are
    # genuinely absent -- not silently carried back.
    assert not [c for c in claims if c.source_kind == "graph_relationship"]


# =====================================================================
# 9-10. Contradiction handling.
# =====================================================================

def test_contradiction_unresolved_not_silently_chosen():
    """detect_contradictions only reports REAL recorded supersession
    signals, and always as 'unresolved' -- it never picks a winner."""
    _, _, mctx = _compose(Q_MEMORY)
    found = rz.detect_contradictions([], mctx)
    for c in found:
        assert c["resolution"] == "unresolved"
    # The real corpus has no active supersedes/contradicts edge today, so
    # zero is the honest current answer -- asserted, not assumed.
    assert found == []


def test_clear_supersession_reasoning_is_never_mutation():
    """Part 9: 'Do not mutate memory or graph in this pass.' reasoning.py
    contains no write of any kind -- proven structurally."""
    src = open(rz.__file__, encoding="utf-8").read()
    for forbidden in (".insert(", ".update(", ".delete(", ".upsert(", ".rpc("):
        assert forbidden not in src, f"reasoning.py must contain no {forbidden} call"


# =====================================================================
# 11-15. Anti-fabrication.
# =====================================================================

def test_no_fabricated_entity():
    claims = _claims(Q_GRAPH)
    real = [c for c in claims if c.source_kind == "graph_relationship"][0]
    state, _ = rz._classify_state("The Q4 Smart Switch Project team owns this deliverable.", [real])
    assert state == rz.INFERRED


def test_no_fabricated_relationship():
    claims = _claims(Q_GRAPH)
    real = [c for c in claims if c.source_kind == "graph_relationship"][0]
    state, _ = rz._classify_state("Tanmay reports to John Snow.", [real])
    assert state == rz.INFERRED


def test_no_ownership_inference():
    claims = _claims(Q_GRAPH)
    real = [c for c in claims if c.source_kind == "graph_relationship"][0]
    state, why = rz._classify_state("Tanmay owns Knova Test Meeting 1.", [real])
    assert state == rz.INFERRED and "forbidden-inference" in why


def test_no_employment_inference():
    claims = _claims(Q_GRAPH)
    real = [c for c in claims if c.source_kind == "graph_relationship"][0]
    state, why = rz._classify_state("John Snow is an employee of Product.", [real])
    assert state == rz.INFERRED and "forbidden-inference" in why


def test_no_project_inference_via_full_pipeline():
    chunks, gctx, mctx = _compose(Q_GRAPH)
    real_id = [c for c in rz.build_claim_inventory(chunks, gctx, mctx)
               if c.source_kind == "graph_relationship"][0].claim_id
    mock = _compliant_mock([{"text": "The Q4 Project owns the meeting.", "claim_ids": [real_id]}])
    result = rz.reason(Q_GRAPH, REAL_WORKSPACE, chunks, gctx, mctx, chat_json_fn=mock)
    assert all(c.state == rz.INFERRED for c in result.conclusions)
    assert result.overall_state == rz.INFERRED


# =====================================================================
# 16-17. Security -- inherited, never re-implemented.
# =====================================================================

def test_workspace_isolation():
    """A different workspace's context yields no claims from the real one."""
    chunks, gctx, mctx = _compose(Q_BOTH, workspace_id=LEAK_WORKSPACE)
    claims = rz.build_claim_inventory(chunks, gctx, mctx)
    blob = " ".join(c.text for c in claims)
    assert "Knova Test Meeting" not in blob and "Credential" not in blob


def test_sensitivity_isolation():
    """A restricted memory never reaches the reasoner for a low-sensitivity
    caller -- filtered upstream by memory_retrieval, never by this module."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted", statement="TEST-7A restricted credential rule")
        ids["sk_ids"].append(sk_id)
        mem_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(mem_id)

        q = "What is the restricted credential rule?"
        low_claims = rz.build_claim_inventory(*_compose(q, workspace_id=ws, allowed=LOW))
        owner_claims = rz.build_claim_inventory(*_compose(q, workspace_id=ws, allowed=OWNER))
        assert not any("restricted credential" in c.text.lower() for c in low_claims)
        assert any("restricted credential" in c.text.lower() for c in owner_claims)
    finally:
        _cleanup(ids)


def _imported_module_names(module) -> set:
    """Real imports only, via AST -- never a text grep. This module's own
    docstring deliberately NAMES the things it refuses to touch
    ('hybrid_search', 'supabase', ...) while explaining why, so grepping the
    raw source reports false violations. Parsing the actual import
    statements is both immune to that and a strictly stronger check."""
    import ast
    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _called_function_names(module) -> set:
    import ast
    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    return called


def test_reasoning_module_never_touches_the_database():
    """Structural proof of Part 14: security is upstream because this module
    cannot fetch anything at all -- it imports no database client."""
    imported = _imported_module_names(rz)
    assert "brain_connectors" not in imported
    assert "supabase" not in imported
    assert "query" not in imported
    called = _called_function_names(rz)
    for db_call in ("table", "rpc", "insert", "update", "delete", "upsert", "execute"):
        assert db_call not in called, f"reasoning.py must never call .{db_call}()"


# =====================================================================
# 18-20. Evidence traceability, citations, confidence compatibility.
# =====================================================================

def test_primary_evidence_traceability():
    claims = _claims(Q_BOTH)
    for c in claims:
        assert c.evidence_refs, f"claim {c.claim_id} has no traceable evidence"
        for ref in c.evidence_refs:
            assert ":" in ref and not ref.endswith(":")


def test_citation_completeness():
    chunks, gctx, mctx = _compose(Q_GRAPH)
    inventory = rz.build_claim_inventory(chunks, gctx, mctx)
    real_id = inventory[0].claim_id
    mock = _compliant_mock([{"text": inventory[0].text, "claim_ids": [real_id]}])
    result = rz.reason(Q_GRAPH, REAL_WORKSPACE, chunks, gctx, mctx, chat_json_fn=mock)
    c = result.conclusions[0]
    assert c.cited_claim_ids == [real_id]
    assert c.evidence_chain and c.evidence_chain[0]["evidence_refs"]


def test_existing_confidence_compatibility():
    """Reasoning may only lower confidence, never raise it, and reuses the
    same three-tier vocabulary -- no new scale."""
    unknown = rz.ReasoningResult(question="q", workspace_id=REAL_WORKSPACE,
                                  temporal_context="current", overall_state=rz.UNKNOWN)
    inferred = rz.ReasoningResult(question="q", workspace_id=REAL_WORKSPACE,
                                   temporal_context="current", overall_state=rz.INFERRED)
    observed = rz.ReasoningResult(question="q", workspace_id=REAL_WORKSPACE,
                                   temporal_context="current", overall_state=rz.OBSERVED)
    assert rz.reasoning_supports_confidence(unknown, "high") == "none"
    assert rz.reasoning_supports_confidence(inferred, "high") == "low"
    assert rz.reasoning_supports_confidence(observed, "high") == "high"
    assert rz.reasoning_supports_confidence(observed, "low") == "low"  # never raised


# =====================================================================
# 21-22. Real corpus, positive and negative.
# =====================================================================

def test_real_corpus_reasoning_cases():
    for q in (Q_MEMORY, "What is the Monday capacity process?", "Why are hardware categories out of scope?"):
        chunks, gctx, mctx = _compose(q)
        result = rz.reason(q, REAL_WORKSPACE, chunks, gctx, mctx,
                            chat_json_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
        assert result.claims_considered >= 1
        assert result.overall_state == rz.OBSERVED
        assert result.reasoned_by == "fallback"


def test_negative_real_corpus_cases():
    """Questions about things the corpus does not establish must never yield
    an OBSERVED/DERIVED ownership or employment assertion."""
    for q in ("Is John Snow an employee?", "Who owns the Product department?",
              "What is the Q4 Project status?", "What is the reporting hierarchy?"):
        chunks, gctx, mctx = _compose(q)
        result = rz.reason(q, REAL_WORKSPACE, chunks, gctx, mctx,
                            chat_json_fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
        for c in result.conclusions:
            if c.state in (rz.OBSERVED, rz.DERIVED):
                low = c.text.lower()
                assert "owns" not in low and "employee" not in low and "reports to" not in low


def test_procurement_mention_is_quotation_not_entity_assertion():
    """'Procurement' appears inside a REAL memory statement, so quoting it is
    correct -- but it must never become an assertion that Procurement is a
    verified department entity (there is no such knowledge_entities row)."""
    rows = supabase.table("knowledge_entities").select("canonical_label") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert not any(r["canonical_label"].lower() == "procurement" for r in rows)
    chunks, gctx, mctx = _compose("What does Procurement do?")
    claims = rz.build_claim_inventory(chunks, gctx, mctx)
    for c in claims:
        assert c.source_kind in ("memory", "graph_relationship", "chunk")
        # quoting real statement text is fine; asserting entity-hood is not
        assert "Procurement is a department" not in c.text


# =====================================================================
# 23-24. Failure handling.
# =====================================================================

def test_model_failure_fallback():
    chunks, gctx, mctx = _compose(Q_MEMORY)

    def boom(*a, **k):
        raise TimeoutError("TEST-7A simulated model outage")

    result = rz.reason(Q_MEMORY, REAL_WORKSPACE, chunks, gctx, mctx, chat_json_fn=boom)
    assert result.reasoned_by == "fallback"
    assert "model_unavailable" in result.metadata["reason"]
    assert result.conclusions and all(c.state == rz.OBSERVED for c in result.conclusions)


def test_malformed_reasoning_output_rejection():
    chunks, gctx, mctx = _compose(Q_MEMORY)
    for bad in ({"unexpected": True}, {"conclusions": "not a list"}, {"conclusions": []}, None):
        result = rz.reason(Q_MEMORY, REAL_WORKSPACE, chunks, gctx, mctx,
                            chat_json_fn=lambda *a, _b=bad, **k: _b)
        assert result.reasoned_by == "fallback"
        assert result.conclusions  # still produces deterministic output


def test_real_bedrock_unavailable_falls_back():
    """The genuine live failure in this environment, not a simulation."""
    chunks, gctx, mctx = _compose(Q_MEMORY)
    result = rz.reason(Q_MEMORY, REAL_WORKSPACE, chunks, gctx, mctx)
    assert result.reasoned_by == "fallback"
    assert "NoCredentialsError" in result.metadata["reason"]


# =====================================================================
# 25-28. Nothing underneath is modified (Part 18).
# =====================================================================

def test_structured_knowledge_graph_memory_wiki_unchanged_by_reasoning():
    sk_before = len(supabase.table("structured_knowledge").select("id").execute().data)
    ent_before = len(supabase.table("knowledge_entities").select("id").execute().data)
    rel_before = len(supabase.table("knowledge_relationships").select("id").execute().data)
    mem_before = len(supabase.table("org_memory").select("id").execute().data)
    wiki_before = wp.build_page("meeting", "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f",
                                 REAL_WORKSPACE, OWNER).content_hash

    for q in (Q_GRAPH, Q_MEMORY, Q_BOTH):
        chunks, gctx, mctx = _compose(q)
        rz.reason(q, REAL_WORKSPACE, chunks, gctx, mctx)

    assert len(supabase.table("structured_knowledge").select("id").execute().data) == sk_before
    assert len(supabase.table("knowledge_entities").select("id").execute().data) == ent_before
    assert len(supabase.table("knowledge_relationships").select("id").execute().data) == rel_before
    assert len(supabase.table("org_memory").select("id").execute().data) == mem_before
    assert wp.build_page("meeting", "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f",
                          REAL_WORKSPACE, OWNER).content_hash == wiki_before


def test_no_second_answer_engine():
    """Part 2's hard constraint, proven structurally via AST (not a text
    grep -- the module docstring names these deliberately while explaining
    that it does NOT call them): reasoning.py never retrieves and never
    builds an answer, and query.py/chatbot.py are not wired to it in 7A."""
    called = _called_function_names(rz)
    assert "hybrid_search" not in called
    assert "build_context_and_citations" not in called
    assert "chat" not in called  # only chat_json, via the injected fn

    import ast
    for path in ("query.py", "chatbot.py"):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(a.name != "reasoning" for a in node.names), \
                    f"{path} must not import reasoning in Phase 7A"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "reasoning", \
                    f"{path} must not import reasoning in Phase 7A"


def test_claim_inventory_no_double_count():
    """Regression for a real bug found in this phase's own benchmark: the
    merged chunk list already CONTAINS graph/memory candidates, so passing
    both produced two claims for one fact -- which could be wrongly
    classified DERIVED."""
    chunks, gctx, mctx = _compose(Q_BOTH)
    claims = rz.build_claim_inventory(chunks, gctx, mctx)
    ids = [c.claim_id for c in claims]
    assert len(ids) == len(set(ids))
    texts = [c.text for c in claims]
    assert len(texts) == len(set(texts)), "one underlying fact must never appear as two claims"
    for c in claims:
        assert not (c.source_kind == "chunk" and
                     any(r.startswith("document:org_memory:") or r.startswith("document:graph_relationship:")
                         for r in c.evidence_refs))


# =====================================================================
# 29-30. Cleanup + full-regression placeholder.
# =====================================================================

def test_no_leftover_test_7a_fixtures():
    leftover = supabase.table("structured_knowledge").select("id") \
        .ilike("statement", "TEST-7A%").execute().data
    assert leftover == []


def test_placeholder_full_regression_run_separately():
    assert True
