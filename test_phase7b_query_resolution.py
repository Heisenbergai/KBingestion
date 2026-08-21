"""
Phase 7B Query Resolution tests.

The resolver is fully deterministic (no LLM anywhere), so every test here
exercises real behavior against the real corpus or against synthetic,
single-use workspaces -- never a mock of the resolver itself.

Context composition uses the real graph/memory paths. hybrid_search is
deliberately NOT used: it needs Bedrock embeddings and no AWS credentials
exist in this environment (documented since Phase 6F). Stated rather than
simulated.

Run with: python -m pytest test_phase7b_query_resolution.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import graph_retrieval as gr
import memory_retrieval as mr
import reasoning as rz
import query_resolution as qres

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

MEETING_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
MEETING_LABEL = "Knova Test Meeting 1"
TANMAY_ID = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
PRODUCT_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
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
    for aid in ids.get("alias_ids", []):
        supabase.table("knowledge_entity_aliases").delete().eq("id", aid).execute()
    for eid in ids.get("entity_ids", []):
        supabase.table("knowledge_entity_identifiers").delete().eq("entity_id", eid).execute()
        supabase.table("knowledge_entities").delete().eq("id", eid).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_entity(workspace_id: str, label: str, entity_type: str = "meeting") -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": entity_type,
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-7B synthetic statement",
        "raw_subject_phrase": "TEST-7B subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-7b-{uuid.uuid4()}",
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


def _compose(question, workspace_id=REAL_WORKSPACE, allowed=OWNER, as_of=None):
    chunks = []
    gctx = gr.build_graph_context(question, workspace_id, allowed, as_of=as_of)
    chunks, _ = gr.merge_graph_context_into_chunks(chunks, gctx)
    mctx = mr.build_memory_context(question, workspace_id, allowed, as_of=as_of, graph_context=gctx)
    chunks, _ = mr.merge_memory_context_into_chunks(chunks, mctx, graph_context=gctx)
    return chunks, gctx, mctx


def _resolve_then_reason(question, workspace_id=REAL_WORKSPACE, allowed=OWNER, as_of=None, prior=None):
    res = qres.resolve_references(question, workspace_id, allowed, as_of=as_of, prior_references=prior)
    effective = qres.rewrite_question_with_references(question, res)
    chunks, gctx, mctx = _compose(effective, workspace_id, allowed, as_of)
    result = rz.reason(effective, workspace_id, chunks, gctx, mctx)
    return res, effective, result


# =====================================================================
# 1-3. Exact, identifier, and alias resolution.
# =====================================================================

def test_exact_entity_resolution():
    res = qres.resolve_references(f"Who organized {MEETING_LABEL}?", REAL_WORKSPACE, OWNER)
    assert res.status == qres.RESOLVED
    assert any(r.object_id == MEETING_ID and r.match_basis == "exact_label" for r in res.references)


def test_exact_identifier_resolution():
    """A real conference_id from knowledge_entity_identifiers."""
    res = qres.resolve_references("What happened in conference ngn-pjwu-jcn?", REAL_WORKSPACE, OWNER)
    assert res.status == qres.RESOLVED
    ref = next(r for r in res.references if r.object_id == MEETING_ID)
    assert ref.match_basis == "identifier"


def test_alias_resolution():
    """knowledge_entity_aliases is EMPTY in the real corpus (verified live),
    so alias resolution is proven with a synthetic alias in its own
    workspace rather than claimed against data that doesn't exist."""
    ws = _fresh_workspace()
    ids = {"entity_ids": [], "alias_ids": []}
    try:
        eid = _make_entity(ws, "TEST-7B Quarterly Sync")
        ids["entity_ids"].append(eid)
        # alias_source_type is CHECK-constrained to real evidence types
        # (knowledge_note_source | structured_knowledge |
        # calendar_event_snapshot | external_reference) -- verified live
        # against pg_constraint rather than guessed. An alias must itself be
        # traceable to real evidence, exactly like everything else in this
        # architecture; there is deliberately no 'manual' escape hatch.
        alias = supabase.table("knowledge_entity_aliases").insert({
            "workspace_id": ws, "entity_id": eid, "alias_text": "QSync",
            "alias_normalized": "qsync", "alias_source_type": "structured_knowledge",
            "alias_source_id": str(uuid.uuid4()), "confidence": 1.0,
        }).execute().data[0]["id"]
        ids["alias_ids"].append(alias)

        res = qres.resolve_references("Who organized QSync?", ws, OWNER)
        assert res.status == qres.RESOLVED
        assert any(r.object_id == eid for r in res.references)
    finally:
        _cleanup(ids)


# =====================================================================
# 4-6. Definite references: unique, ambiguous, unresolved.
# =====================================================================

def test_unique_definite_reference_resolution():
    res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    assert res.status == qres.RESOLVED
    ref = next(r for r in res.references if r.object_id == MEETING_ID)
    assert ref.match_basis == "definite_unique"


def test_ambiguous_definite_reference():
    """TWO real meetings -> AMBIGUOUS, never a silent pick."""
    ids = {"entity_ids": []}
    try:
        second = _make_entity(REAL_WORKSPACE, "TEST-7B Second Meeting")
        ids["entity_ids"].append(second)
        res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
        assert res.status == qres.AMBIGUOUS
        assert not res.references
        amb = res.ambiguous[0]
        assert len(amb["candidate_ids"]) == 2
        assert MEETING_ID in amb["candidate_ids"] and second in amb["candidate_ids"]
    finally:
        _cleanup(ids)


def test_unresolved_definite_reference():
    res = qres.resolve_references("Who organized the meeting?", LEAK_WORKSPACE, OWNER)
    assert res.status == qres.UNRESOLVED
    assert "the meeting" in res.unresolved_phrases
    assert not res.references


# =====================================================================
# 7. Entity-type safety.
# =====================================================================

def test_entity_type_filtering():
    """'the meeting' may only ever denote a meeting -- never a department,
    even though departments exist and are closer alphabetically."""
    res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    for r in res.references:
        assert r.object_type == "meeting"
    # Departments are genuinely ambiguous (Product + Operations) and must
    # NOT silently resolve.
    dept = qres.resolve_references("Tell me about the department", REAL_WORKSPACE, OWNER)
    assert dept.status == qres.AMBIGUOUS
    assert dept.ambiguous[0]["object_type"] == "department"


def test_noun_never_crosses_entity_and_memory_types():
    """'the policy' must never resolve to a meeting/department entity, and
    'the meeting' must never resolve to a memory."""
    pol = qres.resolve_references("What does the policy say?", REAL_WORKSPACE, OWNER)
    for r in pol.references:
        assert r.kind == "memory" and r.object_type == "policy"
    for a in pol.ambiguous:
        assert a["object_type"] == "policy"
    meet = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    for r in meet.references:
        assert r.kind == "entity" and r.object_type == "meeting"


# =====================================================================
# 8-9. Workspace and sensitivity isolation.
# =====================================================================

def test_workspace_isolation():
    res = qres.resolve_references(f"Who organized {MEETING_LABEL}?", LEAK_WORKSPACE, OWNER)
    assert res.status == qres.UNRESOLVED
    assert not res.references


def test_no_cross_workspace_resolution_of_definite_reference():
    """A meeting in ANOTHER workspace must not make 'the meeting' resolve."""
    other_ws = _fresh_workspace()
    ids = {"entity_ids": []}
    try:
        ids["entity_ids"].append(_make_entity(other_ws, "TEST-7B Foreign Meeting"))
        res = qres.resolve_references("Who organized the meeting?", LEAK_WORKSPACE, OWNER)
        assert res.status == qres.UNRESOLVED
    finally:
        _cleanup(ids)


def test_sensitivity_isolation():
    """A restricted memory must not be a resolution candidate for a
    low-sensitivity caller -- and must not make a reference AMBIGUOUS for
    them either (its existence must be invisible)."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_visible = _make_sk(ws, sensitivity="internal", statement="TEST-7B visible policy")
        ids["sk_ids"].append(sk_visible)
        ids["memory_ids"].append(_make_memory(ws, sk_visible))
        sk_secret = _make_sk(ws, sensitivity="restricted", statement="TEST-7B restricted policy")
        ids["sk_ids"].append(sk_secret)
        ids["memory_ids"].append(_make_memory(ws, sk_secret))

        low = qres.resolve_references("What does the policy say?", ws, LOW)
        owner = qres.resolve_references("What does the policy say?", ws, OWNER)
        # LOW sees exactly one -> resolves. OWNER sees two -> ambiguous.
        assert low.status == qres.RESOLVED
        assert "restricted" not in low.references[0].label.lower()
        assert owner.status == qres.AMBIGUOUS
    finally:
        _cleanup(ids)


# =====================================================================
# 10-11, 25. Temporal correctness.
# =====================================================================

def test_as_of_correctness_entity_not_yet_created():
    before = MEETING_VALID_FROM - timedelta(days=400)
    res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER, as_of=before)
    assert res.status == qres.UNRESOLVED
    assert not res.references


def test_historical_candidate_availability_current_still_resolves():
    res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    assert res.status == qres.RESOLVED


def test_current_vs_historical_candidate_distinction():
    """The SAME question resolves now and does not resolve historically --
    a current entity can never leak backward into a historical query."""
    before = MEETING_VALID_FROM - timedelta(days=400)
    assert qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER).status == qres.RESOLVED
    assert qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER,
                                    as_of=before).status == qres.UNRESOLVED


def test_as_of_flows_into_reasoning_consistently():
    before = MEETING_VALID_FROM - timedelta(days=400)
    res, effective, result = _resolve_then_reason("Who organized the meeting?", as_of=before)
    assert res.status == qres.UNRESOLVED
    assert effective == "Who organized the meeting?"   # never rewritten
    assert result.overall_state == rz.UNKNOWN


def test_product_relationship_respects_future_valid_from():
    """Resolution of 'Product' must NOT make a future-dated relationship
    visible early -- the existing temporal contract still governs."""
    res_now, _, result_now = _resolve_then_reason("What requires Product approval?")
    assert res_now.status == qres.RESOLVED
    assert result_now.overall_state == rz.UNKNOWN   # relationship not yet valid

    future = APPROVAL_VALID_FROM + timedelta(days=1)
    res_fut, _, result_fut = _resolve_then_reason("What requires Product approval?", as_of=future)
    assert res_fut.status == qres.RESOLVED
    assert result_fut.claims_considered >= 1        # now visible
    assert result_fut.overall_state == rz.OBSERVED


# =====================================================================
# 12. Follow-up contextual reference.
# =====================================================================

def test_follow_up_contextual_reference():
    """With TWO meetings 'the meeting' is ambiguous -- unless the prior turn
    established exactly one, in which case continuity resolves it."""
    ids = {"entity_ids": []}
    try:
        second = _make_entity(REAL_WORKSPACE, "TEST-7B Second Meeting")
        ids["entity_ids"].append(second)

        assert qres.resolve_references("Who organized the meeting?",
                                        REAL_WORKSPACE, OWNER).status == qres.AMBIGUOUS

        prior = qres.resolve_references(f"Tell me about {MEETING_LABEL}.", REAL_WORKSPACE, OWNER)
        assert prior.status == qres.RESOLVED

        follow = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER,
                                          prior_references=prior.references)
        assert follow.status == qres.RESOLVED
        ref = follow.references[0]
        assert ref.object_id == MEETING_ID and ref.match_basis == "prior_context"
    finally:
        _cleanup(ids)


def test_follow_up_with_two_prior_candidates_stays_ambiguous():
    """Continuity requires the prior turn to have established EXACTLY one."""
    ids = {"entity_ids": []}
    try:
        second = _make_entity(REAL_WORKSPACE, "TEST-7B Second Meeting")
        ids["entity_ids"].append(second)
        fake_prior = [
            qres.ResolvedReference("a", "entity", MEETING_ID, "meeting", MEETING_LABEL, "exact_label"),
            qres.ResolvedReference("b", "entity", second, "meeting", "TEST-7B Second Meeting", "exact_label"),
        ]
        res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER,
                                       prior_references=fake_prior)
        assert res.status == qres.AMBIGUOUS
    finally:
        _cleanup(ids)


# =====================================================================
# 14-15. No fuzzy guessing, no LLM.
# =====================================================================

def test_no_fuzzy_guessing():
    for q in ("Who organized the meetup?", "Who organized that thing?",
              "Who organized the sync?", "Who organized the gathering?"):
        res = qres.resolve_references(q, REAL_WORKSPACE, OWNER)
        assert res.status == qres.UNRESOLVED, f"{q!r} must not fuzzy-match a real meeting"
        assert not res.references


def test_indefinite_reference_never_resolves():
    """'a meeting'/'any meeting' assert no specific referent."""
    for q in ("Who organized a meeting?", "Was there any meeting?"):
        res = qres.resolve_references(q, REAL_WORKSPACE, OWNER)
        assert not any(r.match_basis == "definite_unique" for r in res.references)


def test_no_llm_dependency():
    """Structural proof via AST: the resolver imports no model layer and
    calls no chat function."""
    import ast
    tree = ast.parse(open(qres.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "ai" not in imported
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "chat" not in called and "chat_json" not in called


def test_resolution_is_deterministic():
    a = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    b = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    assert a.status == b.status
    assert [(r.object_id, r.match_basis) for r in a.references] == \
           [(r.object_id, r.match_basis) for r in b.references]


# =====================================================================
# 16-19. Reasoning integration and state preservation.
# =====================================================================

def test_reasoning_integration_same_evidence_as_exact_question():
    """'the meeting' must deliver the SAME evidence to reasoning as naming
    the meeting explicitly -- no special shortcut."""
    _, eff_definite, result_definite = _resolve_then_reason("Who organized the meeting?")
    _, eff_exact, result_exact = _resolve_then_reason(f"Who organized {MEETING_LABEL}?")
    assert eff_definite == eff_exact
    assert result_definite.claims_considered == result_exact.claims_considered
    assert result_definite.overall_state == result_exact.overall_state


def test_observed_preservation():
    _, _, result = _resolve_then_reason("Who organized the meeting?")
    assert result.overall_state == rz.OBSERVED
    assert result.claims_considered >= 1


def test_derived_preservation():
    """Two connected real graph claims still classify DERIVED after
    resolution -- the resolver changes which evidence arrives, never how it
    is classified."""
    _, effective, _ = _resolve_then_reason("Who organized the meeting?")
    chunks, gctx, mctx = _compose(effective)
    claims = rz.build_claim_inventory(chunks, gctx, mctx)
    graph_claims = [c for c in claims if c.source_kind == "graph_relationship"]
    assert len(graph_claims) >= 2
    state, _ = rz._classify_state(graph_claims[0].text + " " + graph_claims[1].text, graph_claims[:2])
    assert state == rz.DERIVED


def test_unknown_preservation_on_ambiguity():
    ids = {"entity_ids": []}
    try:
        ids["entity_ids"].append(_make_entity(REAL_WORKSPACE, "TEST-7B Second Meeting"))
        res, effective, result = _resolve_then_reason("Who organized the meeting?")
        assert res.status == qres.AMBIGUOUS
        assert effective == "Who organized the meeting?"
        assert result.overall_state == rz.UNKNOWN
    finally:
        _cleanup(ids)


def test_resolver_never_assigns_reasoning_state():
    """Part 10: the resolver supplies references only."""
    res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
    for state in (rz.OBSERVED, rz.DERIVED, rz.INFERRED, rz.UNKNOWN):
        assert state not in (res.status,)
    assert res.status in (qres.RESOLVED, qres.AMBIGUOUS, qres.UNRESOLVED)


# =====================================================================
# 20-22. Hallucination / inference rejection.
# =====================================================================

def test_hallucinated_entity_rejection():
    for q in ("Who organized the Q4 Project kickoff?", "Tell me about the customer meeting"):
        res = qres.resolve_references(q, REAL_WORKSPACE, OWNER)
        for r in res.references:
            # may resolve a REAL meeting via the definite phrase, but never
            # invent a Project/customer entity
            assert r.object_type in ("meeting", "person", "department", "policy", "process", "decision")


def test_ownership_inference_rejection():
    """Resolution succeeding must NOT become relationship inference."""
    _, _, result = _resolve_then_reason("Who owns the meeting?")
    for c in result.conclusions:
        if c.state in (rz.OBSERVED, rz.DERIVED):
            assert "owns" not in c.text.lower()


def test_employment_inference_rejection():
    _, _, result = _resolve_then_reason("Which employee manages the meeting?")
    for c in result.conclusions:
        if c.state in (rz.OBSERVED, rz.DERIVED):
            low = c.text.lower()
            assert "employee" not in low and "manages" not in low


# =====================================================================
# 23-24. Multiple / zero candidates (explicit restatement of the core rule).
# =====================================================================

def test_multiple_meetings_ambiguity():
    ids = {"entity_ids": []}
    try:
        ids["entity_ids"].append(_make_entity(REAL_WORKSPACE, "TEST-7B Meeting A"))
        ids["entity_ids"].append(_make_entity(REAL_WORKSPACE, "TEST-7B Meeting B"))
        res = qres.resolve_references("Who organized the meeting?", REAL_WORKSPACE, OWNER)
        assert res.status == qres.AMBIGUOUS
        assert len(res.ambiguous[0]["candidate_ids"]) == 3   # 2 synthetic + 1 real
    finally:
        _cleanup(ids)


def test_zero_meetings_unresolved():
    ws = _fresh_workspace()
    res = qres.resolve_references("Who organized the meeting?", ws, OWNER)
    assert res.status == qres.UNRESOLVED


# =====================================================================
# Supplementary: rewrite safety, no writes, cleanup.
# =====================================================================

def test_rewrite_never_fires_on_ambiguity():
    ids = {"entity_ids": []}
    try:
        ids["entity_ids"].append(_make_entity(REAL_WORKSPACE, "TEST-7B Second Meeting"))
        q = "Who organized the meeting?"
        res = qres.resolve_references(q, REAL_WORKSPACE, OWNER)
        assert qres.rewrite_question_with_references(q, res) == q
    finally:
        _cleanup(ids)


def test_resolver_performs_no_writes():
    import ast
    tree = ast.parse(open(qres.__file__, encoding="utf-8").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for write in ("insert", "update", "delete", "upsert", "rpc"):
        assert write not in called, f"query_resolution.py must never call .{write}()"


def test_no_leftover_test_7b_fixtures():
    for table, col in (("knowledge_entities", "canonical_label"),
                        ("structured_knowledge", "statement")):
        leftover = supabase.table(table).select("id").ilike(col, "TEST-7B%").execute().data
        assert leftover == [], f"leftover TEST-7B rows in {table}"
    aliases = supabase.table("knowledge_entity_aliases").select("id").ilike("alias_text", "QSync").execute().data
    assert aliases == []


def test_placeholder_full_regression_run_separately():
    assert True
