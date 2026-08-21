"""
Phase 7G Organizational Learning tests.

Longitudinal history cannot be faked convincingly with dicts alone, so the
supersession/review/relationship cases build REAL rows in throwaway
workspaces -- supersession chains go through the real
`create_memory_with_evidence` RPC, the same atomic path production uses.
Real production data is never mutated.

Run with: python -m pytest test_phase7g_organizational_learning.py -v
"""
import ast
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import memory_retrieval as mr
import organizational_learning as ol

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _cleanup(ws, ids):
    for m in ids.get("mem", []):
        supabase.table("memory_evidence").delete().eq("memory_id", m).execute()
    for m in reversed(ids.get("mem", [])):
        supabase.table("org_memory").delete().eq("id", m).execute()
    for r in ids.get("rel", []):
        supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", r).execute()
        supabase.table("knowledge_relationships").delete().eq("id", r).execute()
    supabase.table("memory_review_queue").delete().eq("workspace_id", ws).execute()
    for e in ids.get("ent", []):
        supabase.table("knowledge_entities").delete().eq("id", e).execute()
    for s in ids.get("sk", []):
        supabase.table("structured_knowledge").delete().eq("id", s).execute()


def _mk_sk(ws, ids, **kw):
    row = {"workspace_id": ws, "canonical_source_type": "knowledge_note",
           "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
           "primitive_type": "fact", "statement": "TEST-7G stmt",
           "raw_subject_phrase": "TEST-7G", "qualifier_words": [], "sensitivity": "internal",
           "authority": "official", "source_tier": 2, "lifecycle_status": "active",
           "extraction_version": "v2.1", "captured_at": _now_iso(),
           "extraction_run_id": str(uuid.uuid4()), "primitive_fingerprint": f"t7g-{uuid.uuid4()}"}
    row.update(kw)
    s = supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]
    ids.setdefault("sk", []).append(s)
    return s


def _mk_mem(ws, ids, sk, **kw):
    p = {"p_workspace_id": ws, "p_memory_type": "policy",
         "p_promotion_basis": "authoritative_policy", "p_valid_from": None,
         "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
         "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                          "stance": "supports", "captured_at": _now_iso()}]}
    p.update(kw)
    m = supabase.rpc("create_memory_with_evidence", p).execute().data
    ids.setdefault("mem", []).append(m)
    return m


def _mk_ent(ws, ids, label):
    e = supabase.table("knowledge_entities").insert({
        "workspace_id": ws, "entity_type": "department", "canonical_label": label,
        "status": "active"}).execute().data[0]["id"]
    ids.setdefault("ent", []).append(e)
    return e


def _mk_rel(ws, ids, src, tgt, rtype="references", src_type="entity", tgt_type="entity"):
    r = supabase.table("knowledge_relationships").insert({
        "workspace_id": ws, "source_object_type": src_type, "source_object_id": src,
        "target_object_type": tgt_type, "target_object_id": tgt,
        "relationship_type": rtype, "status": "active",
        "valid_from": _now_iso()}).execute().data[0]["id"]
    ids.setdefault("rel", []).append(r)
    return r


def _mem_dict(mid, mtype="policy", supersedes=None, superseded=None, created=0,
              confirmed=None, sens="internal"):
    """A pure in-memory row for the pure detectors."""
    return {"id": mid, "memory_type": mtype, "workspace_id": REAL_WORKSPACE, "sensitivity": sens,
            "supersedes_memory_id": supersedes,
            "superseded_at": (T0 + timedelta(days=superseded)).isoformat() if superseded else None,
            "created_at": (T0 + timedelta(days=created)).isoformat(),
            "last_confirmed_at": (T0 + timedelta(days=confirmed)).isoformat() if confirmed else None,
            "lifecycle_status": "superseded" if superseded else "active"}


def _sk_dict(sid, day, statement="stmt", recurrence=None):
    return {"id": sid, "statement": statement, "sensitivity": "internal",
            "captured_at": (T0 + timedelta(days=day)).isoformat(),
            "recurrence_text": recurrence, "requirement_kind": None}


def _types(signals):
    return sorted({s.learning_type for s in signals})


# =====================================================================
# 1-6. Real corpus: a sparse corpus must produce almost no learning.
# =====================================================================

def test_real_corpus_produces_no_policy_evolution():
    """No supersession has ever occurred in the real workspace, so a
    trajectory claim would be pure invention."""
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    assert [s for s in r.signals if s.learning_type == ol.POLICY_EVOLUTION] == []


def test_real_corpus_produces_no_process_trend():
    """Every real memory has exactly one grounding row, so there is no
    second independent observation to make a trend out of."""
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    assert [s for s in r.signals if s.learning_type == ol.PROCESS_TREND] == []


def test_real_corpus_produces_no_repeated_review():
    """Exactly one item is pending -- below the minimum of two, because one
    pending item is already a Phase 7E REVIEW signal."""
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    assert [s for s in r.signals if s.learning_type == ol.REPEATED_REVIEW] == []


def test_real_corpus_produces_no_relationship_pattern():
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    assert [s for s in r.signals if s.learning_type == ol.RELATIONSHIP_PATTERN] == []


def test_real_corpus_stability_comes_from_real_reconfirmation():
    """All 4 real memories carry last_confirmed_at > created_at -- a real
    revalidation event, not an absence of change."""
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    stable = [s for s in r.signals if s.learning_type == ol.STABILITY_PATTERN]
    assert len(stable) == 4
    for s in stable:
        assert s.observation_window["start"] < s.observation_window["end"]
        assert s.reasoning_state == ol.DERIVED


def test_real_corpus_persistent_uncertainty_reports_real_interval():
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    pu = [s for s in r.signals if s.learning_type == ol.PERSISTENT_UNCERTAINTY]
    assert len(pu) == 1
    assert pu[0].review_required is True
    assert pu[0].observation_window["start"] < pu[0].observation_window["end"]


# =====================================================================
# 7-9. Supersession chains, built through the REAL atomic RPC.
# =====================================================================

def test_three_generations_produce_policy_evolution():
    ws, ids = str(uuid.uuid4()), {}
    try:
        m1 = _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 1"))
        m2 = _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 2"), p_supersedes_memory_id=m1)
        _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 3"), p_supersedes_memory_id=m2)
        r = ol.detect_learning(ws, OWNER)
        evo = [s for s in r.signals if s.learning_type == ol.POLICY_EVOLUTION]
        assert len(evo) == 1
        assert evo[0].support_count == 2
        assert len(evo[0].memory_ids) == 3
    finally:
        _cleanup(ws, ids)


def test_single_supersession_is_a_change_not_learning():
    """One supersession is exactly what Phase 7D already reports as
    POLICY_CHANGED; reporting it again here would double-count it."""
    ws, ids = str(uuid.uuid4()), {}
    try:
        m1 = _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 1"))
        _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 2"), p_supersedes_memory_id=m1)
        r = ol.detect_learning(ws, OWNER)
        assert [s for s in r.signals if s.learning_type == ol.POLICY_EVOLUTION] == []
    finally:
        _cleanup(ws, ids)


def test_chain_read_sees_superseded_ancestors():
    """Regression. memory_retrieval._fetch_memory_rows(as_of=None) filters
    to lifecycle_status='active', which hides every superseded ancestor.
    Feeding POLICY_EVOLUTION from that set made it a permanently dead
    detector that reported 'no evolution' no matter how much history
    existed. _chain_memories must keep non-active generations."""
    ws, ids = str(uuid.uuid4()), {}
    try:
        m1 = _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 1"))
        _mk_mem(ws, ids, _mk_sk(ws, ids, statement="gen 2"), p_supersedes_memory_id=m1)
        active = ol._visible_memories(ws, OWNER, None)
        chain = ol._chain_memories(ws, OWNER, None)
        assert m1 not in [m["id"] for m in active], "ancestor should be inactive"
        assert m1 in [m["id"] for m in chain], "chain read must still see the ancestor"
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 10-11. Process trends need repetition ACROSS TIME.
# =====================================================================

def test_process_trend_requires_distinct_observation_times():
    p = [_mem_dict("p1", mtype="process")]
    g = {"p1": ["k1", "k2"]}
    sks = {"k1": _sk_dict("k1", 0, recurrence="every Monday"),
           "k2": _sk_dict("k2", 30, recurrence="every Monday")}
    out = ol._process_trend(REAL_WORKSPACE, p, g, sks)
    assert _types(out) == [ol.PROCESS_TREND]
    assert out[0].support_count == 2


def test_process_trend_rejects_same_instant_observations():
    """Two rows captured at the same instant are one observation split in
    two, not evidence gathered over time."""
    p = [_mem_dict("p1", mtype="process")]
    out = ol._process_trend(REAL_WORKSPACE, p, {"p1": ["k1", "k2"]},
                             {"k1": _sk_dict("k1", 0), "k2": _sk_dict("k2", 0)})
    assert out == []


# =====================================================================
# 12-14. Stability must never be inferred from silence.
# =====================================================================

def test_stability_requires_real_reconfirmation():
    out = ol._stability(REAL_WORKSPACE, [_mem_dict("a", created=0, confirmed=90)], {}, {})
    assert _types(out) == [ol.STABILITY_PATTERN]


def test_stability_is_not_inferred_from_silence():
    """A memory nobody ever revisited is NOT evidence of stability -- it is
    evidence of nothing (Part 9)."""
    out = ol._stability(REAL_WORKSPACE, [_mem_dict("a", created=0, confirmed=None)], {}, {})
    assert out == []


def test_superseded_memory_is_never_stable():
    out = ol._stability(REAL_WORKSPACE,
                         [_mem_dict("a", created=0, confirmed=90, superseded=95)], {}, {})
    assert out == []


# =====================================================================
# 15-17. Review-derived patterns.
# =====================================================================

def test_repeated_review_requires_two_pending_items():
    ws, ids = str(uuid.uuid4()), {}
    try:
        for i in (1, 2):
            s = _mk_sk(ws, ids, statement=f"ambiguous {i}")
            supabase.table("memory_review_queue").insert({
                "workspace_id": ws, "structured_knowledge_id": s,
                "status": "pending", "reason": "test"}).execute()
        r = ol.detect_learning(ws, OWNER)
        rep = [s for s in r.signals if s.learning_type == ol.REPEATED_REVIEW]
        assert len(rep) == 1 and rep[0].support_count == 2
        assert rep[0].review_required is True
    finally:
        _cleanup(ws, ids)


def test_one_pending_item_yields_uncertainty_but_not_repetition():
    ws, ids = str(uuid.uuid4()), {}
    try:
        s = _mk_sk(ws, ids, statement="ambiguous")
        supabase.table("memory_review_queue").insert({
            "workspace_id": ws, "structured_knowledge_id": s,
            "status": "pending", "reason": "test"}).execute()
        r = ol.detect_learning(ws, OWNER)
        assert [x for x in r.signals if x.learning_type == ol.REPEATED_REVIEW] == []
        assert len([x for x in r.signals if x.learning_type == ol.PERSISTENT_UNCERTAINTY]) == 1
    finally:
        _cleanup(ws, ids)


def test_persistent_uncertainty_is_duration_not_run_count():
    """The consolidation engine is incremental and never re-examines an
    existing pending item, so a run counter would measure how often the
    engine ran, not how often it failed. The signal must report elapsed
    time and must not expose any run-count field."""
    r = ol.detect_learning(REAL_WORKSPACE, OWNER)
    pu = [s for s in r.signals if s.learning_type == ol.PERSISTENT_UNCERTAINTY][0]
    assert "days" in pu.explanation
    assert not any("run" in f.lower() and "count" in f.lower() for f in vars(pu))
    assert "does not re-examine" in pu.explanation


# =====================================================================
# 18-19. Relationship patterns describe interaction ONLY.
# =====================================================================

def test_relationship_pattern_requires_repetition():
    ws, ids = str(uuid.uuid4()), {}
    try:
        a, b = _mk_ent(ws, ids, "Alpha"), _mk_ent(ws, ids, "Beta")
        _mk_rel(ws, ids, a, b)
        assert ol._relationship_patterns(ws, None) == [], "one relationship is one fact"
        _mk_rel(ws, ids, a, b, rtype="requires_approval_from")
        out = ol._relationship_patterns(ws, None)
        assert _types(out) == [ol.RELATIONSHIP_PATTERN]
        assert out[0].support_count == 2
    finally:
        _cleanup(ws, ids)


def test_relationship_pattern_never_asserts_membership():
    """Part 10: repeated interaction must never be upgraded into
    membership, ownership, employment, or management."""
    ws, ids = str(uuid.uuid4()), {}
    try:
        a, b = _mk_ent(ws, ids, "Alpha"), _mk_ent(ws, ids, "Beta")
        _mk_rel(ws, ids, a, b)
        _mk_rel(ws, ids, a, b, rtype="requires_approval_from")
        out = ol._relationship_patterns(ws, None)
        text = out[0].explanation.lower()
        for forbidden in ("member of", "belongs to", "owns", "works for",
                           "employee", "reports to", "manages"):
            assert forbidden not in text
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 20. Contradiction -> UNKNOWN, never a silent winner.
# =====================================================================

def test_contradiction_yields_unknown_and_requires_review():
    ws, ids = str(uuid.uuid4()), {}
    try:
        s1 = _mk_sk(ws, ids, statement="claim A")
        s2 = _mk_sk(ws, ids, statement="claim B")
        m = _mk_mem(ws, ids, s1)
        _mk_rel(ws, ids, s2, s1, rtype="contradicts",
                src_type="structured_knowledge", tgt_type="structured_knowledge")
        r = ol.detect_learning(ws, OWNER)
        cons = [s for s in r.signals
                if s.learning_type == ol.PERSISTENT_UNCERTAINTY and s.reasoning_state == ol.UNKNOWN]
        assert len(cons) == 1
        assert cons[0].memory_ids == [m]
        assert cons[0].contradicting_evidence
        assert cons[0].review_required is True
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 21-23. Security, isolation, temporal correctness.
# =====================================================================

def test_restricted_memory_excluded_before_aggregation():
    """Security is applied before counting, so a restricted memory cannot
    influence a pattern even as an anonymous +1."""
    ws, ids = str(uuid.uuid4()), {}
    try:
        s = _mk_sk(ws, ids, statement="secret policy", sensitivity="restricted")
        _mk_mem(ws, ids, s)
        assert ol._visible_memories(ws, LOW, None) == []
        assert len(ol._visible_memories(ws, OWNER, None)) == 1
        assert ol.detect_learning(ws, LOW).scanned["memories"] == 0
    finally:
        _cleanup(ws, ids)


def test_workspace_isolation():
    r = ol.detect_learning(LEAK_WORKSPACE, OWNER)
    blob = " ".join(str(s.subject.get("label")) for s in r.signals).lower()
    for term in ("credential", "knova", "tanmay", "procurement"):
        assert term not in blob


def test_historical_as_of_excludes_later_generations():
    """Reuses the Phase 6D.1 availability rule verbatim -- a generation
    created after as_of was not known then and cannot count toward a
    trajectory observed at that time."""
    chain = [_mem_dict("m1", created=0, superseded=10),
             _mem_dict("m2", supersedes="m1", created=10, superseded=20),
             _mem_dict("m3", supersedes="m2", created=20)]
    as_of = T0 + timedelta(days=15)
    available = [m for m in chain if mr._created_before_or_at(m, as_of)]
    assert len(available) == 2
    assert ol._policy_evolution(REAL_WORKSPACE, available, {}, {}) == []
    assert len(ol._policy_evolution(REAL_WORKSPACE, chain, {}, {})) == 1


# =====================================================================
# 24-26. Structural guarantees, proven by AST rather than text search
# (a docstring mentioning "insert" must not pass or fail a test).
# =====================================================================

def _module_ast():
    return ast.parse(inspect.getsource(ol))


def test_module_performs_no_database_mutation():
    banned = {"insert", "update", "upsert", "delete", "rpc"}
    found = [n.func.attr for n in ast.walk(_module_ast())
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in banned]
    assert found == [], f"learning must never write: {found}"


def test_module_introduces_no_new_relationship_type():
    """The ontology is frozen. The only relationship_type this module may
    name is `contradicts`, which already exists."""
    frozen = {"references", "requires_approval_from", "supersedes",
              "contradicts", "organized", "attended"}
    forbidden = {"member_of", "works_on", "owns", "manages", "reports_to",
                  "belongs_to", "employed_by"}
    literals = {n.value for n in ast.walk(_module_ast())
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (literals & forbidden)
    assert (literals & frozen) <= {"contradicts", "supersedes"}


def test_propose_for_review_writes_nothing_and_demands_a_human():
    sig = ol._stability(REAL_WORKSPACE, [_mem_dict("a", created=0, confirmed=90)], {}, {})[0]
    prop = ol.propose_for_review(sig)
    assert prop["requires_human_decision"] is True
    assert prop["structured_knowledge_id"] is None
    src = ast.parse(inspect.getsource(ol.propose_for_review))
    assert not [n for n in ast.walk(src) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"insert", "update", "upsert", "delete", "rpc"}]


def test_rejected_types_are_documented_with_reasons():
    assert set(ol.REJECTED_LEARNING_TYPES) == {"RECURRING_PATTERN", "KNOWLEDGE_GAP_PATTERN"}
    for reason in ol.REJECTED_LEARNING_TYPES.values():
        assert len(reason) > 40


# =====================================================================
# 27-30. LLM boundary: explanation only, never discovery.
# =====================================================================

def test_llm_failure_falls_back_to_deterministic_explanation():
    def boom(**kwargs):
        raise RuntimeError("model unavailable")
    r = ol.detect_learning(REAL_WORKSPACE, OWNER, chat_json_fn=boom)
    assert r.signals
    for s in r.signals:
        assert s.explanation_source == "deterministic"
        assert s.explanation


def test_llm_malformed_output_falls_back():
    r = ol.detect_learning(REAL_WORKSPACE, OWNER, chat_json_fn=lambda **k: {"wrong_key": "x"})
    for s in r.signals:
        assert s.explanation_source == "deterministic"


def test_llm_cannot_alter_established_facts():
    """The model may only rephrase. Type, state, support count and window
    are computed before it is ever called and must survive it unchanged.

    Both runs share one explicit as_of. PERSISTENT_UNCERTAINTY's window
    legitimately ends at the evaluation instant, so two calls taken from the
    live clock would differ by however long the first call took -- a
    property of the clock, not of the LLM, and exactly the kind of
    self-inflicted timing bug that would make this test lie."""
    pinned = datetime.now(timezone.utc)
    base = ol.detect_learning(REAL_WORKSPACE, OWNER, as_of=pinned)
    llm = ol.detect_learning(REAL_WORKSPACE, OWNER, as_of=pinned,
                              chat_json_fn=lambda **k: {"explanation": "TOTALLY DIFFERENT TEXT"})
    assert len(base.signals) == len(llm.signals)
    for a, b in zip(base.signals, llm.signals):
        assert (a.learning_type, a.reasoning_state, a.support_count, a.observation_window) == \
               (b.learning_type, b.reasoning_state, b.support_count, b.observation_window)
        assert b.explanation_source == "llm"


def test_company_state_view_shape():
    view = ol.learning_for_company_state(ol.detect_learning(REAL_WORKSPACE, OWNER))
    assert set(view) == {"emerging_patterns", "stable_patterns", "recurring_review",
                          "persistent_uncertainty", "interaction_patterns"}
    assert len(view["stable_patterns"]) == 4
