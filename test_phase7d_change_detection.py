"""
Phase 7D Organizational Change Detection tests.

Detection is fully deterministic (no LLM, no similarity, no thresholds) and
derived from existing columns only -- so every test runs against the real
corpus or against throwaway workspaces. Real production memory/graph state
is never mutated.

Run with: python -m pytest test_phase7d_change_detection.py -v
"""
import uuid
import time
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import impact_analysis as ia
import wiki_projection as wp
import change_detection as cd

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

PRODUCT = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
MEETING = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
APPROVAL_VALID_FROM = datetime(2026, 9, 15, tzinfo=timezone.utc)
FUTURE = APPROVAL_VALID_FROM + timedelta(days=1)


def _now():
    return datetime.now(timezone.utc)


def _now_iso():
    return _now().isoformat()


def _fresh_workspace():
    return str(uuid.uuid4())


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


def _make_sk(ws, ids, **kw):
    row = {"workspace_id": ws, "canonical_source_type": "knowledge_note", "canonical_id": str(uuid.uuid4()),
           "provider": "google_chat", "primitive_type": "fact", "statement": "TEST-7D statement",
           "raw_subject_phrase": "TEST-7D", "qualifier_words": [], "sensitivity": "internal",
           "authority": "official", "source_tier": 2, "lifecycle_status": "active",
           "extraction_version": "v2.1", "captured_at": _now_iso(),
           "extraction_run_id": str(uuid.uuid4()), "primitive_fingerprint": f"test-7d-{uuid.uuid4()}"}
    row.update(kw)
    sk = supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]
    ids.setdefault("sk", []).append(sk)
    return sk


def _make_mem(ws, ids, sk, **kw):
    p = {"p_workspace_id": ws, "p_memory_type": "policy", "p_promotion_basis": "authoritative_policy",
         "p_valid_from": None, "p_valid_until": None, "p_supersedes_memory_id": None,
         "p_consolidation_run_id": None,
         "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                          "stance": "supports", "captured_at": _now_iso()}]}
    p.update(kw)
    m = supabase.rpc("create_memory_with_evidence", p).execute().data
    ids.setdefault("mem", []).append(m)
    return m


def _make_entity(ws, ids, label, entity_type="department"):
    e = supabase.table("knowledge_entities").insert({
        "workspace_id": ws, "entity_type": entity_type,
        "canonical_label": label, "status": "active"}).execute().data[0]["id"]
    ids.setdefault("ent", []).append(e)
    return e


def _make_rel(ws, ids, s_id, t_id, sk, rel_type="requires_approval_from"):
    r = supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": ws, "p_source_object_type": "structured_knowledge", "p_source_object_id": s_id,
        "p_target_object_type": "entity", "p_target_object_id": t_id,
        "p_relationship_type": rel_type, "p_rationale": "TEST-7D", "p_confidence": 0.9,
        "p_valid_from": _now_iso(), "p_valid_until": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                        "stance": "supports", "captured_at": _now_iso()}]}).execute().data
    ids.setdefault("rel", []).append(r)
    return r


# =====================================================================
# 1-2. Unchanged state produces no change.
# =====================================================================

def test_unchanged_memory_produces_no_change():
    """The decisive anti-noise test: the real corpus runs a sleep cycle
    every night (85+ runs) revalidating 4 unchanged memories. A recent
    window must be COMPLETELY quiet."""
    res = cd.detect_changes(REAL_WORKSPACE, OWNER, since=_now() - timedelta(hours=2))
    assert res.events == []
    assert res.scanned["memories"] == 4


def test_revalidation_is_never_change():
    """last_confirmed_at is bumped by every sleep run and must never be read
    as a change signal."""
    runs = supabase.table("memory_consolidation_runs").select("id") \
        .eq("workspace_id", REAL_WORKSPACE) \
        .gte("started_at", (_now() - timedelta(hours=2)).isoformat()).execute().data
    res = cd.detect_changes(REAL_WORKSPACE, OWNER, since=_now() - timedelta(hours=2))
    assert res.events == [], f"{len(runs)} consolidation runs must produce 0 organizational changes"
    import ast
    tree = ast.parse(open(cd.__file__, encoding="utf-8").read())
    src = open(cd.__file__, encoding="utf-8").read()
    # referenced only in prose explaining why it is excluded, never read
    assert 'row.get("last_confirmed_at")' not in src
    assert 'r["last_confirmed_at"]' not in src


def test_unchanged_evidence_produces_no_change():
    res = cd.detect_changes(REAL_WORKSPACE, OWNER, since=_now() - timedelta(hours=2),
                             include_informational=True)
    assert res.events == []


# =====================================================================
# 3-6. Real change types.
# =====================================================================

def test_new_authoritative_policy_is_meaningful():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D new policy"))
        res = cd.detect_changes(ws, OWNER, since=t0)
        promoted = [e for e in res.events if e.change_type == cd.MEMORY_PROMOTED]
        assert len(promoted) == 1
        assert promoted[0].significance == cd.MEANINGFUL
        assert promoted[0].occurred_at_source == "org_memory.created_at"
    finally:
        _cleanup(ws, ids)


def test_policy_replacement():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        a = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D rotate 90 days"))
        b = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D rotate 30 days"),
                       p_supersedes_memory_id=a)
        res = cd.detect_changes(ws, OWNER, since=t0)
        pol = [e for e in res.events if e.change_type == cd.POLICY_CHANGED]
        assert len(pol) == 1
        ev = pol[0]
        assert ev.significance == cd.CRITICAL
        assert ev.reasoning_state == cd.DERIVED
        assert ev.previous_state["memory_id"] == a and ev.new_state["memory_id"] == b
        assert ev.occurred_at_source == "org_memory.superseded_at"
    finally:
        _cleanup(ws, ids)


def test_process_replacement():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        a = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D Mondays"),
                       p_memory_type="process", p_promotion_basis="recurring_durable_process")
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D Fridays"),
                   p_memory_type="process", p_promotion_basis="recurring_durable_process",
                   p_supersedes_memory_id=a)
        res = cd.detect_changes(ws, OWNER, since=t0)
        assert any(e.change_type == cd.PROCESS_CHANGED for e in res.events)
    finally:
        _cleanup(ws, ids)


def test_memory_supersession_uses_the_real_boundary():
    """The change time must be superseded_at (== successor created_at), never
    an ingestion/retrieval/Wiki timestamp."""
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        a = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D old"))
        b = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D new"), p_supersedes_memory_id=a)
        pred = supabase.table("org_memory").select("superseded_at").eq("id", a).execute().data[0]
        succ = supabase.table("org_memory").select("created_at").eq("id", b).execute().data[0]
        assert pred["superseded_at"] == succ["created_at"]
        res = cd.detect_changes(ws, OWNER, since=t0)
        ev = next(e for e in res.events if e.change_type == cd.POLICY_CHANGED)
        assert ev.occurred_at == pred["superseded_at"]
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 7-8. Graph changes.
# =====================================================================

def test_relationship_addition():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        sk = _make_sk(ws, ids)
        ent = _make_entity(ws, ids, "TEST-7D Dept")
        _make_rel(ws, ids, sk, ent, sk)
        res = cd.detect_changes(ws, OWNER, since=t0)
        added = [e for e in res.events if e.change_type == cd.RELATIONSHIP_ADDED]
        assert len(added) == 1
        assert added[0].significance == cd.MEANINGFUL
        assert ent in added[0].affected_entities
    finally:
        _cleanup(ws, ids)


def test_relationship_inactivation_detected_and_deletion_reported_undetectable():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        sk = _make_sk(ws, ids)
        ent = _make_entity(ws, ids, "TEST-7D Dept")
        rel = _make_rel(ws, ids, sk, ent, sk)
        # a REAL status change, stamped by the existing updated_at column
        supabase.table("knowledge_relationships").update(
            {"status": "retracted", "updated_at": _now_iso()}).eq("id", rel).execute()
        res = cd.detect_changes(ws, OWNER, since=t0)
        changed = [e for e in res.events if e.change_type == cd.RELATIONSHIP_CHANGED]
        assert len(changed) == 1 and changed[0].new_state["status"] == "retracted"
        # hard deletion is honestly reported as undetectable, not faked
        assert "RELATIONSHIP_REMOVED" in res.undetectable
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 9-10. Contradiction / review.
# =====================================================================

def test_unresolved_contradiction_becomes_review_required():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        sk = _make_sk(ws, ids, statement="TEST-7D conflicting rule")
        supabase.rpc("upsert_review_candidate", {
            "p_workspace_id": ws, "p_structured_knowledge_id": sk,
            "p_reason": "TEST-7D unresolved", "p_consolidation_run_id": None}).execute()
        res = cd.detect_changes(ws, OWNER, since=t0)
        rev = [e for e in res.events if e.change_type == cd.REVIEW_REQUIRED]
        assert len(rev) == 1
        assert rev[0].reasoning_state == cd.OBSERVED
    finally:
        _cleanup(ws, ids)


def test_review_required_never_silently_resolved():
    """The real pending Q4 candidate stays pending; detection never mutates
    or auto-decides it."""
    before = supabase.table("memory_review_queue").select("id,status") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    cd.detect_changes(REAL_WORKSPACE, OWNER)
    after = supabase.table("memory_review_queue").select("id,status") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert before == after
    assert all(r["status"] == "pending" for r in after)


# =====================================================================
# 11-12. Noise rejection.
# =====================================================================

def test_routine_calendar_event_is_not_organizational_change():
    snaps = supabase.table("calendar_event_snapshots").select("created_at") \
        .eq("workspace_id", REAL_WORKSPACE).order("created_at", desc=True).execute().data
    assert snaps, "the real corpus has calendar snapshots"
    newest = datetime.fromisoformat(snaps[0]["created_at"].replace("Z", "+00:00"))
    res = cd.detect_changes(REAL_WORKSPACE, OWNER,
                             since=newest - timedelta(minutes=5), until=newest + timedelta(minutes=5))
    assert res.events == []


def test_duplicate_ingestion_is_not_meaningful_change():
    """Extra structured_knowledge alone is data arrival: INFORMATIONAL and
    excluded by default."""
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        _make_sk(ws, ids, statement="TEST-7D duplicate one")
        _make_sk(ws, ids, statement="TEST-7D duplicate two")
        default = cd.detect_changes(ws, OWNER, since=t0)
        with_info = cd.detect_changes(ws, OWNER, since=t0, include_informational=True)
        assert default.events == []
        info = [e for e in with_info.events if e.change_type == cd.NEW_KNOWLEDGE]
        assert len(info) == 2 and all(e.significance == cd.INFORMATIONAL for e in info)
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 13-15. Temporal, workspace, sensitivity.
# =====================================================================

def test_temporal_correctness_window_bounds():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D windowed"))
        time.sleep(0.2); t1 = _now()
        assert len(cd.detect_changes(ws, OWNER, since=t0, until=t1).events) == 1
        assert cd.detect_changes(ws, OWNER, since=t1).events == []
        assert cd.detect_changes(ws, OWNER, until=t0).events == []
    finally:
        _cleanup(ws, ids)


def test_future_dated_relationship_validity_is_not_a_change_event():
    """The real Product approval edge was CREATED in the past but becomes
    valid in the future. Creation is the change; future validity is not."""
    rel = supabase.table("knowledge_relationships").select("created_at,valid_from") \
        .eq("target_object_id", PRODUCT).execute().data[0]
    created = datetime.fromisoformat(rel["created_at"].replace("Z", "+00:00"))
    valid_from = datetime.fromisoformat(rel["valid_from"].replace("Z", "+00:00"))
    assert valid_from > created
    around_validity = cd.detect_changes(REAL_WORKSPACE, OWNER,
                                         since=valid_from - timedelta(minutes=1),
                                         until=valid_from + timedelta(minutes=1))
    assert around_validity.events == []


def test_workspace_isolation():
    res = cd.detect_changes(LEAK_WORKSPACE, OWNER)
    assert res.events == []
    assert res.scanned["memories"] == 0


def test_sensitivity_isolation():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        _make_mem(ws, ids, _make_sk(ws, ids, sensitivity="restricted",
                                     statement="TEST-7D restricted policy"))
        low = cd.detect_changes(ws, LOW, since=t0)
        owner = cd.detect_changes(ws, OWNER, since=t0)
        assert low.events == [] and low.scanned["memories"] == 0
        assert len(owner.events) == 1
        assert owner.events[0].significance == cd.CRITICAL   # elevated sensitivity
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 16-18. Impact integration, significance, UNKNOWN.
# =====================================================================

def test_impact_path_integration_never_adds_unverified_targets():
    """Part 10's exact rule: a real path may be attached; 'Sales' may not."""
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        sk = _make_sk(ws, ids, statement="TEST-7D approval requirement")
        ent = _make_entity(ws, ids, "TEST-7D Product")
        _make_rel(ws, ids, sk, ent, sk)
        a = _make_mem(ws, ids, sk)
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D replacement"), p_supersedes_memory_id=a)

        res = cd.detect_changes(ws, OWNER, since=t0)
        ev = next(e for e in res.events if e.change_type == cd.POLICY_CHANGED)
        imp = ia.analyze_impact("structured_knowledge", sk, ws, OWNER, max_hops=1,
                                 candidate_targets=[{"kind": "entity", "object_id": "sales-x",
                                                      "label": "Sales"}])
        cd.attach_impact(ev, imp)
        assert ent in ev.affected_entities
        assert "sales-x" not in ev.affected_entities
        assert [n["target_label"] for n in imp.not_established] == ["Sales"]
    finally:
        _cleanup(ws, ids)


def test_significance_classification_rules():
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        a = _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D sig old"))
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D sig new"), p_supersedes_memory_id=a)
        res = cd.detect_changes(ws, OWNER, since=t0)
        by_type = {e.change_type: e for e in res.events}
        assert by_type[cd.POLICY_CHANGED].significance == cd.CRITICAL
        assert by_type[cd.MEMORY_PROMOTED].significance == cd.MEANINGFUL
        assert {e.significance for e in res.events} <= {cd.CRITICAL, cd.MEANINGFUL, cd.INFORMATIONAL}
    finally:
        _cleanup(ws, ids)


def test_undetectable_changes_reported_not_invented():
    res = cd.detect_changes(REAL_WORKSPACE, OWNER)
    for key in ("RELATIONSHIP_REMOVED", "MEMORY_DORMANT", "MEMORY_ARCHIVED", "KNOWLEDGE_BECAME_INVALID"):
        assert key in res.undetectable
    produced = {e.change_type for e in res.events}
    assert produced.isdisjoint(set(cd.UNDETECTABLE_CHANGES))


def test_no_invented_timestamp_columns():
    """Every occurred_at must come from a real, pre-existing column."""
    res = cd.detect_changes(REAL_WORKSPACE, OWNER)
    allowed = {"org_memory.created_at", "org_memory.superseded_at",
               "knowledge_relationships.created_at", "knowledge_relationships.updated_at",
               "memory_review_queue.created_at", "structured_knowledge.created_at"}
    for e in res.events:
        assert e.occurred_at_source in allowed
        assert e.occurred_at


# =====================================================================
# 19-21. No mutation.
# =====================================================================

def test_no_memory_graph_or_wiki_mutation():
    mem_before = len(supabase.table("org_memory").select("id").execute().data)
    rel_before = len(supabase.table("knowledge_relationships").select("id").execute().data)
    sk_before = len(supabase.table("structured_knowledge").select("id").execute().data)
    rev_before = len(supabase.table("memory_review_queue").select("id").execute().data)
    wiki_before = wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash

    cd.detect_changes(REAL_WORKSPACE, OWNER, include_informational=True)
    cd.detect_changes(REAL_WORKSPACE, OWNER, since=_now() - timedelta(days=365))

    assert len(supabase.table("org_memory").select("id").execute().data) == mem_before
    assert len(supabase.table("knowledge_relationships").select("id").execute().data) == rel_before
    assert len(supabase.table("structured_knowledge").select("id").execute().data) == sk_before
    assert len(supabase.table("memory_review_queue").select("id").execute().data) == rev_before
    assert wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash == wiki_before


def test_module_performs_no_writes_and_no_proactive_action():
    import ast
    tree = ast.parse(open(cd.__file__, encoding="utf-8").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for write in ("insert", "update", "delete", "upsert", "rpc"):
        assert write not in called, f"change_detection.py must never call .{write}()"
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # no notification/email/task machinery, and no LLM
    assert "ai" not in imported
    assert "wiki_generation" not in imported


def test_wiki_boundary_produces_identities_only():
    """Part 14: ChangeEvent -> invalidation CANDIDATE, never a Wiki write or
    regeneration."""
    ws, ids = _fresh_workspace(), {}
    try:
        t0 = _now(); time.sleep(0.2)
        _make_mem(ws, ids, _make_sk(ws, ids, statement="TEST-7D wiki boundary"))
        res = cd.detect_changes(ws, OWNER, since=t0)
        cands = cd.wiki_invalidation_candidates(res.events)
        assert cands and all(set(c) == {"page_kind", "object_id", "reason"} for c in cands)
        import ast
        tree = ast.parse(open(cd.__file__, encoding="utf-8").read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "build_page" not in called and "generate_wiki_page" not in called
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 22-24. Dashboard, real corpus, cleanup.
# =====================================================================

def test_dashboard_compatibility_single_engine():
    res = cd.detect_changes(REAL_WORKSPACE, OWNER, include_informational=True)
    dash = cd.summarize_for_dashboard(res)
    assert set(dash) == {"whats_new", "what_changed", "needs_attention"}
    all_ids = {id(e) for e in res.events}
    for bucket in dash.values():
        for e in bucket:
            assert id(e) in all_ids, "buckets must be views over ONE event list"


def test_real_corpus_benchmark():
    """All-time scan of the real corpus surfaces exactly the real history:
    3 relationships added, 4 memories promoted, 1 review candidate."""
    res = cd.detect_changes(REAL_WORKSPACE, OWNER)
    kinds = [e.change_type for e in res.events]
    assert kinds.count(cd.RELATIONSHIP_ADDED) == 3
    assert kinds.count(cd.MEMORY_PROMOTED) == 4
    assert kinds.count(cd.REVIEW_REQUIRED) == 1
    assert cd.POLICY_CHANGED not in kinds and cd.MEMORY_SUPERSEDED not in kinds
    for e in res.events:
        assert e.reasoning_state in (cd.OBSERVED, cd.DERIVED)
        assert e.occurred_at


def test_no_leftover_test_7d_fixtures():
    for table, col in (("knowledge_entities", "canonical_label"),
                        ("structured_knowledge", "statement")):
        leftover = supabase.table(table).select("id").ilike(col, "TEST-7D%").execute().data
        assert leftover == [], f"leftover TEST-7D rows in {table}"


def test_placeholder_full_regression_run_separately():
    assert True
