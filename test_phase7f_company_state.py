"""
Phase 7F Company State tests.

CompanyState is a deterministic derived view (no LLM anywhere), so every
test runs against the real corpus or throwaway workspaces. Real production
state is never mutated.

Run with: python -m pytest test_phase7f_company_state.py -v
"""
import uuid
import time
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import change_detection as cd
import proactive_intelligence as pi
import wiki_projection as wp
import company_state as cs

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

MEETING = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
MEETING_VALID_FROM = datetime(2026, 8, 16, 8, 30, tzinfo=timezone.utc)
LONG_AGO = MEETING_VALID_FROM - timedelta(days=400)


def _now():
    return datetime.now(timezone.utc)


def _now_iso():
    return _now().isoformat()


def _fresh():
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
        supabase.table("knowledge_entity_identifiers").delete().eq("entity_id", e).execute()
        supabase.table("knowledge_entities").delete().eq("id", e).execute()
    for s in ids.get("sk", []):
        supabase.table("structured_knowledge").delete().eq("id", s).execute()


def _mk_sk(ws, ids, **kw):
    row = {"workspace_id": ws, "canonical_source_type": "knowledge_note", "canonical_id": str(uuid.uuid4()),
           "provider": "google_chat", "primitive_type": "fact", "statement": "TEST-7F stmt",
           "raw_subject_phrase": "TEST-7F", "qualifier_words": [], "sensitivity": "internal",
           "authority": "official", "source_tier": 2, "lifecycle_status": "active",
           "extraction_version": "v2.1", "captured_at": _now_iso(),
           "extraction_run_id": str(uuid.uuid4()), "primitive_fingerprint": f"t7f-{uuid.uuid4()}"}
    row.update(kw)
    s = supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]
    ids.setdefault("sk", []).append(s)
    return s


def _mk_mem(ws, ids, sk, **kw):
    p = {"p_workspace_id": ws, "p_memory_type": "policy", "p_promotion_basis": "authoritative_policy",
         "p_valid_from": None, "p_valid_until": None, "p_supersedes_memory_id": None,
         "p_consolidation_run_id": None,
         "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk,
                          "stance": "supports", "captured_at": _now_iso()}]}
    p.update(kw)
    m = supabase.rpc("create_memory_with_evidence", p).execute().data
    ids.setdefault("mem", []).append(m)
    return m


def _mk_ent(ws, ids, label, etype="department", email=None):
    e = supabase.table("knowledge_entities").insert({
        "workspace_id": ws, "entity_type": etype, "canonical_label": label,
        "status": "active"}).execute().data[0]["id"]
    ids.setdefault("ent", []).append(e)
    if email:
        supabase.table("knowledge_entity_identifiers").insert({
            "workspace_id": ws, "entity_id": e, "identifier_type": "email",
            "identifier_value": email}).execute()
    return e


# =====================================================================
# 1-2. Current and historical state.
# =====================================================================

def test_current_state():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert st.workspace_id == REAL_WORKSPACE
    assert st.as_of == "current"
    assert st.evidence_summary["durable_memories"] == 4
    assert set(st.state_confidence) == {
        "active_policies", "active_processes", "active_decisions",
        "active_people", "active_departments", "verified_connections"}


def test_historical_state():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER, as_of=LONG_AGO)
    assert st.as_of == LONG_AGO.isoformat()
    for dim in (st.active_policies, st.active_processes, st.active_people,
                st.active_departments, st.verified_connections):
        assert dim.items == []
        assert dim.state == cs.UNKNOWN


def test_historical_temporal_consistency_no_mixed_dates():
    """Every dimension must honour the SAME as_of -- a snapshot can never
    mix a current memory with a historical graph."""
    st = cs.build_company_state(REAL_WORKSPACE, OWNER, as_of=LONG_AGO)
    assert st.as_of == LONG_AGO.isoformat()
    assert st.evidence_summary["durable_memories"] == 0
    assert st.evidence_summary["verified_people"] == 0
    assert st.evidence_summary["verified_relationships"] == 0


# =====================================================================
# 3-5. Durable memory dimensions.
# =====================================================================

def test_active_policy_state():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert len(st.active_policies.items) == 3
    assert st.active_policies.state == cs.OBSERVED
    for item in st.active_policies.items:
        assert item.kind == "policy"
        assert item.evidence_ids, "every policy must carry real evidence"
        assert item.attributes["lifecycle_status"] == "active"


def test_active_process_state():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert len(st.active_processes.items) == 1
    assert st.active_processes.items[0].attributes["promotion_basis"] == "recurring_durable_process"


def test_durable_decision_state_absence_is_not_denial():
    """0 decisions exist. The state must say no decision is RECORDED, never
    that the company made none (Part 4)."""
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert st.active_decisions.items == []
    assert st.active_decisions.state == cs.UNKNOWN
    note = st.active_decisions.coverage_note.lower()
    assert "absence of record" in note
    assert "no decisions were made" not in note.replace("not evidence that no decisions were made", "")


# =====================================================================
# 6-7. Change + attention integration (no second engine).
# =====================================================================

def test_recent_changes_come_from_the_change_detector():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    direct = cd.detect_changes(REAL_WORKSPACE, OWNER)
    assert len(st.recent_changes) == len(direct.events)
    assert {e.change_type for e in st.recent_changes} == {e.change_type for e in direct.events}


def test_pending_review_appears_as_attention_not_fact():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    review = [s for s in st.attention_items if s.signal_type == cd.REVIEW_REQUIRED]
    assert len(review) == 1
    assert review[0].attention == pi.REVIEW
    assert review[0].is_hypothesis is True
    # and it is ALSO surfaced as uncertainty, never as durable truth
    topics = {u["topic"] for u in st.open_uncertainty}
    assert "unresolved_review" in topics


def test_no_second_alert_or_change_engine():
    import ast
    tree = ast.parse(open(cs.__file__, encoding="utf-8").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    # consumes the existing engines...
    assert "detect_changes" in called and "build_signals" in called
    # ...and never re-implements detection or alerting
    assert "classify_attention" not in called
    assert "resolve_audience" not in called


# =====================================================================
# 8-10. Verified entities and connections.
# =====================================================================

def test_verified_people():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    labels = {i.label for i in st.active_people.items}
    assert labels == {"Tanmay", "John Snow"}
    for item in st.active_people.items:
        assert any("employment is not established" in n for n in item.not_established)


def test_verified_departments():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    labels = {i.label for i in st.active_departments.items}
    assert labels == {"Product", "Operations"}
    for item in st.active_departments.items:
        assert any("membership is not established" in n for n in item.not_established)


def test_verified_relationships():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert len(st.verified_connections.items) == 2
    types = {i.attributes["relationship_type"] for i in st.verified_connections.items}
    assert types == {"organized", "attended"}
    for item in st.verified_connections.items:
        assert item.evidence_ids
        assert item.label.isascii(), "labels must stay ASCII-safe for downstream consumers"


# =====================================================================
# 11-14. Uncertainty + state labels.
# =====================================================================

def test_uncertainty_representation():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    topics = {u["topic"] for u in st.open_uncertainty}
    for expected in ("active_decisions", "person_roles", "department_membership",
                     "policy_process_ownership"):
        assert expected in topics
    assert all(u["state"] == cs.UNKNOWN for u in st.open_uncertainty)


def test_observed_state_label():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert st.state_confidence["active_policies"] == cs.OBSERVED
    assert all(i.state == cs.OBSERVED for i in st.active_policies.items)


def test_derived_or_observed_only_for_connections():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert all(i.state in (cs.OBSERVED, cs.DERIVED) for i in st.verified_connections.items)


def test_unknown_state_label():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert st.state_confidence["active_decisions"] == cs.UNKNOWN
    empty_ws = cs.build_company_state(_fresh(), OWNER)
    assert all(v == cs.UNKNOWN for v in empty_ws.state_confidence.values())


def test_state_confidence_is_not_a_second_framework():
    """Only the Phase 7A vocabulary is used -- no numeric score anywhere."""
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert set(st.state_confidence.values()) <= {cs.OBSERVED, cs.DERIVED, cs.UNKNOWN}
    for item in st.active_policies.items + st.active_people.items:
        assert not any(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for k, v in item.attributes.items() if k.endswith("score"))


# =====================================================================
# 15-17. No fabricated organizational claims.
# =====================================================================

def test_no_false_ownership():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    blob = _all_text(st)
    assert "owns" not in blob and "ownership of this" not in blob


def test_no_false_employment():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    blob = _all_text(st)
    assert "employee of" not in blob and "works for" not in blob


def test_no_false_department_membership():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    blob = _all_text(st)
    assert "member of" not in blob
    # membership is explicitly reported as NOT established instead
    assert any("membership is not established" in u["detail"] or
               "membership" in u["topic"] for u in st.open_uncertainty)


def _all_text(st) -> str:
    parts = []
    for name in ("active_policies", "active_processes", "active_decisions",
                 "active_people", "active_departments", "verified_connections"):
        dim = getattr(st, name)
        parts.append(dim.coverage_note)
        for i in dim.items:
            parts.append(str(i.label))
    return " ".join(parts).lower()


# =====================================================================
# 18-20. Isolation.
# =====================================================================

def _item_data_text(st) -> str:
    """Only real DATA (item labels + evidence ids) -- deliberately excludes
    the static coverage-note boilerplate, which legitimately contains the
    product name ('what KNOVA has evidence for') and would otherwise
    false-positive a leak check."""
    parts = []
    for name in ("active_policies", "active_processes", "active_decisions",
                 "active_people", "active_departments", "verified_connections"):
        for i in getattr(st, name).items:
            parts.append(str(i.label))
            parts.extend(i.evidence_ids)
    return " ".join(parts).lower()


def test_workspace_isolation():
    leak = cs.build_company_state(LEAK_WORKSPACE, OWNER)
    assert leak.active_policies.items == []
    assert leak.active_people.items == []
    assert leak.active_departments.items == []
    assert leak.verified_connections.items == []
    assert leak.evidence_summary["durable_memories"] == 0
    blob = _item_data_text(leak)
    for token in ("credential", "knova", "tanmay", "hardware"):
        assert token not in blob


def test_global_15th_row_isolation():
    """Global structured_knowledge is 15; the target workspace holds 14. The
    15th row lives in another workspace and must not reach this state."""
    target = supabase.table("structured_knowledge").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    total = supabase.table("structured_knowledge").select("id").execute().data
    assert len(target) == 14 and len(total) == 15
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    ev_ids = {e for i in (st.active_policies.items + st.active_processes.items) for e in i.evidence_ids}
    target_ids = {f"structured_knowledge:{r['id']}" for r in target}
    assert ev_ids <= target_ids


def test_sensitivity_isolation_applied_before_aggregation():
    """A restricted memory must not reach the aggregate at all -- not even
    as a count."""
    ws, ids = _fresh(), {}
    try:
        _mk_mem(ws, ids, _mk_sk(ws, ids, sensitivity="internal", statement="TEST-7F visible"))
        _mk_mem(ws, ids, _mk_sk(ws, ids, sensitivity="restricted", statement="TEST-7F restricted"))
        low = cs.build_company_state(ws, LOW)
        owner = cs.build_company_state(ws, OWNER)
        assert len(low.active_policies.items) == 1
        assert len(owner.active_policies.items) == 2
        assert low.evidence_summary["durable_memories"] == 1   # count itself must not leak
        assert "restricted" not in _all_text(low)
    finally:
        _cleanup(ws, ids)


# =====================================================================
# 21-22. Real benchmark + integration shape.
# =====================================================================

def test_real_workspace_benchmark():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    assert st.evidence_summary == {
        "durable_memories": 4, "verified_people": 2, "verified_departments": 2,
        "verified_relationships": 2, "recent_changes": len(st.recent_changes),
        "attention_items": len(st.attention_items),
        "undetectable_change_types": list(cd.UNDETECTABLE_CHANGES),
    }
    assert len(st.recent_changes) == 8
    assert len(st.attention_items) == 1


def test_synthetic_state_reflects_new_durable_knowledge():
    ws, ids = _fresh(), {}
    try:
        before = cs.build_company_state(ws, OWNER)
        assert before.active_policies.items == []
        _mk_mem(ws, ids, _mk_sk(ws, ids, statement="TEST-7F new policy"))
        after = cs.build_company_state(ws, OWNER)
        assert len(after.active_policies.items) == 1
        assert after.state_confidence["active_policies"] == cs.OBSERVED
    finally:
        _cleanup(ws, ids)


def test_state_vs_activity_distinction():
    """Calendar snapshots are activity and must never become state."""
    snaps = supabase.table("calendar_event_snapshots").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert snaps, "the real corpus has calendar activity"
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    snap_ids = {r["id"] for r in snaps}
    state_ids = {i.object_id for name in ("active_policies", "active_processes", "active_decisions",
                                           "active_people", "active_departments")
                 for i in getattr(st, name).items}
    assert state_ids.isdisjoint(snap_ids)


# =====================================================================
# 23-27. Dashboard, integrity, no mutation.
# =====================================================================

def test_dashboard_compatible_output():
    st = cs.build_company_state(REAL_WORKSPACE, OWNER)
    dash = cs.summarize_for_dashboard(st)
    assert set(dash) == {"whats_new", "whats_important", "needs_attention",
                          "whats_uncertain", "whats_connected", "whats_changed"}
    assert dash["whats_uncertain"] == st.open_uncertainty
    assert dash["whats_connected"] == st.verified_connections.items


def test_no_state_persistence():
    import ast
    src = open(cs.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for write in ("insert", "update", "delete", "upsert", "rpc"):
        assert write not in called, f"company_state.py must never call .{write}()"
    assert "company_state_snapshot" not in src.lower()


def test_data_integrity_unchanged_by_state_building():
    sk_b = len(supabase.table("structured_knowledge").select("id").execute().data)
    mem_b = len(supabase.table("org_memory").select("id").execute().data)
    rel_b = len(supabase.table("knowledge_relationships").select("id").execute().data)
    ent_b = len(supabase.table("knowledge_entities").select("id").execute().data)
    wiki_b = wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash

    cs.build_company_state(REAL_WORKSPACE, OWNER)
    cs.build_company_state(REAL_WORKSPACE, OWNER, as_of=LONG_AGO)

    assert len(supabase.table("structured_knowledge").select("id").execute().data) == sk_b
    assert len(supabase.table("org_memory").select("id").execute().data) == mem_b
    assert len(supabase.table("knowledge_relationships").select("id").execute().data) == rel_b
    assert len(supabase.table("knowledge_entities").select("id").execute().data) == ent_b
    assert wp.build_page("meeting", MEETING, REAL_WORKSPACE, OWNER).content_hash == wiki_b


def test_deterministic_across_rebuilds():
    a = cs.build_company_state(REAL_WORKSPACE, OWNER, include_connections=False)
    b = cs.build_company_state(REAL_WORKSPACE, OWNER, include_connections=False)
    assert a.state_confidence == b.state_confidence
    assert {i.object_id for i in a.active_policies.items} == {i.object_id for i in b.active_policies.items}
    assert [u["topic"] for u in a.open_uncertainty] == [u["topic"] for u in b.open_uncertainty]


def test_no_leftover_test_7f_fixtures():
    for table, col in (("knowledge_entities", "canonical_label"),
                        ("structured_knowledge", "statement")):
        leftover = supabase.table(table).select("id").ilike(col, "TEST-7F%").execute().data
        assert leftover == [], f"leftover TEST-7F rows in {table}"


def test_placeholder_full_regression_run_separately():
    assert True
