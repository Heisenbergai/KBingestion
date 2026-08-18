"""
Phase 6B.1 Memory Temporal + Review Hardening tests -- verifies (1) the
corrected valid_from semantics (NULL means "no known real-world start",
never event_time/captured_at/consolidation-time substituted in), applied
retroactively to the four real Phase 6B memories, and (2) the new durable
memory_review_queue table representing the real Q4 launch approval review
candidate.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase6b1_memory_temporal_review.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from supabase import create_client
import os

from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

REAL_RUN_ID = "36e9e367-41c6-4064-9fae-56f6830f7316"
REAL_REVIEW_QUEUE_ID = "b84e899e-fe92-4e86-90b6-a44a070a4a8e"

# The four real Phase 6B memories (unchanged ids/evidence/promotion_basis).
REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "credential_sharing":  "3d376631-894c-4e32-b3f5-3ecf7cfd5f61",
    "hardware_scope":      "8aef76c9-fda3-44d6-affb-769f2ff09326",
    "monday_capacity":     "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}
REAL_EVIDENCE_IDS = {
    "credential_logging": "cd3ce795-121f-4760-bec3-2fece754427a",
    "credential_sharing":  "d208e27c-0cc5-4270-a9cd-a4024aa94b12",
    "hardware_scope":      "c88b2636-6feb-4f4c-a411-ad6884823f6b",
    "monday_capacity":     "7db9647c-7b7d-4203-9f34-df1b1506cd8e",
}
REAL_EVENT_TIMES = {
    "credential_logging": "2026-08-15T15:47:37.147019+00:00",
    "credential_sharing":  "2026-08-15T15:47:37.147019+00:00",
    "hardware_scope":      "2026-08-15T15:46:27.813299+00:00",
    "monday_capacity":     "2026-08-15T15:41:27.398429+00:00",
}
REAL_CAPTURED_ATS = {
    "credential_logging": "2026-08-16T06:55:29.694031+00:00",
    "credential_sharing":  "2026-08-16T06:55:29.694031+00:00",
    "hardware_scope":      "2026-08-16T06:55:44.281195+00:00",
    "monday_capacity":     "2026-08-16T06:55:34.145793+00:00",
}

SK_Q4_LAUNCH_APPROVAL = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"  # real effective_from=2026-09-15


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _cleanup(ids: dict):
    ordered_keys = ("successor_id", "memory_id", "memory_a_id", "memory_b_id", "predecessor_id")
    memory_keys = [ids[k] for k in ordered_keys if ids.get(k)] + list(ids.get("memory_ids", []))
    for mid in memory_keys:
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in memory_keys:
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for key in ("review_id", "review_id_2"):
        if ids.get(key):
            supabase.table("memory_review_queue").delete().eq("id", ids[key]).execute()
    sk_keys = list(ids.get("sk_ids", [])) + [ids[k] for k in ("sk_a", "sk_b") if ids.get(k)]
    for sk_id in sk_keys:
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_synthetic_sk(sensitivity, label_suffix, **overrides) -> str:
    row = {
        "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
        "canonical_id": SK_Q4_LAUNCH_APPROVAL, "provider": "google_chat",
        "primitive_type": "fact", "statement": f"TEST-6B1 synthetic statement {label_suffix}",
        "raw_subject_phrase": "TEST-6B1", "sensitivity": sensitivity, "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6b1-{label_suffix}-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


def _is_current(valid_from, valid_until, as_of: datetime) -> bool:
    """Direct implementation of Part 5's frozen current/historical predicate,
    used here purely to prove real data against the rule -- not a new
    production module (no retrieval wiring exists yet, deliberately)."""
    vf = datetime.fromisoformat(valid_from) if valid_from else None
    vu = datetime.fromisoformat(valid_until) if valid_until else None
    lower_ok = vf is None or vf <= as_of
    upper_ok = vu is None or vu > as_of
    return lower_ok and upper_ok


# =====================================================================
# 1-2. valid_from source-of-truth rules (proven via temporary, fully
# cleaned-up synthetic memories -- never a permanent promotion)
# =====================================================================

def test_explicit_effective_from_used_as_valid_from():
    """Item 1 -- fc261a0a has a REAL, explicit effective_from ('2026-09-15').
    Proves the rule mechanically: when a caller supplies effective_from as
    valid_from, the RPC stores it verbatim -- never overridden by anything
    else. This temporary memory is fully deleted afterward; it does not
    constitute promoting fc261a0a (which remains a review candidate, per
    Part 11)."""
    ids = {}
    try:
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "process",
            "p_promotion_basis": "explicit_user_keep",
            "p_valid_from": "2026-09-15T00:00:00+00:00", "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": SK_Q4_LAUNCH_APPROVAL,
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("valid_from").eq("id", ids["memory_id"]).execute().data[0]
        assert row["valid_from"] == "2026-09-15T00:00:00+00:00"
    finally:
        _cleanup(ids)


def test_genuine_event_start_can_use_event_time():
    """Item 2 -- a synthetic primitive_type='event' row where event_start
    genuinely represents when the underlying occurrence begins (unlike the
    four real promoted policies/process, whose statements never establish
    a real-world 'begins at' moment -- see Part 1's audit)."""
    ids = {}
    try:
        event_start = "2026-10-01T09:00:00+00:00"
        ids["sk_a"] = _make_synthetic_sk("internal", "event-start", primitive_type="event",
                                         event_start=event_start, effective_from=None)
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "explicit_user_keep", "p_valid_from": event_start, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("valid_from").eq("id", ids["memory_id"]).execute().data[0]
        assert row["valid_from"] == event_start
    finally:
        _cleanup(ids)


def test_no_known_start_stores_null():
    """Item 3 -- the RPC must accept and store NULL valid_from without
    substituting anything."""
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("internal", "no-known-start")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": None, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("valid_from").eq("id", ids["memory_id"]).execute().data[0]
        assert row["valid_from"] is None
    finally:
        _cleanup(ids)


# =====================================================================
# 4-5. No fallback substitution on the REAL corrected memories
# =====================================================================

def test_captured_at_never_used_as_fallback_on_real_memories():
    rows = supabase.table("org_memory").select("id,valid_from").in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    captured_ats = set(REAL_CAPTURED_ATS.values())
    for r in rows:
        assert r["valid_from"] is None, "corrected real memories must have NULL valid_from, not a captured_at fallback"
        assert r["valid_from"] not in captured_ats


def test_consolidation_time_never_used_on_real_memories():
    run = supabase.table("memory_consolidation_runs").select("started_at").eq("id", REAL_RUN_ID).execute().data[0]
    rows = supabase.table("org_memory").select("valid_from").in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    for r in rows:
        assert r["valid_from"] is None
        assert r["valid_from"] != run["started_at"]


# =====================================================================
# 6-7. Current / historical read semantics
# =====================================================================

def test_current_query_treats_null_valid_from_as_no_lower_bound():
    now = datetime.now(timezone.utc)
    rows = supabase.table("org_memory").select("id,valid_from,valid_until") \
        .in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    for r in rows:
        assert _is_current(r["valid_from"], r["valid_until"], now), \
            "a NULL-valid_from memory must be considered CURRENT (no lower bound), never excluded"


def test_historical_query_treats_null_valid_from_as_no_lower_bound_too():
    """A memory whose real start is genuinely unknown cannot be excluded
    from ANY historical as_of on temporal grounds either -- NULL means 'no
    known lower bound', not 'starts now' (which would wrongly exclude it
    from queries about the past)."""
    past_as_of = datetime.now(timezone.utc) - timedelta(days=365)
    rows = supabase.table("org_memory").select("id,valid_from,valid_until") \
        .in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    for r in rows:
        assert _is_current(r["valid_from"], r["valid_until"], past_as_of)


# =====================================================================
# 8-9. Existing memory semantics preserved / correction confirmed
# =====================================================================

def test_existing_four_memories_semantics_preserved():
    rows = supabase.table("org_memory").select("*").in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    assert len(rows) == 4
    by_id = {r["id"]: r for r in rows}
    assert by_id[REAL_MEMORY_IDS["credential_logging"]]["promotion_basis"] == "authoritative_policy"
    assert by_id[REAL_MEMORY_IDS["credential_logging"]]["memory_type"] == "policy"
    assert by_id[REAL_MEMORY_IDS["monday_capacity"]]["promotion_basis"] == "recurring_durable_process"
    assert by_id[REAL_MEMORY_IDS["monday_capacity"]]["memory_type"] == "process"
    for r in rows:
        assert r["lifecycle_status"] == "active"
    # evidence unchanged
    for label, memory_id in REAL_MEMORY_IDS.items():
        ev = supabase.table("memory_evidence").select("evidence_id").eq("memory_id", memory_id).execute().data
        assert len(ev) == 1
        assert ev[0]["evidence_id"] == REAL_EVIDENCE_IDS[label]


def test_valid_from_correction_confirmed():
    rows = supabase.table("org_memory").select("id,valid_from").in_("id", list(REAL_MEMORY_IDS.values())).execute().data
    assert all(r["valid_from"] is None for r in rows), \
        "all four real memories must now have valid_from=NULL -- none of their source statements establish a real-world start"


# =====================================================================
# 10-11. Review candidate persistence + idempotency
# =====================================================================

def test_q4_review_candidate_persists_once():
    rows = supabase.table("memory_review_queue").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).eq("structured_knowledge_id", SK_Q4_LAUNCH_APPROVAL).execute().data
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["id"] == REAL_REVIEW_QUEUE_ID


def test_repeated_cycle_does_not_duplicate_pending_review_item():
    with pytest.raises(Exception):
        supabase.table("memory_review_queue").insert({
            "workspace_id": REAL_WORKSPACE, "structured_knowledge_id": SK_Q4_LAUNCH_APPROVAL,
            "reason": "duplicate attempt", "status": "pending",
        }).execute()


# =====================================================================
# 12. Review candidate remains outside org_memory
# =====================================================================

def test_review_candidate_not_referenced_by_any_memory_evidence():
    ev = supabase.table("memory_evidence").select("id") \
        .eq("evidence_id", SK_Q4_LAUNCH_APPROVAL).execute().data
    assert ev == [], "the Q4 launch approval requirement must remain unpromoted -- review only, per Part 11"


# =====================================================================
# 13-14. Security
# =====================================================================

def test_review_queue_workspace_isolation():
    wrong_ws = supabase.table("memory_review_queue").select("id") \
        .eq("id", REAL_REVIEW_QUEUE_ID).eq("workspace_id", OTHER_REAL_WORKSPACE).execute().data
    assert wrong_ws == []


def test_restricted_review_candidate_never_leaks_via_anon_client():
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("restricted", "review-restricted")
        ids["review_id"] = supabase.table("memory_review_queue").insert({
            "workspace_id": REAL_WORKSPACE, "structured_knowledge_id": ids["sk_a"],
            "reason": "TEST-6B1 restricted review candidate", "status": "pending",
        }).execute().data[0]["id"]

        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        assert url and anon_key
        anon = create_client(url, anon_key)
        result = anon.table("memory_review_queue").select("id").eq("id", ids["review_id"]).execute()
        assert result.data == [], "RLS with zero policies must block anon access entirely -- no exception for 'just a review candidate'"
    finally:
        _cleanup(ids)


# =====================================================================
# 15-17. Existing-data / graph / memory integrity
# =====================================================================

def test_workspace_structured_knowledge_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14


def test_hr_contact_row_cannot_leak_into_review_queue():
    hr_row = supabase.table("structured_knowledge").select("id") \
        .eq("statement", "88994448877 is the HR contact").execute().data
    assert hr_row
    referenced = supabase.table("memory_review_queue").select("id") \
        .eq("structured_knowledge_id", hr_row[0]["id"]).execute().data
    assert referenced == []


def test_graph_and_memory_row_counts_unchanged_except_temporal_correction():
    assert supabase.table("knowledge_entities").select("id", count="exact").execute().count == 5
    assert supabase.table("knowledge_relationships").select("id", count="exact") \
        .eq("status", "active").execute().count == 3
    assert supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count == 4
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count >= 1
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    # memory_consolidation_runs count intentionally NOT pinned to an exact
    # value anymore (Phase 6C fix): true only while Phase 6B's manual run
    # was the sole row ever created for this workspace. Phase 6C's real
    # engine legitimately adds more (its own real safety runs, and a daily
    # scheduled run going forward) -- see test_phase6b_memory_consolidation.
    # py's test_no_test_6b_consolidation_runs_leaked for the fuller
    # explanation of this exact change.
    assert supabase.table("memory_consolidation_runs").select("id", count="exact").execute().count >= 1


# =====================================================================
# 18. Fixture cleanup sentinel
# =====================================================================

def test_no_test_6b1_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-6B1%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_no_test_6b1_review_items_leaked():
    leaked = supabase.table("memory_review_queue").select("id,reason") \
        .like("reason", "TEST-6B1%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_exactly_one_review_queue_row_exists():
    count = supabase.table("memory_review_queue").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count
    assert count == 1


# =====================================================================
# Final state sentinel
# =====================================================================

def test_final_state_matches_expected():
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    # memory_consolidation_runs count intentionally NOT pinned (Phase 6C fix)
    # -- see test_graph_and_memory_row_counts_unchanged_except_temporal_
    # correction above.
    assert supabase.table("memory_consolidation_runs").select("id", count="exact").execute().count >= 1
    assert supabase.table("memory_review_queue").select("id", count="exact").execute().count == 1
