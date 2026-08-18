"""
Phase 6B First Real Memory Consolidation tests -- verifies the actual,
already-completed real consolidation run (memory_consolidation_runs id
36e9e367-41c6-4064-9fae-56f6830f7316) against the live 14-row workspace
corpus: 4 real promotions (2 distinct credential policies, 1 hardware-scope
policy, 1 recurring process), 1 review candidate (Q4 launch approval,
correctly NOT auto-promoted despite real graph connectivity), 9 correctly
rejected candidates.

This suite does NOT re-run consolidation against the real corpus (that
already happened, once, live, before this file was written) -- it verifies
the resulting state, re-exercises idempotency against the SAME real
evidence, and uses synthetic fixtures (fully cleaned) for edge cases the
real corpus doesn't cover (explicit_user_keep, cross_source_corroboration,
NULL-sensitivity rejection using a fresh synthetic row, atomic-failure
verification).

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase6b_memory_consolidation.py -v
"""
import uuid
from datetime import datetime, timezone

import pytest

from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

REAL_RUN_ID = "36e9e367-41c6-4064-9fae-56f6830f7316"

# Real, promoted evidence ids
SK_CREDENTIAL_LOGGING = "cd3ce795-121f-4760-bec3-2fece754427a"
SK_CREDENTIAL_SHARING = "d208e27c-0cc5-4270-a9cd-a4024aa94b12"
SK_HARDWARE_SCOPE = "c88b2636-6feb-4f4c-a411-ad6884823f6b"
SK_MONDAY_CAPACITY = "7db9647c-7b7d-4203-9f34-df1b1506cd8e"

# Real, correctly NOT promoted evidence ids
SK_Q4_LAUNCH_APPROVAL = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"   # review candidate
SK_KITCHEN_RULE = "179530ec-6a14-4b4a-b85e-b851456031e9"
SK_FIRMWARE_1 = "f8bae1bd-efb1-45fd-a76f-8bb5d354e1f6"
SK_FIRMWARE_2 = "da05c326-870f-429f-a40d-eb1795c8eede"
SK_ROADMAP = "3f539ac8-73fb-4ef7-b6df-425111e25d47"
SK_MEETING_1 = "9420ae8a-76f6-40b7-a4b3-0a94cad80921"
SK_MEETING_2 = "9d759f5d-4694-4d53-bb6d-48c37727815b"
SK_DRAFT_SUGGESTION = "1b513964-2cf8-4211-886c-faf9d42fdb3e"
SK_OPERATIONS_ALLOCATION_SIBLING = "5b77b2ca-2c8c-436c-8070-4f61bf5a270d"
SK_OFFICE_MAINTENANCE = "1361ac5e-300e-44d5-94f6-c9100565cba6"

REAL_PROMOTED_SK_IDS = {SK_CREDENTIAL_LOGGING, SK_CREDENTIAL_SHARING, SK_HARDWARE_SCOPE, SK_MONDAY_CAPACITY}
REAL_NOT_PROMOTED_SK_IDS = {
    SK_Q4_LAUNCH_APPROVAL, SK_KITCHEN_RULE, SK_FIRMWARE_1, SK_FIRMWARE_2, SK_ROADMAP,
    SK_MEETING_1, SK_MEETING_2, SK_DRAFT_SUGGESTION, SK_OPERATIONS_ALLOCATION_SIBLING, SK_OFFICE_MAINTENANCE,
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _memory_for_evidence(sk_id: str) -> list:
    """Every org_memory row (real or synthetic) that references this
    structured_knowledge id, via memory_evidence."""
    rows = supabase.table("memory_evidence").select("memory_id").eq("evidence_id", sk_id).execute().data
    return [r["memory_id"] for r in rows]


def _cleanup(ids: dict):
    ordered_keys = ("successor_id", "memory_id", "memory_a_id", "memory_b_id", "predecessor_id")
    memory_keys = [ids[k] for k in ordered_keys if ids.get(k)] + list(ids.get("memory_ids", []))
    for mid in memory_keys:
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in memory_keys:
        supabase.table("org_memory").delete().eq("id", mid).execute()
    if ids.get("run_id"):
        supabase.table("memory_consolidation_runs").delete().eq("id", ids["run_id"]).execute()
    sk_keys = list(ids.get("sk_ids", [])) + [
        ids[k] for k in ("sk_a", "sk_b", "null_sens_sk") if ids.get(k)
    ]
    for sk_id in sk_keys:
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_synthetic_sk(sensitivity, label_suffix) -> str:
    return supabase.table("structured_knowledge").insert({
        "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
        "canonical_id": SK_CREDENTIAL_LOGGING, "provider": "google_chat",
        "primitive_type": "fact", "statement": f"TEST-6B synthetic statement {label_suffix}",
        "raw_subject_phrase": "TEST-6B", "sensitivity": sensitivity, "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6b-{label_suffix}-{uuid.uuid4()}",
    }).execute().data[0]["id"]


# =====================================================================
# 1-3. Real auto-promotions
# =====================================================================

def test_real_authoritative_policy_promoted():
    """Item 1 -- credential-change-logging policy (cd3ce795)."""
    memory_ids = _memory_for_evidence(SK_CREDENTIAL_LOGGING)
    assert len(memory_ids) == 1
    row = supabase.table("org_memory").select("*").eq("id", memory_ids[0]).execute().data[0]
    assert row["memory_type"] == "policy"
    assert row["promotion_basis"] == "authoritative_policy"
    assert row["lifecycle_status"] == "active"
    assert row["sensitivity"] == "internal"


def test_real_recurring_process_promoted():
    """Item 2 -- Monday capacity submission (7db9647c)."""
    memory_ids = _memory_for_evidence(SK_MONDAY_CAPACITY)
    assert len(memory_ids) == 1
    row = supabase.table("org_memory").select("*").eq("id", memory_ids[0]).execute().data[0]
    assert row["memory_type"] == "process"
    assert row["promotion_basis"] == "recurring_durable_process"
    assert row["valid_until"] is None, "a recurring process with no stated end must have valid_until=NULL"


def test_real_hardware_scope_policy_promoted():
    """Item 3 -- hardware-categories-out-of-scope (c88b2636)."""
    memory_ids = _memory_for_evidence(SK_HARDWARE_SCOPE)
    assert len(memory_ids) == 1
    row = supabase.table("org_memory").select("*").eq("id", memory_ids[0]).execute().data[0]
    assert row["memory_type"] == "policy"
    assert row["promotion_basis"] == "authoritative_policy"


# =====================================================================
# 4-9 & meeting. Correctly NOT promoted
# =====================================================================

def test_q4_launch_approval_not_auto_promoted():
    """Item 4 -- the canonical review-not-promote case: official,
    graph-connected, future-effective, but process_step with no
    recurrence_text and no corroboration. Must have zero memory rows."""
    assert _memory_for_evidence(SK_Q4_LAUNCH_APPROVAL) == []


def test_kitchen_rule_not_promoted():
    assert _memory_for_evidence(SK_KITCHEN_RULE) == []


def test_firmware_target_not_promoted():
    assert _memory_for_evidence(SK_FIRMWARE_1) == []
    assert _memory_for_evidence(SK_FIRMWARE_2) == []


def test_roadmap_not_promoted():
    assert _memory_for_evidence(SK_ROADMAP) == []


def test_meeting_not_promoted():
    """Meeting primitives are evidence/activity context, never a memory
    object in their own right."""
    assert _memory_for_evidence(SK_MEETING_1) == []
    assert _memory_for_evidence(SK_MEETING_2) == []


def test_draft_suggestion_not_promoted():
    """authority='working', lifecycle_status='draft' -- fails every path
    at the authority check alone."""
    assert _memory_for_evidence(SK_DRAFT_SUGGESTION) == []


def test_operations_allocation_sibling_not_promoted():
    """Same canonical note as the promoted Monday-capacity row, but this
    specific primitive has recurrence_text=NULL on its own fields -- a
    sibling's recurrence tag is never borrowed."""
    assert _memory_for_evidence(SK_OPERATIONS_ALLOCATION_SIBLING) == []


def test_office_maintenance_not_promoted():
    assert _memory_for_evidence(SK_OFFICE_MAINTENANCE) == []


# =====================================================================
# 10. Duplicate credential-policy determination
# =====================================================================

def test_credential_policies_are_two_distinct_memories():
    """cd3ce795 ('credential CHANGES must be logged and reviewed') and
    d208e27c ('SHARING credentials in Slack is not permitted') are
    different actions governed by different rules -- confirmed as two
    independent org_memory rows, never merged, and never sharing a
    grounding_fingerprint."""
    logging_ids = _memory_for_evidence(SK_CREDENTIAL_LOGGING)
    sharing_ids = _memory_for_evidence(SK_CREDENTIAL_SHARING)
    assert len(logging_ids) == 1 and len(sharing_ids) == 1
    assert logging_ids[0] != sharing_ids[0]

    rows = supabase.table("org_memory").select("id,grounding_fingerprint") \
        .in_("id", [logging_ids[0], sharing_ids[0]]).execute().data
    fingerprints = {r["grounding_fingerprint"] for r in rows}
    assert len(fingerprints) == 2, "distinct claims must never share a grounding fingerprint"


# =====================================================================
# 11-12. Idempotency / distinct grounding
# =====================================================================

def test_reissuing_real_promotion_call_is_idempotent():
    """Item 11 -- re-calling create_memory_with_evidence with the EXACT
    same real params used for the original real promotion must return the
    SAME memory id, not create a duplicate.

    STALE VALUE (Phase 6B.1): this originally passed valid_from=event_time
    ("2026-08-15T15:47:37..."), matching Phase 6B's own original (later
    corrected) choice. Phase 6B.1 retroactively corrected all four real
    memories' valid_from to NULL, since none of their source statements
    establish a real-world effective start (event_time records only when
    the statement was MADE, not when the claim became true -- see the
    Phase 6B.1 report). Re-running this test with the OLD value against the
    corrected real row genuinely created a duplicate memory (found live,
    cleaned up by exact id) -- this is now updated to match the real,
    corrected logical key."""
    original_id = _memory_for_evidence(SK_CREDENTIAL_LOGGING)[0]
    res = supabase.rpc("create_memory_with_evidence", {
        "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
        "p_promotion_basis": "authoritative_policy",
        "p_valid_from": None, "p_valid_until": None,
        "p_supersedes_memory_id": None, "p_consolidation_run_id": REAL_RUN_ID,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": SK_CREDENTIAL_LOGGING,
                        "stance": "supports", "captured_at": "2026-08-16T06:55:29.694031+00:00"}],
    }).execute()
    assert res.data == original_id
    count = supabase.table("org_memory").select("id", count="exact") \
        .eq("id", original_id).execute().count
    assert count == 1


def test_different_synthetic_grounding_remains_distinct():
    """Item 12 -- reconfirms Phase 6A.1's Case B behavior against this
    pass's own promotion logic, with fresh synthetic fixtures."""
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("internal", "grounding-a")
        ids["sk_b"] = _make_synthetic_sk("internal", "grounding-b")
        valid_from = "2024-01-01T00:00:00+00:00"
        res_a = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": valid_from, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        res_b = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": valid_from, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_b"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_ids"] = [res_a.data, res_b.data]
        assert res_a.data != res_b.data
    finally:
        _cleanup(ids)


# =====================================================================
# 13. NULL sensitivity rejected
# =====================================================================

def test_null_sensitivity_candidate_rejected():
    """Uses the real Calendar-sourced Meeting primitive (genuinely NULL
    sensitivity) as an attempted promotion -- must reject, write nothing."""
    before_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    before_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count
    with pytest.raises(Exception):
        supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "explicit_user_keep", "p_valid_from": _now_iso(), "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": SK_MEETING_1,
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
    after_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    after_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count
    assert after_memory == before_memory
    assert after_evidence == before_evidence


# =====================================================================
# 14-15. Evidence atomicity + workspace isolation
# =====================================================================

def test_all_four_real_memories_have_exactly_one_evidence_row():
    for sk_id in REAL_PROMOTED_SK_IDS:
        memory_ids = _memory_for_evidence(sk_id)
        assert len(memory_ids) == 1
        ev = supabase.table("memory_evidence").select("id", count="exact") \
            .eq("memory_id", memory_ids[0]).execute().count
        assert ev == 1


def test_all_real_memories_workspace_scoped():
    all_ids = set()
    for sk_id in REAL_PROMOTED_SK_IDS:
        all_ids.update(_memory_for_evidence(sk_id))
    rows = supabase.table("org_memory").select("id,workspace_id").in_("id", list(all_ids)).execute().data
    assert len(rows) == 4
    assert all(r["workspace_id"] == REAL_WORKSPACE for r in rows)
    ev_rows = supabase.table("memory_evidence").select("workspace_id").in_("memory_id", list(all_ids)).execute().data
    assert all(r["workspace_id"] == REAL_WORKSPACE for r in ev_rows)


# =====================================================================
# 16. No global 15th-row leak
# =====================================================================

def test_hr_contact_row_never_referenced_by_any_memory():
    hr_row = supabase.table("structured_knowledge").select("id") \
        .eq("statement", "88994448877 is the HR contact").execute().data
    assert hr_row
    referenced = supabase.table("memory_evidence").select("id") \
        .eq("evidence_id", hr_row[0]["id"]).execute().data
    assert referenced == []


# =====================================================================
# 17. Run stats accurate
# =====================================================================

def test_real_run_stats_accurate():
    run = supabase.table("memory_consolidation_runs").select("*").eq("id", REAL_RUN_ID).execute().data[0]
    assert run["workspace_id"] == REAL_WORKSPACE
    assert run["status"] == "completed"
    assert run["completed_at"] is not None
    stats = run["stats"]
    assert stats["evaluated"] == 14
    assert stats["eligible"] == 4
    assert stats["promoted"] == 4
    assert stats["review_candidates"] == 1
    assert stats["rejected"] == 9
    assert stats["already_promoted"] == 0
    assert stats["contradiction_flagged"] == 0
    assert stats["superseded"] == 0
    assert stats["failed"] == 0
    assert SK_Q4_LAUNCH_APPROVAL in stats["review_candidate_ids"]
    # internal consistency: every row accounted for exactly once
    assert stats["promoted"] + stats["review_candidates"] + stats["rejected"] == stats["evaluated"]


def test_original_real_consolidation_run_still_exists():
    """STALE ASSERTION FIXED (Phase 6C): this used to assert count==1 for
    this workspace's memory_consolidation_runs -- true only while Phase 6B's
    manual run was the sole row ever created. Phase 6C added a real,
    deterministic consolidation engine that legitimately creates additional
    rows for this same workspace (its own real safety runs, and a daily
    scheduled run going forward) -- an exact total count is no longer a
    stable invariant this suite can assert. What still matters, and is
    checked here: the ORIGINAL real Phase 6B run (REAL_RUN_ID) still exists,
    unchanged, never deleted or replaced by a later run."""
    rows = supabase.table("memory_consolidation_runs").select("id, status") \
        .eq("id", REAL_RUN_ID).eq("workspace_id", REAL_WORKSPACE).execute().data
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


# =====================================================================
# 18. Repeat run creates zero duplicates
# =====================================================================

def test_repeating_all_four_real_promotions_is_a_full_no_op():
    """STALE VALUES (Phase 6B.1): each valid_from below originally used
    event_time, matching Phase 6B's own original (later corrected) choice.
    Phase 6B.1 retroactively corrected all four real memories' valid_from
    to NULL -- updated here to match, otherwise this test would itself
    create four duplicate memories against the corrected real rows (found
    live during this exact correction, cleaned up by exact id)."""
    before_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    before_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count

    calls = [
        ("policy", "authoritative_policy", None, SK_CREDENTIAL_LOGGING, "2026-08-16T06:55:29.694031+00:00"),
        ("policy", "authoritative_policy", None, SK_CREDENTIAL_SHARING, "2026-08-16T06:55:29.694031+00:00"),
        ("policy", "authoritative_policy", None, SK_HARDWARE_SCOPE, "2026-08-16T06:55:44.281195+00:00"),
        ("process", "recurring_durable_process", None, SK_MONDAY_CAPACITY, "2026-08-16T06:55:34.145793+00:00"),
    ]
    for memory_type, basis, valid_from, sk_id, captured_at in calls:
        supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": memory_type,
            "p_promotion_basis": basis, "p_valid_from": valid_from, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": REAL_RUN_ID,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk_id,
                            "stance": "supports", "captured_at": captured_at}],
        }).execute()

    after_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    after_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count
    assert after_memory == before_memory
    assert after_evidence == before_evidence


# =====================================================================
# 19-21. Existing-data integrity
# =====================================================================

def test_structured_knowledge_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14


def test_graph_unchanged():
    assert supabase.table("knowledge_entities").select("id", count="exact").execute().count == 5
    assert supabase.table("knowledge_relationships").select("id", count="exact") \
        .eq("status", "active").execute().count == 3
    assert supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count == 4


def test_calendar_snapshot_unchanged():
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count >= 1


# =====================================================================
# 22. Failure does not create evidence-less memory
# =====================================================================

def test_failed_candidate_creates_no_evidence_less_memory():
    before_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("internal", "fail-check")
        # Intentionally malformed: evidence_id doesn't exist.
        with pytest.raises(Exception):
            supabase.rpc("create_memory_with_evidence", {
                "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
                "p_promotion_basis": "authoritative_policy", "p_valid_from": _now_iso(), "p_valid_until": None,
                "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
                "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": str(uuid.uuid4()),
                                "stance": "supports", "captured_at": _now_iso()}],
            }).execute()
        after_memory = supabase.table("org_memory").select("id", count="exact").execute().count
        assert after_memory == before_memory
        # an unrelated, valid promotion afterward must still succeed --
        # one failed candidate never poisons the write path for others.
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2024-02-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        assert res.data is not None
    finally:
        _cleanup(ids)


# =====================================================================
# 23-24. Synthetic explicit_user_keep / cross_source_corroboration paths
# =====================================================================

def test_synthetic_explicit_user_keep_path():
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("public", "explicit-keep")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "explicit_user_keep", "p_valid_from": "2024-03-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("promotion_basis,memory_type").eq("id", ids["memory_id"]).execute().data[0]
        assert row["promotion_basis"] == "explicit_user_keep"
        assert row["memory_type"] == "decision"
    finally:
        _cleanup(ids)


def test_synthetic_cross_source_corroboration_path():
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("internal", "corroboration-a")
        ids["sk_b"] = _make_synthetic_sk("internal", "corroboration-b")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "cross_source_corroboration", "p_valid_from": "2024-04-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [
                {"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                 "stance": "supports", "captured_at": _now_iso()},
                {"evidence_type": "structured_knowledge", "evidence_id": ids["sk_b"],
                 "stance": "supports", "captured_at": _now_iso()},
            ],
        }).execute()
        ids["memory_id"] = res.data
        ev = supabase.table("memory_evidence").select("id", count="exact") \
            .eq("memory_id", ids["memory_id"]).execute().count
        assert ev == 2, "corroboration is the one basis that legitimately grounds a memory in multiple structured_knowledge rows"
    finally:
        _cleanup(ids)


# =====================================================================
# 25. No centrality-only promotion
# =====================================================================

def test_no_centrality_only_promotion_for_graph_connected_row():
    """The real Q4 launch approval requirement is the graph's single most-
    connected structured_knowledge row (it anchors the only structured_
    knowledge-sourced relationship in the whole graph) -- confirms it is
    STILL not promoted, proving graph centrality alone never triggers
    automatic promotion, exactly as frozen."""
    assert _memory_for_evidence(SK_Q4_LAUNCH_APPROVAL) == []
    # sanity: confirm it really IS graph-connected, so this test means what it claims
    rel = supabase.table("knowledge_relationships").select("id") \
        .eq("source_object_id", SK_Q4_LAUNCH_APPROVAL).execute().data
    assert rel, "sanity check: this row must actually be graph-connected for the test to be meaningful"


# =====================================================================
# 26. Synthetic contradiction / review-event representation
# =====================================================================

def test_no_automatic_contradiction_detection_exists_yet():
    """Two synthetic memories about an unrelated topic, both independently
    valid and promoted, must simply coexist -- no automatic contradiction
    detection/rejection exists in this pass (that is 6C's job). This
    confirms the write path doesn't silently invent conflict logic beyond
    what Phase 6A.1's Case C (same grounding, different promotion_basis)
    already checks."""
    ids = {}
    try:
        ids["sk_a"] = _make_synthetic_sk("internal", "topic-x")
        ids["sk_b"] = _make_synthetic_sk("internal", "topic-y")
        res_a = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "explicit_user_keep", "p_valid_from": "2024-05-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_a"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        res_b = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "decision",
            "p_promotion_basis": "explicit_user_keep", "p_valid_from": "2024-05-02T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["sk_b"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_ids"] = [res_a.data, res_b.data]
        rows = supabase.table("org_memory").select("lifecycle_status").in_("id", ids["memory_ids"]).execute().data
        assert all(r["lifecycle_status"] == "active" for r in rows)
    finally:
        _cleanup(ids)


def test_consolidation_run_stats_can_represent_contradiction_flagged():
    """The real run correctly recorded contradiction_flagged=0 (Part 11 --
    do not invent contradictions in a corpus that has none). This confirms
    the stats SHAPE itself can represent a nonzero value when a future
    (6C) run's real contradiction-detection logic needs to report one --
    a synthetic run row only, never touching the real run."""
    ids = {}
    try:
        run = supabase.table("memory_consolidation_runs").insert({
            "workspace_id": REAL_WORKSPACE, "started_at": _now_iso(), "status": "completed",
            "completed_at": _now_iso(),
            "stats": {"evaluated": 2, "eligible": 2, "promoted": 1, "review_candidates": 0,
                     "rejected": 0, "already_promoted": 0, "contradiction_flagged": 1,
                     "superseded": 1, "failed": 0},
        }).execute().data[0]
        ids["run_id"] = run["id"]
        assert run["stats"]["contradiction_flagged"] == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 27. Fixture cleanup sentinel
# =====================================================================

def test_no_test_6b_entities_or_structured_knowledge_leaked():
    leaked_sk = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-6B%").execute().data
    assert leaked_sk == [], f"fixture cleanup failed, leaked rows: {leaked_sk}"


def test_no_test_6b_consolidation_runs_leaked():
    """STALE ASSERTION FIXED (Phase 6C): count==1 was only ever true because
    Phase 6B's manual run was the sole row ever created for this workspace.
    Phase 6C's real engine legitimately adds more (its own real safety runs,
    and a daily scheduled run going forward), so total count is no longer a
    leak signal this suite can use on its own. Fixture-level cleanup (see
    _cleanup()'s ids["run_id"] handling above) is what actually prevents a
    leaked SYNTHETIC run from this suite's own fixtures -- this test now
    checks the thing that's still a real invariant: every run row for this
    workspace has a schema-valid status (nothing left mid-write / corrupted
    by a failed test)."""
    rows = supabase.table("memory_consolidation_runs").select("id, status") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert len(rows) >= 1
    for r in rows:
        assert r["status"] in ("running", "completed", "failed")


# =====================================================================
# Final state sentinel (the real full pytest run across every
# test_phase*.py file remains the authoritative regression gate)
# =====================================================================

def test_final_memory_state_matches_expected():
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    # memory_consolidation_runs count intentionally NOT pinned to an exact
    # value anymore -- see test_no_test_6b_consolidation_runs_leaked's
    # docstring (Phase 6C made this table a legitimately growing audit log,
    # not a fixed-count table).
    assert supabase.table("memory_consolidation_runs").select("id", count="exact").execute().count >= 1
