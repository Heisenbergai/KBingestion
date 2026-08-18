"""
Phase 6C Sleep / Memory Consolidation Engine tests.

Two kinds of fixtures, per this codebase's established convention:
  - REAL data (REAL_WORKSPACE's frozen 14-row corpus, the 4 real Phase 6B
    memories, the real Q4 review candidate) for the no-op/idempotency/
    revalidation/leak-isolation tests that must be proven against production
    state, not a synthetic stand-in.
  - A brand-new, single-use synthetic workspace_id per isolated test
    (`_fresh_workspace()`) for promotion/rejection/contradiction/failure/
    concurrency edge cases. workspace_id has no cross-DB FK to the app DB's
    real `workspaces` table (confirmed impossible -- see drive_app_db.py's
    module docstring on why this service never joins across projects), so a
    random UUID behaves identically to a real one for every table this
    engine touches, with zero risk of colliding with real data and zero
    pollution of the real audit log beyond this suite's own, fully cleaned,
    fixture rows.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Item 30 of the required matrix ("full regression") is NOT a single pytest
function in this file -- exactly like every prior phase's own "Full
regression" report section, it is the separate, real, sequential run of
every test_phase*.py file together (see the Phase 6C final report's
"Full regression" section). test_placeholder_full_regression_runs_separately
below documents this rather than silently dropping the numbered item.

Run with: python -m pytest test_phase6c_sleep_cycle.py -v
"""
import uuid
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from query import supabase
import memory_consolidation as mc

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"   # the real 15th-row workspace
LEAK_SK_ID = "8715f3e7-1d08-4a87-b326-a95375be8e73"

REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "credential_sharing":  "3d376631-894c-4e32-b3f5-3ecf7cfd5f61",
    "hardware_scope":      "8aef76c9-fda3-44d6-affb-769f2ff09326",
    "monday_capacity":     "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}
SK_Q4_LAUNCH_APPROVAL = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _fresh_workspace() -> str:
    return str(uuid.uuid4())


def _cleanup(ids: dict) -> None:
    for mid in ids.get("memory_ids", []):
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    # Reverse insertion order: a superseding memory (appended after its
    # predecessor) carries supersedes_memory_id -> predecessor, an
    # ON DELETE RESTRICT FK -- the successor must be deleted first (the
    # Phase 6A.1 lesson, repeated here).
    for mid in reversed(ids.get("memory_ids", [])):
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for rid in ids.get("review_ids", []):
        supabase.table("memory_review_queue").delete().eq("id", rid).execute()
    for run_id in ids.get("run_ids", []):
        supabase.table("memory_consolidation_runs").delete().eq("id", run_id).execute()
    for rel_id in ids.get("relationship_ids", []):
        supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", rel_id).execute()
        supabase.table("knowledge_relationships").delete().eq("id", rel_id).execute()
    for ent_id in ids.get("entity_ids", []):
        supabase.table("knowledge_entities").delete().eq("id", ent_id).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()
    # Sweep for any fully-synthetic single-use workspace, in case a step
    # above threw before its id was recorded.
    for ws in ids.get("workspace_ids", []):
        mem_ids = [r["id"] for r in supabase.table("org_memory").select("id")
                   .eq("workspace_id", ws).execute().data or []]
        for mid in mem_ids:
            supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
        supabase.table("org_memory").delete().eq("workspace_id", ws).execute()
        supabase.table("memory_review_queue").delete().eq("workspace_id", ws).execute()
        supabase.table("memory_consolidation_runs").delete().eq("workspace_id", ws).execute()
        rel_ids = [r["id"] for r in supabase.table("knowledge_relationships").select("id")
                   .eq("workspace_id", ws).execute().data or []]
        for rid in rel_ids:
            supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", rid).execute()
        supabase.table("knowledge_relationships").delete().eq("workspace_id", ws).execute()
        supabase.table("knowledge_entities").delete().eq("workspace_id", ws).execute()
        supabase.table("structured_knowledge").delete().eq("workspace_id", ws).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-6C synthetic statement",
        "raw_subject_phrase": "TEST-6C subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6c-{uuid.uuid4()}",
    }
    row.update(overrides)
    return supabase.table("structured_knowledge").insert(row).execute().data[0]["id"]


def _make_entity(workspace_id: str, label: str) -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": "person",
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_relationship(workspace_id: str, sk_id: str, entity_id: str) -> str:
    """Graph-connects sk_id, mirroring the real Product<-requires_approval_from
    precedent -- the exact deterministic signal classify_candidate() uses to
    escalate a failed-promotion candidate to REVIEW instead of REJECT."""
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": "structured_knowledge", "p_source_object_id": sk_id,
        "p_target_object_type": "entity", "p_target_object_id": entity_id,
        "p_relationship_type": "requires_approval_from", "p_rationale": "TEST-6C",
        "p_confidence": 0.9, "p_valid_from": _now_iso(), "p_valid_until": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }).execute().data


def _is_current(valid_from, valid_until, as_of: datetime) -> bool:
    vf = datetime.fromisoformat(valid_from) if valid_from else None
    vu = datetime.fromisoformat(valid_until) if valid_until else None
    return (vf is None or vf <= as_of) and (vu is None or vu > as_of)


# =====================================================================
# 1. Unchanged run is idempotent (real corpus)
# =====================================================================

def test_unchanged_run_is_idempotent():
    before_memories = supabase.table("org_memory").select("id", count="exact").execute().count
    before_review = supabase.table("memory_review_queue").select("id", count="exact").eq("status", "pending").execute().count

    result_1 = mc.run_consolidation(REAL_WORKSPACE)
    result_2 = mc.run_consolidation(REAL_WORKSPACE)

    assert result_1["status"] == "completed"
    assert result_2["status"] == "completed"
    for result in (result_1, result_2):
        s = result["stats"]
        assert s["promoted"] == 0
        assert s["review_candidates"] == 0
        assert s["superseded"] == 0
        assert s["contradiction_flagged"] == 0
        assert s["failed"] == 0

    after_memories = supabase.table("org_memory").select("id", count="exact").execute().count
    after_review = supabase.table("memory_review_queue").select("id", count="exact").eq("status", "pending").execute().count
    assert after_memories == before_memories == 4
    assert after_review == before_review == 1


# =====================================================================
# 2-5. Promotion / rejection classification
# =====================================================================

def test_new_authoritative_policy_promotes():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws, requirement_kind="policy", authority="official",
                                       statement="TEST-6C all deploys require sign-off"))
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 1
        rows = supabase.table("org_memory").select("id, promotion_basis, memory_type") \
            .eq("workspace_id", ws).execute().data
        assert len(rows) == 1
        ids["memory_ids"].append(rows[0]["id"])
        assert rows[0]["promotion_basis"] == "authoritative_policy"
        assert rows[0]["memory_type"] == "policy"
    finally:
        _cleanup(ids)


def test_new_recurring_process_promotes():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws, requirement_kind="process_step", recurrence_text="every Monday",
                                       statement="TEST-6C submit the weekly report"))
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 1
        rows = supabase.table("org_memory").select("id, promotion_basis, memory_type") \
            .eq("workspace_id", ws).execute().data
        assert len(rows) == 1
        ids["memory_ids"].append(rows[0]["id"])
        assert rows[0]["promotion_basis"] == "recurring_durable_process"
        assert rows[0]["memory_type"] == "process"
    finally:
        _cleanup(ids)


def test_kitchen_like_process_does_not_promote():
    """Mirrors the real 179530ec ('keep the kitchen area clear') shape --
    process_step with no recurrence_text fails recurring_durable_process."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws, requirement_kind="process_step", recurrence_text=None,
                                       statement="TEST-6C keep the kitchen area clear"))
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 0
        assert result["stats"]["rejected"] == 1
        assert supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws).execute().count == 0
    finally:
        _cleanup(ids)


def test_new_non_durable_fact_rejected():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws, primitive_type="fact", requirement_kind=None,
                                       statement="TEST-6C the release target is September 12"))
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["rejected"] == 1
        assert result["stats"]["promoted"] == 0
    finally:
        _cleanup(ids)


# =====================================================================
# 6-7. Review queue integration
# =====================================================================

def test_review_candidate_created_once():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "run_ids": [], "review_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="process_step", recurrence_text=None,
                          statement="TEST-6C Q4 launch requires Product approval")
        ids["sk_ids"].append(sk_id)
        entity_id = _make_entity(ws, "TEST-6C Product")
        ids["entity_ids"].append(entity_id)
        ids["relationship_ids"].append(_make_relationship(ws, sk_id, entity_id))

        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        rows = supabase.table("memory_review_queue").select("id, status") \
            .eq("workspace_id", ws).execute().data
        ids["review_ids"].extend(r["id"] for r in rows)
        assert result["stats"]["promoted"] == 0
        assert result["stats"]["review_candidates"] == 1
        assert len(rows) == 1
        assert rows[0]["status"] == "pending"
    finally:
        _cleanup(ids)


def test_duplicate_review_avoided():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "run_ids": [], "review_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="process_step", recurrence_text=None,
                          statement="TEST-6C Q4 launch requires Product approval")
        ids["sk_ids"].append(sk_id)
        entity_id = _make_entity(ws, "TEST-6C Product")
        ids["entity_ids"].append(entity_id)
        ids["relationship_ids"].append(_make_relationship(ws, sk_id, entity_id))

        result_1 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_1["run_id"])
        result_2 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_2["run_id"])

        review_rows = supabase.table("memory_review_queue").select("id").eq("workspace_id", ws).execute().data
        ids["review_ids"].extend(r["id"] for r in review_rows)

        assert result_1["stats"]["review_candidates"] == 1
        assert result_2["stats"]["review_candidates"] == 0
        rows = supabase.table("memory_review_queue").select("id", count="exact") \
            .eq("workspace_id", ws).execute().count
        assert rows == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 8-9. Grounding dedup
# =====================================================================

def test_same_grounding_does_not_duplicate_memory():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6C production changes require review")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 1
        row = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(row["id"])

        candidate = supabase.table("structured_knowledge").select("*").eq("id", sk_id).execute().data[0]
        bucket, detail = mc.classify_candidate(ws, candidate)
        assert bucket == "already_durable"
        assert detail["id"] == row["id"]
    finally:
        _cleanup(ids)


def test_different_grounding_remains_separate():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws, requirement_kind="policy", authority="official",
                                       raw_subject_phrase="TEST-6C policy A subject",
                                       statement="TEST-6C statement A about policy A"))
        ids["sk_ids"].append(_make_sk(ws, requirement_kind="policy", authority="official",
                                       raw_subject_phrase="TEST-6C policy B subject",
                                       statement="TEST-6C statement B about policy B"))
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 2
        rows = supabase.table("org_memory").select("id, grounding_fingerprint").eq("workspace_id", ws).execute().data
        assert len(rows) == 2
        ids["memory_ids"].extend(r["id"] for r in rows)
        assert rows[0]["grounding_fingerprint"] != rows[1]["grounding_fingerprint"]
    finally:
        _cleanup(ids)


# =====================================================================
# 10. Explicit user keep -- extension-point capability, not a live signal
# =====================================================================

def test_explicit_user_keep_path():
    """Part 7: no real upstream signal exists yet -- this proves the engine
    is CAPABLE of consuming one the moment it's wired in, by injecting the
    future signal exactly where a real UI/service contract would."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, primitive_type="fact", requirement_kind=None,
                          statement="TEST-6C a human explicitly said keep this")
        ids["sk_ids"].append(sk_id)
        with patch("memory_consolidation._check_explicit_user_keep", return_value=True):
            result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["promoted"] == 1
        row = supabase.table("org_memory").select("id, promotion_basis, memory_type") \
            .eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(row["id"])
        assert row["promotion_basis"] == "explicit_user_keep"
        assert row["memory_type"] == "decision"
    finally:
        _cleanup(ids)


# =====================================================================
# 11-12. Contradiction detection + supersession
# =====================================================================

def test_contradiction_unresolved_routes_to_review():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": [], "review_ids": []}
    try:
        sk_a = _make_sk(ws, requirement_kind="policy", authority="official",
                         raw_subject_phrase="TEST-6C widget policy",
                         statement="TEST-6C widgets must be shipped in blue boxes")
        ids["sk_ids"].append(sk_a)
        result_1 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_1["run_id"])
        assert result_1["stats"]["promoted"] == 1
        memory_a = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]["id"]
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, requirement_kind="policy", authority="official",
                         raw_subject_phrase="TEST-6C widget policy",
                         statement="TEST-6C widgets must be shipped in red boxes")
        ids["sk_ids"].append(sk_b)
        with patch("ai.chat_json", return_value={"verdict": "unresolved", "rationale": "ambiguous"}):
            result_2 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_2["run_id"])

        assert result_2["stats"]["promoted"] == 0
        assert result_2["stats"]["contradiction_flagged"] == 1
        # memory A untouched
        row_a = supabase.table("org_memory").select("lifecycle_status").eq("id", memory_a).execute().data[0]
        assert row_a["lifecycle_status"] == "active"
        assert supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws).execute().count == 1
        review_rows = supabase.table("memory_review_queue").select("id").eq("workspace_id", ws).execute().data
        assert len(review_rows) == 1
        ids["review_ids"].extend(r["id"] for r in review_rows)
    finally:
        _cleanup(ids)


def test_authoritative_later_replacement_creates_supersession():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_a = _make_sk(ws, requirement_kind="policy", authority="official",
                         raw_subject_phrase="TEST-6C vacation policy",
                         statement="TEST-6C vacation requests need 2 weeks notice")
        ids["sk_ids"].append(sk_a)
        result_1 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_1["run_id"])
        memory_a = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]["id"]
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, requirement_kind="policy", authority="official",
                         raw_subject_phrase="TEST-6C vacation policy",
                         statement="TEST-6C vacation requests now need only 1 week notice, effective immediately, "
                                    "superseding the prior 2-week policy")
        ids["sk_ids"].append(sk_b)
        with patch("ai.chat_json", return_value={"verdict": "resolved_supersession", "rationale": "explicit replacement"}):
            result_2 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_2["run_id"])

        assert result_2["stats"]["promoted"] == 1
        assert result_2["stats"]["superseded"] == 1
        memory_b = supabase.table("org_memory").select("id, supersedes_memory_id, lifecycle_status") \
            .eq("workspace_id", ws).eq("lifecycle_status", "active").execute().data[0]
        ids["memory_ids"].append(memory_b["id"])
        assert memory_b["supersedes_memory_id"] == memory_a

        row_a = supabase.table("org_memory").select("lifecycle_status").eq("id", memory_a).execute().data[0]
        assert row_a["lifecycle_status"] == "superseded"
        # both preserved, never deleted
        assert supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws).execute().count == 2
        assert supabase.table("memory_evidence").select("id", count="exact").eq("memory_id", memory_a).execute().count == 1
        assert supabase.table("memory_evidence").select("id", count="exact").eq("memory_id", memory_b["id"]).execute().count == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 13-15. Read-only over source evidence / structured_knowledge / graph
# =====================================================================

def test_source_evidence_never_mutated():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6C evidence must not be mutated")
        ids["sk_ids"].append(sk_id)
        before = supabase.table("structured_knowledge").select("*").eq("id", sk_id).execute().data[0]

        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        row = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(row["id"])

        after = supabase.table("structured_knowledge").select("*").eq("id", sk_id).execute().data[0]
        assert before == after
    finally:
        _cleanup(ids)


def test_structured_knowledge_never_mutated_globally():
    before = {r["id"]: r["updated_at"] for r in
              supabase.table("structured_knowledge").select("id, updated_at")
              .eq("workspace_id", REAL_WORKSPACE).execute().data}
    before_count = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count

    mc.run_consolidation(REAL_WORKSPACE)

    after = {r["id"]: r["updated_at"] for r in
             supabase.table("structured_knowledge").select("id, updated_at")
             .eq("workspace_id", REAL_WORKSPACE).execute().data}
    after_count = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count

    assert before_count == after_count == 14
    assert before == after


def test_graph_relationships_never_mutated():
    before = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    mc.run_consolidation(REAL_WORKSPACE)
    after = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    assert before == after
    assert len(after) == 3


# =====================================================================
# 16-17. Temporal semantics
# =====================================================================

def test_null_valid_from_remains_valid():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official", effective_from=None,
                          statement="TEST-6C no known real-world start")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        row = supabase.table("org_memory").select("id, valid_from").eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(row["id"])
        assert row["valid_from"] is None

        candidate = supabase.table("structured_knowledge").select("*").eq("id", sk_id).execute().data[0]
        bucket, _ = mc.classify_candidate(ws, candidate)
        assert bucket == "already_durable"
    finally:
        _cleanup(ids)


def test_historical_semantics_remain_correct():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official", effective_from=None,
                          statement="TEST-6C historical semantics check")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        row = supabase.table("org_memory").select("id, valid_from, valid_until") \
            .eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(row["id"])

        assert _is_current(row["valid_from"], row["valid_until"], _now())
        assert _is_current(row["valid_from"], row["valid_until"], _now() - timedelta(days=365))
    finally:
        _cleanup(ids)


# =====================================================================
# 18. Sensitivity NULL rejected
# =====================================================================

def test_sensitivity_null_rejected():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official", sensitivity=None,
                          statement="TEST-6C NULL sensitivity must reject")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        assert result["status"] == "completed"
        assert result["stats"]["rejected"] == 1
        assert result["stats"]["promoted"] == 0
        assert result["stats"]["failed"] == 0
        assert supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws).execute().count == 0
    finally:
        _cleanup(ids)


# =====================================================================
# 19-20. Workspace isolation
# =====================================================================

def test_workspace_isolation():
    ws_a = _fresh_workspace()
    ws_b = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        ids["sk_ids"].append(_make_sk(ws_a, requirement_kind="policy", authority="official",
                                       statement="TEST-6C workspace A policy"))
        ids["sk_ids"].append(_make_sk(ws_b, requirement_kind="policy", authority="official",
                                       statement="TEST-6C workspace B policy"))

        result = mc.run_consolidation(ws_a)
        ids["run_ids"].append(result["run_id"])
        assert result["stats"]["evaluated"] == 1
        assert result["stats"]["promoted"] == 1

        row_a = supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws_a).execute().count
        row_b = supabase.table("org_memory").select("id", count="exact").eq("workspace_id", ws_b).execute().count
        assert row_a == 1
        assert row_b == 0

        mem_a = supabase.table("org_memory").select("id").eq("workspace_id", ws_a).execute().data[0]["id"]
        ids["memory_ids"].append(mem_a)

        b_runs = supabase.table("memory_consolidation_runs").select("id", count="exact") \
            .eq("workspace_id", ws_b).execute().count
        assert b_runs == 0
    finally:
        _cleanup(ids)


def test_global_15th_row_cannot_leak():
    far_future = (_now() + timedelta(days=3650)).isoformat()
    real_ws_candidates = mc._fetch_candidates(REAL_WORKSPACE, None, far_future)
    assert LEAK_SK_ID not in {c["id"] for c in real_ws_candidates}

    leak_ws_candidates = mc._fetch_candidates(LEAK_WORKSPACE, None, far_future)
    assert {c["id"] for c in leak_ws_candidates} == {LEAK_SK_ID}


# =====================================================================
# 21-23. Failure / retry
# =====================================================================

def test_partial_candidate_failure_does_not_create_orphan_memory():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        good_id = _make_sk(ws, requirement_kind="policy", authority="official",
                            raw_subject_phrase="TEST-6C good candidate subject",
                            statement="TEST-6C the good candidate")
        bad_id = _make_sk(ws, requirement_kind="policy", authority="official",
                           raw_subject_phrase="TEST-6C bad candidate subject",
                           statement="TEST-6C the bad candidate")
        ids["sk_ids"].extend([good_id, bad_id])

        real_promote = mc._promote

        def _flaky_promote(workspace_id, run_id, candidate, basis, supersedes_id):
            if candidate["id"] == bad_id:
                raise RuntimeError("TEST-6C injected failure")
            return real_promote(workspace_id, run_id, candidate, basis, supersedes_id)

        with patch("memory_consolidation._promote", side_effect=_flaky_promote):
            result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])

        assert result["status"] == "failed"
        assert result["stats"]["failed"] == 1
        assert result["stats"]["promoted"] == 1

        good_memory = supabase.table("org_memory").select("id") \
            .eq("workspace_id", ws).execute().data
        assert len(good_memory) == 1
        ids["memory_ids"].append(good_memory[0]["id"])

        bad_evidence = supabase.table("memory_evidence").select("id").eq("evidence_id", bad_id).execute().data
        assert bad_evidence == [], "the failed candidate must never leave an orphaned evidence/memory row"
    finally:
        _cleanup(ids)


def test_failed_run_is_marked_failed():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "run_ids": []}
    try:
        bad_id = _make_sk(ws, requirement_kind="policy", authority="official",
                           statement="TEST-6C always fails")
        ids["sk_ids"].append(bad_id)

        def _always_raise(*args, **kwargs):
            raise RuntimeError("TEST-6C injected failure")

        with patch("memory_consolidation._promote", side_effect=_always_raise):
            result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])

        assert result["status"] == "failed"
        db_row = supabase.table("memory_consolidation_runs").select("status, stats") \
            .eq("id", result["run_id"]).execute().data[0]
        assert db_row["status"] == "failed"
        assert db_row["stats"]["failed"] == 1
    finally:
        _cleanup(ids)


def test_retry_succeeds():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        good_id = _make_sk(ws, requirement_kind="policy", authority="official",
                            raw_subject_phrase="TEST-6C retry good subject",
                            statement="TEST-6C retry good candidate")
        bad_id = _make_sk(ws, requirement_kind="policy", authority="official",
                           raw_subject_phrase="TEST-6C retry bad subject",
                           statement="TEST-6C retry bad candidate")
        ids["sk_ids"].extend([good_id, bad_id])

        real_promote = mc._promote

        def _flaky_once(workspace_id, run_id, candidate, basis, supersedes_id):
            if candidate["id"] == bad_id:
                raise RuntimeError("TEST-6C injected failure (first attempt only)")
            return real_promote(workspace_id, run_id, candidate, basis, supersedes_id)

        with patch("memory_consolidation._promote", side_effect=_flaky_once):
            first = mc.run_consolidation(ws)
        ids["run_ids"].append(first["run_id"])
        assert first["status"] == "failed"

        # Retry: boundary was never advanced past the failed run, so both
        # candidates are in scope again -- the good one resolves via
        # ALREADY_DURABLE, the bad one gets a real chance to succeed.
        second = mc.run_consolidation(ws)
        ids["run_ids"].append(second["run_id"])
        assert second["status"] == "completed"
        assert second["stats"]["already_durable"] == 1
        assert second["stats"]["promoted"] == 1

        rows = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data
        ids["memory_ids"].extend(r["id"] for r in rows)
        assert len(rows) == 2, "no duplicates from the good candidate, and the bad one now has exactly one memory"
    finally:
        _cleanup(ids)


# =====================================================================
# 24. Concurrency
# =====================================================================

def test_concurrent_promotion_does_not_duplicate():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6C concurrent promotion race")
        ids["sk_ids"].append(sk_id)

        rpc_args = {
            "p_workspace_id": ws, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy",
            "p_valid_from": None, "p_valid_until": None,
            "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk_id,
                            "stance": "supports", "captured_at": _now_iso()}],
        }
        results = [None, None]

        def _call(idx):
            results[idx] = supabase.rpc("create_memory_with_evidence", rpc_args).execute().data

        t1 = threading.Thread(target=_call, args=(0,))
        t2 = threading.Thread(target=_call, args=(1,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results[0] is not None and results[1] is not None
        assert results[0] == results[1], "two concurrent calls for the same logical key must resolve to the same memory id"

        rows = supabase.table("org_memory").select("id") \
            .eq("workspace_id", ws).eq("grounding_fingerprint", f"structured_knowledge:{sk_id}").execute().data
        assert len(rows) == 1
        ids["memory_ids"].append(rows[0]["id"])
    finally:
        _cleanup(ids)


# =====================================================================
# 25. Review item promotion/resolution
# =====================================================================

def test_review_item_promoted_and_resolved_correctly():
    """Structured_knowledge is insert-only in this system (confirmed via
    live audit -- no code path ever mutates an existing row, see the Phase
    6C report's Part 1 schema audit), so a candidate 'later qualifying'
    never happens via an in-place UPDATE the incremental window would
    re-see. This test isolates the classify -> promote -> resolve-review
    LOGIC itself (Part 6) by re-presenting the same candidate directly to
    _process_candidate with the field that newly qualifies it added --
    exactly what a corrected/updated re-extraction would look like, without
    fighting the (separately, already tested) incremental boundary."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "entity_ids": [], "relationship_ids": [], "memory_ids": [], "review_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="process_step", recurrence_text=None,
                          statement="TEST-6C review then later qualifies")
        ids["sk_ids"].append(sk_id)
        entity_id = _make_entity(ws, "TEST-6C entity")
        ids["entity_ids"].append(entity_id)
        ids["relationship_ids"].append(_make_relationship(ws, sk_id, entity_id))

        result_1 = mc.run_consolidation(ws)
        ids["run_ids"].append(result_1["run_id"])
        assert result_1["stats"]["review_candidates"] == 1
        review_row = supabase.table("memory_review_queue").select("id, status").eq("workspace_id", ws).execute().data[0]
        ids["review_ids"].append(review_row["id"])
        assert review_row["status"] == "pending"

        stats = {"promoted": 0, "review_candidates": 0, "already_durable": 0, "rejected": 0,
                 "superseded": 0, "contradiction_flagged": 0}
        candidate = supabase.table("structured_knowledge").select("*").eq("id", sk_id).execute().data[0]
        candidate["recurrence_text"] = "every Friday"  # the newly-qualifying field
        run_id_2 = mc._start_run(ws, None, _now_iso(), _now())
        ids["run_ids"].append(run_id_2)
        mc._process_candidate(ws, run_id_2, candidate, stats)

        assert stats["promoted"] == 1
        memory_row = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]
        ids["memory_ids"].append(memory_row["id"])

        resolved = supabase.table("memory_review_queue").select("status, resolved_at, resolution") \
            .eq("id", review_row["id"]).execute().data[0]
        assert resolved["status"] == "promoted"
        assert resolved["resolved_at"] is not None
    finally:
        _cleanup(ids)


# =====================================================================
# 26-27. Revalidation / age / expiry
# =====================================================================

def test_durable_memory_remains_after_working_memory_horizon():
    """Not forgotten by age (Part 11): an old last_confirmed_at must not
    trigger any demotion on its own -- only evidence-existence matters."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6C long-lived durable memory")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        memory_id = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]["id"]
        ids["memory_ids"].append(memory_id)

        long_ago = (_now() - timedelta(days=400)).isoformat()
        supabase.table("org_memory").update({"last_confirmed_at": long_ago}).eq("id", memory_id).execute()

        stats = {}
        mc._revalidate(ws, stats)

        row = supabase.table("org_memory").select("lifecycle_status, last_confirmed_at").eq("id", memory_id).execute().data[0]
        assert row["lifecycle_status"] == "active", "age alone must never demote a memory"
        assert row["last_confirmed_at"] > long_ago
    finally:
        _cleanup(ids)


def test_expired_validity_does_not_imply_deletion():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "run_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6C expired validity window")
        ids["sk_ids"].append(sk_id)
        result = mc.run_consolidation(ws)
        ids["run_ids"].append(result["run_id"])
        memory_id = supabase.table("org_memory").select("id").eq("workspace_id", ws).execute().data[0]["id"]
        ids["memory_ids"].append(memory_id)

        past_until = (_now() - timedelta(days=1)).isoformat()
        supabase.table("org_memory").update({"valid_until": past_until}).eq("id", memory_id).execute()

        row = supabase.table("org_memory").select("valid_from, valid_until").eq("id", memory_id).execute().data[0]
        assert not _is_current(row["valid_from"], row["valid_until"], _now()), \
            "an expired window must be excluded from CURRENT reads"

        mc.run_consolidation(ws)  # revalidation must not delete it for being expired

        still_there = supabase.table("org_memory").select("id").eq("id", memory_id).execute().data
        assert len(still_there) == 1
        assert supabase.table("memory_evidence").select("id", count="exact").eq("memory_id", memory_id).execute().count == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 28. Real 4-memory corpus unchanged in the no-op run
# =====================================================================

def test_real_four_memory_corpus_remains_unchanged_in_noop_run():
    before = {r["id"]: r for r in supabase.table("org_memory").select("*")
              .in_("id", list(REAL_MEMORY_IDS.values())).execute().data}
    assert len(before) == 4

    mc.run_consolidation(REAL_WORKSPACE)

    after = {r["id"]: r for r in supabase.table("org_memory").select("*")
             .in_("id", list(REAL_MEMORY_IDS.values())).execute().data}
    assert len(after) == 4
    for memory_id, before_row in before.items():
        after_row = after[memory_id]
        for field in before_row:
            if field == "last_confirmed_at":
                assert after_row[field] >= before_row[field]
            else:
                assert after_row[field] == before_row[field], f"{field} changed on {memory_id}"


# =====================================================================
# 29. Fixture cleanup sentinel
# =====================================================================

def test_no_test_6c_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id, statement") \
        .like("statement", "TEST-6C%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_real_workspace_counts_still_exact():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_review_queue").select("id", count="exact").execute().count == 1


# =====================================================================
# 30. Full regression -- see this file's module docstring
# =====================================================================

def test_placeholder_full_regression_runs_separately():
    """Item 30 of the required matrix. Exactly like every prior phase's own
    'Full regression' report section, this is the separate, real, sequential
    run of every test_phase*.py file together -- not something one pytest
    function inside this file can meaningfully do by invoking pytest on
    itself. See the Phase 6C final report's Full regression section for the
    real, live run this item is satisfied by."""
    assert callable(mc.run_consolidation)
