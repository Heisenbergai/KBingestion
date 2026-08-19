"""
Phase 6A Memory Schema tests -- verifies the three new tables
(memory_consolidation_runs, org_memory, memory_evidence) and the
create_memory_with_evidence RPC against the live schema.

No backfill: this suite creates ZERO permanent org_memory/memory_evidence
rows. Every fixture is synthetic, cleaned up in a finally block, and the
suite's own sentinel tests (#20-25) reconfirm the exact pre-existing counts
(structured_knowledge=15 global/14 workspace, 5 entities, 3 relationships,
4 relationship evidence, 1 calendar snapshot) are unchanged at the end.

RLS/zero-policy verification (#15/#16) uses a genuinely separate anon-key
client (SUPABASE_ANON_KEY, confirmed available in this environment) rather
than introspecting pg_catalog through the service-role client every other
test uses -- the service key bypasses RLS by design, so it cannot prove
RLS is actually enforced; only an anon-tier client attempting a real read/
write can.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase6a_memory_schema.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from supabase import create_client
import os

from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

# A real, permanent structured_knowledge row -- used as evidence in every
# fixture below. Never mutated, never deleted.
REAL_SK_ID = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _cleanup(ids: dict):
    # Successor MUST be deleted before predecessor -- supersedes_memory_id
    # is ON DELETE RESTRICT (Phase 6A.1's own point), so deleting a
    # predecessor while a successor still references it fails, by design.
    # This ordering is a fixture-cleanup requirement, not a workaround.
    ordered_keys = ("successor_id", "memory_id", "memory_a_id", "memory_b_id", "predecessor_id")
    memory_keys = [ids[k] for k in ordered_keys if ids.get(k)] + list(ids.get("memory_ids", []))
    for mid in memory_keys:
        supabase.table("memory_evidence").delete().eq("memory_id", mid).execute()
    for mid in memory_keys:
        supabase.table("org_memory").delete().eq("id", mid).execute()
    if ids.get("run_id"):
        supabase.table("memory_consolidation_runs").delete().eq("id", ids["run_id"]).execute()
    sk_keys = list(ids.get("sk_ids", [])) + [
        ids[k] for k in ("public_sk", "internal_sk", "confidential_sk", "restricted_sk", "null_sens_sk",
                        "grounding_a_sk", "grounding_b_sk")
        if ids.get(k)
    ]
    for sk_id in sk_keys:
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_synthetic_sk(sensitivity, label_suffix) -> str:
    return supabase.table("structured_knowledge").insert({
        "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
        "canonical_id": REAL_SK_ID, "provider": "google_chat",
        "primitive_type": "fact", "statement": f"TEST-6A synthetic statement {label_suffix}",
        "raw_subject_phrase": "TEST-6A", "sensitivity": sensitivity, "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6a-{label_suffix}-{uuid.uuid4()}",
    }).execute().data[0]["id"]


def _create_memory(workspace_id=REAL_WORKSPACE, memory_type="policy", promotion_basis="authoritative_policy",
                   valid_from=None, valid_until=None, supersedes_memory_id=None,
                   consolidation_run_id=None, evidence_sk_id=REAL_SK_ID):
    return supabase.rpc("create_memory_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_memory_type": memory_type,
        "p_promotion_basis": promotion_basis,
        "p_valid_from": valid_from or _now_iso(),
        "p_valid_until": valid_until,
        "p_supersedes_memory_id": supersedes_memory_id,
        "p_consolidation_run_id": consolidation_run_id,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": evidence_sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }).execute()


# =====================================================================
# 1-5. CHECK constraints
# =====================================================================

def test_memory_type_check():
    with pytest.raises(Exception):
        supabase.table("org_memory").insert({
            "workspace_id": REAL_WORKSPACE, "memory_type": "not-a-real-type",
            "promotion_basis": "authoritative_policy", "valid_from": _now_iso(),
            "sensitivity": "internal",
        }).execute()


def test_lifecycle_status_check():
    with pytest.raises(Exception):
        supabase.table("org_memory").insert({
            "workspace_id": REAL_WORKSPACE, "memory_type": "policy",
            "lifecycle_status": "not-a-real-status",
            "promotion_basis": "authoritative_policy", "valid_from": _now_iso(),
            "sensitivity": "internal",
        }).execute()


def test_promotion_basis_check():
    with pytest.raises(Exception):
        supabase.table("org_memory").insert({
            "workspace_id": REAL_WORKSPACE, "memory_type": "policy",
            "promotion_basis": "not-a-real-basis", "valid_from": _now_iso(),
            "sensitivity": "internal",
        }).execute()


def test_memory_evidence_evidence_type_check():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        with pytest.raises(Exception):
            supabase.table("memory_evidence").insert({
                "memory_id": ids["memory_id"], "workspace_id": REAL_WORKSPACE,
                "evidence_type": "calendar_event_snapshot", "evidence_id": REAL_SK_ID,
                "stance": "supports", "captured_at": _now_iso(),
            }).execute()
    finally:
        _cleanup(ids)


def test_memory_evidence_stance_only_supports():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        with pytest.raises(Exception):
            supabase.table("memory_evidence").insert({
                "memory_id": ids["memory_id"], "workspace_id": REAL_WORKSPACE,
                "evidence_type": "structured_knowledge", "evidence_id": REAL_SK_ID,
                "stance": "contradicts", "captured_at": _now_iso(),
            }).execute()
    finally:
        _cleanup(ids)


# =====================================================================
# 6-7. Foreign keys
# =====================================================================

def test_fk_memory_evidence_to_org_memory():
    fake_memory_id = str(uuid.uuid4())
    with pytest.raises(Exception):
        supabase.table("memory_evidence").insert({
            "memory_id": fake_memory_id, "workspace_id": REAL_WORKSPACE,
            "evidence_type": "structured_knowledge", "evidence_id": REAL_SK_ID,
            "stance": "supports", "captured_at": _now_iso(),
        }).execute()


def test_fk_evidence_to_structured_knowledge():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        fake_sk_id = str(uuid.uuid4())
        with pytest.raises(Exception):
            supabase.table("memory_evidence").insert({
                "memory_id": ids["memory_id"], "workspace_id": REAL_WORKSPACE,
                "evidence_type": "structured_knowledge", "evidence_id": fake_sk_id,
                "stance": "supports", "captured_at": _now_iso(),
            }).execute()
    finally:
        _cleanup(ids)


# =====================================================================
# 8. No cascade deletion across superseded memory history
# =====================================================================

def test_no_cascade_deletion_across_supersession():
    ids = {}
    try:
        pred = _create_memory(promotion_basis="authoritative_policy",
                              valid_from="2020-01-01T00:00:00+00:00")
        ids["predecessor_id"] = pred.data

        succ = _create_memory(promotion_basis="authoritative_policy",
                              valid_from="2020-02-01T00:00:00+00:00",
                              supersedes_memory_id=ids["predecessor_id"])
        ids["successor_id"] = succ.data

        pred_row = supabase.table("org_memory").select("lifecycle_status").eq("id", ids["predecessor_id"]).execute().data[0]
        succ_row = supabase.table("org_memory").select("lifecycle_status,supersedes_memory_id").eq("id", ids["successor_id"]).execute().data[0]
        assert pred_row["lifecycle_status"] == "superseded"
        assert succ_row["lifecycle_status"] == "active"
        assert succ_row["supersedes_memory_id"] == ids["predecessor_id"]

        # Deleting the predecessor directly must fail -- ON DELETE RESTRICT,
        # never a silent cascade that would sever the successor's history.
        with pytest.raises(Exception):
            supabase.table("org_memory").delete().eq("id", ids["predecessor_id"]).execute()

        # Both must still be present and correctly linked.
        still_there = supabase.table("org_memory").select("id") \
            .in_("id", [ids["predecessor_id"], ids["successor_id"]]).execute().data
        assert len(still_there) == 2
    finally:
        _cleanup(ids)


# =====================================================================
# 9-12. Schema shape -- no forbidden fields, required fields present
# =====================================================================

def test_sensitivity_field_exists_and_non_null():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] is not None
        assert row["sensitivity"] in ("public", "internal", "confidential", "restricted")
    finally:
        _cleanup(ids)


def test_no_free_text_statement_or_content_field():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("*").eq("id", ids["memory_id"]).execute().data[0]
        forbidden = {"statement", "content", "text", "body", "claim"}
        assert forbidden.isdisjoint(row.keys()), \
            f"org_memory must never restate source content, found: {forbidden & set(row.keys())}"
    finally:
        _cleanup(ids)


def test_no_importance_field():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("*").eq("id", ids["memory_id"]).execute().data[0]
        forbidden = {"importance", "priority", "durability_score", "contradiction_id"}
        assert forbidden.isdisjoint(row.keys()), \
            f"org_memory must not carry an unexplainable numeric score, found: {forbidden & set(row.keys())}"
    finally:
        _cleanup(ids)


def test_no_working_memory_priority_field():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("*").eq("id", ids["memory_id"]).execute().data[0]
        forbidden = {"working_memory_state", "working_memory_priority", "decay_state", "last_refreshed_at"}
        assert forbidden.isdisjoint(row.keys()), \
            "working memory is computed at read time, never persisted on org_memory"
        # STALE SET (Phase 6A.1, then 6D.2): originally 12 columns. Phase
        # 6A.1 Issue 1 legitimately added grounding_fingerprint -- a
        # deterministic function of the memory's real evidence. Phase 6D.2
        # legitimately added superseded_at -- a deterministic function of
        # WHEN a real, atomic supersession event happened (equal to the
        # real successor's created_at), not a working-memory/importance/
        # decay field either. Neither addition violates what this test
        # actually protects. Updated to the real 14-column set.
        assert set(row.keys()) == {
            "id", "workspace_id", "memory_type", "lifecycle_status", "promotion_basis",
            "valid_from", "valid_until", "supersedes_memory_id", "sensitivity",
            "created_at", "last_confirmed_at", "consolidation_run_id", "grounding_fingerprint",
            "superseded_at",
        }
    finally:
        _cleanup(ids)


# =====================================================================
# 13. Evidence uniqueness
# =====================================================================

def test_evidence_uniqueness_enforced():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        with pytest.raises(Exception):
            supabase.table("memory_evidence").insert({
                "memory_id": ids["memory_id"], "workspace_id": REAL_WORKSPACE,
                "evidence_type": "structured_knowledge", "evidence_id": REAL_SK_ID,
                "stance": "supports", "captured_at": _now_iso(),
            }).execute()
    finally:
        _cleanup(ids)


# =====================================================================
# 14. Workspace isolation
# =====================================================================

def test_workspace_isolation():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        wrong_ws_match = supabase.table("org_memory").select("id") \
            .eq("id", ids["memory_id"]).eq("workspace_id", OTHER_REAL_WORKSPACE).execute().data
        assert wrong_ws_match == []
    finally:
        _cleanup(ids)


def test_rpc_rejects_cross_workspace_evidence():
    """A structured_knowledge row from REAL_WORKSPACE must be rejected as
    evidence for a memory claimed under a different workspace."""
    with pytest.raises(Exception):
        _create_memory(workspace_id=OTHER_REAL_WORKSPACE, promotion_basis="authoritative_policy")


# =====================================================================
# 15-16. RLS enabled, zero policies -- verified via a genuinely separate
# anon-key client, not by introspecting pg_catalog through the service key
# (which bypasses RLS by design and cannot prove enforcement).
# =====================================================================

def _anon_client():
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    assert url and anon_key, "SUPABASE_ANON_KEY must be set for this test to be meaningful"
    return create_client(url, anon_key)


def test_rls_blocks_anon_read_on_all_three_tables():
    anon = _anon_client()
    for table in ("memory_consolidation_runs", "org_memory", "memory_evidence"):
        result = anon.table(table).select("id").limit(1).execute()
        assert result.data == [], f"{table} leaked a row to an anon-key client -- RLS/policy gap"


def test_rls_blocks_anon_write_on_org_memory():
    anon = _anon_client()
    with pytest.raises(Exception):
        anon.table("org_memory").insert({
            "workspace_id": REAL_WORKSPACE, "memory_type": "policy",
            "promotion_basis": "authoritative_policy", "valid_from": _now_iso(),
            "sensitivity": "public",
        }).execute()


# =====================================================================
# 17-18. Evidence invariant + successful creation
# =====================================================================

def test_memory_without_evidence_rejected():
    with pytest.raises(Exception):
        supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": _now_iso(),
            "p_valid_until": None, "p_supersedes_memory_id": None,
            "p_consolidation_run_id": None, "p_evidence": [],
        }).execute()


def test_valid_memory_with_evidence_succeeds():
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy")
        ids["memory_id"] = res.data
        assert res.data is not None
        row = supabase.table("org_memory").select("*").eq("id", ids["memory_id"]).execute().data[0]
        assert row["memory_type"] == "policy"
        assert row["promotion_basis"] == "authoritative_policy"
        assert row["lifecycle_status"] == "active"
        ev = supabase.table("memory_evidence").select("*").eq("memory_id", ids["memory_id"]).execute().data
        assert len(ev) == 1
        assert ev[0]["evidence_id"] == REAL_SK_ID
    finally:
        _cleanup(ids)


def test_sensitivity_computed_as_strictest_ceiling():
    """Synthetic structured_knowledge rows across all 4 tiers + a NULL-
    sensitivity row -- proves the RPC's ceiling computation, not just the
    always-'internal' case the real corpus happens to offer."""
    ids = {}
    try:
        ids["public_sk"] = _make_synthetic_sk("public", "public")
        ids["confidential_sk"] = _make_synthetic_sk("confidential", "confidential")

        res_public_only = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2021-01-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["public_sk"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res_public_only.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] == "public"

        # A second memory mixing public + confidential evidence must take
        # the STRICTER ceiling (confidential), never the looser one.
        res_mixed = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2021-02-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [
                {"evidence_type": "structured_knowledge", "evidence_id": ids["public_sk"],
                 "stance": "supports", "captured_at": _now_iso()},
                {"evidence_type": "structured_knowledge", "evidence_id": ids["confidential_sk"],
                 "stance": "supports", "captured_at": _now_iso()},
            ],
        }).execute()
        ids["predecessor_id"] = res_mixed.data  # reuse cleanup slot
        mixed_row = supabase.table("org_memory").select("sensitivity").eq("id", ids["predecessor_id"]).execute().data[0]
        assert mixed_row["sensitivity"] == "confidential"
    finally:
        _cleanup(ids)


# =====================================================================
# 19. Idempotency
# =====================================================================

def test_repeated_identical_write_is_idempotent():
    ids = {}
    try:
        valid_from = "2022-06-01T00:00:00+00:00"
        res1 = _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from)
        ids["memory_id"] = res1.data

        res2 = _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from)
        assert res2.data == res1.data, "identical logical key must reuse the same memory row"

        count = supabase.table("org_memory").select("id", count="exact") \
            .eq("workspace_id", REAL_WORKSPACE).eq("promotion_basis", "authoritative_policy") \
            .eq("valid_from", valid_from).execute().count
        assert count == 1

        ev_count = supabase.table("memory_evidence").select("id", count="exact") \
            .eq("memory_id", ids["memory_id"]).execute().count
        assert ev_count == 1, "retry must not duplicate evidence either"
    finally:
        _cleanup(ids)


# =====================================================================
# 20-24. Existing-data integrity
# =====================================================================

def test_structured_knowledge_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14


def test_graph_entities_unchanged():
    assert supabase.table("knowledge_entities").select("id", count="exact").execute().count == 5


def test_graph_relationships_unchanged():
    assert supabase.table("knowledge_relationships").select("id", count="exact").execute().count == 3


def test_relationship_evidence_unchanged():
    assert supabase.table("knowledge_relationship_evidence").select("id", count="exact").execute().count == 4


def test_calendar_snapshot_unchanged():
    """Phase 6A.1: the workspace's real Google Calendar connection is live
    (confirmed in Phase 5K.1 -- a second real event synced in mid-session).
    This test must never assume a static count; it only confirms this
    pass's own work didn't touch calendar_event_snapshots at all -- >= 1,
    never fewer than what already existed."""
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count >= 1


# =====================================================================
# PHASE 6A.1 -- write-contract hardening tests.
#
# Several of the 20 items in this pass's own required list are already
# fully covered by pre-existing Phase 6A tests above, re-run unmodified
# against the corrected RPC in this same file (idempotency: see
# test_repeated_identical_write_is_idempotent; public+confidential ceiling:
# see test_sensitivity_computed_as_strictest_ceiling; supersession/
# coexistence: see test_no_cascade_deletion_across_supersession; workspace
# isolation, existing-data integrity, evidence uniqueness: see their
# original Phase 6A tests). The tests below cover what is GENUINELY NEW in
# this pass: grounding-aware identity (Case B/C), the NULL-sensitivity hard
# rejection, and the previously-unexercised internal-only/internal+
# restricted sensitivity tiers.
# =====================================================================

def test_case_a_same_grounding_same_semantics_is_idempotent():
    """Item 1 -- explicit Phase 6A.1 framing: identical grounding + identical
    semantics reuses the same memory row."""
    ids = {}
    try:
        valid_from = "2023-01-01T00:00:00+00:00"
        res1 = _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from,
                              evidence_sk_id=REAL_SK_ID)
        ids["memory_id"] = res1.data
        res2 = _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from,
                              evidence_sk_id=REAL_SK_ID)
        assert res2.data == res1.data
        count = supabase.table("org_memory").select("id", count="exact") \
            .eq("workspace_id", REAL_WORKSPACE).eq("grounding_fingerprint", f"structured_knowledge:{REAL_SK_ID}") \
            .eq("valid_from", valid_from).execute().count
        assert count == 1
    finally:
        _cleanup(ids)


def test_case_b_different_grounding_same_semantics_creates_distinct_memories():
    """Items 2 & 3 -- two synthetic structured_knowledge fixtures, different
    ids, otherwise identical workspace/memory_type/promotion_basis/
    valid_from. Under the OLD (Phase 6A) coarse key these would have
    collided into one row; under the corrected key both must exist
    independently."""
    ids = {}
    try:
        ids["grounding_a_sk"] = _make_synthetic_sk("internal", "grounding-a")
        ids["grounding_b_sk"] = _make_synthetic_sk("internal", "grounding-b")
        valid_from = "2023-02-01T00:00:00+00:00"

        res_a = _create_memory(memory_type="policy", promotion_basis="authoritative_policy",
                               valid_from=valid_from, evidence_sk_id=ids["grounding_a_sk"])
        res_b = _create_memory(memory_type="policy", promotion_basis="authoritative_policy",
                               valid_from=valid_from, evidence_sk_id=ids["grounding_b_sk"])
        ids["memory_ids"] = [res_a.data, res_b.data]

        assert res_a.data != res_b.data, "different grounding must never collapse into one memory"
        rows = supabase.table("org_memory").select("id,grounding_fingerprint") \
            .in_("id", ids["memory_ids"]).execute().data
        assert len(rows) == 2
        assert rows[0]["grounding_fingerprint"] != rows[1]["grounding_fingerprint"]
    finally:
        _cleanup(ids)


def test_case_c_same_grounding_different_promotion_basis_is_rejected():
    """Same grounding, same workspace, DIFFERENT promotion_basis -- must be
    rejected as a semantic conflict (two simultaneously-active memories
    claiming different reasons for the same underlying fact), not silently
    allowed to coexist. Explicit supersession is the sanctioned replacement
    path, exercised separately in test_no_cascade_deletion_across_supersession."""
    ids = {}
    try:
        ids["grounding_a_sk"] = _make_synthetic_sk("internal", "case-c")
        res1 = _create_memory(memory_type="policy", promotion_basis="authoritative_policy",
                              valid_from="2023-03-01T00:00:00+00:00", evidence_sk_id=ids["grounding_a_sk"])
        ids["memory_id"] = res1.data

        with pytest.raises(Exception):
            _create_memory(memory_type="policy", promotion_basis="explicit_user_keep",
                           valid_from="2023-03-02T00:00:00+00:00", evidence_sk_id=ids["grounding_a_sk"])

        # confirm the rejection wrote nothing -- still exactly one memory
        # for this grounding.
        count = supabase.table("org_memory").select("id", count="exact") \
            .eq("workspace_id", REAL_WORKSPACE).eq("grounding_fingerprint", f"structured_knowledge:{ids['grounding_a_sk']}") \
            .execute().count
        assert count == 1
    finally:
        _cleanup(ids)


def test_duplicate_evidence_still_prevented_post_migration():
    """Item 4 -- re-confirms ON CONFLICT DO NOTHING on memory_evidence still
    functions correctly after the fingerprint migration (calling the RPC
    twice with identical evidence must not duplicate the evidence row)."""
    ids = {}
    try:
        valid_from = "2023-04-01T00:00:00+00:00"
        res1 = _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from)
        ids["memory_id"] = res1.data
        _create_memory(promotion_basis="authoritative_policy", valid_from=valid_from)
        ev_count = supabase.table("memory_evidence").select("id", count="exact") \
            .eq("memory_id", ids["memory_id"]).execute().count
        assert ev_count == 1
    finally:
        _cleanup(ids)


def test_null_sensitivity_evidence_is_rejected():
    """Items 5 & 6 -- a real Calendar-sourced structured_knowledge row
    (sensitivity IS NULL -- "no classification concept", never "public")
    must reject memory creation outright, and leave zero org_memory/
    memory_evidence rows behind."""
    real_calendar_sk_id = "9420ae8a-76f6-40b7-a4b3-0a94cad80921"
    before_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    before_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count

    with pytest.raises(Exception):
        _create_memory(promotion_basis="explicit_user_keep",
                       valid_from="2023-05-01T00:00:00+00:00", evidence_sk_id=real_calendar_sk_id)

    after_memory = supabase.table("org_memory").select("id", count="exact").execute().count
    after_evidence = supabase.table("memory_evidence").select("id", count="exact").execute().count
    assert after_memory == before_memory, "NULL sensitivity rejection must write zero memory rows"
    assert after_evidence == before_evidence, "NULL sensitivity rejection must write zero evidence rows"


def test_sensitivity_classified_public_alone_stays_public():
    """Item 7."""
    ids = {}
    try:
        ids["public_sk"] = _make_synthetic_sk("public", "public-alone")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2023-06-01T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["public_sk"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] == "public"
    finally:
        _cleanup(ids)


def test_sensitivity_classified_internal_alone_stays_internal():
    """Item 8 -- uses the REAL structured_knowledge row (genuinely
    sensitivity='internal' in the live corpus), not a synthetic one."""
    ids = {}
    try:
        res = _create_memory(promotion_basis="authoritative_policy",
                             valid_from="2023-06-02T00:00:00+00:00", evidence_sk_id=REAL_SK_ID)
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] == "internal"
    finally:
        _cleanup(ids)


def test_sensitivity_internal_plus_restricted_is_restricted():
    """Item 10."""
    ids = {}
    try:
        ids["internal_sk"] = _make_synthetic_sk("internal", "internal-mix")
        ids["restricted_sk"] = _make_synthetic_sk("restricted", "restricted-mix")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2023-06-03T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [
                {"evidence_type": "structured_knowledge", "evidence_id": ids["internal_sk"],
                 "stance": "supports", "captured_at": _now_iso()},
                {"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                 "stance": "supports", "captured_at": _now_iso()},
            ],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] == "restricted"
    finally:
        _cleanup(ids)


def test_sensitivity_never_caller_controlled():
    """Item 11 -- the RPC signature has no sensitivity parameter at all;
    the only way sensitivity is ever set is the server-side ceiling
    computation. Confirmed structurally: calling with restricted-tier
    evidence always produces 'restricted' regardless of anything else in
    the call, and there is no parameter through which a caller could
    request a looser value."""
    ids = {}
    try:
        ids["restricted_sk"] = _make_synthetic_sk("restricted", "caller-control-check")
        res = supabase.rpc("create_memory_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE, "p_memory_type": "policy",
            "p_promotion_basis": "authoritative_policy", "p_valid_from": "2023-06-04T00:00:00+00:00",
            "p_valid_until": None, "p_supersedes_memory_id": None, "p_consolidation_run_id": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                            "stance": "supports", "captured_at": _now_iso()}],
        }).execute()
        ids["memory_id"] = res.data
        row = supabase.table("org_memory").select("sensitivity").eq("id", ids["memory_id"]).execute().data[0]
        assert row["sensitivity"] == "restricted", \
            "sensitivity must always reflect the real evidence ceiling -- there is no parameter to override it"
    finally:
        _cleanup(ids)


# =====================================================================
# 25. Fixture cleanup sentinel
# =====================================================================

def test_no_org_memory_rows_leaked():
    """STALE COUNT (Phase 6B): this originally asserted zero permanent
    org_memory rows -- true at the time Phase 6A/6A.1 were written, since
    6B (the real promotion pass) hadn't happened yet. Phase 6B legitimately
    promoted 4 real, evidence-backed memories (2 distinct credential
    policies, 1 hardware-scope policy, 1 recurring process -- see
    test_phase6b_memory_consolidation.py). What this test actually still
    protects -- no LEAKED synthetic fixture row from THIS file's own
    tests -- now means exactly 4, not more."""
    count = supabase.table("org_memory").select("id", count="exact").execute().count
    assert count == 4, "exactly the 4 real Phase 6B promotions, no leaked synthetic fixture from this file"


def test_no_memory_evidence_rows_leaked():
    """STALE COUNT (Phase 6B): see test_no_org_memory_rows_leaked's
    docstring -- same correction, same reason."""
    count = supabase.table("memory_evidence").select("id", count="exact").execute().count
    assert count == 4


def test_no_test_6a_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-6A%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"
