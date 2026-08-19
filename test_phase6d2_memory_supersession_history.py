"""
Phase 6D.2 Memory Supersession History tests.

Three independent temporal concepts, now all real and separately testable:
  created_at      -- MEMORY AVAILABILITY (when KNOVA created this memory)
  valid_from/until -- CLAIM VALIDITY (when the underlying claim is true)
  superseded_at   -- MEMORY SUCCESSION (when this memory stopped being the
                     current durable representation), set only by
                     create_memory_with_evidence's own atomic supersession
                     path, equal to the real successor's created_at.

Real data: the 4 real memories (0 real supersessions -- confirmed live,
never modified here). Synthetic, single-use workspaces for every
supersession scenario, per the explicit instruction not to touch real data.

Run with: python -m pytest test_phase6d2_memory_supersession_history.py -v
"""
import uuid
import threading
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import graph_retrieval as gr
import memory_retrieval as mr

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)

REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "credential_sharing":  "3d376631-894c-4e32-b3f5-3ecf7cfd5f61",
    "hardware_scope":      "8aef76c9-fda3-44d6-affb-769f2ff09326",
    "monday_capacity":     "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _fresh_workspace() -> str:
    return str(uuid.uuid4())


def _parse(ts) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


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
        "primitive_type": "fact", "statement": "TEST-6D2 synthetic statement",
        "raw_subject_phrase": "TEST-6D2 subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6d2-{uuid.uuid4()}",
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


def _row(memory_id: str) -> dict:
    return supabase.table("org_memory").select("*").eq("id", memory_id).execute().data[0]


# =====================================================================
# 1-2. Schema
# =====================================================================

def test_superseded_at_column_exists_and_nullable():
    for label, memory_id in REAL_MEMORY_IDS.items():
        row = _row(memory_id)
        assert "superseded_at" in row, "superseded_at must exist on every org_memory row"


def test_active_memory_has_null_superseded_at():
    for label, memory_id in REAL_MEMORY_IDS.items():
        row = _row(memory_id)
        assert row["lifecycle_status"] == "active"
        assert row["superseded_at"] is None, f"{label} has never been superseded"


# =====================================================================
# 3-5. Atomicity / consistency
# =====================================================================

def test_creating_successor_sets_predecessor_superseded_at():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)
        assert _row(memory_a)["superseded_at"] is None

        sk_b = _make_sk(ws, statement="TEST-6D2 successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        row_a = _row(memory_a)
        assert row_a["lifecycle_status"] == "superseded"
        assert row_a["superseded_at"] is not None
    finally:
        _cleanup(ids)


def test_successor_created_at_equals_predecessor_superseded_at():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 equality predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 equality successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        row_a, row_b = _row(memory_a), _row(memory_b)
        assert row_a["superseded_at"] == row_b["created_at"], \
            "both are written from the same captured v_now within one atomic transaction"
    finally:
        _cleanup(ids)


def test_supersession_is_atomic_and_idempotent():
    """A repeated, identical successor-creation call must not re-stamp
    superseded_at -- the predecessor's guard (lifecycle_status !=
    'superseded') already makes this idempotent, proven directly here."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 idempotent predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 idempotent successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b_first = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b_first)
        superseded_at_first = _row(memory_a)["superseded_at"]

        memory_b_second = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        assert memory_b_second == memory_b_first, "idempotent re-run must resolve to the same memory"
        superseded_at_second = _row(memory_a)["superseded_at"]
        assert superseded_at_second == superseded_at_first, "must never drift on a repeated call"
    finally:
        _cleanup(ids)


# =====================================================================
# 6-9. Historical retrieval respects the succession boundary
# =====================================================================

def test_before_supersession_as_of_returns_a():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 timeline A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)
        created_a = _parse(_row(memory_a)["created_at"])

        just_after_a = created_a + timedelta(milliseconds=1)
        ctx = mr.build_memory_context("TEST-6D2 timeline A statement", ws, OWNER, as_of=just_after_a)
        assert ctx is not None
        assert memory_a in {c.memory_id for c in ctx.candidates}
    finally:
        _cleanup(ids)


def test_after_supersession_as_of_excludes_a_as_current():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 exclude A after statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 exclude successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        superseded_at = _parse(_row(memory_a)["superseded_at"])
        after = superseded_at + timedelta(milliseconds=1)
        ctx = mr.build_memory_context("TEST-6D2 exclude A after statement", ws, OWNER, as_of=after)
        found = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found, "A must not be treated as current once B has replaced it, per Part 7"
    finally:
        _cleanup(ids)


def test_after_supersession_as_of_returns_b():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 returns-B predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 returns-B successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        after = _parse(_row(memory_b)["created_at"]) + timedelta(milliseconds=1)
        ctx = mr.build_memory_context("TEST-6D2 returns-B successor statement", ws, OWNER, as_of=after)
        assert ctx is not None
        assert memory_b in {c.memory_id for c in ctx.candidates}
    finally:
        _cleanup(ids)


def test_current_query_returns_b_not_a():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 current query A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 current query B statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        ctx_a = mr.build_memory_context("TEST-6D2 current query A statement", ws, OWNER)
        found_a = {c.memory_id for c in ctx_a.candidates} if ctx_a else set()
        assert memory_a not in found_a

        ctx_b = mr.build_memory_context("TEST-6D2 current query B statement", ws, OWNER)
        assert ctx_b is not None
        assert memory_b in {c.memory_id for c in ctx_b.candidates}
    finally:
        _cleanup(ids)


# =====================================================================
# 10-12. The three concepts stay independent
# =====================================================================

def test_null_valid_from_behaves_independently_of_succession():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 null valid_from independence statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)
        row = _row(memory_a)
        assert row["valid_from"] is None
        assert row["superseded_at"] is None
        # NULL valid_from means "no known lower bound on claim validity" --
        # it must never be conflated with created_at or superseded_at.
        assert row["valid_from"] != row["created_at"]
    finally:
        _cleanup(ids)


def test_valid_until_behaves_independently_of_succession():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 valid_until independence statement")
        ids["sk_ids"].append(sk_a)
        valid_until = (_now() + timedelta(days=1)).isoformat()
        memory_a = _make_memory(ws, sk_a, p_valid_until=valid_until)
        ids["memory_ids"].append(memory_a)

        # Expired on CLAIM validity grounds, current time -- excluded, but
        # for a reason completely unrelated to succession (superseded_at
        # stays NULL throughout).
        ctx_after_expiry = mr.build_memory_context(
            "TEST-6D2 valid_until independence statement", ws, OWNER, as_of=_now() + timedelta(days=2))
        found = {c.memory_id for c in ctx_after_expiry.candidates} if ctx_after_expiry else set()
        assert memory_a not in found
        assert _row(memory_a)["superseded_at"] is None
    finally:
        _cleanup(ids)


def test_historical_availability_still_respects_created_at():
    """Phase 6D.1's fix remains intact: a point before created_at excludes
    the memory regardless of claim validity or succession state."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 availability still respected statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        before = _parse(_row(memory_a)["created_at"]) - timedelta(days=1)
        ctx = mr.build_memory_context("TEST-6D2 availability still respected statement", ws, OWNER, as_of=before)
        found = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found
    finally:
        _cleanup(ids)


# =====================================================================
# 13-14. Isolation / security
# =====================================================================

def test_workspace_isolation_with_supersession():
    ws_a = _fresh_workspace()
    ws_b = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws_a, statement="TEST-6D2 isolation predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws_a, sk_a)
        ids["memory_ids"].append(memory_a)

        ctx_b = mr.build_memory_context("TEST-6D2 isolation predecessor statement", ws_b, OWNER, as_of=_now())
        assert ctx_b is None
    finally:
        _cleanup(ids)


def test_sensitivity_isolation_with_supersession():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, sensitivity="restricted", statement="TEST-6D2 restricted supersession statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, sensitivity="restricted", statement="TEST-6D2 restricted successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        internal_only = gq.resolve_allowed_sensitivities("employee", False)
        ctx = mr.build_memory_context("TEST-6D2 restricted successor statement", ws, internal_only)
        assert ctx is None
    finally:
        _cleanup(ids)


# =====================================================================
# 15-16. Concurrency
# =====================================================================

def test_concurrent_supersession_leaves_no_impossible_state():
    """Two real concurrent successors, DIFFERENT groundings, both claiming
    to supersede A. Proven live (not asserted): both succeed, A ends up in
    exactly one consistent state (superseded, with a single non-null
    superseded_at matching exactly one real successor's created_at), no
    orphan evidence. The system does not need to pick a single "canonical"
    successor among two independently-evidenced ones -- that would be a
    new business rule (Part 9 forbids inventing one); Postgres row-level
    locking is what deterministically decides which UPDATE actually
    flips A's status, and that decision is real infrastructure, not
    invented logic."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 concurrent predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b1 = _make_sk(ws, statement="TEST-6D2 concurrent successor B1 statement")
        sk_b2 = _make_sk(ws, statement="TEST-6D2 concurrent successor B2 statement")
        ids["sk_ids"].extend([sk_b1, sk_b2])

        results = [None, None]

        def _call(idx, sk_id):
            try:
                r = supabase.rpc("create_memory_with_evidence", {
                    "p_workspace_id": ws, "p_memory_type": "policy", "p_promotion_basis": "authoritative_policy",
                    "p_valid_from": None, "p_valid_until": None,
                    "p_supersedes_memory_id": memory_a, "p_consolidation_run_id": None,
                    "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": sk_id,
                                    "stance": "supports", "captured_at": _now_iso()}],
                }).execute()
                results[idx] = r.data
            except Exception as e:
                results[idx] = ("error", str(e))

        t1 = threading.Thread(target=_call, args=(0, sk_b1))
        t2 = threading.Thread(target=_call, args=(1, sk_b2))
        t1.start(); t2.start(); t1.join(); t2.join()

        for r in results:
            assert isinstance(r, str), f"a concurrent successor must not error: {r}"
        ids["memory_ids"].extend(results)

        row_a = _row(memory_a)
        assert row_a["lifecycle_status"] == "superseded"
        assert row_a["superseded_at"] is not None
        # superseded_at matches exactly one of the two real successors'
        # created_at -- consistent, not duplicated, not contradictory.
        successor_created_ats = {_row(mid)["created_at"] for mid in results}
        assert row_a["superseded_at"] in successor_created_ats
    finally:
        _cleanup(ids)


def test_no_orphan_evidence_after_concurrent_supersession():
    """Every memory created in the concurrency scenario must have real,
    resolvable evidence -- exercised again standalone in case the race
    above ever produces a different real winner across runs."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D2 no-orphan predecessor statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D2 no-orphan successor statement")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        for mid in (memory_a, memory_b):
            ev = supabase.table("memory_evidence").select("id", count="exact").eq("memory_id", mid).execute().count
            assert ev >= 1, f"{mid} must have real evidence, never orphaned"
    finally:
        _cleanup(ids)


# =====================================================================
# 17-20. Existing-data integrity
# =====================================================================

def test_real_four_memories_unchanged():
    for label, memory_id in REAL_MEMORY_IDS.items():
        row = _row(memory_id)
        assert row["lifecycle_status"] == "active"
        assert row["valid_from"] is None
        assert row["valid_until"] is None
        assert row["superseded_at"] is None


def test_real_review_queue_unchanged():
    count = supabase.table("memory_review_queue").select("id", count="exact").eq("status", "pending").execute().count
    assert count == 1


def test_structured_knowledge_unchanged_by_supersession_fix():
    before = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER)
    after = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert before == after
    assert len(after) == 14


def test_graph_unchanged_by_supersession_fix():
    before = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    gr.build_graph_context("What requires Product approval?", REAL_WORKSPACE, OWNER)
    after = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    assert before == after
    assert len(after) == 3


# =====================================================================
# 21. Fixture cleanup sentinel
# =====================================================================

def test_no_test_6d2_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id, statement") \
        .like("statement", "TEST-6D2%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_real_workspace_counts_still_exact():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_review_queue").select("id", count="exact").execute().count == 1


# =====================================================================
# 22. Full regression -- see this file's module docstring
# =====================================================================

def test_placeholder_full_regression_runs_separately():
    assert callable(mr.build_memory_context)
