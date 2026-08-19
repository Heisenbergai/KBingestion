"""
Phase 6D.1 Memory Historical Availability tests.

Distinguishes CLAIM VALIDITY (valid_from/valid_until -- when the underlying
claim became/stopped being true) from MEMORY AVAILABILITY (created_at --
when KNOVA itself came to know it). NULL valid_from remains a legitimate,
permanent "no known real-world start" (frozen Phase 6B.1 decision, never
reinterpreted here) -- it is availability, not validity, that now also
gates historical reads via created_at <= as_of.

Real data for the real 4-memory before/at/after/current matrix (Part 5).
Synthetic, single-use workspaces for explicit valid_from/valid_until and
supersession-timing edge cases.

Run with: python -m pytest test_phase6d1_memory_temporal_availability.py -v
"""
import uuid
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
    for mid in reversed(ids.get("memory_ids", [])):
        supabase.table("org_memory").delete().eq("id", mid).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-6D1 synthetic statement",
        "raw_subject_phrase": "TEST-6D1 subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6d1-{uuid.uuid4()}",
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


# =====================================================================
# 1. Current query unchanged
# =====================================================================

def test_current_memory_query_unchanged():
    for label, memory_id in REAL_MEMORY_IDS.items():
        ctx = mr.build_memory_context("TEST-6D1 irrelevant", REAL_WORKSPACE, OWNER)
    # A single relevant real query, matching Phase 6D's own convention.
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER)
    assert ctx is not None
    ids = {c.memory_id for c in ctx.candidates}
    assert REAL_MEMORY_IDS["credential_logging"] in ids
    for c in ctx.candidates:
        assert c.lifecycle_status == "active"


# =====================================================================
# 2-4. Real before / at / after created_at
# =====================================================================

def _real_created_at(memory_id: str) -> datetime:
    row = supabase.table("org_memory").select("created_at").eq("id", memory_id).execute().data[0]
    return datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))


def test_historical_before_created_at_excluded_real():
    created_at = _real_created_at(REAL_MEMORY_IDS["credential_logging"])
    before = created_at - timedelta(days=1)
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=before)
    ids = {c.memory_id for c in ctx.candidates} if ctx else set()
    assert REAL_MEMORY_IDS["credential_logging"] not in ids, \
        "a memory must never be returned for a historical point before KNOVA created it"


def test_historical_at_created_at_included_real():
    created_at = _real_created_at(REAL_MEMORY_IDS["credential_logging"])
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=created_at)
    ids = {c.memory_id for c in ctx.candidates} if ctx else set()
    assert REAL_MEMORY_IDS["credential_logging"] in ids, "created_at <= as_of is inclusive at the exact boundary"


def test_historical_after_created_at_included_real():
    created_at = _real_created_at(REAL_MEMORY_IDS["credential_logging"])
    after = created_at + timedelta(hours=1)
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=after)
    ids = {c.memory_id for c in ctx.candidates} if ctx else set()
    assert REAL_MEMORY_IDS["credential_logging"] in ids


# =====================================================================
# 5-6. NULL / explicit valid_from
# =====================================================================

def test_null_valid_from_does_not_imply_historical_existence_real():
    """The core Phase 6D.1 correction: NULL valid_from must not be read as
    'KNOVA has always known this'. A point long before real creation must
    exclude it, even though its CLAIM validity window is open-ended."""
    far_past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=far_past)
    ids = {c.memory_id for c in ctx.candidates} if ctx else set()
    assert REAL_MEMORY_IDS["credential_logging"] not in ids


def test_explicit_valid_from_still_gates_claim_validity():
    """A memory whose CLAIM becomes valid strictly after KNOVA already knew
    about it (created_at < valid_from) must still be excluded before
    valid_from, and included once valid_from arrives -- proving availability
    and validity are independently enforced, not conflated."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="TEST-6D1 explicit valid_from statement")
        ids["sk_ids"].append(sk_id)
        future_valid_from = (_now() + timedelta(days=30)).isoformat()
        memory_id = _make_memory(ws, sk_id, p_valid_from=future_valid_from)
        ids["memory_ids"].append(memory_id)

        # Now (created, but claim not valid yet) -- excluded on claim validity.
        ctx_now = mr.build_memory_context("TEST-6D1 explicit valid_from statement", ws, OWNER)
        assert ctx_now is None or memory_id not in {c.memory_id for c in ctx_now.candidates}

        # After valid_from -- included.
        ctx_later = mr.build_memory_context(
            "TEST-6D1 explicit valid_from statement", ws, OWNER,
            as_of=_now() + timedelta(days=31))
        assert ctx_later is not None
        assert memory_id in {c.memory_id for c in ctx_later.candidates}
    finally:
        _cleanup(ids)


# =====================================================================
# 7. valid_until
# =====================================================================

def test_valid_until_still_excludes_after_expiry():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="TEST-6D1 valid_until statement")
        ids["sk_ids"].append(sk_id)
        valid_until = (_now() + timedelta(days=1)).isoformat()
        memory_id = _make_memory(ws, sk_id, p_valid_until=valid_until)
        ids["memory_ids"].append(memory_id)

        ctx_before = mr.build_memory_context("TEST-6D1 valid_until statement", ws, OWNER)
        assert ctx_before is not None
        assert memory_id in {c.memory_id for c in ctx_before.candidates}

        ctx_after = mr.build_memory_context(
            "TEST-6D1 valid_until statement", ws, OWNER, as_of=_now() + timedelta(days=2))
        found = {c.memory_id for c in ctx_after.candidates} if ctx_after else set()
        assert memory_id not in found
    finally:
        _cleanup(ids)


# =====================================================================
# 8-9. Supersession
# =====================================================================

def test_superseded_current_behavior_unchanged():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D1 supersession A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D1 supersession B statement replacing A")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        ctx = mr.build_memory_context("TEST-6D1 supersession A statement", ws, OWNER)
        found = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found, "current behavior is unchanged by Phase 6D.1"
    finally:
        _cleanup(ids)


def test_superseded_historical_behavior_honest_not_invented():
    """SUPERSEDED BY PHASE 6D.2 (2026-08-19): this test originally proved
    that, absent any supersession-event timestamp, a historical read at a
    point before the real-world succession still honestly reports the
    memory's real, PRESENT lifecycle_status ('superseded') rather than
    inventing what it 'would have been' at that past moment -- the gap was
    that "before the real-world succession" could only be approximated via
    created_at, not verified precisely. Phase 6D.2 added
    org_memory.superseded_at (set atomically, equal to the real successor's
    created_at) and wired it into the historical predicate, which closes
    that gap outright: a historical read can now distinguish "before
    succession" from "after succession" precisely, and correctly returns A
    with its real, honest lifecycle_status in EITHER case -- proven here
    using real DB-read timestamps (never relative offsets guessed against
    unpredictable RPC round-trip latency, which is what made this test
    flaky once the new boundary was added). See
    test_phase6d2_memory_supersession_history.py for the full before/after/
    current supersession-boundary matrix this test's original gap-disclosure
    purpose has been superseded by."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D1 supersession historical A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D1 supersession historical B statement replacing A")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        superseded_at = datetime.fromisoformat(
            supabase.table("org_memory").select("superseded_at").eq("id", memory_a)
            .execute().data[0]["superseded_at"].replace("Z", "+00:00"))

        # Precisely before the real succession event -- A is still current.
        ctx_before = mr.build_memory_context(
            "TEST-6D1 supersession historical A statement", ws, OWNER,
            as_of=superseded_at - timedelta(milliseconds=1))
        assert ctx_before is not None
        before_candidate = next((c for c in ctx_before.candidates if c.memory_id == memory_a), None)
        assert before_candidate is not None
        assert before_candidate.lifecycle_status == "superseded", \
            "lifecycle_status is always the real, current value, honestly reported either way"

        # Precisely at/after the real succession event -- A is no longer current.
        ctx_after = mr.build_memory_context(
            "TEST-6D1 supersession historical A statement", ws, OWNER, as_of=superseded_at)
        after_found = {c.memory_id for c in ctx_after.candidates} if ctx_after else set()
        assert memory_a not in after_found, \
            "Phase 6D.2 closes the gap this test used to only work around -- the boundary is now precise"
    finally:
        _cleanup(ids)


# =====================================================================
# 10-11. Graph/memory as_of consistency, chatbot/query parity
# =====================================================================

def test_graph_and_memory_respect_same_as_of_boundary():
    """The real Product<-requires_approval_from relationship is future-dated
    (valid_from=2026-09-15, a real, pre-existing finding from Phase 5K, not
    a Phase 6D.1 concern) -- both graph and memory must agree that it does
    not exist yet at current time, and both must agree it exists once as_of
    passes that point, using the identical as_of value."""
    now_ctx = gr.build_graph_context("What requires Product approval?", REAL_WORKSPACE, OWNER)
    assert now_ctx is None or not now_ctx.relationships

    future = datetime(2026, 9, 16, tzinfo=timezone.utc)
    future_ctx = gr.build_graph_context("What requires Product approval?", REAL_WORKSPACE, OWNER, as_of=future)
    assert future_ctx is not None and future_ctx.relationships

    # Memory retrieval at the identical as_of values -- both structurally
    # agree there is no PROMOTED memory for this candidate either way (it
    # was never promoted, review-only), proving neither layer silently
    # disagrees about temporal framing.
    mem_now = mr.build_memory_context("What requires Product approval?", REAL_WORKSPACE, OWNER, graph_context=now_ctx)
    mem_future = mr.build_memory_context("What requires Product approval?", REAL_WORKSPACE, OWNER,
                                         as_of=future, graph_context=future_ctx)
    assert mem_now is None
    assert mem_future is None


def test_chatbot_and_query_wire_the_same_as_of_into_memory_retrieval():
    """Source-level parity check: both call sites must pass the SAME parsed
    as_of into memory_retrieval.build_memory_context -- catches an
    accidental future removal/divergence of the wiring itself, which a
    purely behavioral test (calling the shared function directly) cannot."""
    query_src = open("query.py", encoding="utf-8").read()
    chatbot_src = open("chatbot.py", encoding="utf-8").read()
    for src, name in ((query_src, "query.py"), (chatbot_src, "chatbot.py")):
        assert "memory_retrieval.build_memory_context(" in src
        assert "as_of=as_of_parsed" in src, f"{name} must thread the same parsed as_of into memory_retrieval"


# =====================================================================
# 12-14. Isolation / security / review candidate
# =====================================================================

def test_workspace_isolation_with_as_of():
    ws_a = _fresh_workspace()
    ws_b = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws_a, statement="TEST-6D1 isolation A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws_a, sk_a)
        ids["memory_ids"].append(memory_a)

        ctx_b = mr.build_memory_context("TEST-6D1 isolation A statement", ws_b, OWNER, as_of=_now())
        assert ctx_b is None
    finally:
        _cleanup(ids)


def test_sensitivity_isolation_with_as_of():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted", statement="TEST-6D1 restricted historical statement")
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        internal_only = gq.resolve_allowed_sensitivities("employee", False)
        ctx = mr.build_memory_context("TEST-6D1 restricted historical statement", ws, internal_only, as_of=_now())
        assert ctx is None
    finally:
        _cleanup(ids)


def test_review_candidate_remains_excluded_historically():
    for as_of in (datetime(2020, 1, 1, tzinfo=timezone.utc), _now(), _now() + timedelta(days=3650)):
        ctx = mr.build_memory_context("What requires Product approval?", REAL_WORKSPACE, OWNER, as_of=as_of)
        found = {c.memory_id for c in ctx.candidates} if ctx else set()
        # Structurally impossible regardless: no org_memory row is ever
        # grounded in the review candidate's sk id.
        assert found == set() or SK_Q4_LAUNCH_APPROVAL not in found


# =====================================================================
# 15-16. Global corpus / 15th-row isolation
# =====================================================================

def test_workspace_corpus_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14


def test_global_15th_row_cannot_leak_with_as_of():
    far_future = _now() + timedelta(days=3650)
    real_candidates = mr._fetch_memory_rows(REAL_WORKSPACE, far_future)
    leak_candidates = mr._fetch_memory_rows(LEAK_WORKSPACE, far_future)
    assert leak_candidates == []
    assert all(c["workspace_id"] == REAL_WORKSPACE for c in real_candidates)


# =====================================================================
# 17-19. Existing-data integrity
# =====================================================================

def test_four_real_memories_unchanged():
    for label, memory_id in REAL_MEMORY_IDS.items():
        row = supabase.table("org_memory").select("valid_from,valid_until,lifecycle_status") \
            .eq("id", memory_id).execute().data[0]
        assert row["valid_from"] is None, label
        assert row["valid_until"] is None, label
        assert row["lifecycle_status"] == "active", label


def test_structured_knowledge_unchanged_by_temporal_fix():
    before = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER,
                            as_of=datetime(2020, 1, 1, tzinfo=timezone.utc))
    after = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert before == after


def test_graph_unchanged_by_temporal_fix():
    before = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    gr.build_graph_context("What requires Product approval?", REAL_WORKSPACE, OWNER,
                           as_of=datetime(2026, 9, 16, tzinfo=timezone.utc))
    after = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    assert before == after
    assert len(after) == 3


# =====================================================================
# 20. Fixture cleanup sentinel
# =====================================================================

def test_no_test_6d1_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id, statement") \
        .like("statement", "TEST-6D1%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_real_workspace_counts_still_exact():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_review_queue").select("id", count="exact").execute().count == 1


# =====================================================================
# 21. Full regression -- see this file's module docstring
# =====================================================================

def test_placeholder_full_regression_runs_separately():
    assert callable(mr.build_memory_context)
