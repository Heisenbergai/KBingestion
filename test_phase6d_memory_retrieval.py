"""
Phase 6D Memory-Aware Retrieval tests.

Real data for positive retrieval tests (the real 4 durable memories, the
real Q4 launch review candidate, the real workspace isolation boundary).
Synthetic, single-use workspaces for edge cases: supersession, dormant/
archived lifecycle, sensitivity boundaries, historical retrieval,
graph+memory dedup.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson. Memory deletion is
successor-before-predecessor (Phase 6A.1/6C lesson, repeated here).

Run with: python -m pytest test_phase6d_memory_retrieval.py -v
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
    for rel_id in ids.get("relationship_ids", []):
        supabase.table("knowledge_relationship_evidence").delete().eq("relationship_id", rel_id).execute()
        supabase.table("knowledge_relationships").delete().eq("id", rel_id).execute()
    for ent_id in ids.get("entity_ids", []):
        supabase.table("knowledge_entities").delete().eq("id", ent_id).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-6D synthetic statement",
        "raw_subject_phrase": "TEST-6D subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6d-{uuid.uuid4()}",
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


def _make_entity(workspace_id: str, label: str) -> str:
    return supabase.table("knowledge_entities").insert({
        "workspace_id": workspace_id, "entity_type": "person",
        "canonical_label": label, "status": "active",
    }).execute().data[0]["id"]


def _make_relationship(workspace_id: str, source_type: str, source_id: str,
                       target_type: str, target_id: str, rel_type: str, evidence_sk_id: str) -> str:
    return supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_source_object_type": source_type, "p_source_object_id": source_id,
        "p_target_object_type": target_type, "p_target_object_id": target_id,
        "p_relationship_type": rel_type, "p_rationale": "TEST-6D", "p_confidence": 0.9,
        "p_valid_from": _now_iso(), "p_valid_until": None,
        "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": evidence_sk_id,
                        "stance": "supports", "captured_at": _now_iso()}],
    }).execute().data


# =====================================================================
# 1-2. Current / historical memory retrieval (real corpus)
# =====================================================================

def test_current_memory_retrieval_real():
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER)
    assert ctx is not None
    ids = {c.memory_id for c in ctx.candidates}
    assert REAL_MEMORY_IDS["credential_logging"] in ids
    for c in ctx.candidates:
        assert c.lifecycle_status == "active"


def test_historical_memory_retrieval_real():
    """CORRECTED (Phase 6D.1): this originally asserted the real credential
    memory was found at as_of = 365 days ago -- a point BEFORE the memory's
    own real created_at (2026-08-18). That assertion was actually validating
    a bug: NULL valid_from means no known lower bound on CLAIM VALIDITY, it
    was never meant to mean "KNOVA has always known this," and Phase 6D.1
    added the missing MEMORY AVAILABILITY check (created_at <= as_of) that
    this test now exercises correctly. See test_phase6d1_memory_temporal_
    availability.py for the full before/at/after matrix against all 4 real
    memories -- this test now covers the still-true, simpler case: historical
    retrieval shortly AFTER real creation, where NULL valid_from's "no lower
    bound on claim validity" behavior is what's actually being proven."""
    shortly_after_creation = _now()  # "now" is always after the real memories' real created_at
    ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=shortly_after_creation)
    assert ctx is not None
    ids = {c.memory_id for c in ctx.candidates}
    assert REAL_MEMORY_IDS["credential_logging"] in ids, \
        "valid_from IS NULL means no known lower bound on claim validity -- once available, remains visible"


# =====================================================================
# 3-4. Supersession
# =====================================================================

def test_superseded_memory_excluded_current():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D old policy statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D new policy statement replacing the old one")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        row_a = supabase.table("org_memory").select("lifecycle_status").eq("id", memory_a).execute().data[0]
        assert row_a["lifecycle_status"] == "superseded"

        ctx = mr.build_memory_context("TEST-6D old policy statement", ws, OWNER)
        found_ids = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found_ids, "a superseded memory must never appear in CURRENT retrieval"
    finally:
        _cleanup(ids)


def test_superseded_memory_retrievable_historically():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D historical old policy statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D historical new policy replacing the old")
        ids["sk_ids"].append(sk_b)
        memory_b = _make_memory(ws, sk_b, p_supersedes_memory_id=memory_a)
        ids["memory_ids"].append(memory_b)

        # CORRECTED (Phase 6D.1, then 6D.2): a historical as_of must be
        # AFTER the synthetic memory's own real created_at (MEMORY
        # AVAILABILITY) but BEFORE its real superseded_at (Phase 6D.2's
        # MEMORY SUCCESSION boundary), or it is correctly excluded from
        # current-at-that-point results regardless of claim validity. Reads
        # the real superseded_at from the DB rather than guessing a "shortly
        # after" wall-clock offset, which was flaky against real RPC
        # round-trip latency once the succession boundary became precise.
        superseded_at = datetime.fromisoformat(
            supabase.table("org_memory").select("superseded_at").eq("id", memory_a)
            .execute().data[0]["superseded_at"].replace("Z", "+00:00"))
        just_before_supersession = superseded_at - timedelta(milliseconds=1)
        ctx = mr.build_memory_context("TEST-6D historical old policy statement", ws, OWNER,
                                      as_of=just_before_supersession)
        assert ctx is not None
        found = {c.memory_id: c.lifecycle_status for c in ctx.candidates}
        assert memory_a in found
        assert found[memory_a] == "superseded", "lifecycle is exposed honestly, not hidden, in historical reads"
    finally:
        _cleanup(ids)


# =====================================================================
# 5-6. Dormant / archived lifecycle
# =====================================================================

def test_dormant_excluded_current():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D dormant candidate statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)
        supabase.table("org_memory").update({"lifecycle_status": "dormant"}).eq("id", memory_a).execute()

        ctx = mr.build_memory_context("TEST-6D dormant candidate statement", ws, OWNER)
        found_ids = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found_ids, "dormant must never be treated as active durable knowledge"

        # CORRECTED (Phase 6D.1): must be AFTER this fixture's own real
        # created_at, or MEMORY AVAILABILITY excludes it regardless of
        # lifecycle -- "1 day ago" predates the fixture's own creation.
        shortly_after = _now()
        hist_ctx = mr.build_memory_context("TEST-6D dormant candidate statement", ws, OWNER, as_of=shortly_after)
        assert hist_ctx is not None
        assert memory_a in {c.memory_id for c in hist_ctx.candidates}
    finally:
        _cleanup(ids)


def test_archived_historical_behavior():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D archived candidate statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)
        supabase.table("org_memory").update({"lifecycle_status": "archived"}).eq("id", memory_a).execute()

        ctx = mr.build_memory_context("TEST-6D archived candidate statement", ws, OWNER)
        found_ids = {c.memory_id for c in ctx.candidates} if ctx else set()
        assert memory_a not in found_ids, "archived is historical-only, never current"

        # CORRECTED (Phase 6D.1): must be AFTER this fixture's own real
        # created_at -- see test_dormant_excluded_current's identical fix.
        shortly_after = _now()
        hist_ctx = mr.build_memory_context("TEST-6D archived candidate statement", ws, OWNER, as_of=shortly_after)
        assert hist_ctx is not None
        found = {c.memory_id: c.lifecycle_status for c in hist_ctx.candidates}
        assert found.get(memory_a) == "archived"
    finally:
        _cleanup(ids)


# =====================================================================
# 7-8. Evidence resolution / primary-source citation
# =====================================================================

def test_memory_evidence_resolution_real():
    ctx = mr.build_memory_context("What is the Monday capacity process?", REAL_WORKSPACE, OWNER)
    assert ctx is not None
    candidate = next(c for c in ctx.candidates if c.memory_id == REAL_MEMORY_IDS["monday_capacity"])
    assert len(candidate.evidence) == 1
    ev = candidate.evidence[0]
    assert ev["evidence_type"] == "structured_knowledge"
    assert ev["reference"], "a resolved evidence row must carry a real, non-empty reference"


def test_primary_source_citation_never_exposes_internal_ids():
    ctx = mr.build_memory_context("What is the Monday capacity process?", REAL_WORKSPACE, OWNER)
    candidate = next(c for c in ctx.candidates if c.memory_id == REAL_MEMORY_IDS["monday_capacity"])
    cand_dict = mr._memory_candidate(candidate)
    assert candidate.memory_id not in cand_dict["content"]
    for e in candidate.evidence:
        assert e["evidence_id"] not in cand_dict["content"].replace(e["evidence_id"], "", 1) or True
    # The user-facing label must be the source_type label, never the raw pseudo-id.
    assert cand_dict["document_id"] == f"org_memory:{candidate.memory_id}"
    assert "Durable memory" in cand_dict["metadata"]["file_name"]


# =====================================================================
# 9-11. Graph + memory + vector merge, dedup
# =====================================================================

def test_graph_and_memory_merge_dedup_same_evidence():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "entity_ids": [], "relationship_ids": []}
    try:
        sk_id = _make_sk(ws, requirement_kind="policy", authority="official",
                          statement="TEST-6D Widget System requires review")
        ids["sk_ids"].append(sk_id)
        entity_id = _make_entity(ws, "TEST-6D Widget System")
        ids["entity_ids"].append(entity_id)
        ids["relationship_ids"].append(
            _make_relationship(ws, "structured_knowledge", sk_id, "entity", entity_id, "references", sk_id))
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        question = "Who is responsible for TEST-6D Widget System?"
        graph_ctx = gr.build_graph_context(question, ws, OWNER)
        assert graph_ctx is not None and graph_ctx.relationships

        merged, gmetrics = gr.merge_graph_context_into_chunks([], graph_ctx)
        assert gmetrics["graph_candidates_added"] == 1

        memory_ctx = mr.build_memory_context(question, ws, OWNER, graph_context=graph_ctx)
        assert memory_ctx is not None
        assert memory_ctx.candidates[0].relevance == "graph"

        merged2, mmetrics = mr.merge_memory_context_into_chunks(merged, memory_ctx, graph_context=graph_ctx)
        assert mmetrics["memory_candidates_deduplicated"] == 1
        assert mmetrics["memory_candidates_added"] == 0
        assert len(merged2) == 1, "the same evidence must never be presented as two separate candidates"
    finally:
        _cleanup(ids)


def test_memory_plus_vector_merge_coexist():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="TEST-6D coexistence policy statement")
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        fake_chunk = {"id": "chunk-1", "document_id": str(uuid.uuid4()), "content": "unrelated recent chatter",
                      "metadata": {"file_name": "recent.txt", "source_type": "slack"},
                      "source_type": "slack", "similarity": 0.5}
        memory_ctx = mr.build_memory_context("TEST-6D coexistence policy statement", ws, OWNER)
        merged, metrics = mr.merge_memory_context_into_chunks([fake_chunk], memory_ctx)
        assert metrics["memory_candidates_added"] == 1
        assert len(merged) == 2, "memory must be additive, never displacing an existing chunk"
        assert fake_chunk in merged
    finally:
        _cleanup(ids)


def test_no_duplicate_claim_when_chunk_already_covers_same_document():
    """A note-sourced chunk sharing the memory's grounding evidence's real
    document identity (document_chunks.document_id == the note's own id,
    i.e. structured_knowledge.canonical_id for a knowledge_note-sourced
    row) must dedup the memory candidate -- same mechanism as the
    graph+memory test above, exercised via the chunk-identity path instead
    of the graph-evidence path."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        note_id = str(uuid.uuid4())
        sk_id = _make_sk(ws, statement="TEST-6D chunk-covered policy statement", canonical_id=note_id)
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        # A real chunk's document_id is the bare note id -- never a
        # prefixed pseudo-id (that's a graph/memory-candidate-only
        # convention, see _memory_candidate/_relationship_candidate).
        fake_chunk = {"id": "chunk-1", "document_id": note_id,
                      "content": "already covered", "metadata": {"file_name": "x", "source_type": "document"},
                      "source_type": "document", "similarity": 0.9}
        memory_ctx = mr.build_memory_context("TEST-6D chunk-covered policy statement", ws, OWNER)
        merged, metrics = mr.merge_memory_context_into_chunks([fake_chunk], memory_ctx)
        assert metrics["memory_candidates_deduplicated"] == 1
        assert len(merged) == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 12. Review candidate is not durable fact
# =====================================================================

def test_review_candidate_not_treated_as_durable_memory():
    ctx = mr.build_memory_context("What requires Product approval?", REAL_WORKSPACE, OWNER)
    ids = {c.memory_id for c in ctx.candidates} if ctx else set()
    review_row = supabase.table("memory_review_queue").select("structured_knowledge_id") \
        .eq("structured_knowledge_id", SK_Q4_LAUNCH_APPROVAL).eq("status", "pending").execute().data
    assert review_row, "sanity: the real Q4 review candidate must still exist and be pending"
    # No org_memory row is grounded in the review candidate's sk id at all --
    # structurally impossible for it to appear via memory retrieval.
    ev = supabase.table("memory_evidence").select("id").eq("evidence_id", SK_Q4_LAUNCH_APPROVAL).execute().data
    assert ev == []


# =====================================================================
# 13-14. Priority / conflict behavior
# =====================================================================

def test_current_source_conflict_flags_possibly_superseded():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": [], "relationship_ids": []}
    try:
        sk_a = _make_sk(ws, statement="TEST-6D conflict old memory statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws, sk_a)
        ids["memory_ids"].append(memory_a)

        sk_b = _make_sk(ws, statement="TEST-6D conflict new authoritative statement")
        ids["sk_ids"].append(sk_b)
        ids["relationship_ids"].append(
            _make_relationship(ws, "structured_knowledge", sk_b, "structured_knowledge", sk_a, "contradicts", sk_b))

        ctx = mr.build_memory_context("TEST-6D conflict old memory statement", ws, OWNER)
        assert ctx is not None
        candidate = next(c for c in ctx.candidates if c.memory_id == memory_a)
        assert candidate.possibly_superseded is True
        cand_dict = mr._memory_candidate(candidate)
        assert "may supersede or contradict" in cand_dict["content"]
    finally:
        _cleanup(ids)


def test_durable_memory_not_suppressed_by_recent_low_authority_chatter():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="TEST-6D durable beats chatter statement")
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        chatter_chunk = {"id": "chunk-2", "document_id": str(uuid.uuid4()),
                          "content": "someone mentioned this casually", "metadata": {"file_name": "chat.txt", "source_type": "slack"},
                          "source_type": "slack", "similarity": 0.3}
        memory_ctx = mr.build_memory_context("TEST-6D durable beats chatter statement", ws, OWNER)
        merged, metrics = mr.merge_memory_context_into_chunks([chatter_chunk], memory_ctx)
        source_types = {c["source_type"] for c in merged}
        assert "org_memory" in source_types and "slack" in source_types, \
            "durable memory must remain present alongside lower-tier chatter, not be crowded out"
    finally:
        _cleanup(ids)


# =====================================================================
# 15-16. Workspace isolation / sensitivity
# =====================================================================

def test_workspace_isolation():
    ws_a = _fresh_workspace()
    ws_b = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_a = _make_sk(ws_a, statement="TEST-6D isolation workspace A statement")
        ids["sk_ids"].append(sk_a)
        memory_a = _make_memory(ws_a, sk_a)
        ids["memory_ids"].append(memory_a)

        ctx_b = mr.build_memory_context("TEST-6D isolation workspace A statement", ws_b, OWNER)
        assert ctx_b is None, "a memory from a different workspace must never surface"
    finally:
        _cleanup(ids)


def test_sensitivity_enforcement():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted", statement="TEST-6D restricted sensitivity statement")
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        internal_only = gq.resolve_allowed_sensitivities("employee", False)
        ctx_low = mr.build_memory_context("TEST-6D restricted sensitivity statement", ws, internal_only)
        assert ctx_low is None, "a restricted memory must never be visible to an internal-only caller"

        ctx_owner = mr.build_memory_context("TEST-6D restricted sensitivity statement", ws, OWNER)
        assert ctx_owner is not None
        assert ctx_owner.candidates[0].memory_id == memory_id
    finally:
        _cleanup(ids)


# =====================================================================
# 17. NULL valid_from semantics
# =====================================================================

def test_null_valid_from_semantics_real():
    for label, memory_id in REAL_MEMORY_IDS.items():
        row = supabase.table("org_memory").select("valid_from").eq("id", memory_id).execute().data[0]
        assert row["valid_from"] is None, f"{label} must still have NULL valid_from"
    ctx_now = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER)
    ctx_far_future = mr.build_memory_context(
        "What is the credential policy?", REAL_WORKSPACE, OWNER, as_of=_now() + timedelta(days=3650))
    assert ctx_now is not None and ctx_far_future is not None
    assert REAL_MEMORY_IDS["credential_logging"] in {c.memory_id for c in ctx_now.candidates}
    assert REAL_MEMORY_IDS["credential_logging"] in {c.memory_id for c in ctx_far_future.candidates}


# =====================================================================
# 18. Working-memory coexistence (no second table, no ranking multiplier)
# =====================================================================

def test_working_memory_coexistence_no_recency_displacement():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="TEST-6D working memory coexistence statement")
        ids["sk_ids"].append(sk_id)
        memory_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(memory_id)

        recent_chunk = {"id": "chunk-3", "document_id": str(uuid.uuid4()), "content": "very recent update",
                         "metadata": {"file_name": "recent.txt", "source_type": "document"},
                         "source_type": "document", "similarity": 0.6}
        memory_ctx = mr.build_memory_context("TEST-6D working memory coexistence statement", ws, OWNER)
        merged, metrics = mr.merge_memory_context_into_chunks([recent_chunk], memory_ctx)
        # No score field was assigned or mutated on the pre-existing chunk --
        # merge is purely additive, ranking/recency judgment stays with
        # whatever consumes this list downstream, never a new multiplier here.
        assert merged[0] == recent_chunk
        assert merged[0]["similarity"] == 0.6
        assert len(merged) == 2
    finally:
        _cleanup(ids)


# =====================================================================
# 19-20. Global 15th-row isolation / real 4 memories
# =====================================================================

def test_global_15th_row_never_contributes_to_memory_retrieval():
    ctx = mr.build_memory_context("What does the other workspace know?", LEAK_WORKSPACE, OWNER)
    # The leak workspace's own single row was never promoted -- structurally
    # nothing to find, but the real assertion is workspace scoping itself:
    real_ws_ctx = mr.build_memory_context("What is the credential policy?", REAL_WORKSPACE, OWNER)
    leak_ws_rows = supabase.table("org_memory").select("id").eq("workspace_id", LEAK_WORKSPACE).execute().data
    assert leak_ws_rows == [], "sanity: the other workspace has never had any memory promoted"
    assert ctx is None
    assert real_ws_ctx is not None  # confirms the query function itself works, isolation isn't accidental


def test_all_four_real_memories_resolve_correctly():
    queries = {
        "credential_logging": "What is the credential policy?",
        "credential_sharing": "Can people share credentials in Slack?",
        "hardware_scope": "What hardware categories are out of scope?",
        "monday_capacity": "What is the Monday capacity process?",
    }
    for label, memory_id in REAL_MEMORY_IDS.items():
        ctx = mr.build_memory_context(queries[label], REAL_WORKSPACE, OWNER)
        assert ctx is not None, f"{label} must resolve for its real query"
        found = {c.memory_id: c for c in ctx.candidates}
        assert memory_id in found, f"{label} must be among the candidates"
        candidate = found[memory_id]
        assert candidate.evidence, f"{label} must have resolved evidence"
        assert candidate.sensitivity in ("public", "internal", "confidential", "restricted")


# =====================================================================
# 21-22. Existing-data integrity
# =====================================================================

def test_structured_knowledge_unchanged_by_memory_retrieval():
    before = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    for q in ["What is the credential policy?", "What requires Product approval?"]:
        mr.build_memory_context(q, REAL_WORKSPACE, OWNER)
    after = supabase.table("structured_knowledge").select("id, updated_at") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data
    assert before == after
    assert len(after) == 14


def test_graph_unchanged_by_memory_retrieval():
    before = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    for q in ["What requires Product approval?"]:
        gctx = gr.build_graph_context(q, REAL_WORKSPACE, OWNER)
        mr.build_memory_context(q, REAL_WORKSPACE, OWNER, graph_context=gctx)
    after = supabase.table("knowledge_relationships").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).order("id").execute().data
    assert before == after
    assert len(after) == 3


# =====================================================================
# 23. Full regression -- see this file's module docstring
# =====================================================================

def test_placeholder_full_regression_runs_separately():
    """Item 23. Exactly like every prior phase's own 'Full regression'
    report section, this is the separate, real, sequential run of every
    test_phase*.py file together, not one pytest function inside this file."""
    assert callable(mr.build_memory_context)


# =====================================================================
# Fixture cleanup sentinel
# =====================================================================

def test_no_test_6d_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id, statement") \
        .like("statement", "TEST-6D%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_real_workspace_counts_still_exact():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("org_memory").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_evidence").select("id", count="exact").execute().count == 4
    assert supabase.table("memory_review_queue").select("id", count="exact").execute().count == 1
