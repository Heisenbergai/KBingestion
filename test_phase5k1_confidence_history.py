"""
Phase 5K.1 Confidence + Historical Chatbot tests -- verifies (1) graph
candidates carry NO fake `similarity` value and instead expose a dedicated,
honestly-labeled graph confidence signal (graph_confidence/combine_confidence
in graph_retrieval.py), and (2) chatbot.run_rag_query() now accepts and
propagates an explicit `as_of` all the way to graph_query's real temporal
filter, identically to query.py's /query.

This environment has no live AWS Bedrock credentials (confirmed:
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY unset) -- ai.embed_texts()/ai.chat()
are monkeypatched (dummy 1024-dim embedding, canned chat reply) so
chatbot.run_rag_query() can be exercised fully and honestly end to end; every
Supabase/graph call is real and live, matching the precedent already
established in test_phase5k_graph_retrieval_completion.py.

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5k1_confidence_history.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import chatbot
import graph_query as gq
import graph_retrieval as gr
from query import supabase, build_context_and_citations

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_ENTITY_ID = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
JOHN_SNOW_ENTITY_ID = "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d"
PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"

REAL_NOTE_SOURCE_7A9EAA34 = "2419c928-5bf1-4b13-bdd6-f1a2b88b1bfb"
REAL_NOTE_ID_7A9EAA34 = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"

OWNER_SENSITIVITIES = gq.resolve_allowed_sensitivities("owner", False)
EMPLOYEE_SENSITIVITIES = gq.resolve_allowed_sensitivities("employee", False)

_DUMMY_EMBEDDING = [0.0] * 1024
FUTURE_AS_OF = "2026-09-16T00:00:00Z"


def _cleanup(ids: dict):
    if ids.get("relationship_id"):
        supabase.table("knowledge_relationships").delete().eq("id", ids["relationship_id"]).execute()
    for key in ("public_sk", "restricted_sk", "derived_sk"):
        if ids.get(key):
            supabase.table("structured_knowledge").delete().eq("id", ids[key]).execute()
    for key in ("src_entity", "tgt_entity"):
        if ids.get(key):
            supabase.table("knowledge_entities").delete().eq("id", ids[key]).execute()


def _make_status_fixture(status: str, valid_from, valid_until=None) -> dict:
    ids = {}
    ids["src_entity"] = supabase.table("knowledge_entities").insert({
        "workspace_id": REAL_WORKSPACE, "entity_type": "department",
        "canonical_label": f"TEST-5K1-{status.upper()}-SRC", "status": "active",
    }).execute().data[0]["id"]
    ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
        "workspace_id": REAL_WORKSPACE, "entity_type": "department",
        "canonical_label": f"TEST-5K1-{status.upper()}-TGT", "status": "active",
    }).execute().data[0]["id"]
    rel = supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": REAL_WORKSPACE,
        "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
        "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
        "p_relationship_type": "references", "p_rationale": f"synthetic 5K1 {status} fixture",
        "p_confidence": None, "p_valid_from": valid_from, "p_valid_until": valid_until,
        "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                        "stance": "supports", "captured_at": valid_from}],
    }).execute().data
    ids["relationship_id"] = rel
    if status != "active":
        supabase.table("knowledge_relationships").update({"status": status}).eq("id", rel).execute()
    return ids


def _run_chatbot(question, filter_sensitivities=None, as_of=None):
    bot = chatbot.BotConfig(id="test-bot", name="Test Bot", workspace_id=REAL_WORKSPACE)
    captured = {}

    def fake_chat(messages, system=None, **kwargs):
        captured["system"] = system
        return "canned answer"

    with patch("ai.embed_texts", return_value=[_DUMMY_EMBEDDING]), \
         patch("ai.chat", side_effect=fake_chat):
        answer, sources, confidence, corroboration = chatbot.run_rag_query(
            question, bot, filter_sensitivities=filter_sensitivities or OWNER_SENSITIVITIES,
            as_of=as_of,
        )
    return confidence, captured.get("system", "")


# =====================================================================
# 1. Graph candidate has no fake similarity
# =====================================================================

def test_graph_candidate_similarity_is_always_none():
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, metrics = gr.merge_graph_context_into_chunks([], ctx)
    assert metrics["graph_candidates_added"] >= 1
    graph_candidates = [c for c in merged if c["source_type"] == "graph_relationship"]
    assert graph_candidates
    for c in graph_candidates:
        assert c["similarity"] is None
        assert c["evidence_strength"] in ("primary", "derived")


# =====================================================================
# 2. Real vector similarity remains untouched
# =====================================================================

def test_real_chunk_similarity_passes_through_merge_unchanged():
    real_chunk = {
        "id": "c1", "document_id": "doc-1", "content": "text",
        "metadata": {"file_name": "f.txt"}, "source_type": "slack",
        "source_tier": 3, "similarity": 0.62,
    }
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, _ = gr.merge_graph_context_into_chunks([real_chunk], ctx)
    kept = next(c for c in merged if c["id"] == "c1")
    assert kept["similarity"] == 0.62


# =====================================================================
# 3. Graph-only deterministic answer -> HIGH
# =====================================================================

def test_graph_only_deterministic_answer_is_high():
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert gr.combine_confidence("none", ctx) == "high"
    assert gr.combine_confidence("low", ctx) == "high"


# =====================================================================
# 4. Weak graph context does not become HIGH
# =====================================================================

def test_weak_or_empty_graph_context_never_upgrades_confidence():
    # None context (non-graph-relevant question)
    assert gr.combine_confidence("low", None) == "low"
    # Real entity, but zero CURRENT relationships (Product, future-dated)
    ctx = gr.build_graph_context("What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert ctx is not None and ctx.relationships == []
    assert gr.combine_confidence("low", ctx) == "low"
    assert gr.combine_confidence("medium", ctx) == "medium"


# =====================================================================
# 5. Mixed graph + vector does not distort similarity
# =====================================================================

def test_mixed_retrieval_top_sim_unaffected_by_graph_candidate():
    strong_chunk = {
        "id": "c1", "document_id": "doc-1", "content": "text",
        "metadata": {"file_name": "f.txt"}, "source_type": "slack",
        "source_tier": 3, "similarity": 0.9,
    }
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, _ = gr.merge_graph_context_into_chunks([strong_chunk], ctx)
    top_sim = max((c.get("similarity") or 0) for c in merged)
    assert top_sim == 0.9, "the real chunk's similarity must remain the max -- no graph value can outrank or blend with it"
    # And the graph candidate(s) present don't collide with or overwrite it
    assert any(c["source_type"] == "graph_relationship" and c["similarity"] is None for c in merged)


# =====================================================================
# 6 & 7. Chatbot historical as_of
# =====================================================================

def test_chatbot_current_excludes_future_dated_product_relationship():
    confidence, system_text = _run_chatbot("What relationships does Product have?")
    assert "requires_approval_from" not in system_text


def test_chatbot_historical_as_of_reveals_product_relationship():
    confidence, system_text = _run_chatbot("What relationships does Product have?", as_of=FUTURE_AS_OF)
    assert "requires_approval_from" in system_text
    assert "Product" in system_text


# =====================================================================
# 8. Current excludes superseded/contradicted/retracted (reconfirmed
# through the shared graph_query layer, which chatbot/query.py both rely on)
# =====================================================================

def test_current_excludes_all_non_active_statuses():
    for status in ("superseded", "contradicted", "retracted"):
        ids = _make_status_fixture(status, datetime.now(timezone.utc).isoformat())
        try:
            ge = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
            assert ge.outbound_relationships == [], f"{status} must be excluded from a current read"
        finally:
            _cleanup(ids)


# =====================================================================
# 9. Historical respects temporal validity (through the chatbot path too)
# =====================================================================

def test_chatbot_historical_as_of_respects_temporal_window_for_superseded():
    past_from = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    past_until = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    ids = _make_status_fixture("superseded", past_from, past_until)
    try:
        label = ids and supabase.table("knowledge_entities").select("canonical_label") \
            .eq("id", ids["src_entity"]).execute().data[0]["canonical_label"]
        as_of_during = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()

        _, current_text = _run_chatbot(f"Who owns {label}?", as_of=None)
        assert label not in current_text or "references" not in current_text

        _, historical_text = _run_chatbot(f"Who owns {label}?", as_of=as_of_during)
        assert label in historical_text and "references" in historical_text
    finally:
        _cleanup(ids)


# =====================================================================
# 10. /query and chatbot semantics match
# =====================================================================

def test_query_and_chatbot_agree_on_product_temporal_state():
    query_ctx_current = gr.build_graph_context("What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    query_ctx_future = gr.build_graph_context(
        "What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
        as_of=datetime(2026, 9, 16, tzinfo=timezone.utc),
    )
    assert query_ctx_current.relationships == []
    assert any(r.relationship_type == "requires_approval_from" for r in query_ctx_future.relationships)

    _, chatbot_current_text = _run_chatbot("What relationships does Product have?")
    _, chatbot_future_text = _run_chatbot("What relationships does Product have?", as_of=FUTURE_AS_OF)
    assert "requires_approval_from" not in chatbot_current_text
    assert "requires_approval_from" in chatbot_future_text


# =====================================================================
# 11. Graph evidence remains citable
# =====================================================================

def test_graph_evidence_remains_citable():
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    _, citations = build_context_and_citations(merged)
    assert any(c["source_type"] == "graph_relationship" for c in citations)


# =====================================================================
# 12. Workspace isolation remains intact
# =====================================================================

def test_workspace_isolation_intact():
    resolved = gr.resolve_entity_mentions("Who owns Product?", OTHER_REAL_WORKSPACE)
    assert resolved == []


# =====================================================================
# 13. Restricted evidence remains hidden
# =====================================================================

def test_restricted_evidence_remains_hidden():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K1-RESTRICTED-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K1-RESTRICTED-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["restricted_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5K1-RESTRICTED statement text",
            "raw_subject_phrase": "TEST-5K1", "sensitivity": "restricted", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5k1-restricted-{uuid.uuid4()}",
        }).execute().data[0]["id"]
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5K1 restricted fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        low_ctx = gr.build_graph_context("Who owns TEST-5K1-RESTRICTED-SRC?", REAL_WORKSPACE, EMPLOYEE_SENSITIVITIES)
        merged, _ = gr.merge_graph_context_into_chunks([], low_ctx)
        combined = "\n".join(c["content"] for c in merged)
        assert "TEST-5K1-RESTRICTED statement text" not in combined
    finally:
        _cleanup(ids)


# =====================================================================
# 14 & 15. Workspace corpus safety
# =====================================================================

def test_workspace_structured_knowledge_is_14_rows():
    count = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count
    assert count == 14


def test_unrelated_15th_row_cannot_leak():
    global_count = supabase.table("structured_knowledge").select("id", count="exact").execute().count
    assert global_count == 15
    resolved = gr.resolve_entity_mentions("Who owns 88994448877?", REAL_WORKSPACE)
    assert resolved == []


# =====================================================================
# 16-18. Existing state unchanged
# =====================================================================

def test_existing_entities_unchanged():
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count
    assert count == 5


def test_existing_real_relationships_unchanged():
    rows = supabase.table("knowledge_relationships").select("relationship_type,status") \
        .eq("workspace_id", REAL_WORKSPACE) \
        .not_.like("rationale", "%synthetic 5K%").execute().data
    assert {r["relationship_type"] for r in rows} == {"requires_approval_from", "organized", "attended"}
    assert all(r["status"] == "active" for r in rows)


def test_structured_knowledge_counts_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15


# =====================================================================
# 19. Fixture cleanup sentinel
# =====================================================================

def test_no_test_5k1_entities_leaked():
    leaked = supabase.table("knowledge_entities").select("id,canonical_label") \
        .like("canonical_label", "TEST-5K1-%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_no_test_5k1_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-5K1%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


# =====================================================================
# 20. Full-state regression sentinel (the real full pytest run across every
# test_phase5*.py file remains the authoritative regression gate)
# =====================================================================

def test_full_known_state_after_phase_5k1():
    assert supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 5
    assert supabase.table("knowledge_relationships").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 3
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count == 1
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14
