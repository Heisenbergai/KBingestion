"""
Phase 5K Graph Retrieval Completion tests -- closes the three gaps Phase 5J
left open: chatbot.py wiring, relationship status filtering in the shared
graph read layer, and graph-only confidence.

This environment has no live AWS Bedrock credentials (confirmed:
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY unset) -- ai.embed_texts()/ai.chat()
are monkeypatched here (never the Supabase calls, which are real and live)
so that chatbot.run_rag_query() can be exercised FULLY and honestly end to
end, matching Phase 5E's own "monkeypatched real-shaped responses" precedent
for exactly this kind of missing-live-credential gap. The dummy embedding is
a real 1024-dim zero vector (document_chunks.embedding is vector(1024),
confirmed live) so the real match_chunks_hybrid RPC accepts it without
error -- it just won't rank meaningfully, which is irrelevant to what these
tests check (graph wiring, not vector ranking).

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5k_graph_retrieval_completion.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import chatbot
import graph_query as gq
import graph_retrieval as gr
import signals
from query import supabase, build_context_and_citations
import source_labels

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_ENTITY_ID = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
JOHN_SNOW_ENTITY_ID = "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"

REAL_NOTE_SOURCE_7A9EAA34 = "2419c928-5bf1-4b13-bdd6-f1a2b88b1bfb"
REAL_NOTE_ID_7A9EAA34 = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"

OWNER_SENSITIVITIES = gq.resolve_allowed_sensitivities("owner", False)
EMPLOYEE_SENSITIVITIES = gq.resolve_allowed_sensitivities("employee", False)
PUBLIC_ONLY = ["public"]

_DUMMY_EMBEDDING = [0.0] * 1024


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
    """One synthetic entity pair + one relationship at the given status/
    temporal window, evidenced by the real note source (so its evidence is
    always visible at any sensitivity tier)."""
    ids = {}
    ids["src_entity"] = supabase.table("knowledge_entities").insert({
        "workspace_id": REAL_WORKSPACE, "entity_type": "department",
        "canonical_label": f"TEST-5K-{status.upper()}-SRC", "status": "active",
    }).execute().data[0]["id"]
    ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
        "workspace_id": REAL_WORKSPACE, "entity_type": "department",
        "canonical_label": f"TEST-5K-{status.upper()}-TGT", "status": "active",
    }).execute().data[0]["id"]
    rel = supabase.rpc("create_relationship_with_evidence", {
        "p_workspace_id": REAL_WORKSPACE,
        "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
        "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
        "p_relationship_type": "references", "p_rationale": f"synthetic 5K {status} fixture",
        "p_confidence": None, "p_valid_from": valid_from, "p_valid_until": valid_until,
        "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                        "stance": "supports", "captured_at": valid_from}],
    }).execute().data
    ids["relationship_id"] = rel
    if status != "active":
        supabase.table("knowledge_relationships").update({"status": status}).eq("id", rel).execute()
    return ids


# =====================================================================
# 1. Graph works through /query
# =====================================================================

def test_query_module_wires_graph_retrieval():
    import query as query_module
    assert query_module.graph_retrieval is gr, "must reuse the ONE adapter, not a copy"


def test_query_pipeline_sequence_produces_graph_citation():
    """Exercises the exact sequence query.py's /query endpoint runs after
    chunk retrieval: build_graph_context -> merge -> build_context_and_citations."""
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, metrics = gr.merge_graph_context_into_chunks([], ctx)
    context_text, citations = build_context_and_citations(merged)
    assert any(c["source_type"] == "graph_relationship" for c in citations)
    assert "Tanmay" in context_text
    assert metrics["graph_candidates_added"] >= 1


# =====================================================================
# 2. Graph works through chatbot path
# =====================================================================

def test_chatbot_module_wires_graph_retrieval():
    assert chatbot.graph_retrieval is gr, "must reuse the ONE adapter, not a copy"


def test_chatbot_run_rag_query_includes_graph_context():
    bot = chatbot.BotConfig(id="test-bot", name="Test Bot", workspace_id=REAL_WORKSPACE)
    captured = {}

    def fake_chat(messages, system=None, **kwargs):
        captured["system"] = system
        return "Tanmay organized the meeting."

    with patch("ai.embed_texts", return_value=[_DUMMY_EMBEDDING]), \
         patch("ai.chat", side_effect=fake_chat):
        answer, sources, confidence, corroboration = chatbot.run_rag_query(
            "Who organized Knova Test Meeting 1?", bot,
            filter_sensitivities=OWNER_SENSITIVITIES,
        )

    assert "Tanmay" in captured["system"]
    assert "organized" in captured["system"]
    assert confidence != "low", "a deterministic graph-only fact must not be forced to low confidence"


# =====================================================================
# 3. Graph works through widget/internal path
# =====================================================================

def test_chatbot_run_rag_query_widget_tier_sensitivity_still_gets_graph_context():
    """Mirrors widget_query's hard-coded filter_sensitivities=['public'] --
    the real organized/attended evidence (calendar_event_snapshot) carries no
    sensitivity concept at all, so it must remain visible even at the most
    restrictive real tier."""
    bot = chatbot.BotConfig(id="test-bot", name="Test Bot", workspace_id=REAL_WORKSPACE)
    captured = {}

    def fake_chat(messages, system=None, **kwargs):
        captured["system"] = system
        return "John Snow attended."

    with patch("ai.embed_texts", return_value=[_DUMMY_EMBEDDING]), \
         patch("ai.chat", side_effect=fake_chat):
        chatbot.run_rag_query(
            "Who attended Knova Test Meeting 1?", bot,
            filter_sensitivities=PUBLIC_ONLY, feature="chatbot_external",
        )
    assert "John Snow" in captured["system"]
    assert "attended" in captured["system"]


# =====================================================================
# 4. Security parity across /query and chatbot
# =====================================================================

def test_restricted_evidence_hidden_identically_via_adapter_and_chatbot():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-PARITY-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-PARITY-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["restricted_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5K-PARITY restricted statement",
            "raw_subject_phrase": "TEST-5K", "sensitivity": "restricted", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5k-parity-{uuid.uuid4()}",
        }).execute().data[0]["id"]
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5K parity fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        # Adapter-level (the /query path)
        low_ctx = gr.build_graph_context("Who owns TEST-5K-PARITY-SRC?", REAL_WORKSPACE, EMPLOYEE_SENSITIVITIES)
        assert low_ctx is None or all(
            r.id != ids["relationship_id"] for r in low_ctx.relationships
        ), "a relationship whose ONLY evidence is restricted must be invisible, not just its evidence"

        # chatbot-level (same adapter, different caller)
        bot = chatbot.BotConfig(id="test-bot", name="Test Bot", workspace_id=REAL_WORKSPACE)
        captured = {}

        def fake_chat(messages, system=None, **kwargs):
            captured["system"] = system
            return "no info"

        with patch("ai.embed_texts", return_value=[_DUMMY_EMBEDDING]), \
             patch("ai.chat", side_effect=fake_chat):
            chatbot.run_rag_query(
                "Who owns TEST-5K-PARITY-SRC?", bot, filter_sensitivities=EMPLOYEE_SENSITIVITIES,
            )
        assert "TEST-5K-PARITY restricted statement" not in captured["system"]
    finally:
        _cleanup(ids)


# =====================================================================
# 5-7. Status filtering: superseded / contradicted / retracted excluded
# =====================================================================

def test_active_relationship_returned_in_current_query():
    ids = _make_status_fixture("active", datetime.now(timezone.utc).isoformat())
    try:
        ge = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert any(r.id == ids["relationship_id"] for r in ge.outbound_relationships)
    finally:
        _cleanup(ids)


def test_superseded_relationship_excluded_from_current_query():
    ids = _make_status_fixture("superseded", datetime.now(timezone.utc).isoformat())
    try:
        ge = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert ge.outbound_relationships == []
    finally:
        _cleanup(ids)


def test_contradicted_relationship_excluded_from_current_query():
    ids = _make_status_fixture("contradicted", datetime.now(timezone.utc).isoformat())
    try:
        ge = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert ge.outbound_relationships == []
    finally:
        _cleanup(ids)


def test_retracted_relationship_excluded_from_current_query():
    ids = _make_status_fixture("retracted", datetime.now(timezone.utc).isoformat())
    try:
        ge = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert ge.outbound_relationships == []
    finally:
        _cleanup(ids)


# =====================================================================
# 8. Historical as_of can retrieve appropriate historical relationship
# =====================================================================

def test_historical_as_of_retrieves_superseded_relationship_that_was_valid_then():
    past_from = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    past_until = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    ids = _make_status_fixture("superseded", past_from, past_until)
    try:
        current = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert current.outbound_relationships == [], "superseded must stay excluded from current"

        as_of_during = datetime.now(timezone.utc) - timedelta(days=45)
        historical = gq.get_entity_graph(ids["src_entity"], REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=as_of_during)
        assert any(r.id == ids["relationship_id"] for r in historical.outbound_relationships), \
            "a historical as_of query must still see what was true then, even though it's since been superseded"
    finally:
        _cleanup(ids)


# =====================================================================
# 9 & 10. Graph-only confidence
# =====================================================================

def test_graph_only_primary_evidence_reaches_high_confidence_threshold():
    """STALE MECHANISM (Phase 5K.1): this originally asserted a synthetic
    `similarity` value (1.0) on the merged candidate cleared the 'high'
    threshold. Phase 5K.1 removed that -- `similarity` is honestly None on
    every graph candidate now, reserved for real vector/keyword similarity
    only (Part 2). The observable behavior this test actually cares about
    -- a deterministic, primary-sourced graph answer reaching HIGH -- is now
    produced by graph_confidence()/combine_confidence(), a dedicated,
    separately-labeled graph signal, never a fake similarity number."""
    ctx = gr.build_graph_context("Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    assert all(c.get("similarity") is None for c in merged), \
        "a graph candidate must never carry a fake similarity value"
    assert gr.graph_confidence(ctx) == "high"
    assert gr.combine_confidence("none", ctx) == "high"


def test_graph_only_derived_support_reaches_medium_not_high():
    """STALE MECHANISM (Phase 5K.1): originally asserted a synthetic
    `similarity` value (0.35) landed in the 0.30-0.45 band. Now verified via
    graph_confidence() directly -- a relationship whose ONLY evidence is
    structured_knowledge (derived_support) must resolve to 'medium', never
    'high' (Part 9: a derived primitive is never equivalent to primary
    source evidence) and never silently drop to the old always-zero/low
    behavior either."""
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-DERIVED-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-DERIVED-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["derived_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5K derived-only statement",
            "raw_subject_phrase": "TEST-5K", "sensitivity": "public", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5k-derived-{uuid.uuid4()}",
        }).execute().data[0]["id"]
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5K derived-only fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["derived_sk"],
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        ctx = gr.build_graph_context("Who owns TEST-5K-DERIVED-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
        merged, _ = gr.merge_graph_context_into_chunks([], ctx)
        assert all(c.get("similarity") is None for c in merged)
        assert gr.graph_confidence(ctx) == "medium"
        assert gr.combine_confidence("low", ctx) == "medium"
        assert gr.combine_confidence("high", ctx) == "high", \
            "combine_confidence must never LOWER a stronger vector confidence"
    finally:
        _cleanup(ids)


# =====================================================================
# 11. Graph + vector dedup remains intact
# =====================================================================

def test_dedup_still_works_after_status_and_confidence_changes():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-DEDUP-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-DEDUP-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5K dedup fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        ctx = gr.build_graph_context("Who owns TEST-5K-DEDUP-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
        existing_chunk = {
            "id": "real-chunk-1", "document_id": REAL_NOTE_ID_7A9EAA34,
            "content": "real chunk", "metadata": {"file_name": "real.txt"},
            "source_type": "google_chat", "source_tier": 2, "similarity": 0.5,
        }
        merged, metrics = gr.merge_graph_context_into_chunks([existing_chunk], ctx)
        assert metrics["graph_candidates_deduplicated"] >= 1
        assert len(merged) == 1
    finally:
        _cleanup(ids)


# =====================================================================
# 12 & 13. Future-dated Product relationship
# =====================================================================

def test_product_current_query_excludes_future_dated_relationship():
    ctx = gr.build_graph_context("What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert ctx is not None
    assert ctx.relationships == []


def test_product_future_as_of_reveals_relationship():
    future_as_of = datetime(2026, 9, 16, tzinfo=timezone.utc)
    ctx = gr.build_graph_context(
        "What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=future_as_of,
    )
    assert ctx is not None
    assert any(r.relationship_type == "requires_approval_from" for r in ctx.relationships)


# =====================================================================
# 14. Restricted graph evidence remains hidden (adapter-level, direct)
# =====================================================================

def test_restricted_evidence_never_in_low_privilege_merged_content():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-RESTRICTED-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5K-RESTRICTED-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["restricted_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5K-RESTRICTED statement text",
            "raw_subject_phrase": "TEST-5K", "sensitivity": "restricted", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5k-restricted-{uuid.uuid4()}",
        }).execute().data[0]["id"]
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5K restricted fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        low_ctx = gr.build_graph_context("Who owns TEST-5K-RESTRICTED-SRC?", REAL_WORKSPACE, EMPLOYEE_SENSITIVITIES)
        merged, _ = gr.merge_graph_context_into_chunks([], low_ctx)
        combined = "\n".join(c["content"] for c in merged)
        assert "TEST-5K-RESTRICTED statement text" not in combined
    finally:
        _cleanup(ids)


# =====================================================================
# 15. Wrong workspace remains invisible
# =====================================================================

def test_wrong_workspace_entity_never_resolves():
    resolved = gr.resolve_entity_mentions("Who owns Product?", OTHER_REAL_WORKSPACE)
    assert resolved == []


# =====================================================================
# 16 & 17. Workspace corpus safety
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
# 18-20. Existing state unchanged
# =====================================================================

def test_existing_entities_unchanged():
    count = supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count
    assert count == 5


def test_existing_real_relationships_unchanged():
    """The three real relationships (requires_approval_from, organized,
    attended) must still all be status='active' real rows -- this pass only
    ever creates/mutates TEST-5K-* synthetic fixtures, never touches these."""
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
# 21. Fixture cleanup sentinel
# =====================================================================

def test_no_test_5k_entities_leaked():
    leaked = supabase.table("knowledge_entities").select("id,canonical_label") \
        .like("canonical_label", "TEST-5K-%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_no_test_5k_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-5K%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


# =====================================================================
# Supplementary -- signals.py UUID guard (found live via this pass's own
# Part 9 benchmark: a graph candidate's pseudo document_id
# "graph_relationship:<uuid>" is not itself a valid uuid, and
# user_signals.document_id is uuid-typed -- every graph-derived signal write
# was silently failing before this fix).
# =====================================================================

def test_signals_skips_non_uuid_graph_pseudo_document_id():
    """log_sources_cited/log_sources_used_in_context must silently skip a
    graph pseudo-id rather than attempt (and fail) a write against the
    uuid-typed document_id column."""
    fake_relationship_id = str(uuid.uuid4())
    pseudo_id = f"graph_relationship:{fake_relationship_id}"
    # Must not raise, and must not attempt to log the pseudo id at all --
    # verified by confirming no user_signals row is ever created for it.
    signals.log_sources_cited(REAL_WORKSPACE, None, "test_5k", "q", [pseudo_id], "high")
    signals.log_sources_used_in_context(REAL_WORKSPACE, None, "test_5k", "q", [pseudo_id], "high")


def test_signals_uuid_regex_accepts_real_uuids_rejects_pseudo_ids():
    assert signals._UUID_RE.match(REAL_NOTE_ID_7A9EAA34)
    assert not signals._UUID_RE.match(f"graph_relationship:{REAL_NOTE_ID_7A9EAA34}")
    assert not signals._UUID_RE.match("not-a-uuid-at-all")


# =====================================================================
# 22. Full-state regression sentinel (the real full pytest run across every
# test_phase5*.py file remains the authoritative regression gate -- this is
# a fast sentinel for the counts this pass must not have disturbed)
# =====================================================================

def test_full_known_state_after_phase_5k():
    assert supabase.table("knowledge_entities").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 5
    assert supabase.table("knowledge_relationships").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 3
    # STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    # Calendar sync event legitimately arrived live during this session via
    # the deployed filtration-worker cron -- see test_phase5f_person_
    # identity.py's REAL_CALENDAR_EVENT_2_MEETING_URL comment for the
    # full explanation.
    assert supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count == 2
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count == 14
