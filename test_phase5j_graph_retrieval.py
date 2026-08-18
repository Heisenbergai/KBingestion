"""
Phase 5J Graph + Retrieval Integration tests -- verifies graph_retrieval.py
(the adapter that folds graph_query.py's read layer into query.py's existing
hybrid retrieval as an ADDITIVE context source) against the real corpus for
positive/end-to-end cases, and synthetic fixtures for security/temporal/
ambiguity boundary cases the real corpus can't exercise on its own.

This environment has no live AWS Bedrock credentials (confirmed before
writing this suite: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are unset), so
ai.embed_texts()/ai.chat() cannot be called here -- hybrid_search()'s own
embedding call and the final LLM answer synthesis are NOT exercised live in
this suite, same honesty precedent as Phase 5E's "no live OAuth" notes.
Every test below exercises real code against real data through paths that
need no embedding/LLM call: entity resolution, graph traversal, evidence
resolution, and the merge/dedup logic (fed a real chunk-shaped dict built
directly from a real document_chunks row where a test needs one, never a
fabricated one).

Every fixture helper builds its id dict incrementally with cleanup-on-failure
from the first write, per the Phase 5D-incident lesson.

Run with: python -m pytest test_phase5j_graph_retrieval.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import graph_query as gq
import graph_retrieval as gr
from query import supabase, build_context_and_citations
import source_labels

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"

TANMAY_ENTITY_ID = "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0"
JOHN_SNOW_ENTITY_ID = "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"
PRODUCT_ENTITY_ID = "c25f1ce7-6bcc-4a08-a80c-03db321c15f3"
OPERATIONS_ENTITY_ID = "1034346e-5731-45b8-9ee5-2e7d1413ca81"

REAL_SNAPSHOT_ID = "ffcf42b9-62dc-446e-9378-292d70d1d2ca"
REAL_NOTE_SOURCE_7A9EAA34 = "2419c928-5bf1-4b13-bdd6-f1a2b88b1bfb"
REAL_NOTE_ID_7A9EAA34 = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"  # == document_chunks.document_id for this note

OWNER_SENSITIVITIES = gq.resolve_allowed_sensitivities("owner", False)
EMPLOYEE_SENSITIVITIES = gq.resolve_allowed_sensitivities("employee", False)


def _cleanup(ids: dict):
    if ids.get("relationship_id"):
        supabase.table("knowledge_relationships").delete().eq("id", ids["relationship_id"]).execute()
    for key in ("public_sk", "restricted_sk"):
        if ids.get(key):
            supabase.table("structured_knowledge").delete().eq("id", ids[key]).execute()
    for key in ("src_entity", "tgt_entity", "src_entity_b", "tgt_entity_b"):
        if ids.get(key):
            supabase.table("knowledge_entities").delete().eq("id", ids[key]).execute()


# =====================================================================
# 1. Graph-relevant query invokes graph
# =====================================================================

def test_relational_query_is_detected_graph_relevant():
    assert gr.is_graph_relevant("Who organized Knova Test Meeting 1?")
    assert gr.is_graph_relevant("What relationships does Product have?")
    assert gr.is_graph_relevant("Who is responsible for the Q4 launch approval?")


def test_relational_query_produces_real_graph_context():
    ctx = gr.build_graph_context(
        "Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is not None
    assert ctx.traversal_depth == 2
    assert any(r.relationship_type == "organized" for r in ctx.relationships)


# =====================================================================
# 2. Non-graph query does not unnecessarily invoke graph
# =====================================================================

def test_non_relational_query_is_not_graph_relevant():
    assert not gr.is_graph_relevant("What is the production credential policy?")
    assert not gr.is_graph_relevant("Tell me about the Q4 smart-switch launch.")
    assert not gr.is_graph_relevant("What does the Q4 roadmap say?")


def test_non_relational_query_returns_no_graph_context():
    ctx = gr.build_graph_context(
        "What is the production credential policy?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is None


# =====================================================================
# 3. Entity resolution is deterministic
# =====================================================================

def test_entity_resolution_is_repeatable():
    r1 = gr.resolve_entity_mentions("Who owns Product?", REAL_WORKSPACE)
    r2 = gr.resolve_entity_mentions("Who owns Product?", REAL_WORKSPACE)
    assert [e["id"] for e in r1] == [e["id"] for e in r2]
    assert [e["id"] for e in r1] == [PRODUCT_ENTITY_ID]


def test_entity_resolution_finds_all_real_labels():
    for label, eid in [
        ("Tanmay", TANMAY_ENTITY_ID), ("John Snow", JOHN_SNOW_ENTITY_ID),
        ("Product", PRODUCT_ENTITY_ID), ("Operations", OPERATIONS_ENTITY_ID),
        ("Knova Test Meeting 1", MEETING_ENTITY_ID),
    ]:
        resolved = gr.resolve_entity_mentions(f"What is the relationship for {label}?", REAL_WORKSPACE)
        assert [e["id"] for e in resolved] == [eid], f"failed to resolve {label!r}"


def test_nested_mention_prefers_more_specific_match():
    """'John Snow' contains 'John' as a substring, but only the real entity
    ('John Snow') exists -- no synthetic 'John' entity is created here, so
    this just confirms the longer real label wins cleanly, not a fuzzy
    partial match on 'John' alone producing something unintended."""
    resolved = gr.resolve_entity_mentions("Who is John Snow related to?", REAL_WORKSPACE)
    assert [e["id"] for e in resolved] == [JOHN_SNOW_ENTITY_ID]


# =====================================================================
# 4. Ambiguous entity falls back safely
# =====================================================================

def test_ambiguous_mention_resolves_to_nothing():
    """Synthetic: two entities sharing the exact same canonical_label in the
    same workspace. A mention of that label must resolve to ZERO entities,
    never guess between them."""
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-AMBIGUOUS", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-AMBIGUOUS", "status": "active",
        }).execute().data[0]["id"]

        resolved = gr.resolve_entity_mentions("Who owns TEST-5J-AMBIGUOUS?", REAL_WORKSPACE)
        assert resolved == []

        ctx = gr.build_graph_context("Who owns TEST-5J-AMBIGUOUS?", REAL_WORKSPACE, OWNER_SENSITIVITIES)
        assert ctx is None, "ambiguous entity must fall back to no graph context, never guess"
    finally:
        _cleanup(ids)


# =====================================================================
# 5. Graph + vector candidates merge without duplication
# =====================================================================

def test_graph_candidate_added_when_not_covered_by_existing_chunks():
    ctx = gr.build_graph_context(
        "Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is not None
    merged, metrics = gr.merge_graph_context_into_chunks([], ctx)
    assert metrics["graph_candidates_added"] >= 1
    assert any(c["id"].startswith("graph:") for c in merged)


def test_graph_candidate_deduplicated_when_fully_covered_by_existing_chunk():
    """Synthetic relationship whose ONLY evidence is a real, existing
    knowledge_note_source (the real note behind the Product approval
    requirement) -- proves the true full-coverage dedup path precisely,
    which the real organized/attended relationships can never trigger
    (their evidence is a calendar snapshot, never embedded as a chunk)."""
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-DEDUP-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-DEDUP-TGT", "status": "active",
        }).execute().data[0]["id"]
        now = datetime.now(timezone.utc).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5J dedup fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                            "stance": "supports", "captured_at": now}],
        }).execute().data

        ctx = gr.build_graph_context(
            "Who owns TEST-5J-DEDUP-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
        )
        assert ctx is not None
        assert any(r.id == ids["relationship_id"] for r in ctx.relationships)

        existing_chunk = {
            "id": "real-chunk-1", "document_id": REAL_NOTE_ID_7A9EAA34,
            "content": "real chunk already covering this note", "metadata": {"file_name": "real.txt"},
            "source_type": "google_chat", "source_tier": 2, "similarity": 0.5,
        }
        merged, metrics = gr.merge_graph_context_into_chunks([existing_chunk], ctx)
        assert metrics["graph_candidates_deduplicated"] >= 1
        assert len(merged) == 1, "the synthetic relationship must NOT add a duplicate candidate"
    finally:
        _cleanup(ids)


# =====================================================================
# 6. Graph evidence keeps provenance
# =====================================================================

def test_graph_candidate_content_quotes_real_evidence_reference():
    ctx = gr.build_graph_context(
        "Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    graph_candidates = [c for c in merged if c["id"].startswith("graph:")]
    assert graph_candidates
    combined = "\n".join(c["content"] for c in graph_candidates)
    assert "https://meet.google.com/ngn-pjwu-jcn" in combined, \
        "the real calendar evidence's own meeting_url must appear verbatim, never paraphrased away"
    assert "organized" in combined


# =====================================================================
# 7. Graph citations resolve to source
# =====================================================================

def test_graph_candidate_survives_citation_assembly_with_correct_label():
    ctx = gr.build_graph_context(
        "Who organized Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    context_text, citations = build_context_and_citations(merged)
    graph_citations = [c for c in citations if c["source_type"] == "graph_relationship"]
    assert graph_citations
    assert source_labels.source_type_label("graph_relationship") == "Knowledge graph"
    assert "Tanmay" in graph_citations[0]["file_name"]
    assert "Tanmay" in context_text


# =====================================================================
# 8. Workspace isolation
# =====================================================================

def test_entity_resolution_is_workspace_scoped():
    resolved = gr.resolve_entity_mentions("Who owns Product?", OTHER_REAL_WORKSPACE)
    assert resolved == [], "Product exists only in REAL_WORKSPACE -- must not resolve under a different one"

    resolved_wrong_way = gr.resolve_entity_mentions("Who is responsible for Tanmay?", OTHER_REAL_WORKSPACE)
    assert resolved_wrong_way == []


# =====================================================================
# 9. Restricted evidence does not leak
# =====================================================================

def test_restricted_evidence_invisible_to_low_privilege_but_relationship_still_visible():
    """The critical Part 7 test: a relationship with one public + one
    restricted evidence record must remain VISIBLE to a low-privilege caller
    via its public evidence alone, while the restricted record itself never
    appears in that caller's evidence list -- inherited entirely from
    graph_query.py's existing per-evidence-record filtering, re-verified
    here at the retrieval-adapter boundary."""
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-RESTRICTED-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-RESTRICTED-TGT", "status": "active",
        }).execute().data[0]["id"]

        now_ts = datetime.now(timezone.utc).isoformat()
        ids["public_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5J public synthetic statement",
            "raw_subject_phrase": "TEST-5J", "sensitivity": "public", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now_ts, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5j-public-{uuid.uuid4()}",
        }).execute().data[0]["id"]
        ids["restricted_sk"] = supabase.table("structured_knowledge").insert({
            "workspace_id": REAL_WORKSPACE, "canonical_source_type": "knowledge_note",
            "canonical_id": REAL_NOTE_ID_7A9EAA34, "provider": "google_chat",
            "primitive_type": "fact", "statement": "TEST-5J restricted synthetic statement",
            "raw_subject_phrase": "TEST-5J", "sensitivity": "restricted", "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": now_ts, "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"test-5j-restricted-{uuid.uuid4()}",
        }).execute().data[0]["id"]

        now = datetime.now(timezone.utc).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5J restricted-leak fixture",
            "p_confidence": None, "p_valid_from": now, "p_valid_until": None,
            "p_evidence": [
                {"evidence_type": "structured_knowledge", "evidence_id": ids["public_sk"],
                 "stance": "supports", "captured_at": now},
                {"evidence_type": "structured_knowledge", "evidence_id": ids["restricted_sk"],
                 "stance": "supports", "captured_at": now},
            ],
        }).execute().data

        low_ctx = gr.build_graph_context(
            "Who owns TEST-5J-RESTRICTED-SRC?", REAL_WORKSPACE, EMPLOYEE_SENSITIVITIES,
        )
        assert low_ctx is not None, "relationship must still be visible via its public evidence"
        rel = next(r for r in low_ctx.relationships if r.id == ids["relationship_id"])
        assert len(rel.evidence) == 1
        assert rel.evidence[0].evidence_id == ids["public_sk"]
        assert all(ev.evidence_id != ids["restricted_sk"] for ev in rel.evidence)

        merged, _ = gr.merge_graph_context_into_chunks([], low_ctx)
        combined = "\n".join(c["content"] for c in merged)
        assert "TEST-5J restricted synthetic statement" not in combined
        assert "restricted" not in combined.lower() or ids["restricted_sk"] not in combined

        high_ctx = gr.build_graph_context(
            "Who owns TEST-5J-RESTRICTED-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
        )
        high_rel = next(r for r in high_ctx.relationships if r.id == ids["relationship_id"])
        assert len(high_rel.evidence) == 2
    finally:
        _cleanup(ids)


# =====================================================================
# 10 & 11. Temporal filtering / expired relationship excluded from current
# =====================================================================

def test_expired_relationship_excluded_from_current_graph_context():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-TEMPORAL-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-TEMPORAL-TGT", "status": "active",
        }).execute().data[0]["id"]

        past_from = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        past_until = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5J expired fixture",
            "p_confidence": None, "p_valid_from": past_from, "p_valid_until": past_until,
            "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                            "stance": "supports", "captured_at": past_from}],
        }).execute().data

        current_ctx = gr.build_graph_context(
            "Who owns TEST-5J-TEMPORAL-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
        )
        # The entity itself still resolves (it exists) -- what must be
        # excluded is the EXPIRED relationship specifically, not the whole
        # context. A resolved entity with zero current relationships is a
        # real, honest result (same "empty is valid" contract get_entity_graph
        # already documents), not the same as "nothing resolved at all".
        assert current_ctx is not None
        assert current_ctx.relationships == [], \
            "an expired relationship must not appear in a current-time graph context"
        merged, metrics = gr.merge_graph_context_into_chunks([], current_ctx)
        assert metrics["graph_candidates_added"] == 0
        assert merged == []
    finally:
        _cleanup(ids)


# =====================================================================
# 12. Historical query can retrieve historical relationship
# =====================================================================

def test_historical_as_of_retrieves_expired_relationship():
    ids = {}
    try:
        ids["src_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-HISTORICAL-SRC", "status": "active",
        }).execute().data[0]["id"]
        ids["tgt_entity"] = supabase.table("knowledge_entities").insert({
            "workspace_id": REAL_WORKSPACE, "entity_type": "department",
            "canonical_label": "TEST-5J-HISTORICAL-TGT", "status": "active",
        }).execute().data[0]["id"]

        past_from = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        past_until = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        ids["relationship_id"] = supabase.rpc("create_relationship_with_evidence", {
            "p_workspace_id": REAL_WORKSPACE,
            "p_source_object_type": "entity", "p_source_object_id": ids["src_entity"],
            "p_target_object_type": "entity", "p_target_object_id": ids["tgt_entity"],
            "p_relationship_type": "references", "p_rationale": "synthetic 5J historical fixture",
            "p_confidence": None, "p_valid_from": past_from, "p_valid_until": past_until,
            "p_evidence": [{"evidence_type": "knowledge_note_source", "evidence_id": REAL_NOTE_SOURCE_7A9EAA34,
                            "stance": "supports", "captured_at": past_from}],
        }).execute().data

        as_of_during = datetime.now(timezone.utc) - timedelta(days=45)
        historical_ctx = gr.build_graph_context(
            "Who owns TEST-5J-HISTORICAL-SRC?", REAL_WORKSPACE, OWNER_SENSITIVITIES, as_of=as_of_during,
        )
        assert historical_ctx is not None
        assert any(r.id == ids["relationship_id"] for r in historical_ctx.relationships)
        assert historical_ctx.temporal_context == as_of_during.isoformat()
    finally:
        _cleanup(ids)


# =====================================================================
# 13. Product approval query does not imply ownership
# =====================================================================

def test_product_current_query_correctly_shows_zero_relationships():
    """Real, load-bearing finding: the one real Product relationship
    (requires_approval_from) has valid_from=2026-09-15, which is FUTURE
    relative to 'now' in this corpus (2026-08-18) -- confirmed independently
    in Phase 5D as intentional, correct temporal behavior, not a bug. A
    CURRENT-time graph context for Product must therefore show zero
    relationships today -- exactly mirroring Part 8's "don't show an expired
    relationship as current" rule, applied symmetrically to a not-yet-valid
    one."""
    ctx = gr.build_graph_context(
        "What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is not None
    assert ctx.relationships == []


def test_product_relationship_candidate_never_says_owns():
    """Same query, evaluated as_of a point AFTER the relationship's real
    valid_from -- this is the actual, correct way to observe it (Part 8:
    historical/future-relative queries use an explicit as_of)."""
    future_as_of = datetime(2026, 9, 16, tzinfo=timezone.utc)
    ctx = gr.build_graph_context(
        "What relationships does Product have?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
        as_of=future_as_of,
    )
    assert ctx is not None
    assert any(r.relationship_type == "requires_approval_from" for r in ctx.relationships)
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    combined = "\n".join(c["content"] for c in merged).lower()
    assert "requires_approval_from" in combined
    assert "owns" not in combined
    assert "owned by" not in combined
    assert "manages" not in combined


# =====================================================================
# 14. John Snow work query does not invent a Project relationship
# =====================================================================

def test_john_snow_work_query_never_invents_works_on():
    ctx = gr.build_graph_context(
        "What does John Snow work on?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is not None
    types = {r.relationship_type for r in ctx.relationships}
    assert types == {"attended"}, \
        "John Snow's only real relationship is 'attended' -- no works_on type exists anywhere in this graph"
    merged, _ = gr.merge_graph_context_into_chunks([], ctx)
    combined = "\n".join(c["content"] for c in merged).lower()
    assert "works_on" not in combined
    assert "work on" not in combined


# =====================================================================
# 15. Operations query does not create unsupported relationships
# =====================================================================

def test_operations_query_not_graph_relevant_and_has_zero_relationships():
    assert not gr.is_graph_relevant("What does Operations know about the Q4 launch?")
    # Even if it HAD been graph-relevant, Operations legitimately has zero
    # relationships -- verified directly, independent of the wording check.
    ge = gq.get_entity_graph(OPERATIONS_ENTITY_ID, REAL_WORKSPACE, OWNER_SENSITIVITIES)
    assert ge.inbound_relationships == []
    assert ge.outbound_relationships == []


# =====================================================================
# 16. Real Meeting query returns real organizer/attendee evidence
# =====================================================================

def test_meeting_query_returns_both_real_activity_relationships():
    ctx = gr.build_graph_context(
        "Who attended Knova Test Meeting 1?", REAL_WORKSPACE, OWNER_SENSITIVITIES,
    )
    assert ctx is not None
    labels = {(r.relationship_type, r.source.label) for r in ctx.relationships}
    assert ("organized", "Tanmay") in labels
    assert ("attended", "John Snow") in labels
    for r in ctx.relationships:
        assert all(ev.evidence_type == "calendar_event_snapshot" for ev in r.evidence)


# =====================================================================
# 17. Exact source citations remain intact (no graph involvement)
# =====================================================================

def test_normal_chunk_citations_unaffected_by_graph_module_presence():
    plain_chunk = {
        "id": "c1", "document_id": "doc-1", "content": "plain content",
        "metadata": {"file_name": "plain.txt"}, "source_type": "slack",
        "source_tier": 3, "similarity": 0.6,
    }
    context_text, citations = build_context_and_citations([plain_chunk])
    assert citations == [{
        "index": 1, "file_name": "plain.txt", "snippet": "plain content",
        "source_type": "slack", "source_tier": 3,
    }]
    assert "plain content" in context_text


# =====================================================================
# 18. Existing Phase 1-5 behavior unchanged (import/signature sanity)
# =====================================================================

def test_query_module_imports_and_signatures_unchanged():
    import query as query_module
    assert hasattr(query_module, "hybrid_search")
    assert hasattr(query_module, "build_context_and_citations")
    assert hasattr(query_module, "QueryRequest")
    # as_of is new and optional -- must default to None, never break a
    # caller that omits it entirely (every caller before this phase).
    req = query_module.QueryRequest(question="x", workspace_id=REAL_WORKSPACE)
    assert req.as_of is None


# =====================================================================
# 19. 14-row workspace corpus is respected
# =====================================================================

def test_workspace_structured_knowledge_is_14_rows():
    count = supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).execute().count
    assert count == 14


# =====================================================================
# 20. The unrelated 15th global row cannot leak into workspace retrieval
# =====================================================================

def test_hr_contact_row_excluded_from_workspace_corpus_and_never_resolvable():
    global_count = supabase.table("structured_knowledge").select("id", count="exact").execute().count
    assert global_count == 15
    hr_row = supabase.table("structured_knowledge").select("workspace_id") \
        .eq("statement", "88994448877 is the HR contact").execute().data
    assert hr_row
    assert hr_row[0]["workspace_id"] != REAL_WORKSPACE

    # graph_retrieval's entity loader is workspace-scoped by construction --
    # confirm it can never see anything from that other workspace even if a
    # query happened to mention text from it.
    resolved = gr.resolve_entity_mentions("Who owns 88994448877?", REAL_WORKSPACE)
    assert resolved == []


# =====================================================================
# Fixture-leak sentinel
# =====================================================================

def test_no_test_5j_entities_leaked():
    leaked = supabase.table("knowledge_entities").select("id,canonical_label") \
        .like("canonical_label", "TEST-5J-%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"


def test_no_test_5j_structured_knowledge_leaked():
    leaked = supabase.table("structured_knowledge").select("id,statement") \
        .like("statement", "TEST-5J%").execute().data
    assert leaked == [], f"fixture cleanup failed, leaked rows: {leaked}"
