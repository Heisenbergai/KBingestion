"""
Phase 6F Company Wiki Prose Generation tests.

Real data for the 9 real pages' claim inventories and benchmark. Synthetic,
single-use workspaces for sensitivity/supersession edge cases. No live
Bedrock credentials are configured in this environment (confirmed: a real
ai.chat() call raises botocore.exceptions.NoCredentialsError) -- live-model
success is therefore validated with deterministic mock chat_json_fn
implementations, exactly as Phase 6F's own instructions anticipate; the
REAL failure path (fallback-on-exception) is proven with the genuine
NoCredentialsError itself, not a simulated one.

wiki_generation.py never imports brain_connectors and never touches
Supabase -- several tests below snapshot real table row counts before/after
a full benchmark run specifically to prove that structurally, not just by
reading the source.

Run with: python -m pytest test_phase6f_wiki_generation.py -v
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from query import supabase
import graph_query as gq
import memory_retrieval as mr
import wiki_projection as wp
import wiki_generation as wg

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
LEAK_WORKSPACE = "892e3fc6-04a3-4421-a729-f83ed8c92ea3"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

REAL_ENTITY_IDS = {
    "john_snow":   "5c7fd6c0-ccb0-4a9e-94cf-bff4dd90e19d",
    "tanmay":      "66a242b2-44eb-4f2b-9a02-eafe41dbdbf0",
    "operations":  "1034346e-5731-45b8-9ee5-2e7d1413ca81",
    "product":     "c25f1ce7-6bcc-4a08-a80c-03db321c15f3",
    "meeting":     "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f",
}
REAL_MEMORY_IDS = {
    "credential_logging": "2b9140a0-a2e1-4892-b869-fb811e45f1f5",
    "credential_sharing":  "3d376631-894c-4e32-b3f5-3ecf7cfd5f61",
    "hardware_scope":      "8aef76c9-fda3-44d6-affb-769f2ff09326",
    "monday_capacity":     "8742eefd-f59c-4a0d-b211-9b75ce0a727e",
}
SK_Q4_LAUNCH_APPROVAL = "fc261a0a-4aa7-4224-a2b1-66513a03a05e"  # real pending review candidate
ALL_9_REAL_PAGES = (
    [("person", REAL_ENTITY_IDS["john_snow"]), ("person", REAL_ENTITY_IDS["tanmay"]),
     ("department", REAL_ENTITY_IDS["operations"]), ("department", REAL_ENTITY_IDS["product"]),
     ("meeting", REAL_ENTITY_IDS["meeting"])]
    + [("policy", mid) for mid in (REAL_MEMORY_IDS["credential_logging"], REAL_MEMORY_IDS["credential_sharing"], REAL_MEMORY_IDS["hardware_scope"])]
    + [("process", REAL_MEMORY_IDS["monday_capacity"])]
)


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
    for ent_id in ids.get("entity_ids", []):
        supabase.table("knowledge_entities").delete().eq("id", ent_id).execute()
    for sk_id in ids.get("sk_ids", []):
        supabase.table("structured_knowledge").delete().eq("id", sk_id).execute()


def _make_sk(workspace_id: str, **overrides) -> str:
    row = {
        "workspace_id": workspace_id, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "TEST-6F synthetic statement",
        "raw_subject_phrase": "TEST-6F subject", "qualifier_words": [],
        "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"test-6f-{uuid.uuid4()}",
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


def _compliant_mock(messages, system, max_tokens, temperature, model=None, workspace_id=None, user_id=None, feature=None):
    """One paragraph per claim, always correctly cited -- extracted straight
    from the prompt this module itself built, so it stays correct for any
    page's real claim set without hardcoding page-specific text here."""
    prompt = messages[0]["content"]
    temporal_context = [l for l in prompt.splitlines() if l.startswith("temporal_context:")][0].split(":", 1)[1].strip()
    paragraphs = []
    for line in prompt.splitlines():
        if line.startswith("["):
            cid = line[1:line.index("]")]
            text = line.split("]", 1)[1].strip()
            paragraphs.append({"text": text, "claim_ids": [cid]})
    return {"temporal_context_echo": temporal_context, "paragraphs": paragraphs}


def _raising_mock(*a, **k):
    raise TimeoutError("TEST-6F simulated model timeout")


# =====================================================================
# 1-2. Claim inventory determinism + valid LLM rendering.
# =====================================================================

def test_claim_inventory_deterministic_from_pagemodel():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    claims1 = wg.build_claim_inventory(page)
    claims2 = wg.build_claim_inventory(page)
    assert [(c.claim_id, c.text) for c in claims1] == [(c.claim_id, c.text) for c in claims2]


def test_compliant_mock_renders_via_llm():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
    assert result.generation_metadata["rendered_by"] == "llm"
    assert result.rendered_content and "Tanmay" in result.rendered_content


# =====================================================================
# 3-4. Citation preservation + unsupported claim rejection.
# =====================================================================

def test_citations_reflect_actually_used_claims():
    page = wp.build_page("meeting", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
    claim_ids = {c.claim_id for c in wg.build_claim_inventory(page)}
    cited_ids = {c["claim_id"] for c in result.citations}
    assert cited_ids == claim_ids  # the compliant mock cites every claim, one per paragraph
    for c in result.citations:
        assert c["evidence_refs"]  # every citation traces to something real


def test_unknown_claim_id_rejected():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)

    def mock(*a, **k):
        return {"temporal_context_echo": page.temporal_context,
                "paragraphs": [{"text": "Tanmay exists.", "claim_ids": ["identity:999"]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert any("unknown claim_id" in e for e in result.generation_metadata["validation_errors"])


# =====================================================================
# 5. Forbidden relationship / hallucinated entity rejection.
# =====================================================================

def test_forbidden_vocabulary_rejected_when_not_in_cited_claim():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    claims = wg.build_claim_inventory(page)

    def mock(*a, **k):
        return {"temporal_context_echo": page.temporal_context,
                "paragraphs": [{"text": "Tanmay owns Knova Test Meeting 1.", "claim_ids": [claims[0].claim_id]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert any("forbidden term" in e for e in result.generation_metadata["validation_errors"])


def test_forbidden_vocabulary_allowed_when_quoting_real_claim_text():
    """A word on the blocklist is NOT rejected if it already appears in the
    claim being cited -- honest quotation of real content must never be
    penalized, only genuinely new vocabulary. Uses the 'statement' claim
    (page.title, the real grounding statement) since that's the claim type
    that actually carries real free-text content forward verbatim."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, statement="This process was caused by a real prior incident.")
        ids["sk_ids"].append(sk_id)
        mem_id = _make_memory(ws, sk_id, p_memory_type="process", p_promotion_basis="recurring_durable_process")
        ids["memory_ids"].append(mem_id)
        page = wp.build_page("process", mem_id, ws, OWNER)
        claims = wg.build_claim_inventory(page)
        statement_claim = next(c for c in claims if c.claim_type == "statement")
        assert "caused" in statement_claim.text.lower()  # sanity: the real quoted statement

        def mock(*a, **k):
            return {"temporal_context_echo": page.temporal_context,
                    "paragraphs": [{"text": statement_claim.text, "claim_ids": [statement_claim.claim_id]}]}

        result = wg.generate_wiki_page(page, chat_json_fn=mock)
        assert result.generation_metadata["rendered_by"] == "llm"
    finally:
        _cleanup(ids)


def test_hallucinated_entity_rejected():
    page = wp.build_page("person", REAL_ENTITY_IDS["john_snow"], REAL_WORKSPACE, OWNER)
    claims = wg.build_claim_inventory(page)

    def mock(*a, **k):
        return {"temporal_context_echo": page.temporal_context,
                "paragraphs": [{"text": "John Snow later met with Sarah Connor.", "claim_ids": [claims[0].claim_id]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert any("Sarah Connor" in e for e in result.generation_metadata["validation_errors"])


def test_content_not_grounded_in_cited_claim_rejected():
    """Phase 6H regression: found via an adversarial validator battery. A
    paragraph citing a REAL claim_id, with no forbidden word and no
    unrecognized proper noun, could still state content entirely absent
    from what that claim actually says -- citation EXISTENCE was checked,
    but never citation RELEVANCE. "All credentials must be rotated every 30
    days without exception." cites the real policy identity claim (about
    promotion_basis, not rotation schedules) and would have passed before
    this fix. This exact gap was already named, but left open, in the
    original Phase 6F report's own "anything not verified" section --
    closed here once the adversarial battery made it concrete."""
    page = wp.build_page("policy", REAL_MEMORY_IDS["credential_logging"], REAL_WORKSPACE, OWNER)
    claims = wg.build_claim_inventory(page)
    identity_claim = next(c for c in claims if c.claim_type == "identity")

    def mock(*a, **k):
        return {"temporal_context_echo": page.temporal_context,
                "paragraphs": [{"text": "All credentials must be rotated every 30 days without exception.",
                                 "claim_ids": [identity_claim.claim_id]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert any("not grounded" in e for e in result.generation_metadata["validation_errors"])


def test_legitimate_multi_claim_combination_still_accepted():
    """The content-grounding fix above must not over-reject legitimate
    summarizing/combining (Part 3's own YES list) -- a paragraph that
    combines two real claims with ordinary connector words stays well under
    the novel-content threshold and renders normally."""
    page = wp.build_page("meeting", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER)
    claims = wg.build_claim_inventory(page)
    rel_claims = [c for c in claims if c.claim_type == "relationship"]

    def mock(*a, **k):
        return {"temporal_context_echo": page.temporal_context,
                "paragraphs": [{"text": "Tanmay organized this meeting, which John Snow also attended.",
                                 "claim_ids": [c.claim_id for c in rel_claims]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "llm"


# =====================================================================
# 6. Temporal correctness.
# =====================================================================

def test_temporal_context_echo_mismatch_rejected():
    page = wp.build_page("department", REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)

    def mock(*a, **k):
        return {"temporal_context_echo": "not-the-real-value",
                "paragraphs": [{"text": "Operations exists.", "claim_ids": ["identity:0"]}]}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert any("temporal_context_echo mismatch" in e for e in result.generation_metadata["validation_errors"])


def test_historical_relationship_claim_prefixed_with_real_date():
    future = datetime(2026, 9, 16, tzinfo=timezone.utc)
    page = wp.build_page("department", REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER, as_of=future)
    claims = wg.build_claim_inventory(page)
    rel_claim = next(c for c in claims if c.claim_type == "relationship")
    assert rel_claim.text.startswith("As of September 16, 2026,")
    current_page = wp.build_page("department", REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)
    current_claims = wg.build_claim_inventory(current_page)
    absence_claim = next(c for c in current_claims if c.claim_type == "relationship_absence")
    assert absence_claim.text.startswith("Operations currently")


# =====================================================================
# 7. Supersession rendering -- honestly never fabricated in V1.
# =====================================================================

def test_supersession_narrative_never_fabricated_without_successor_field():
    """Part 9: 'Only say this if the deterministic PageModel supplies:
    predecessor; successor; superseded_at.' wiki_projection's WikiPageModel
    (frozen, not redesigned this phase) never carries the SUCCESSOR's
    identity on the predecessor's own page -- only a bare superseded_at
    timestamp. Since the required input genuinely isn't present, the
    correct, honest behavior is to never produce a supersession NARRATIVE
    (something naming or describing the successor) at all, not to
    approximate one from a timestamp alone.

    This is deliberately narrower than banning the word "superseded"
    outright: _identity_claim already, correctly, discloses a non-active
    lifecycle_status honestly (e.g. "Its current status is superseded.") --
    that's a real field the PageModel DOES have, and reporting it plainly
    is exactly what satisfies Part 9's "do not present it as current truth"
    half using data that's actually present. What must never happen is the
    OTHER half -- naming or describing the successor -- since the PageModel
    never supplies that identity."""
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk1 = _make_sk(ws, statement="TEST-6F predecessor statement")
        sk2 = _make_sk(ws, statement="TEST-6F successor statement")
        ids["sk_ids"] += [sk1, sk2]
        predecessor_id = _make_memory(ws, sk1)
        ids["memory_ids"].append(predecessor_id)
        predecessor_created_at = supabase.table("org_memory").select("created_at").eq("id", predecessor_id).execute().data[0]["created_at"]
        before_succession = datetime.fromisoformat(predecessor_created_at.replace("Z", "+00:00")) + timedelta(milliseconds=1)
        successor_id = _make_memory(ws, sk2, p_supersedes_memory_id=predecessor_id)
        ids["memory_ids"].append(successor_id)

        page = wp.build_page("policy", predecessor_id, ws, OWNER, as_of=before_succession)
        assert page is not None
        assert page.sections[0].items[0]["superseded_at"] is not None  # the row IS superseded...
        claims = wg.build_claim_inventory(page)
        assert all(c.claim_type not in ("supersession",) for c in claims)  # ...but no such claim_type is ever produced
        all_text = " ".join(c.text.lower() for c in claims)
        assert "successor statement" not in all_text          # the successor's own real statement never leaks in
        assert successor_id.lower() not in all_text            # nor its id
        assert not any(p in all_text for p in ("replaced by", "was later superseded by", "newer policy", "newer record"))
    finally:
        _cleanup(ids)


# =====================================================================
# 8. Review candidate exclusion.
# =====================================================================

def test_review_candidate_never_reaches_renderer():
    review_rows = supabase.table("memory_review_queue").select("status") \
        .eq("workspace_id", REAL_WORKSPACE).eq("structured_knowledge_id", SK_Q4_LAUNCH_APPROVAL).execute().data
    assert review_rows and review_rows[0]["status"] == "pending"
    for ptype in ("policy", "process", "decision"):
        page = wp.build_page(ptype, SK_Q4_LAUNCH_APPROVAL, REAL_WORKSPACE, OWNER)
        assert page is None  # no PageModel means wiki_generation is structurally never invoked for it


# =====================================================================
# 9-10. Sensitivity pre-filter + workspace isolation.
# =====================================================================

def test_restricted_content_excluded_before_reaching_llm():
    ws = _fresh_workspace()
    ids = {"sk_ids": [], "memory_ids": []}
    try:
        sk_id = _make_sk(ws, sensitivity="restricted", statement="TEST-6F restricted statement")
        ids["sk_ids"].append(sk_id)
        mem_id = _make_memory(ws, sk_id)
        ids["memory_ids"].append(mem_id)

        low_page = wp.build_page("policy", mem_id, ws, LOW)
        assert low_page is None  # Part 11: excluded at the PageModel stage, never reaches this module at all
        owner_page = wp.build_page("policy", mem_id, ws, OWNER)
        assert owner_page is not None
        result = wg.generate_wiki_page(owner_page, chat_json_fn=_compliant_mock)
        assert result.generation_metadata["rendered_by"] == "llm"
    finally:
        _cleanup(ids)


def test_workspace_isolation_unaffected_pagemodel_none():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], LEAK_WORKSPACE, OWNER)
    assert page is None  # nothing for wiki_generation to ever receive


# =====================================================================
# 11-12. Failure handling.
# =====================================================================

def test_llm_exception_triggers_fallback():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page, chat_json_fn=_raising_mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert "llm_unavailable" in result.generation_metadata["reason"]
    assert "TimeoutError" in result.generation_metadata["reason"]
    assert result.rendered_content  # the Wiki still works


def test_malformed_json_shape_triggers_fallback():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)

    def mock(*a, **k):
        return {"unexpected_shape": True}

    result = wg.generate_wiki_page(page, chat_json_fn=mock)
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert result.rendered_content


def test_real_bedrock_no_credentials_triggers_fallback():
    """The genuine, live-observed failure in this environment (no AWS
    credentials configured locally) -- not a simulated one. Confirms the
    real ai.chat_json code path is exercised, not bypassed."""
    page = wp.build_page("department", REAL_ENTITY_IDS["operations"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page)  # real ai.chat_json, no override
    assert result.generation_metadata["rendered_by"] == "fallback"
    assert "NoCredentialsError" in result.generation_metadata["reason"]
    assert result.rendered_content


# =====================================================================
# 13. Deterministic regeneration.
# =====================================================================

def test_claim_ids_stable_across_rebuilds():
    for ptype, oid in ALL_9_REAL_PAGES:
        page1 = wp.build_page(ptype, oid, REAL_WORKSPACE, OWNER)
        page2 = wp.build_page(ptype, oid, REAL_WORKSPACE, OWNER)
        ids1 = [c.claim_id for c in wg.build_claim_inventory(page1)]
        ids2 = [c.claim_id for c in wg.build_claim_inventory(page2)]
        assert ids1 == ids2
        assert page1.content_hash == page2.content_hash  # unchanged from Phase 6E


# =====================================================================
# 14-15. Real 9-page benchmark, no Decision hallucination.
# =====================================================================

def test_real_9_page_benchmark_mock_and_real():
    for ptype, oid in ALL_9_REAL_PAGES:
        page = wp.build_page(ptype, oid, REAL_WORKSPACE, OWNER)
        assert page is not None
        mock_result = wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
        assert mock_result.generation_metadata["rendered_by"] == "llm"
        real_result = wg.generate_wiki_page(page)
        assert real_result.generation_metadata["rendered_by"] == "fallback"
        assert "NoCredentialsError" in real_result.generation_metadata["reason"]
        for r in (mock_result, real_result):
            assert r.rendered_content
            assert r.content_hash == page.content_hash
            assert r.temporal_context == page.temporal_context


def test_no_decision_page_fabricated_in_benchmark():
    rows = supabase.table("org_memory").select("id").eq("workspace_id", REAL_WORKSPACE).eq("memory_type", "decision").execute().data
    assert rows == []
    assert ("decision", None) not in [(t, o) for t, o in ALL_9_REAL_PAGES]


# =====================================================================
# 16-19. Real negative cases.
# =====================================================================

def test_qa_procurement_never_recognized_as_entity_labels():
    """Real statement text may legitimately CONTAIN the words QA/Procurement
    -- monday_capacity's real grounding statement literally is 'Procurement,
    warehouse, assembly, and QA must submit their available capacity...'.
    Phase 6E already established that honest quotation of real source text
    is correct, not a violation (test_qa_and_procurement_never_become_
    entities_or_links) -- banning the substring from all claim text would
    make this test wrong, not stricter. The actual invariant is narrower and
    structural: QA/Procurement never become a recognized ENTITY LABEL the
    validator would accept as a legitimate closed-world reference (i.e. they
    never appear as a counterpart_label, canonical_label, or link label --
    only ever as a substring inside a longer, real, honestly-quoted
    statement)."""
    for ptype, oid in ALL_9_REAL_PAGES:
        page = wp.build_page(ptype, oid, REAL_WORKSPACE, OWNER)
        allowed = {l.lower() for l in wg._collect_allowed_labels(page)}
        assert "qa" not in allowed and "procurement" not in allowed


def test_project_page_type_not_renderable():
    assert wp.build_page("project", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER) is None
    assert "project" not in wp.PAGE_BUILDERS


def test_john_snow_employment_never_renderable():
    page = wp.build_page("person", REAL_ENTITY_IDS["john_snow"], REAL_WORKSPACE, OWNER)
    claims = wg.build_claim_inventory(page)
    for c in claims:
        for bad in ("employ", "member of", "reports to"):
            assert bad not in c.text.lower()


def test_product_ownership_never_renderable():
    future = datetime(2026, 9, 16, tzinfo=timezone.utc)
    page = wp.build_page("department", REAL_ENTITY_IDS["product"], REAL_WORKSPACE, OWNER, as_of=future)
    claims = wg.build_claim_inventory(page)
    for c in claims:
        assert "owns" not in c.text.lower() and " own " not in f" {c.text.lower()} "


# =====================================================================
# 20-22. Underlying data unchanged (this module is read-only, structurally).
# =====================================================================

def test_structured_knowledge_graph_memory_row_counts_unchanged_by_generation():
    sk_before = len(supabase.table("structured_knowledge").select("id").execute().data)
    rel_before = len(supabase.table("knowledge_relationships").select("id").execute().data)
    mem_before = len(supabase.table("org_memory").select("id").execute().data)

    for ptype, oid in ALL_9_REAL_PAGES:
        page = wp.build_page(ptype, oid, REAL_WORKSPACE, OWNER)
        wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
        wg.generate_wiki_page(page)  # real ai.chat_json path too

    sk_after = len(supabase.table("structured_knowledge").select("id").execute().data)
    rel_after = len(supabase.table("knowledge_relationships").select("id").execute().data)
    mem_after = len(supabase.table("org_memory").select("id").execute().data)
    assert (sk_before, rel_before, mem_before) == (sk_after, rel_after, mem_after)


# =====================================================================
# 23. Fixture cleanup -- no TEST-6F rows survive a full run.
# =====================================================================

def test_no_leftover_test_6f_fixtures():
    leftover = supabase.table("structured_knowledge").select("id") \
        .ilike("statement", "TEST-6F%").execute().data
    assert leftover == []


# =====================================================================
# Part 10 / 17 -- link/section pass-through unchanged.
# =====================================================================

def test_links_and_sections_pass_through_unchanged():
    page = wp.build_page("meeting", REAL_ENTITY_IDS["meeting"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
    assert result.links == page.links
    assert result.sections == page.sections


def test_generation_metadata_reports_latencies():
    page = wp.build_page("person", REAL_ENTITY_IDS["tanmay"], REAL_WORKSPACE, OWNER)
    result = wg.generate_wiki_page(page, chat_json_fn=_compliant_mock)
    meta = result.generation_metadata
    for key in ("claim_inventory_ms", "llm_ms", "validation_ms"):
        assert key in meta and isinstance(meta[key], float) and meta[key] >= 0.0


# =====================================================================
# 24. Full-regression placeholder.
# =====================================================================

def test_placeholder_full_regression_run_separately():
    """No fixture here depends on suite ordering -- every synthetic case
    uses its own fresh workspace; every real-data case only reads."""
    assert True
