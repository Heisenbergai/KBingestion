"""
Regression suite for the drive_sync_* SECURITY DEFINER RPCs (drive_app_db.py)
and ingest.process_document_bytes's auto_classify branch.

STATUS NOTE (Google Workspace scope lock, 2026-08-15): the bulk Drive
ingestion flow that used to call these RPCs from connector_google.py has
been neutralized -- Drive is reference-only now (see
connector_google.resolve_drive_reference, tested in
test_google_workspace.py). This suite still exercises the RPC
infrastructure directly, since it's left in place for a possible future
individual-file-import feature and remains real, tested, deployed code.

Two layers, matching what's actually testable in this environment:

1. REAL-DB tests against the live drive_sync_* RPCs (app DB project). Need
   APP_SUPABASE_URL / APP_SUPABASE_SERVICE_KEY set -- SKIP (not fail, not
   fake-pass) when unconfigured.
2. HERMETIC unit tests for ingest.process_document_bytes's auto_classify
   branch -- no DB, no external API calls, everything monkeypatched.

Run with: python -m pytest test_google_drive_ingestion.py -v
"""
import os
import uuid
from types import SimpleNamespace

import pytest

import drive_app_db
import ingest

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"
MAGIC_SMART_HOMES_WS = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"

_APP_DB_CONFIGURED = bool(os.getenv("APP_SUPABASE_URL") and os.getenv("APP_SUPABASE_SERVICE_KEY"))
_skip_no_app_db = pytest.mark.skipif(
    not _APP_DB_CONFIGURED,
    reason="APP_SUPABASE_URL/APP_SUPABASE_SERVICE_KEY not configured in this environment",
)


def _new_id() -> str:
    return str(uuid.uuid4())


def _cleanup(document_id: str):
    try:
        import httpx
        httpx.delete(
            f"{drive_app_db.APP_SUPABASE_URL}/rest/v1/knowledge_items?id=eq.{document_id}",
            headers=drive_app_db._headers(), timeout=15,
        )
    except Exception:
        pass


# =====================================================================
# Layer 1 — real RPC, real app DB
# =====================================================================

@_skip_no_app_db
class TestDriveSyncRpcRealDb:
    def test_first_sync_defaults_to_safe_baseline(self):
        doc_id = _new_id()
        try:
            item = drive_app_db.upsert_knowledge_item(
                workspace_id=TEST_COMPANY_1_WS, document_id=doc_id,
                title="t.pdf", file_name="t.pdf", file_size=100,
                mime_type="application/pdf", file_type="pdf", storage_path="x/t.pdf",
            )
            assert item["sensitivity"] == "internal"
            assert item["authority"] == "working"
            assert item["doc_class"] is None
            assert item["lifecycle_status"] == "active"
            assert item["uploaded_by"] is None
        finally:
            _cleanup(doc_id)

    def test_classification_can_raise_sensitivity(self):
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            raised = drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
                sensitivity="confidential",
            )
            assert raised["sensitivity"] == "confidential"
        finally:
            _cleanup(doc_id)

    def test_classification_never_lowers_sensitivity(self):
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf", sensitivity="confidential",
            )
            lowered = drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf", sensitivity="public",
            )
            assert lowered["sensitivity"] == "confidential"
        finally:
            _cleanup(doc_id)

    def test_baseline_read_call_preserves_authority_doc_class_lifecycle(self):
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
                sensitivity="confidential", authority="canonical",
                doc_class="financial", lifecycle_status="active",
            )
            baseline = drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            assert baseline["sensitivity"] == "confidential"
            assert baseline["authority"] == "canonical"
            assert baseline["doc_class"] == "financial"
            assert baseline["lifecycle_status"] == "active"
        finally:
            _cleanup(doc_id)

    def test_malformed_sensitivity_rejected_by_postgrest_type_system(self):
        doc_id = _new_id()
        try:
            with pytest.raises(Exception):
                drive_app_db.upsert_knowledge_item(
                    TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                    "application/pdf", "pdf", "x/t.pdf", sensitivity="not_a_real_value",
                )
        finally:
            _cleanup(doc_id)

    def test_invalid_workspace_rejected(self):
        with pytest.raises(Exception):
            drive_app_db.upsert_knowledge_item(
                "00000000-0000-0000-0000-000000000000", _new_id(),
                "t.pdf", "t.pdf", 100, "application/pdf", "pdf", "x/t.pdf",
            )

    def test_cross_workspace_document_id_rejected(self):
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            with pytest.raises(Exception):
                drive_app_db.upsert_knowledge_item(
                    MAGIC_SMART_HOMES_WS, doc_id, "hijack.pdf", "hijack.pdf", 100,
                    "application/pdf", "pdf", "x/hijack.pdf",
                )
        finally:
            _cleanup(doc_id)

    def test_soft_delete_and_reappear_undeletes(self):
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            drive_app_db.soft_delete_knowledge_item(TEST_COMPANY_1_WS, doc_id)
            reappeared = drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            assert reappeared["deleted_at"] is None
        finally:
            _cleanup(doc_id)

    def test_stable_identity_idempotent_across_repeated_calls(self):
        doc_id = _new_id()
        try:
            for _ in range(3):
                drive_app_db.upsert_knowledge_item(
                    TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                    "application/pdf", "pdf", "x/t.pdf",
                )
            import httpx
            res = httpx.get(
                f"{drive_app_db.APP_SUPABASE_URL}/rest/v1/knowledge_items?id=eq.{doc_id}&select=id",
                headers=drive_app_db._headers(), timeout=15,
            )
            assert len(res.json()) == 1
        finally:
            _cleanup(doc_id)


class TestDriveSyncRpcPrivileges:
    @_skip_no_app_db
    def test_rpc_not_executable_by_anon_or_authenticated(self):
        import httpx
        res = httpx.post(
            f"{drive_app_db.APP_SUPABASE_URL}/rest/v1/rpc/drive_sync_upsert_knowledge_item",
            headers={"apikey": os.getenv("APP_SUPABASE_ANON_KEY", ""), "Content-Type": "application/json"},
            json={"p_workspace_id": TEST_COMPANY_1_WS, "p_document_id": _new_id(),
                  "p_title": "x", "p_file_name": "x", "p_file_size": 1, "p_mime_type": "text/plain",
                  "p_file_type": "other", "p_storage_path": "x", "p_sensitivity": None,
                  "p_authority": None, "p_doc_class": None, "p_lifecycle_status": None},
            timeout=15,
        )
        assert res.status_code in (401, 403, 404), (
            f"anon must NOT be able to execute this RPC, got {res.status_code}: {res.text[:200]}"
        )


# =====================================================================
# Layer 2 — hermetic unit tests, no DB / no external APIs
# =====================================================================

class _FakeTable:
    def table(self, *_a, **_kw): return self
    def delete(self): return self
    def insert(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def execute(self): return SimpleNamespace(data=[])


def test_auto_classify_raises_sensitivity_and_applies_authority_doc_class(monkeypatch):
    monkeypatch.setattr(ingest, "extract_text", lambda *_a, **_kw: "some real extracted text")
    monkeypatch.setattr(ingest, "extract_doc_date", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "is_spreadsheet", lambda *_a, **_kw: False)
    monkeypatch.setattr(ingest, "clean_text", lambda t: t)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk one"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks, on_progress=None, workspace_id=None: [[0.0] * 8])
    monkeypatch.setattr(ingest, "extract_section_label", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "supabase", _FakeTable())
    monkeypatch.setattr(ingest, "classify_document", lambda *_a, **_kw: {
        "sensitivity": "confidential", "authority": "canonical",
        "doc_class": "financial", "lifecycle_status": "active",
        "confidence": "high", "signals": ["test"],
    })

    result = ingest.process_document_bytes(
        b"fake bytes", document_id="doc-1", asset_id="doc-1", workspace_id="ws-1",
        mime_type="application/pdf", file_name="t.pdf",
        sensitivity="internal", authority="working", doc_class=None, lifecycle_status="active",
        auto_classify=True,
    )

    assert result["effective_sensitivity"] == "confidential"
    assert result["effective_authority"] == "canonical"
    assert result["effective_doc_class"] == "financial"
    assert result["effective_lifecycle_status"] == "active"
    assert result["proposed_sensitivity"] == "confidential"


def test_auto_classify_never_lowers_baseline_sensitivity(monkeypatch):
    monkeypatch.setattr(ingest, "extract_text", lambda *_a, **_kw: "text")
    monkeypatch.setattr(ingest, "extract_doc_date", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "is_spreadsheet", lambda *_a, **_kw: False)
    monkeypatch.setattr(ingest, "clean_text", lambda t: t)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks, on_progress=None, workspace_id=None: [[0.0] * 8])
    monkeypatch.setattr(ingest, "extract_section_label", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "supabase", _FakeTable())
    monkeypatch.setattr(ingest, "classify_document", lambda *_a, **_kw: {
        "sensitivity": "public", "authority": "working", "doc_class": None, "lifecycle_status": "active",
        "confidence": "high", "signals": [],
    })

    result = ingest.process_document_bytes(
        b"fake bytes", document_id="doc-2", asset_id="doc-2", workspace_id="ws-1",
        mime_type="application/pdf", file_name="t.pdf",
        sensitivity="confidential", authority="working", doc_class=None, lifecycle_status="active",
        auto_classify=True,
    )

    assert result["effective_sensitivity"] == "confidential"


def test_manual_upload_path_unaffected_by_auto_classify_default(monkeypatch):
    monkeypatch.setattr(ingest, "extract_text", lambda *_a, **_kw: "text")
    monkeypatch.setattr(ingest, "extract_doc_date", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "is_spreadsheet", lambda *_a, **_kw: False)
    monkeypatch.setattr(ingest, "clean_text", lambda t: t)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk"])
    monkeypatch.setattr(ingest, "embed_chunks", lambda chunks, on_progress=None, workspace_id=None: [[0.0] * 8])
    monkeypatch.setattr(ingest, "extract_section_label", lambda *_a, **_kw: None)
    monkeypatch.setattr(ingest, "supabase", _FakeTable())
    monkeypatch.setattr(ingest, "classify_document", lambda *_a, **_kw: {
        "sensitivity": "restricted", "authority": "canonical",
        "doc_class": "legal", "lifecycle_status": "archived",
        "confidence": "high", "signals": [],
    })

    result = ingest.process_document_bytes(
        b"fake bytes", document_id="doc-3", asset_id="doc-3", workspace_id="ws-1",
        mime_type="application/pdf", file_name="t.pdf",
        sensitivity="public", authority="informal", doc_class="policy_sop",
        lifecycle_status="draft",
    )

    assert "effective_sensitivity" not in result
    assert result["proposed_sensitivity"] == "restricted"
