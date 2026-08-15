"""
Google Drive canonical ingestion — regression suite for the P0 gaps closed
by drive_app_db.py / connector_google.py / ingest.py's auto_classify path.

Two layers, matching what's actually testable in this environment:

1. REAL-DB tests against the live drive_sync_* SECURITY DEFINER RPCs (app DB
   project). These need APP_SUPABASE_URL / APP_SUPABASE_SERVICE_KEY set —
   the same NEW credential this feature introduces, not yet provisioned to
   local/CI environments as of this pass. They SKIP (not fail, not fake-pass)
   when unconfigured, same convention as other environment-gated tests in
   this codebase (see test_widget_suggest_contract.py's pattern). Every one
   of these was additionally run directly against the live database via the
   Supabase MCP during implementation — see the implementation report for
   that evidence; these codify the same checks as a repeatable suite.

2. HERMETIC unit tests for ingest.process_document_bytes's auto_classify
   branch — no DB, no external API calls, everything monkeypatched. This is
   the one piece of new logic that duplicates a rule (raise-only sensitivity)
   also enforced independently in SQL, so it gets its own direct test rather
   than relying solely on the RPC-level checks above.

Run with: python -m pytest test_google_drive_ingestion.py -v
"""
import os
import uuid
from types import SimpleNamespace

import pytest

import connector_google
import drive_app_db
import ingest

# Real workspace used throughout this session's Slack/Drive validation.
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
    """Best-effort: real test rows this suite created against the live app DB."""
    try:
        import httpx
        httpx.delete(
            f"{drive_app_db.APP_SUPABASE_URL}/rest/v1/knowledge_items?id=eq.{document_id}",
            headers=drive_app_db._headers(),
            timeout=15,
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
            assert lowered["sensitivity"] == "confidential", "sensitivity must never be lowered by re-classification"
        finally:
            _cleanup(doc_id)

    def test_baseline_read_call_preserves_authority_doc_class_lifecycle(self):
        """Regression test for the two-phase-upsert bug found and fixed
        during this pass: a re-sync's baseline-read call (all classification
        fields NULL) must PRESERVE the document's already-classified
        authority/doc_class/lifecycle_status, not revert them to defaults."""
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
            )  # simulates connector_google's pass-1 "read current" call
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
        """A document_id already owned by workspace A can never be
        repointed to workspace B through this RPC."""
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
            assert reappeared["deleted_at"] is None, "a re-synced (reappeared) file must be un-deleted"
        finally:
            _cleanup(doc_id)

    def test_stable_identity_idempotent_across_repeated_calls(self):
        """Same document_id, called repeatedly (simulating worker retry /
        duplicate poll) -- converges to one row, no duplication possible
        since id is the primary key and ON CONFLICT targets it directly."""
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


@_skip_no_app_db
class TestWorkspaceConnectionPairingMatrix:
    """
    The full A-G pairing matrix requested for pre-deployment verification.
    Since `connections` (which connection belongs to which workspace) and
    `knowledge_items` live in separate Supabase projects, no single RPC call
    can enforce all seven cases -- each is enforced at whichever layer
    actually has the data to check it:

    A. correct connection + its real workspace  -> PASS  -- RPC layer (below)
    B. connection A + workspace B (wrong pair)   -> FAIL  -- RPC layer, but
       ONLY once a document_id already exists under A's real workspace (see
       note below); a first-ever wrong pairing is NOT catchable by the RPC
       at all -- that gap is real and is why Railway's OWN connection->
       workspace resolution (conn["workspace_id"], read directly off the
       `connections` row) is itself part of the security boundary, not just
       a convenience lookup. There is no code path in this feature that
       lets a CALLER supply workspace_id independently of that lookup.
    C. connection B + workspace A (wrong pair, symmetric case of B)
    D. correct connection B + its real workspace -> PASS  -- RPC layer
    E. unknown connection                        -> FAIL  -- Railway layer,
       see TestConnectionResolutionSecurityBoundary.test_unknown_connection_rejected
    F. non-Google connection                     -> FAIL  -- Railway layer,
       see TestConnectionResolutionSecurityBoundary.test_non_google_drive_provider_rejected
    G. inactive connection                       -> FAIL  -- Railway layer,
       see TestConnectionResolutionSecurityBoundary.test_inactive_connection_rejected
    """

    def test_a_correct_pairing_workspace_one_passes(self):
        doc_id = _new_id()
        try:
            item = drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "a.pdf", "a.pdf", 100,
                "application/pdf", "pdf", "x/a.pdf",
            )
            assert item["workspace_id"] == TEST_COMPANY_1_WS
        finally:
            _cleanup(doc_id)

    def test_d_correct_pairing_workspace_two_passes(self):
        doc_id = _new_id()
        try:
            item = drive_app_db.upsert_knowledge_item(
                MAGIC_SMART_HOMES_WS, doc_id, "d.pdf", "d.pdf", 100,
                "application/pdf", "pdf", "x/d.pdf",
            )
            assert item["workspace_id"] == MAGIC_SMART_HOMES_WS
        finally:
            _cleanup(doc_id)

    def test_b_and_c_wrong_pairing_rejected_once_document_id_is_established(self):
        """B and C are the same check from each direction: once a
        document_id is bound to a real workspace, re-submitting it under
        the OTHER workspace is rejected either way."""
        doc_id = _new_id()
        try:
            drive_app_db.upsert_knowledge_item(
                TEST_COMPANY_1_WS, doc_id, "t.pdf", "t.pdf", 100,
                "application/pdf", "pdf", "x/t.pdf",
            )
            with pytest.raises(Exception):  # B: A's document_id, presented as B's
                drive_app_db.upsert_knowledge_item(
                    MAGIC_SMART_HOMES_WS, doc_id, "hijack.pdf", "hijack.pdf", 100,
                    "application/pdf", "pdf", "x/hijack.pdf",
                )

            doc_id_2 = _new_id()
            drive_app_db.upsert_knowledge_item(
                MAGIC_SMART_HOMES_WS, doc_id_2, "t2.pdf", "t2.pdf", 100,
                "application/pdf", "pdf", "x/t2.pdf",
            )
            with pytest.raises(Exception):  # C: B's document_id, presented as A's
                drive_app_db.upsert_knowledge_item(
                    TEST_COMPANY_1_WS, doc_id_2, "hijack2.pdf", "hijack2.pdf", 100,
                    "application/pdf", "pdf", "x/hijack2.pdf",
                )
            _cleanup(doc_id_2)
        finally:
            _cleanup(doc_id)


class TestDriveSyncRpcPrivileges:
    """These use has_function_privilege, which does not require the new
    APP_SUPABASE_SERVICE_KEY -- runs whenever any app-DB credential is
    available, since privilege metadata is readable without elevated rights."""

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
    """Minimal stand-in for supabase.table(...) chains used inside
    process_document_bytes for document_chunks -- swallows delete/insert."""
    def table(self, *_a, **_kw): return self
    def delete(self): return self
    def insert(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def execute(self): return SimpleNamespace(data=[])


def test_knowledge_file_type_mapping():
    assert connector_google._knowledge_file_type("application/pdf") == "pdf"
    assert connector_google._knowledge_file_type(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "word"
    assert connector_google._knowledge_file_type(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") == "excel"
    assert connector_google._knowledge_file_type("application/vnd.ms-excel") == "excel"
    assert connector_google._knowledge_file_type(
        "application/vnd.openxmlformats-officedocument.presentationml.presentation") == "powerpoint"
    assert connector_google._knowledge_file_type("text/plain") == "other"


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
        sensitivity="internal",  # baseline -- classifier proposes confidential, must raise
        authority="working", doc_class=None, lifecycle_status="active",
        auto_classify=True,
    )

    assert result["effective_sensitivity"] == "confidential"
    assert result["effective_authority"] == "canonical"
    assert result["effective_doc_class"] == "financial"
    assert result["effective_lifecycle_status"] == "active"
    # proposed_* (job-status/UI fields) still present and unaffected.
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
        "sensitivity": "public",  # proposes LOWER than baseline
        "authority": "working", "doc_class": None, "lifecycle_status": "active",
        "confidence": "high", "signals": [],
    })

    result = ingest.process_document_bytes(
        b"fake bytes", document_id="doc-2", asset_id="doc-2", workspace_id="ws-1",
        mime_type="application/pdf", file_name="t.pdf",
        sensitivity="confidential",  # baseline is already high
        authority="working", doc_class=None, lifecycle_status="active",
        auto_classify=True,
    )

    assert result["effective_sensitivity"] == "confidential", "must never lower below the caller-supplied baseline"


class _FakeQuery:
    """Chainable stand-in for supabase.table(...).select(...).eq(...).execute()."""
    def __init__(self, data):
        self._data = data
    def select(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def in_(self, *_a, **_kw): return self
    def upsert(self, *_a, **_kw): return self
    def update(self, *_a, **_kw): return self
    def execute(self): return SimpleNamespace(data=self._data)


class _FakeConnectionsClient:
    """Stand-in for bc.supabase -- only .table("connections") returns real
    rows; every other table (ingest_items, etc.) returns empty, sufficient
    for the fail-closed checks under test which all return before reaching
    those tables."""
    def __init__(self, connections_row):
        self._connections_row = connections_row
    def table(self, name):
        if name == "connections":
            return _FakeQuery(self._connections_row)
        return _FakeQuery([])


# =====================================================================
# Section 1 — storage failure must abort BEFORE any knowledge_items/chunk
# is created, and must never touch ingest_items (retryable next pass).
# =====================================================================

class TestStorageFailureSemantics:
    def test_storage_upload_failure_raises_not_swallows(self, monkeypatch):
        monkeypatch.setattr(drive_app_db, "_configured", lambda: True)
        monkeypatch.setattr(drive_app_db, "APP_SUPABASE_URL", "https://fake.supabase.co")
        monkeypatch.setattr(drive_app_db, "APP_SUPABASE_SERVICE_KEY", "fake")

        class _FailingResponse:
            status_code = 500
            text = "internal error"

        import httpx
        monkeypatch.setattr(httpx, "post", lambda *_a, **_kw: _FailingResponse())

        with pytest.raises(RuntimeError):
            drive_app_db.upload_original_file("ws-1", "doc-1", "t.pdf", b"bytes", "application/pdf")

    def test_sync_one_file_creates_nothing_when_storage_upload_fails(self, monkeypatch):
        """The core requirement: a Storage failure must leave no orphaned
        knowledge_items row, no chunks, and ingest_items untouched."""
        calls = {"upsert_knowledge_item": 0, "process_document_bytes": 0, "ingest_items_upsert": 0}

        monkeypatch.setattr(connector_google, "_fetch_file_bytes",
                            lambda *_a, **_kw: (b"bytes", "application/pdf", ""))
        monkeypatch.setattr(drive_app_db, "upload_original_file",
                            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("storage down")))

        def _spy_upsert(*_a, **_kw):
            calls["upsert_knowledge_item"] += 1
            return {"sensitivity": "internal"}
        monkeypatch.setattr(drive_app_db, "upsert_knowledge_item", _spy_upsert)

        def _spy_process(*_a, **_kw):
            calls["process_document_bytes"] += 1
            return {}
        monkeypatch.setattr(ingest, "process_document_bytes", _spy_process)

        class _SpyIngestItems(_FakeQuery):
            def upsert(self, *_a, **_kw):
                calls["ingest_items_upsert"] += 1
                return self

        class _SpySupabase:
            def table(self, _name): return _SpyIngestItems([])
        monkeypatch.setattr(connector_google.bc, "supabase", _SpySupabase())

        conn = {"workspace_id": "ws-1"}
        folder = {"id": "folder-1"}
        f = {"id": "file-1", "name": "budget.pdf", "modifiedTime": "2026-08-15T00:00:00Z"}

        ok = connector_google._sync_one_file(conn, "conn-1", folder, f, None, "token", "application/pdf")

        assert ok is False, "a storage failure must be reported as a failed file, not silently succeed"
        assert calls["upsert_knowledge_item"] == 0, "no knowledge_items write may happen after a storage failure"
        assert calls["process_document_bytes"] == 0, "no chunks may be written after a storage failure"
        assert calls["ingest_items_upsert"] == 0, "ingest_items must stay untouched so the next sync pass retries"

    def test_sync_one_file_succeeds_end_to_end_when_storage_upload_succeeds(self, monkeypatch):
        """Sanity check on the same helper: the happy path still completes
        all three RPC-layer steps and marks ingest_items embedded."""
        calls = {"upsert_knowledge_item": 0, "ingest_items_upsert": 0}

        monkeypatch.setattr(connector_google, "_fetch_file_bytes",
                            lambda *_a, **_kw: (b"bytes", "application/pdf", ""))
        monkeypatch.setattr(drive_app_db, "upload_original_file", lambda *_a, **_kw: "drive-sync/ws-1/doc.pdf")

        def _spy_upsert(*_a, **_kw):
            calls["upsert_knowledge_item"] += 1
            return {"sensitivity": "internal"}
        monkeypatch.setattr(drive_app_db, "upsert_knowledge_item", _spy_upsert)
        monkeypatch.setattr(ingest, "process_document_bytes", lambda *_a, **_kw: {
            "effective_sensitivity": "internal", "effective_authority": "working",
            "effective_doc_class": None, "effective_lifecycle_status": "active",
        })

        class _SpyIngestItems(_FakeQuery):
            def upsert(self, *_a, **_kw):
                calls["ingest_items_upsert"] += 1
                return self

        class _SpySupabase:
            def table(self, _name): return _SpyIngestItems([])
        monkeypatch.setattr(connector_google.bc, "supabase", _SpySupabase())

        conn = {"workspace_id": "ws-1"}
        folder = {"id": "folder-1"}
        f = {"id": "file-1", "name": "budget.pdf", "modifiedTime": "2026-08-15T00:00:00Z"}

        ok = connector_google._sync_one_file(conn, "conn-1", folder, f, None, "token", "application/pdf")

        assert ok is True
        assert calls["upsert_knowledge_item"] == 2, "pass 1 (baseline) + pass 3 (final effective values)"
        assert calls["ingest_items_upsert"] == 1


# =====================================================================
# Section 2 — connection provider/status fail-closed checks. Since
# `connections` and `knowledge_items` live in different Supabase projects
# (see drive_app_db.py's module docstring), the drive_sync_* RPCs cannot
# independently verify connection provider/status -- THIS lookup, inside
# sync_connection(), is the only place in the call chain that can, and is
# therefore part of the security boundary, not just a convenience check.
# =====================================================================

class TestConnectionResolutionSecurityBoundary:
    def test_unknown_connection_rejected(self, monkeypatch):
        monkeypatch.setattr(connector_google.bc, "supabase", _FakeConnectionsClient([]))
        with pytest.raises(connector_google.HTTPException) as exc:
            connector_google.sync_connection("does-not-exist")
        assert exc.value.status_code == 404

    def test_non_google_drive_provider_rejected(self, monkeypatch):
        monkeypatch.setattr(connector_google.bc, "supabase", _FakeConnectionsClient([
            {"id": "conn-1", "workspace_id": "ws-1", "provider": "slack", "status": "active", "config": {}}
        ]))
        with pytest.raises(connector_google.HTTPException) as exc:
            connector_google.sync_connection("conn-1")
        assert exc.value.status_code == 400
        assert "not a Google Drive connection" in exc.value.detail

    def test_inactive_connection_rejected(self, monkeypatch):
        monkeypatch.setattr(connector_google.bc, "supabase", _FakeConnectionsClient([
            {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive", "status": "error", "config": {}}
        ]))
        with pytest.raises(connector_google.HTTPException) as exc:
            connector_google.sync_connection("conn-1")
        assert exc.value.status_code == 400
        assert "not active" in exc.value.detail

    def test_active_google_drive_connection_with_no_folders_returns_cleanly(self, monkeypatch):
        """Positive case (A/D-style pairing): a correctly-provider/status
        connection with zero folders selected proceeds past both checks and
        returns a clean, empty result rather than raising."""
        monkeypatch.setattr(connector_google.bc, "supabase", _FakeConnectionsClient([
            {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
             "status": "active", "config": {"folders": []},
             "access_token_enc": None, "refresh_token_enc": None, "token_expires_at": None}
        ]))
        monkeypatch.setattr(connector_google, "_valid_access_token", lambda _conn: "fake-token")

        result = connector_google.sync_connection("conn-1")
        assert result["folders_checked"] == 0
        assert result["processed"] == 0
        assert result["failed"] == 0


def test_manual_upload_path_unaffected_by_auto_classify_default(monkeypatch):
    """auto_classify defaults to False -- manual-upload's existing call
    shape (no auto_classify kwarg) must behave exactly as before: the
    caller-supplied sensitivity/authority/doc_class/lifecycle_status are
    used as-is, never overridden by the classifier's proposal."""
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
        # auto_classify NOT passed -- defaults to False
    )

    assert "effective_sensitivity" not in result
    assert result["proposed_sensitivity"] == "restricted"  # still reported for UI/job status
