"""
Google Workspace scope-lock regression suite — Calendar (structured
metadata), Meet (durable transcript knowledge), Chat (reused Slack
filtration), and Drive (reference-only, including the MANDATORY negative
test that no bulk knowledge_items/Storage activity is possible anymore).

calendar_events / external_references live in the VECTOR DB project (same
SUPABASE_URL/SUPABASE_SERVICE_KEY every other test in this repo already
uses) — no new credential gate needed for those, unlike drive_app_db.py's
app-DB RPC tests in test_google_drive_ingestion.py.

Run with: python -m pytest test_google_workspace.py -v
"""
import uuid
from types import SimpleNamespace

import pytest

import connector_google as google
import connector_google_calendar as gcal
import connector_google_meet as gmeet
import connector_google_chat as gchat
import worker
import brain_connectors as bc
import integrations
from auth import AuthContext

TEST_COMPANY_1_WS = "4053915c-044b-4bb5-b2d5-8db8750ed5fa"


def _new_id() -> str:
    return str(uuid.uuid4())


# =====================================================================
# OAuth scope-union
# =====================================================================

def test_scopes_for_surfaces_builds_union_from_enabled_only():
    scope = google.scopes_for_surfaces(["meet", "chat"])
    assert "meetings.space.readonly" in scope
    assert "chat.messages.readonly" in scope
    assert "chat.spaces.readonly" in scope
    assert "calendar.events.readonly" not in scope
    assert "drive.readonly" not in scope


def test_scopes_for_surfaces_all_four_covers_every_scope():
    scope = google.scopes_for_surfaces(["calendar", "meet", "chat", "drive"])
    for expected in ("calendar.events.readonly", "meetings.space.readonly",
                     "chat.messages.readonly", "chat.spaces.readonly", "drive.readonly"):
        assert expected in scope


def test_scopes_for_surfaces_empty_list_requests_nothing():
    assert google.scopes_for_surfaces([]) == ""


def test_scopes_for_surfaces_ignores_unknown_surface_names():
    scope = google.scopes_for_surfaces(["calendar", "not_a_real_surface"])
    assert "calendar.events.readonly" in scope
    assert "not_a_real_surface" not in scope


def test_scopes_for_surfaces_drive_only():
    scope = google.scopes_for_surfaces(["drive"])
    assert scope == "https://www.googleapis.com/auth/drive.readonly"


# =====================================================================
# Connection resolution / fail-closed checks — shared by all three pollers
# =====================================================================

class _FakeQuery:
    def __init__(self, data):
        self._data = data
    def select(self, *_a, **_kw): return self
    def eq(self, *_a, **_kw): return self
    def neq(self, *_a, **_kw): return self
    def upsert(self, *_a, **_kw): return self
    def execute(self): return SimpleNamespace(data=self._data)


class _FakeConnectionsClient:
    def __init__(self, rows):
        self._rows = rows
    def table(self, name):
        if name == "connections":
            return _FakeQuery(self._rows)
        return _FakeQuery([])


def test_get_active_connection_requires_surface_enabled(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["calendar"]}},
    ]))
    assert google.get_active_connection("ws-1", "calendar") is not None
    assert google.get_active_connection("ws-1", "meet") is None


def test_get_active_connection_rejects_inactive_or_wrong_provider(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([]))
    assert google.get_active_connection("ws-1", "drive") is None


def test_get_active_connection_rejects_unknown_surface(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    assert google.get_active_connection("ws-1", "not_a_real_surface") is None


@pytest.mark.parametrize("surface,poll_fn", [
    ("calendar", gcal.poll_connection),
    ("meet", gmeet.poll_connection),
    ("chat", gchat.poll_connection),
])
def test_each_poller_rejects_surface_not_enabled(monkeypatch, surface, poll_fn):
    other_surfaces = [s for s in ("calendar", "meet", "chat", "drive") if s != surface]
    monkeypatch.setattr(bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": other_surfaces}},
    ]))
    with pytest.raises(Exception, match="not enabled"):
        poll_fn("conn-1", "ws-1")


@pytest.mark.parametrize("surface,poll_fn", [
    ("calendar", gcal.poll_connection),
    ("meet", gmeet.poll_connection),
    ("chat", gchat.poll_connection),
])
def test_each_poller_rejects_inactive_connection(monkeypatch, surface, poll_fn):
    monkeypatch.setattr(bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "error", "config": {"enabled_surfaces": [surface]}},
    ]))
    with pytest.raises(Exception, match="not.*active"):
        poll_fn("conn-1", "ws-1")


# =====================================================================
# Surface selector: adding surfaces to an EXISTING connection must union
# with what's already granted, never silently drop it. Also: OAuth
# cancellation/failure must leave the existing connection's
# enabled_surfaces completely untouched, and only ONE row ever exists.
# =====================================================================

def test_build_install_url_merges_existing_surfaces_with_newly_requested(monkeypatch):
    """Requirement 4: existing Drive connection adds Meet -> Drive remains
    enabled. build_install_url is what decides the scope string AND what
    gets sealed into state for the callback to save."""
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    monkeypatch.setattr(bc, "encode_oauth_state", lambda ws, uid, extra=None: f"STATE:{extra}")

    url = google.build_install_url("ws-1", "user-1", enabled_surfaces=["meet"])

    assert "drive.readonly" in url, "existing Drive scope must be preserved on reconsent"
    assert "meetings.space.readonly" in url, "newly requested Meet scope must be included"
    assert "STATE:{'surfaces': " in url
    assert "'drive'" in url and "'meet'" in url


def test_build_install_url_first_time_connect_uses_only_requested_surfaces(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([]))  # no existing connection
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    monkeypatch.setattr(bc, "encode_oauth_state", lambda ws, uid, extra=None: f"STATE:{extra}")

    url = google.build_install_url("ws-1", "user-1", enabled_surfaces=["chat"])

    assert "chat.messages.readonly" in url
    assert "drive.readonly" not in url, "first-ever connect must not silently add Drive"
    assert "calendar.events.readonly" not in url


def test_oauth_callback_error_leaves_existing_connection_untouched(monkeypatch):
    """Requirement 5: OAuth cancellation -> previous enabled_surfaces unchanged."""
    calls = {"upsert": 0}

    class _SpyConnections(_FakeConnectionsClient):
        def table(self, name):
            if name == "connections":
                class _T(_FakeQuery):
                    def upsert(self, *_a, **_kw):
                        calls["upsert"] += 1
                        return self
                return _T(self._rows)
            return _FakeQuery([])

    monkeypatch.setattr(google.bc, "supabase", _SpyConnections([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))

    import asyncio
    result = asyncio.run(google.google_callback(code="", state="", error="access_denied"))

    assert calls["upsert"] == 0, "a cancelled/errored OAuth flow must never touch the connections table"


def test_google_connection_stays_one_row_across_reconnects(monkeypatch):
    """Requirement 7/8: still exactly one connections row, provider still
    google_drive, after a surface is added via re-consent -- verified at
    the upsert on_conflict-key level (workspace_id, provider,
    external_team_id all unchanged means the SAME row is updated)."""
    captured_rows = []

    class _SpyConnections(_FakeConnectionsClient):
        def table(self, name):
            if name == "connections":
                class _T(_FakeQuery):
                    def upsert(self, row, on_conflict=None):
                        captured_rows.append((row, on_conflict))
                        return self
                return _T(self._rows)
            return _FakeQuery([])

    monkeypatch.setattr(google.bc, "supabase", _SpyConnections([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    monkeypatch.setattr(bc, "decode_oauth_state", lambda state: {"w": "ws-1", "u": "user-1", "surfaces": ["meet"]})

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "post", lambda *_a, **_kw: SimpleNamespace(
        json=lambda: {"access_token": "tok", "refresh_token": "rt", "expires_in": 3600, "scope": "x"}))
    monkeypatch.setattr(_httpx, "get", lambda *_a, **_kw: SimpleNamespace(json=lambda: {"email": "a@co.com"}))
    monkeypatch.setattr(bc, "encrypt_secret", lambda v: f"enc:{v}")

    import asyncio
    asyncio.run(google.google_callback(code="abc", state="signed-state", error=""))

    assert len(captured_rows) == 1
    row, on_conflict = captured_rows[0]
    assert row["provider"] == "google_drive"
    assert on_conflict == "workspace_id,provider,external_team_id"
    assert set(row["config"]["enabled_surfaces"]) == {"drive", "meet"}, "must union, not replace"


def test_non_admin_cannot_modify_surfaces_on_existing_connection(monkeypatch):
    """Requirement 6: a plain member (not owner/admin/super_admin) may not
    re-consent to change an EXISTING connection's surfaces."""
    monkeypatch.setattr(integrations.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    member_auth = AuthContext(user_id="u1", workspaces={"ws-1": "member"})
    body = integrations.OAuthUrlRequest(
        workspace_id="ws-1", provider="google_drive", enabled_surfaces=["meet"],
    )

    import asyncio
    with pytest.raises(Exception) as exc:
        asyncio.run(integrations.oauth_url(body, member_auth))
    assert getattr(exc.value, "status_code", None) == 403


def test_admin_can_modify_surfaces_on_existing_connection(monkeypatch):
    monkeypatch.setattr(integrations.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    monkeypatch.setattr(google, "build_install_url", lambda *a, **kw: "https://accounts.google.com/fake")
    admin_auth = AuthContext(user_id="u1", workspaces={"ws-1": "admin"})
    body = integrations.OAuthUrlRequest(
        workspace_id="ws-1", provider="google_drive", enabled_surfaces=["meet"],
    )

    import asyncio
    result = asyncio.run(integrations.oauth_url(body, admin_auth))
    assert result["url"] == "https://accounts.google.com/fake"


def test_first_time_connect_does_not_require_admin(monkeypatch):
    """A brand-new connection (no existing row) stays open to any workspace
    member -- only MODIFYING an existing one is admin-gated, matching every
    other provider's un-gated first Connect."""
    monkeypatch.setattr(integrations.bc, "supabase", _FakeConnectionsClient([]))
    monkeypatch.setattr(google, "build_install_url", lambda *a, **kw: "https://accounts.google.com/fake")
    member_auth = AuthContext(user_id="u1", workspaces={"ws-1": "member"})
    body = integrations.OAuthUrlRequest(
        workspace_id="ws-1", provider="google_drive", enabled_surfaces=["drive"],
    )

    import asyncio
    result = asyncio.run(integrations.oauth_url(body, member_auth))
    assert result["url"] == "https://accounts.google.com/fake"


# =====================================================================
# Calendar — structured metadata only
# =====================================================================

def test_calendar_event_start_end_handles_timed_and_all_day():
    timed = {"start": {"dateTime": "2026-08-20T10:00:00Z"}, "end": {"dateTime": "2026-08-20T11:00:00Z"}}
    assert gcal._event_start_end(timed) == ("2026-08-20T10:00:00Z", "2026-08-20T11:00:00Z")
    all_day = {"start": {"date": "2026-08-20"}, "end": {"date": "2026-08-21"}}
    assert gcal._event_start_end(all_day) == ("2026-08-20", "2026-08-21")


def test_calendar_poll_writes_structured_row_and_never_touches_knowledge_notes(monkeypatch):
    """Regression for Decision 2: no embeddings, no LLM filtration, no
    knowledge_notes row -- only calendar_events."""
    calls = {"calendar_events_upsert": 0, "knowledge_notes_insert": 0, "classify_document_called": 0}

    monkeypatch.setattr(bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["calendar"]},
         "access_token_enc": "enc", "refresh_token_enc": "enc", "token_expires_at": None},
    ]))
    monkeypatch.setattr(google, "_valid_access_token", lambda _c: "fake-token")
    monkeypatch.setattr(gcal, "_list_events", lambda token: [{
        "id": "evt-1", "status": "confirmed", "summary": "Q4 planning",
        "start": {"dateTime": "2026-08-20T10:00:00Z"}, "end": {"dateTime": "2026-08-20T11:00:00Z"},
        "organizer": {"email": "a@co.com"}, "attendees": [{"email": "b@co.com", "responseStatus": "accepted"}],
        "hangoutLink": "https://meet.google.com/xyz", "updated": "2026-08-19T00:00:00Z",
    }])

    class _SpySupabase(_FakeConnectionsClient):
        def table(self, name):
            if name == "calendar_events":
                class _T(_FakeQuery):
                    def upsert(self, *_a, **_kw):
                        calls["calendar_events_upsert"] += 1
                        return self
                return _T([])
            if name == "knowledge_notes":
                class _T2(_FakeQuery):
                    def insert(self, *_a, **_kw):
                        calls["knowledge_notes_insert"] += 1
                        return self
                return _T2([])
            return super().table(name)

    monkeypatch.setattr(bc, "supabase", _SpySupabase([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["calendar"]}},
    ]))

    result = gcal.poll_connection("conn-1", "ws-1")

    assert calls["calendar_events_upsert"] == 1
    assert calls["knowledge_notes_insert"] == 0, "Calendar must never create a knowledge_notes row"
    assert result["processed"] == 1


# =====================================================================
# Meet — durable transcript knowledge
# =====================================================================

def test_assemble_transcript_orders_by_starttime_and_skips_empty():
    entries = [
        {"participant": "Bob", "text": "second", "startTime": "2026-08-20T10:01:00Z"},
        {"participant": "Alice", "text": "first", "startTime": "2026-08-20T10:00:00Z"},
        {"participant": "Carl", "text": "  ", "startTime": "2026-08-20T10:02:00Z"},
    ]
    result = gmeet._assemble_transcript(entries)
    assert result == "Alice: first\nBob: second"


def test_meet_transcript_becomes_durable_note_with_provenance(monkeypatch):
    """Core Meet acceptance requirement: transcript captured -> durable
    knowledge_notes row created via the EXISTING distill_meeting_transcript/
    create_note_and_embed pipeline (not reimplemented), with provenance
    sources carrying conference/transcript/entry identifiers."""
    captured = {}

    monkeypatch.setattr(gmeet, "_list_transcripts", lambda token, name: [{"name": "conferenceRecords/c1/transcripts/t1"}])
    monkeypatch.setattr(gmeet, "_list_transcript_entries", lambda token, name: [
        {"participant": "Alice", "text": "Let's ship the Q4 plan.",
         "startTime": "2026-08-20T10:00:00Z", "name": "conferenceRecords/c1/transcripts/t1/entries/e1"},
    ])
    monkeypatch.setattr(bc, "distill_meeting_transcript", lambda transcript, title, workspace_id=None: {
        "title": "Q4 plan", "body": "Team agreed to ship the Q4 plan.",
    })

    def _fake_create_note(workspace_id, connection_id, provider, note, **kwargs):
        captured["provider"] = provider
        captured["sources"] = kwargs.get("sources")
        captured["source_type"] = kwargs.get("source_type")
        return "note-123"
    monkeypatch.setattr(bc, "create_note_and_embed", _fake_create_note)
    monkeypatch.setattr(google, "resolve_drive_references_in_text", lambda *_a, **_kw: [])

    class _SpyIngestItems(_FakeQuery):
        def upsert(self, *_a, **_kw): return self
    class _SpySupabase:
        def table(self, _n): return _SpyIngestItems([])
    monkeypatch.setattr(bc, "supabase", _SpySupabase())

    conn = {"id": "conn-1"}
    created = gmeet._process_one_conference(conn, "token", "ws-1", {"name": "conferenceRecords/c1", "startTime": "2026-08-20T10:00:00Z"})

    assert created is True
    assert captured["provider"] == "google_meet"
    assert captured["source_type"] == "meeting"
    assert len(captured["sources"]) == 1
    assert captured["sources"][0]["channel_id"] == "conferenceRecords/c1"
    assert captured["sources"][0]["thread_ts"] == "conferenceRecords/c1/transcripts/t1"
    assert captured["sources"][0]["occurred_at"] == "2026-08-20T10:00:00Z"


def test_meet_discarded_transcript_creates_no_note(monkeypatch):
    monkeypatch.setattr(gmeet, "_list_transcripts", lambda token, name: [{"name": "conferenceRecords/c1/transcripts/t1"}])
    monkeypatch.setattr(gmeet, "_list_transcript_entries", lambda token, name: [
        {"participant": "Alice", "text": "hi", "startTime": "2026-08-20T10:00:00Z", "name": "e1"},
    ])
    monkeypatch.setattr(bc, "distill_meeting_transcript", lambda *_a, **_kw: None)  # discarded

    created_note = {"called": False}
    monkeypatch.setattr(bc, "create_note_and_embed", lambda *_a, **_kw: created_note.update(called=True))

    class _SpyIngestItems(_FakeQuery):
        def upsert(self, *_a, **_kw): return self
    class _SpySupabase:
        def table(self, _n): return _SpyIngestItems([])
    monkeypatch.setattr(bc, "supabase", _SpySupabase())

    created = gmeet._process_one_conference({"id": "conn-1"}, "token", "ws-1", {"name": "conferenceRecords/c1"})
    assert created is False
    assert created_note["called"] is False


# =====================================================================
# Chat — reuses Slack filtration verbatim
# =====================================================================

def test_normalize_message_produces_slack_compatible_raw_shape():
    space = {"name": "spaces/abc", "displayName": "#general"}
    message = {
        "name": "spaces/abc/messages/msg1",
        "sender": {"name": "users/u1", "displayName": "Alice"},
        "text": "The launch is Friday.",
        "createTime": "2026-08-20T10:00:00Z",
        "thread": {"name": "spaces/abc/threads/th1"},
    }
    raw = gchat._normalize_message(message, space)["raw"]
    # These are exactly the keys brain_connectors.batch_conversations/_format_batch read.
    assert raw["channel"] == "spaces/abc"
    assert raw["channel_name"] == "#general"
    assert raw["user_name"] == "Alice"
    assert raw["text"] == "The launch is Friday."
    assert raw["ts"] == "2026-08-20T10:00:00Z"
    assert raw["thread_ts"] == "spaces/abc/threads/th1"


def test_normalize_message_thread_ts_none_when_not_actually_threaded():
    space = {"name": "spaces/abc"}
    message = {"name": "spaces/abc/messages/msg1", "sender": {"name": "users/u1"},
               "text": "hi", "createTime": "t", "thread": {"name": "spaces/abc"}}
    raw = gchat._normalize_message(message, space)["raw"]
    assert raw["thread_ts"] is None


def test_chat_poll_reuses_run_filtration_not_a_new_engine(monkeypatch):
    """Core Chat acceptance requirement: no second filtration engine --
    save_ingest_items + run_filtration are called, nothing else."""
    calls = {"save_ingest_items": 0, "run_filtration": 0, "run_filtration_provider": None}

    monkeypatch.setattr(bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["chat"]},
         "access_token_enc": "enc", "refresh_token_enc": "enc", "token_expires_at": None},
    ]))
    monkeypatch.setattr(google, "_valid_access_token", lambda _c: "fake-token")
    monkeypatch.setattr(gchat, "_list_spaces", lambda token: [{"name": "spaces/abc", "displayName": "#general"}])
    monkeypatch.setattr(gchat, "_list_messages", lambda token, space: [{
        "name": "spaces/abc/messages/m1", "sender": {"name": "u1", "displayName": "Alice"},
        "text": "Real message content", "createTime": "2026-08-20T10:00:00Z", "thread": {},
    }])

    def _fake_save(workspace_id, connection_id, provider, items):
        calls["save_ingest_items"] += 1
        assert provider == "google_chat"
        return len(items)
    monkeypatch.setattr(bc, "save_ingest_items", _fake_save)

    def _fake_filtration(workspace_id, connection_id, provider, resolve_permalink=None):
        calls["run_filtration"] += 1
        calls["run_filtration_provider"] = provider
        return {"kept": 1, "discarded": 0}
    monkeypatch.setattr(bc, "run_filtration", _fake_filtration)

    result = gchat.poll_connection("conn-1", "ws-1")

    assert calls["save_ingest_items"] == 1
    assert calls["run_filtration"] == 1
    assert calls["run_filtration_provider"] == "google_chat"
    assert result["kept"] == 1


# =====================================================================
# Drive — reference-only, including the MANDATORY negative test
# =====================================================================

def test_extract_drive_file_ids_finds_real_shaped_links():
    text = ("The plan is here: https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view "
            "and the sheet: https://docs.google.com/spreadsheets/d/1QrStUvWxYz1234567/edit")
    ids = google.extract_drive_file_ids(text)
    assert ids == ["1AbCdEfGhIjKlMnOp", "1QrStUvWxYz1234567"]


def test_extract_drive_file_ids_empty_for_no_links():
    assert google.extract_drive_file_ids("just a normal sentence with no links") == []


def test_resolve_drive_reference_returns_none_without_active_connection(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([]))
    result = google.resolve_drive_reference("ws-1", "file123", "knowledge_note", "note-1")
    assert result is None


def test_resolve_drive_reference_creates_exactly_one_row_real_db(monkeypatch):
    """Real write against the vector DB's external_references table --
    proves the 'created exactly once' requirement, including on a repeat
    resolution of the same (linked_object, file) pair."""
    monkeypatch.setattr(google, "get_active_connection", lambda ws, surface: {"id": "conn-1"})
    monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token")
    monkeypatch.setattr(google, "_drive_get", lambda path, token, params=None: {
        "id": "file123", "name": "Q4 Plan.docx",
        "webViewLink": "https://drive.google.com/file/d/file123/view",
        "modifiedTime": "2026-08-01T00:00:00Z",
    })

    linked_id = str(uuid.uuid4())
    try:
        ref1 = google.resolve_drive_reference(TEST_COMPANY_1_WS, "file123", "knowledge_note", linked_id)
        ref2 = google.resolve_drive_reference(TEST_COMPANY_1_WS, "file123", "knowledge_note", linked_id)
        assert ref1["title"] == "Q4 Plan.docx"

        res = bc.supabase.table("external_references").select("id") \
            .eq("linked_object_id", linked_id).eq("external_file_id", "file123").execute()
        assert len(res.data) == 1, "resolving the same file for the same linked object twice must not duplicate the row"
    finally:
        bc.supabase.table("external_references").delete().eq("linked_object_id", linked_id).execute()


def test_drive_negative_no_bulk_ingestion_capability_exists_anymore():
    """MANDATORY negative test: the bulk Drive folder-polling code path must
    not exist at all anymore -- not disabled, not gated, GONE."""
    assert not hasattr(google, "sync_connection")
    assert not hasattr(google, "_sync_one_file")
    assert not hasattr(google, "_reconcile_deleted_files")
    assert not hasattr(google, "_list_folder_files")
    # worker.py's scheduled step list must not include a drive-poll step.
    assert not hasattr(worker, "run_google_drive_polling")


def test_drive_negative_reference_resolution_never_calls_knowledge_items_write(monkeypatch):
    """Even the retained reference-resolution path must never touch
    knowledge_items or Storage -- only external_references."""
    import drive_app_db
    calls = {"upsert_knowledge_item": 0, "upload_original_file": 0}
    monkeypatch.setattr(drive_app_db, "upsert_knowledge_item", lambda *_a, **_kw: calls.__setitem__("upsert_knowledge_item", calls["upsert_knowledge_item"] + 1))
    monkeypatch.setattr(drive_app_db, "upload_original_file", lambda *_a, **_kw: calls.__setitem__("upload_original_file", calls["upload_original_file"] + 1))

    monkeypatch.setattr(google, "get_active_connection", lambda ws, surface: {"id": "conn-1"})
    monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token")
    monkeypatch.setattr(google, "_drive_get", lambda path, token, params=None: {
        "id": "file123", "name": "x.docx", "webViewLink": "url", "modifiedTime": "t",
    })

    class _SpyRefs(_FakeQuery):
        def upsert(self, *_a, **_kw): return self
    class _SpySupabase:
        def table(self, name): return _SpyRefs([{"id": "ref-1"}])
    monkeypatch.setattr(google.bc, "supabase", _SpySupabase())

    google.resolve_drive_reference("ws-1", "file123", "knowledge_note", "note-1")

    assert calls["upsert_knowledge_item"] == 0
    assert calls["upload_original_file"] == 0
