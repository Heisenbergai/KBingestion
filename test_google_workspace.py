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
from typing import Optional

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

    url = google.build_install_url("ws-1", "user-1", "conn-1", enabled_surfaces=["meet"])

    assert "drive.readonly" in url, "existing Drive scope must be preserved on reconsent"
    assert "meetings.space.readonly" in url, "newly requested Meet scope must be included"
    assert "'connection_id': 'conn-1'" in url
    assert "'drive'" in url and "'meet'" in url


def test_build_install_url_first_time_connect_uses_only_requested_surfaces(monkeypatch):
    """A fresh pending connection (created via POST /integrations/connections
    before OAuth starts) has no enabled_surfaces yet -- only what's newly
    requested gets scoped."""
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-new", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "pending", "config": {}},
    ]))
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    monkeypatch.setattr(bc, "encode_oauth_state", lambda ws, uid, extra=None: f"STATE:{extra}")

    url = google.build_install_url("ws-1", "user-1", "conn-new", enabled_surfaces=["chat"])

    assert "chat.messages.readonly" in url
    assert "drive.readonly" not in url, "first-ever connect must not silently add Drive"
    assert "calendar.events.readonly" not in url


def test_build_install_url_rejects_connection_not_owned_by_workspace(monkeypatch):
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "OTHER-workspace", "provider": "google_drive",
         "status": "pending", "config": {}},
    ]))
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    with pytest.raises(google.HTTPException) as exc:
        google.build_install_url("ws-1", "user-1", "conn-1", enabled_surfaces=["drive"])
    assert exc.value.status_code == 404


def test_oauth_callback_error_leaves_existing_connection_untouched(monkeypatch):
    """Requirement 5: OAuth cancellation -> previous enabled_surfaces unchanged."""
    calls = {"update": 0}

    class _SpyConnections(_FakeConnectionsClient):
        def table(self, name):
            if name == "connections":
                class _T(_FakeQuery):
                    def update(self, *_a, **_kw):
                        calls["update"] += 1
                        return self
                return _T(self._rows)
            return _FakeQuery([])

    monkeypatch.setattr(google.bc, "supabase", _SpyConnections([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))

    import asyncio
    result = asyncio.run(google.google_callback(code="", state="", error="access_denied"))

    assert calls["update"] == 0, "a cancelled/errored OAuth flow must never touch the connections table"


def test_google_connection_stays_one_row_across_reconnects(monkeypatch):
    """Requirement 7/8: still exactly one connections row, provider still
    google_drive, after a surface is added via re-consent -- verified at
    the UPDATE-by-connection_id level (never an upsert/insert -- the SAME
    row named in state is the only one ever touched)."""
    captured_updates = []

    class _SpyConnections(_FakeConnectionsClient):
        def table(self, name):
            if name == "connections":
                class _T(_FakeQuery):
                    def update(self, row):
                        captured_updates.append(row)
                        return self
                    def insert(self, *_a, **_kw):
                        raise AssertionError("must never INSERT a new row on reconnect")
                return _T(self._rows)
            return _FakeQuery([])

    monkeypatch.setattr(google.bc, "supabase", _SpyConnections([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    monkeypatch.setattr(google, "_google_credentials", lambda ws: ("client-id", "secret"))
    monkeypatch.setattr(bc, "decode_oauth_state",
                        lambda state: {"w": "ws-1", "u": "user-1", "connection_id": "conn-1", "surfaces": ["meet"]})

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "post", lambda *_a, **_kw: SimpleNamespace(
        json=lambda: {"access_token": "tok", "refresh_token": "rt", "expires_in": 3600, "scope": "x"}))
    monkeypatch.setattr(_httpx, "get", lambda *_a, **_kw: SimpleNamespace(json=lambda: {"email": "a@co.com"}))
    monkeypatch.setattr(bc, "encrypt_secret", lambda v: f"enc:{v}")

    import asyncio
    asyncio.run(google.google_callback(code="abc", state="signed-state", error=""))

    assert len(captured_updates) == 1
    row = captured_updates[0]
    assert set(row["config"]["enabled_surfaces"]) == {"drive", "meet"}, "must union, not replace"


def test_oauth_callback_missing_connection_id_fails_closed(monkeypatch):
    """A malformed/legacy state with no connection_id must not fall back to
    guessing a connection -- fails closed."""
    monkeypatch.setattr(bc, "decode_oauth_state", lambda state: {"w": "ws-1", "u": "user-1"})
    import asyncio
    result = asyncio.run(google.google_callback(code="abc", state="signed-state", error=""))
    # oauth_complete_html("google_drive", "error") -- just confirm it doesn't raise/crash
    assert result is not None


def test_non_admin_cannot_modify_surfaces_on_existing_connection(monkeypatch):
    """Requirement 6: a plain member (not owner/admin/super_admin) may not
    re-consent to change an EXISTING (active) connection's surfaces."""
    monkeypatch.setattr(integrations.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["drive"]}},
    ]))
    member_auth = AuthContext(user_id="u1", workspaces={"ws-1": "member"})
    body = integrations.OAuthUrlRequest(
        workspace_id="ws-1", provider="google_drive", connection_id="conn-1", enabled_surfaces=["meet"],
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
        workspace_id="ws-1", provider="google_drive", connection_id="conn-1", enabled_surfaces=["meet"],
    )

    import asyncio
    result = asyncio.run(integrations.oauth_url(body, admin_auth))
    assert result["url"] == "https://accounts.google.com/fake"


def test_first_time_connect_does_not_require_admin(monkeypatch):
    """A brand-new PENDING connection (first-time setup) stays open to any
    workspace member -- only MODIFYING an already-ACTIVE one is admin-gated,
    matching every other provider's un-gated first Connect."""
    monkeypatch.setattr(integrations.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-new", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "pending", "config": {}},
    ]))
    monkeypatch.setattr(google, "build_install_url", lambda *a, **kw: "https://accounts.google.com/fake")
    member_auth = AuthContext(user_id="u1", workspaces={"ws-1": "member"})
    body = integrations.OAuthUrlRequest(
        workspace_id="ws-1", provider="google_drive", connection_id="conn-new", enabled_surfaces=["drive"],
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

    def _fake_filtration(workspace_id, connection_id, provider, resolve_permalink=None,
                         on_note_created=None):
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
    monkeypatch.setattr(google, "get_active_connection", lambda ws, surface, connection_id=None: {"id": "conn-1"})
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

    monkeypatch.setattr(google, "get_active_connection", lambda ws, surface, connection_id=None: {"id": "conn-1"})
    monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token")
    monkeypatch.setattr(google, "_drive_get", lambda path, token, params=None: {
        "id": "file123", "name": "x.docx", "webViewLink": "url", "modifiedTime": "2026-08-01T00:00:00Z",
    })

    class _SpyRefs(_FakeQuery):
        def upsert(self, *_a, **_kw): return self
    class _SpySupabase:
        def table(self, name): return _SpyRefs([{"id": "ref-1"}])
    monkeypatch.setattr(google.bc, "supabase", _SpySupabase())

    google.resolve_drive_reference("ws-1", "file123", "knowledge_note", "note-1")

    assert calls["upsert_knowledge_item"] == 0
    assert calls["upload_original_file"] == 0


# =====================================================================
# Chat -> Drive-reference resolution fix (2026-08-16): a real Chat KEEP
# message containing a real Drive link exposed the gap live -- Chat never
# called resolve_drive_references_in_text() at all (only Meet did).
# run_filtration() now takes an optional on_note_created(note_id,
# contributing_items) hook; connector_google_chat.poll_connection() wires
# it to scan the RAW message text of exactly the items that contributed to
# each note (never the distilled note body -- the classifier may paraphrase
# the URL out of it). These tests exercise that wiring end-to-end against
# the real vector DB (same pattern test_phase1_retrieval.py's provenance
# tests already use for Slack), with only the Drive metadata GET and the
# LLM/embedding calls mocked.
# =====================================================================

from test_phase1_retrieval import _stand_in_vector


def _make_test_chat_connection(workspace_id: str = TEST_COMPANY_1_WS) -> str:
    row = bc.supabase.table("connections").insert({
        "workspace_id": workspace_id,
        "provider": "google_drive",
        "external_team_id": f"TEST-CHAT-DRIVE-{_new_id()}",
        "external_team_name": "Chat Drive-Ref Test Connection",
        "access_token_enc": "not-a-real-token-never-decrypted-in-these-tests",
        "refresh_token_enc": "not-a-real-token-never-decrypted-in-these-tests",
        "status": "inactive",  # never picked up by the real scheduled worker
        "config": {"enabled_surfaces": ["chat"]},
    }).execute().data
    return row[0]["id"]


def _delete_test_chat_connection(connection_id: str) -> None:
    bc.supabase.table("connections").delete().eq("id", connection_id).execute()


def _insert_chat_ingest_items(connection_id: str, workspace_id: str, texts: list[str]) -> list[dict]:
    rows = [{
        "workspace_id":  workspace_id,
        "connection_id": connection_id,
        "provider":      "google_chat",
        "external_id":   f"spaces/test-{connection_id}/messages/{i}",
        "kind":          "message",
        "raw": {
            "channel": "spaces/test", "channel_name": "#test",
            "user": "users/u1", "user_name": "Alice",
            "text": text, "ts": f"2026-08-16T20:{40 + i:02d}:00Z",
            "thread_ts": None,
        },
        "status": "pending",
    } for i, text in enumerate(texts)]
    return bc.supabase.table("ingest_items").insert(rows).execute().data


def _run_chat_filtration_with_drive_hook(connection_id: str, workspace_id: str, monkeypatch,
                                         classify_response: dict) -> dict:
    """Mirrors exactly what connector_google_chat.poll_connection wires up,
    without the Google Chat API fetch itself (ingest_items are inserted
    directly, matching test_phase1_retrieval.py's Slack pattern)."""
    monkeypatch.setattr(bc.ai, "chat_json", lambda **k: classify_response)
    monkeypatch.setattr(bc, "embed_chunks", lambda chunks, **k: [_stand_in_vector()] * len(chunks))

    def _on_note_created(note_id, contributing):
        raw_text = "\n".join((it.get("raw", {}).get("text") or "") for it in contributing)
        if raw_text.strip():
            google.resolve_drive_references_in_text(
                workspace_id, raw_text, "knowledge_note", note_id, connection_id=connection_id,
            )

    return bc.run_filtration(workspace_id, connection_id, "google_chat",
                             on_note_created=_on_note_created)


def _mock_drive_metadata_lookup(monkeypatch, files: dict, calls: Optional[dict] = None):
    """files: {file_id: {"name":..., "webViewLink":..., "modifiedTime":...}}.
    Mocks exactly the two things resolve_drive_reference touches (the
    connection lookup and the single-file metadata GET) -- never a real
    Google API call, and calls['drive_get_paths'] records every path/params
    passed so tests can assert no byte-download endpoint was ever hit."""
    monkeypatch.setattr(google, "get_active_connection",
                        lambda ws, surface, connection_id=None: {"id": connection_id or "conn-1"})
    monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token")

    def _fake_drive_get(path, token, params=None):
        if calls is not None:
            calls.setdefault("drive_get_calls", []).append({"path": path, "params": params or {}})
        file_id = path.split("/")[-1]
        if file_id not in files:
            raise Exception("404 not found")
        meta = files[file_id]
        return {"id": file_id, "name": meta["name"], "webViewLink": meta["webViewLink"],
                "modifiedTime": meta["modifiedTime"]}
    monkeypatch.setattr(google, "_drive_get", _fake_drive_get)


def _refs_for(linked_id: str) -> list[dict]:
    return bc.supabase.table("external_references").select("*") \
        .eq("linked_object_id", linked_id).eq("linked_object_type", "knowledge_note").execute().data


def _cleanup_refs(*linked_ids):
    for lid in linked_ids:
        if lid:
            bc.supabase.table("external_references").delete().eq("linked_object_id", lid).execute()


def test_chat_keep_with_drive_url_creates_external_reference(monkeypatch):
    """Test 1: Chat KEEP + Drive URL -> external reference created."""
    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        _mock_drive_metadata_lookup(monkeypatch, {
            "1bdzBnUwqBT2WBXfgWelMgMDdU0l3": {
                "name": "Q4 Deck.pptx",
                "webViewLink": "https://docs.google.com/presentation/d/1bdzBnUwqBT2WBXfgWelMgMDdU0l3/edit",
                "modifiedTime": "2026-08-01T00:00:00Z",
            },
        })
        items = _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "Starting Sept 15 the launch needs QA sign-off: "
            "https://docs.google.com/presentation/d/1bdzBnUwqBT2WBXfgWelMgMDdU0l3/edit?usp=drive_link",
        ])
        result = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Launch QA gate", "note": "Launch requires QA sign-off starting Sept 15.",
                       "category": "process", "participants": [], "source_message_indices": [0]}]
        })
        assert result["notes_created"] == 1
        note_id = bc.supabase.table("ingest_items").select("note_id").eq("id", items[0]["id"]) \
            .execute().data[0]["note_id"]
        assert note_id

        refs = _refs_for(note_id)
        assert len(refs) == 1
        assert refs[0]["external_file_id"] == "1bdzBnUwqBT2WBXfgWelMgMDdU0l3"
        assert refs[0]["title"] == "Q4 Deck.pptx"
        assert refs[0]["workspace_id"] == TEST_COMPANY_1_WS
        assert refs[0]["provider"] == "google_drive"
        assert refs[0]["linked_object_type"] == "knowledge_note"
        assert refs[0]["linked_object_id"] == note_id

        # Repeat poll: no more pending items, hook never fires again, no duplicate.
        result2 = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {"items": []})
        assert result2["notes_created"] == 0
        assert len(_refs_for(note_id)) == 1
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)


def test_chat_keep_without_drive_url_creates_no_reference(monkeypatch):
    """Test 2: Chat KEEP without a Drive URL -> no external reference."""
    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        calls = {}
        _mock_drive_metadata_lookup(monkeypatch, {}, calls)
        _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "Starting Sept 15 the launch needs QA sign-off, no link this time.",
        ])
        result = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Launch QA gate", "note": "Launch requires QA sign-off starting Sept 15.",
                       "category": "process", "participants": [], "source_message_indices": [0]}]
        })
        assert result["notes_created"] == 1
        note_id = bc.supabase.table("knowledge_notes").select("id").eq("connection_id", conn_id) \
            .execute().data[0]["id"]
        assert _refs_for(note_id) == []
        assert calls.get("drive_get_calls", []) == [], "no Drive link in text -> Drive API must never be called"
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)


def test_chat_discard_with_drive_url_creates_no_reference(monkeypatch):
    """Test 3: Chat DISCARD + Drive URL -> no external reference (hook only
    fires for items that actually became a KEEP note)."""
    conn_id = _make_test_chat_connection()
    try:
        calls = {}
        _mock_drive_metadata_lookup(monkeypatch, {"fileABC1234567890": {
            "name": "x", "webViewLink": "url", "modifiedTime": "2026-08-01T00:00:00Z"}}, calls)
        items = _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "lol check this out https://drive.google.com/file/d/fileABC1234567890/view",
        ])
        result = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch,
                                                      {"items": []})  # pure noise, nothing kept
        assert result["notes_created"] == 0
        status = bc.supabase.table("ingest_items").select("status").eq("id", items[0]["id"]) \
            .execute().data[0]["status"]
        assert status == "discarded"
        assert calls.get("drive_get_calls", []) == [], "a discarded message must never trigger Drive resolution"
    finally:
        _delete_test_chat_connection(conn_id)


def test_chat_drive_url_dropped_from_distilled_note_still_creates_reference(monkeypatch):
    """Test 4 (THE regression this whole fix targets): the LLM's distilled
    note.body omits the Drive URL entirely, but the reference must still be
    created because resolution scans the ORIGINAL raw message text, not the
    distilled body."""
    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        _mock_drive_metadata_lookup(monkeypatch, {"1bdzBnUwqBT2WBXfgWelMgMDdU0l3": {
            "name": "Q4 Deck.pptx",
            "webViewLink": "https://docs.google.com/presentation/d/1bdzBnUwqBT2WBXfgWelMgMDdU0l3/edit",
            "modifiedTime": "2026-08-01T00:00:00Z",
        }})
        _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "Starting Sept 15 the launch needs QA sign-off: "
            "https://docs.google.com/presentation/d/1bdzBnUwqBT2WBXfgWelMgMDdU0l3/edit?usp=drive_link",
        ])
        # Distilled note body deliberately has NO URL in it -- exactly what
        # the live incident showed the real classifier produces.
        result = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Launch QA gate",
                       "note": "Launch requires QA sign-off from Product and QA starting Sept 15.",
                       "category": "process", "participants": [], "source_message_indices": [0]}]
        })
        assert result["notes_created"] == 1
        note_row = bc.supabase.table("knowledge_notes").select("id,body").eq("connection_id", conn_id) \
            .execute().data[0]
        note_id = note_row["id"]
        assert "docs.google.com" not in note_row["body"], "fixture sanity check: body really has no URL"

        refs = _refs_for(note_id)
        assert len(refs) == 1
        assert refs[0]["external_file_id"] == "1bdzBnUwqBT2WBXfgWelMgMDdU0l3"
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)


def test_chat_multiple_drive_urls_one_message_creates_one_reference_per_file(monkeypatch):
    """Test 5: multiple distinct Drive URLs in one raw message -> one
    external_references row per distinct file."""
    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        _mock_drive_metadata_lookup(monkeypatch, {
            "fileAAAAAAAAAAAAAAAAA": {"name": "A", "webViewLink": "urlA", "modifiedTime": "2026-08-01T00:00:00Z"},
            "fileBBBBBBBBBBBBBBBBB": {"name": "B", "webViewLink": "urlB", "modifiedTime": "2026-08-01T00:00:00Z"},
        })
        _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "Spec: https://drive.google.com/file/d/fileAAAAAAAAAAAAAAAAA/view "
            "and data: https://drive.google.com/file/d/fileBBBBBBBBBBBBBBBBB/view",
        ])
        result = _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Spec + data", "note": "Team reviewed the spec and data files.",
                       "category": "fact", "participants": [], "source_message_indices": [0]}]
        })
        assert result["notes_created"] == 1
        note_id = bc.supabase.table("knowledge_notes").select("id").eq("connection_id", conn_id) \
            .execute().data[0]["id"]

        refs = _refs_for(note_id)
        assert len(refs) == 2
        assert {r["external_file_id"] for r in refs} == {"fileAAAAAAAAAAAAAAAAA", "fileBBBBBBBBBBBBBBBBB"}
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)


def test_chat_drive_reference_repeat_resolution_does_not_duplicate(monkeypatch):
    """Test 6: calling resolution twice for the same (note, file) pair --
    e.g. a hook misfire or manual replay -- must not create a second row."""
    conn_id = _make_test_chat_connection()
    try:
        _mock_drive_metadata_lookup(monkeypatch, {"fileCCCCCCCCCCCCCCCCC": {
            "name": "C", "webViewLink": "urlC", "modifiedTime": "2026-08-01T00:00:00Z"}})
        note_id = str(uuid.uuid4())
        text = "doc: https://drive.google.com/file/d/fileCCCCCCCCCCCCCCCCC/view"
        google.resolve_drive_references_in_text(TEST_COMPANY_1_WS, text, "knowledge_note", note_id,
                                                 connection_id=conn_id)
        google.resolve_drive_references_in_text(TEST_COMPANY_1_WS, text, "knowledge_note", note_id,
                                                 connection_id=conn_id)
        refs = _refs_for(note_id)
        assert len(refs) == 1
    finally:
        _cleanup_refs(note_id)
        _delete_test_chat_connection(conn_id)


def test_chat_drive_reference_wrong_workspace_rejected(monkeypatch):
    """Test 7: resolving through a connection that doesn't belong to the
    claimed workspace must fail closed -- zero rows, no exception."""
    conn_id = _make_test_chat_connection(workspace_id=TEST_COMPANY_1_WS)
    try:
        monkeypatch.setattr(google, "get_active_connection",
                            lambda ws, surface, connection_id=None: None)  # simulates workspace mismatch
        note_id = str(uuid.uuid4())
        refs = google.resolve_drive_references_in_text(
            "some-other-workspace-id", "doc: https://drive.google.com/file/d/fileDDDDDDDDDDDDDDDDD/view",
            "knowledge_note", note_id, connection_id=conn_id,
        )
        assert refs == []
        assert _refs_for(note_id) == []
    finally:
        _delete_test_chat_connection(conn_id)


def test_chat_drive_reference_inactive_connection_rejected(monkeypatch):
    """Test 8: an inactive/wrong connection_id must fail closed -- this is
    exactly what get_active_connection already enforces (see
    test_get_active_connection_rejects_inactive_or_wrong_provider); here we
    confirm Chat's hook surfaces that as zero references, not an exception."""
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([]))  # no matching active row
    note_id = str(uuid.uuid4())
    refs = google.resolve_drive_references_in_text(
        "ws-1", "doc: https://drive.google.com/file/d/fileEEEEEEEEEEEEEEEEE/view",
        "knowledge_note", note_id, connection_id="conn-does-not-exist-or-inactive",
    )
    assert refs == []


def test_chat_drive_reference_surface_disabled_ignored_safely(monkeypatch):
    """Test 9: a real, active Chat connection that never had 'drive' added
    to enabled_surfaces must not resolve references -- ignored safely, no
    exception, no row."""
    monkeypatch.setattr(google.bc, "supabase", _FakeConnectionsClient([
        {"id": "conn-1", "workspace_id": "ws-1", "provider": "google_drive",
         "status": "active", "config": {"enabled_surfaces": ["chat"]}},  # drive NOT enabled
    ]))
    note_id = str(uuid.uuid4())
    refs = google.resolve_drive_references_in_text(
        "ws-1", "doc: https://drive.google.com/file/d/fileFFFFFFFFFFFFFFFFF/view",
        "knowledge_note", note_id, connection_id="conn-1",
    )
    assert refs == []


def test_chat_drive_reference_never_downloads_file_bytes(monkeypatch):
    """Test 10: the metadata GET must only ever request id/name/webViewLink/
    modifiedTime fields -- never alt=media (the byte-download form)."""
    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        calls = {}
        _mock_drive_metadata_lookup(monkeypatch, {"fileGGGGGGGGGGGGGGGGG": {
            "name": "G", "webViewLink": "urlG", "modifiedTime": "2026-08-01T00:00:00Z"}}, calls)
        _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "doc: https://drive.google.com/file/d/fileGGGGGGGGGGGGGGGGG/view",
        ])
        _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Doc", "note": "A doc was shared.", "category": "fact",
                       "participants": [], "source_message_indices": [0]}]
        })
        note_id = bc.supabase.table("knowledge_notes").select("id").eq("connection_id", conn_id) \
            .execute().data[0]["id"]

        drive_calls = calls.get("drive_get_calls", [])
        assert len(drive_calls) == 1
        assert drive_calls[0]["path"] == "files/fileGGGGGGGGGGGGGGGGG"
        assert drive_calls[0]["params"].get("fields") == "id,name,webViewLink,modifiedTime"
        assert "alt" not in drive_calls[0]["params"], "must never request file content (alt=media)"
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)


def test_chat_drive_reference_creates_no_knowledge_items_or_storage_writes(monkeypatch):
    """Tests 11 + 12: even when a real reference IS created from a real
    Chat KEEP note, no knowledge_items row and no Storage write ever
    happens -- Drive stays reference-only end to end through this new
    code path, not just in resolve_drive_reference() in isolation."""
    import drive_app_db
    calls = {"upsert_knowledge_item": 0, "upload_original_file": 0}
    monkeypatch.setattr(drive_app_db, "upsert_knowledge_item",
                        lambda *_a, **_kw: calls.__setitem__("upsert_knowledge_item", calls["upsert_knowledge_item"] + 1))
    monkeypatch.setattr(drive_app_db, "upload_original_file",
                        lambda *_a, **_kw: calls.__setitem__("upload_original_file", calls["upload_original_file"] + 1))

    conn_id = _make_test_chat_connection()
    note_id = None
    try:
        _mock_drive_metadata_lookup(monkeypatch, {"fileHHHHHHHHHHHHHHHHH": {
            "name": "H", "webViewLink": "urlH", "modifiedTime": "2026-08-01T00:00:00Z"}})
        _insert_chat_ingest_items(conn_id, TEST_COMPANY_1_WS, [
            "doc: https://drive.google.com/file/d/fileHHHHHHHHHHHHHHHHH/view",
        ])
        _run_chat_filtration_with_drive_hook(conn_id, TEST_COMPANY_1_WS, monkeypatch, {
            "items": [{"title": "Doc", "note": "A doc was shared.", "category": "fact",
                       "participants": [], "source_message_indices": [0]}]
        })
        note_id = bc.supabase.table("knowledge_notes").select("id").eq("connection_id", conn_id) \
            .execute().data[0]["id"]
        assert len(_refs_for(note_id)) == 1, "fixture sanity check: a reference really was created"

        assert calls["upsert_knowledge_item"] == 0
        assert calls["upload_original_file"] == 0
    finally:
        _cleanup_refs(note_id)
        bc.delete_note(note_id) if note_id else None
        _delete_test_chat_connection(conn_id)
