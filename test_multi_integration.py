"""
Multi-integration management (2026-08-16): a workspace may have up to 10
ACTIVE integrations total, across any mix of providers -- no more "one
connection per provider" assumption. connection_id is the only real
identity; display_name is descriptive metadata only; Disconnect must always
exist, for any connection state, and must only ever touch that one row.

Most tests here hit the REAL vector DB (same SUPABASE_URL/SUPABASE_SERVICE_KEY
every other test in this repo uses) with synthetic, random workspace_ids —
`connections` has no foreign key to a real `workspaces` row (confirmed live:
only a primary key on id, plus the unique index on
workspace_id+provider+external_team_id), so this is safe and self-cleaning.

Run with: python -m pytest test_multi_integration.py -v
"""
import uuid
from types import SimpleNamespace

import pytest

import brain_connectors as bc
import integrations
from auth import AuthContext


def _ws() -> str:
    return str(uuid.uuid4())


def _cleanup(*workspace_ids: str):
    for ws in workspace_ids:
        bc.supabase.table("connections").delete().eq("workspace_id", ws).execute()


def _admin_auth(ws: str) -> AuthContext:
    return AuthContext(user_id="u1", workspaces={ws: "admin"})


def _member_auth(ws: str) -> AuthContext:
    return AuthContext(user_id="u2", workspaces={ws: "member"})


# =====================================================================
# 1-2. Basic + same-provider coexistence
# =====================================================================

def test_one_integration_works():
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "slack", display_name="Main Slack")
        assert conn["status"] == "pending"
        assert conn["display_name"] == "Main Slack"
        assert bc.count_active_connections(ws) == 1
    finally:
        _cleanup(ws)


def test_two_same_provider_integrations_work():
    ws = _ws()
    try:
        c1 = bc.create_pending_connection(ws, "slack", display_name="Main Slack")
        c2 = bc.create_pending_connection(ws, "slack", display_name="Engineering Slack")
        assert c1["id"] != c2["id"]
        assert bc.count_active_connections(ws) == 2

        rows = bc.supabase.table("connections").select("id,provider").eq("workspace_id", ws).execute().data
        assert len(rows) == 2
        assert all(r["provider"] == "slack" for r in rows)
    finally:
        _cleanup(ws)


# =====================================================================
# 3-4. 10-active-integration limit, enforced server-side
# =====================================================================

def test_up_to_ten_active_integrations_works():
    ws = _ws()
    try:
        for i in range(10):
            bc.create_pending_connection(ws, "slack", display_name=f"Slack {i}")
        assert bc.count_active_connections(ws) == 10
    finally:
        _cleanup(ws)


def test_eleventh_active_integration_rejected_server_side():
    ws = _ws()
    try:
        for i in range(10):
            bc.create_pending_connection(ws, "slack", display_name=f"Slack {i}")
        with pytest.raises(Exception) as exc:
            bc.create_pending_connection(ws, "zoom", display_name="One too many")
        assert getattr(exc.value, "status_code", None) == 400
        assert bc.count_active_connections(ws) == 10, "the rejected 11th must not have been created"
    finally:
        _cleanup(ws)


def test_revoked_connections_do_not_count_toward_limit():
    ws = _ws()
    try:
        conns = [bc.create_pending_connection(ws, "slack", display_name=f"Slack {i}") for i in range(10)]
        # Disconnect one -- frees a slot immediately (locked requirement:
        # "if a pending/stuck connection counts temporarily, Disconnect must
        # immediately free that slot").
        bc.supabase.table("connections").update({"status": "revoked"}).eq("id", conns[0]["id"]).execute()
        assert bc.count_active_connections(ws) == 9
        new_conn = bc.create_pending_connection(ws, "zoom", display_name="Now it fits")
        assert new_conn is not None
        assert bc.count_active_connections(ws) == 10
    finally:
        _cleanup(ws)


def test_error_status_connections_count_toward_limit():
    """'error' (broken, needs reconnect/disconnect) occupies a slot -- it's
    a real row the user must act on, not a free pass."""
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "slack")
        bc.supabase.table("connections").update({"status": "error"}).eq("id", conn["id"]).execute()
        assert bc.count_active_connections(ws) == 1
    finally:
        _cleanup(ws)


# =====================================================================
# 5-6. Display name
# =====================================================================

def test_optional_name_persists():
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "google_drive", display_name="CEO Personal Google")
        row = bc.supabase.table("connections").select("display_name").eq("id", conn["id"]).execute().data[0]
        assert row["display_name"] == "CEO Personal Google"
    finally:
        _cleanup(ws)


def test_name_is_optional():
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "slack")
        assert conn["display_name"] is None
    finally:
        _cleanup(ws)


def test_rename_only_affects_that_connection():
    ws = _ws()
    try:
        a = bc.create_pending_connection(ws, "slack", display_name="A")
        b = bc.create_pending_connection(ws, "slack", display_name="B")

        bc.rename_connection(a["id"], ws, "A renamed")

        row_a = bc.supabase.table("connections").select("display_name").eq("id", a["id"]).execute().data[0]
        row_b = bc.supabase.table("connections").select("display_name").eq("id", b["id"]).execute().data[0]
        assert row_a["display_name"] == "A renamed"
        assert row_b["display_name"] == "B", "renaming A must never touch B"
    finally:
        _cleanup(ws)


# =====================================================================
# 7-11. Disconnect always exists, for every state, and only touches ONE row
# =====================================================================

@pytest.mark.parametrize("initial_status", ["active", "pending", "error"])
def test_disconnect_works_regardless_of_connection_state(initial_status):
    """Requirements 7/8/9: disconnect must always be available -- active,
    pending (mid-OAuth), or error (failed/stuck) -- and always succeeds."""
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "slack")
        bc.supabase.table("connections").update({"status": initial_status}).eq("id", conn["id"]).execute()

        import asyncio
        result = asyncio.run(bc.disconnect(conn["id"], delete_notes=False, auth=_admin_auth(ws)))
        assert result["success"] is True

        row = bc.supabase.table("connections").select("status").eq("id", conn["id"]).execute().data[0]
        assert row["status"] == "revoked"
    finally:
        _cleanup(ws)


def test_reconnect_after_disconnect_creates_a_clean_setup():
    """Requirement 10: disconnect, then start over -- a fresh
    create_pending_connection call works cleanly, independent of the
    revoked row, and the revoked row is left alone (history preserved)."""
    ws = _ws()
    try:
        old = bc.create_pending_connection(ws, "slack", display_name="First attempt")
        import asyncio
        asyncio.run(bc.disconnect(old["id"], delete_notes=False, auth=_admin_auth(ws)))

        fresh = bc.create_pending_connection(ws, "slack", display_name="Second attempt")
        assert fresh["id"] != old["id"]
        assert fresh["status"] == "pending"

        old_row = bc.supabase.table("connections").select("status,display_name").eq("id", old["id"]).execute().data[0]
        assert old_row["status"] == "revoked"
        assert old_row["display_name"] == "First attempt", "history preserved, not deleted"
    finally:
        _cleanup(ws)


def test_disconnect_connection_a_does_not_affect_b():
    ws = _ws()
    try:
        a = bc.create_pending_connection(ws, "slack", display_name="A")
        b = bc.create_pending_connection(ws, "slack", display_name="B")
        bc.supabase.table("connections").update({"status": "active"}).eq("id", a["id"]).execute()
        bc.supabase.table("connections").update({"status": "active"}).eq("id", b["id"]).execute()

        import asyncio
        asyncio.run(bc.disconnect(a["id"], delete_notes=False, auth=_admin_auth(ws)))

        row_a = bc.supabase.table("connections").select("status").eq("id", a["id"]).execute().data[0]
        row_b = bc.supabase.table("connections").select("status").eq("id", b["id"]).execute().data[0]
        assert row_a["status"] == "revoked"
        assert row_b["status"] == "active", "disconnecting A must never touch B"
    finally:
        _cleanup(ws)


# =====================================================================
# 13-14. Cross-workspace security
# =====================================================================

def test_cross_workspace_connection_access_rejected():
    ws_a, ws_b = _ws(), _ws()
    try:
        conn = bc.create_pending_connection(ws_a, "slack")
        assert bc.get_connection_for_workspace(conn["id"], ws_b) is None, \
            "workspace B must never resolve workspace A's connection"
        assert bc.get_connection_for_workspace(conn["id"], ws_a) is not None
    finally:
        _cleanup(ws_a, ws_b)


def test_cross_workspace_disconnect_rejected():
    ws_a, ws_b = _ws(), _ws()
    try:
        conn = bc.create_pending_connection(ws_a, "slack")
        wrong_workspace_auth = AuthContext(user_id="attacker", workspaces={ws_b: "admin"})

        import asyncio
        with pytest.raises(Exception) as exc:
            asyncio.run(bc.disconnect(conn["id"], delete_notes=False, auth=wrong_workspace_auth))
        assert getattr(exc.value, "status_code", None) in (403, 404)

        row = bc.supabase.table("connections").select("status").eq("id", conn["id"]).execute().data[0]
        assert row["status"] == "pending", "the disconnect attempt from the wrong workspace must not have applied"
    finally:
        _cleanup(ws_a, ws_b)


def test_cross_workspace_rename_rejected():
    ws_a, ws_b = _ws(), _ws()
    try:
        conn = bc.create_pending_connection(ws_a, "slack", display_name="Original")
        with pytest.raises(Exception) as exc:
            bc.rename_connection(conn["id"], ws_b, "Hijacked name")
        assert getattr(exc.value, "status_code", None) == 404

        row = bc.supabase.table("connections").select("display_name").eq("id", conn["id"]).execute().data[0]
        assert row["display_name"] == "Original"
    finally:
        _cleanup(ws_a, ws_b)


# =====================================================================
# 15-17. Same-provider connections don't collide
# =====================================================================

def test_multiple_google_connections_with_different_enabled_surfaces():
    ws = _ws()
    try:
        personal = bc.create_pending_connection(ws, "google_drive", display_name="CEO Personal")
        company = bc.create_pending_connection(ws, "google_drive", display_name="Company Workspace")

        bc.supabase.table("connections").update(
            {"status": "active", "config": {"enabled_surfaces": ["drive"]}, "external_team_id": "ceo@personal.com"}
        ).eq("id", personal["id"]).execute()
        bc.supabase.table("connections").update(
            {"status": "active", "config": {"enabled_surfaces": ["calendar", "meet", "chat", "drive"]},
             "external_team_id": "admin@company.com"}
        ).eq("id", company["id"]).execute()

        p = bc.supabase.table("connections").select("config").eq("id", personal["id"]).execute().data[0]
        c = bc.supabase.table("connections").select("config").eq("id", company["id"]).execute().data[0]
        assert p["config"]["enabled_surfaces"] == ["drive"]
        assert set(c["config"]["enabled_surfaces"]) == {"calendar", "meet", "chat", "drive"}
    finally:
        _cleanup(ws)


def test_multiple_slack_connections_do_not_collide():
    ws = _ws()
    try:
        a = bc.create_pending_connection(ws, "slack", display_name="Main")
        b = bc.create_pending_connection(ws, "slack", display_name="Engineering")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "T_MAIN", "app_id": "A_MAIN"}
        ).eq("id", a["id"]).execute()
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "T_ENG", "app_id": "A_ENG"}
        ).eq("id", b["id"]).execute()

        rows = bc.supabase.table("connections").select("id,external_team_id") \
            .eq("workspace_id", ws).eq("provider", "slack").execute().data
        team_ids = {r["external_team_id"] for r in rows}
        assert team_ids == {"T_MAIN", "T_ENG"}
    finally:
        _cleanup(ws)


def test_multiple_zoom_connections_do_not_collide():
    ws = _ws()
    try:
        a = bc.create_pending_connection(ws, "zoom", display_name="Exec Zoom")
        b = bc.create_pending_connection(ws, "zoom", display_name="Support Zoom")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "ACCT_EXEC"}
        ).eq("id", a["id"]).execute()
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "ACCT_SUPPORT"}
        ).eq("id", b["id"]).execute()

        rows = bc.supabase.table("connections").select("id,external_team_id") \
            .eq("workspace_id", ws).eq("provider", "zoom").execute().data
        assert {r["external_team_id"] for r in rows} == {"ACCT_EXEC", "ACCT_SUPPORT"}
    finally:
        _cleanup(ws)


# =====================================================================
# integrations.py endpoint-level: list/create/rename with real data
# =====================================================================

def test_list_integrations_shows_all_connections_per_provider_not_collapsed():
    """The pre-refactor bug: by_provider = {c['provider']: c for c in conns}
    silently dropped every connection but the last per provider."""
    ws = _ws()
    try:
        bc.create_pending_connection(ws, "slack", display_name="Main")
        bc.create_pending_connection(ws, "slack", display_name="Engineering")

        import asyncio
        result = asyncio.run(integrations.list_integrations(ws, _admin_auth(ws)))
        slack_entry = next(i for i in result["integrations"] if i["id"] == "slack")
        assert len(slack_entry["connections"]) == 2
        names = {c["display_name"] for c in slack_entry["connections"]}
        assert names == {"Main", "Engineering"}
    finally:
        _cleanup(ws)


def test_create_connection_endpoint_enforces_limit_and_returns_id():
    ws = _ws()
    try:
        import asyncio
        for i in range(10):
            body = integrations.CreateConnectionRequest(workspace_id=ws, provider="slack", display_name=f"S{i}")
            result = asyncio.run(integrations.create_connection(body, _admin_auth(ws)))
            assert result["connection_id"]

        body_11 = integrations.CreateConnectionRequest(workspace_id=ws, provider="zoom", display_name="11th")
        with pytest.raises(Exception) as exc:
            asyncio.run(integrations.create_connection(body_11, _admin_auth(ws)))
        assert getattr(exc.value, "status_code", None) == 400
    finally:
        _cleanup(ws)


def test_rename_endpoint_is_workspace_scoped():
    ws_a, ws_b = _ws(), _ws()
    try:
        import asyncio
        create_body = integrations.CreateConnectionRequest(workspace_id=ws_a, provider="slack", display_name="Mine")
        created = asyncio.run(integrations.create_connection(create_body, _admin_auth(ws_a)))

        rename_body = integrations.RenameConnectionRequest(workspace_id=ws_b, display_name="Stolen")
        with pytest.raises(Exception) as exc:
            asyncio.run(integrations.rename_connection_route(created["connection_id"], rename_body, _admin_auth(ws_b)))
        assert getattr(exc.value, "status_code", None) == 404
    finally:
        _cleanup(ws_a, ws_b)


# =====================================================================
# Webhook credential/routing ambiguity -- shared helper (used by both
# Slack and Zoom) and Zoom's own recording.completed connection selection,
# closing the "6. Provider-specific routing" acceptance items.
# =====================================================================

def test_get_provider_credentials_by_external_team_fails_closed_on_cross_workspace_ambiguity():
    """The shared webhook-signature-lookup helper (used by BOTH Slack's and
    Zoom's webhook handlers): the same external_team_id legitimately
    resolving to connections in TWO DIFFERENT workspaces must not pick one
    arbitrarily."""
    ws_a, ws_b = _ws(), _ws()
    try:
        a = bc.create_pending_connection(ws_a, "zoom")
        b = bc.create_pending_connection(ws_b, "zoom")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SHARED_ACCT"}
        ).eq("id", a["id"]).execute()
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SHARED_ACCT"}
        ).eq("id", b["id"]).execute()

        result = bc.get_provider_credentials_by_external_team("zoom", "SHARED_ACCT")
        assert result is None, "ambiguous across workspaces -- must fail closed, never guess"
    finally:
        _cleanup(ws_a, ws_b)


def test_get_provider_credentials_by_external_team_resolves_when_unambiguous(monkeypatch):
    # CONNECTOR_ENCRYPTION_KEY isn't provisioned in this local test
    # environment (a real secret, deliberately not available here -- see
    # this session's established pattern) -- not what this test is about,
    # so encryption is stubbed to identity for just this narrow check.
    monkeypatch.setattr(bc, "encrypt_secret", lambda v: v)
    monkeypatch.setattr(bc, "decrypt_secret", lambda v: v)
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "zoom")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SOLO_ACCT"}
        ).eq("id", conn["id"]).execute()
        bc.save_provider_credentials(ws, "zoom", client_id="cid", client_secret="secret", webhook_secret="whsec")

        result = bc.get_provider_credentials_by_external_team("zoom", "SOLO_ACCT")
        assert result is not None
        assert result["client_id"] == "cid"
    finally:
        _cleanup(ws)
        bc.supabase.table("provider_credentials").delete().eq("workspace_id", ws).eq("provider", "zoom").execute()


class _FakeZoomRequest:
    """Minimal stand-in for FastAPI's Request -- only what zoom_events reads."""
    def __init__(self, body_bytes: bytes):
        self._body = body_bytes
        self.headers = {}
    async def body(self):
        return self._body


def test_zoom_webhook_ambiguous_recording_completed_fails_closed(monkeypatch):
    """Requirement 6 (Zoom): an ambiguous recording.completed event (same
    account_id active in more than one workspace) must never guess which
    workspace's brain to write into -- _process_recording must not be
    called at all."""
    import connector_zoom as zoom
    ws_a, ws_b = _ws(), _ws()
    try:
        a = bc.create_pending_connection(ws_a, "zoom")
        b = bc.create_pending_connection(ws_b, "zoom")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SHARED_ACCT_2"}
        ).eq("id", a["id"]).execute()
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SHARED_ACCT_2"}
        ).eq("id", b["id"]).execute()

        called = {"count": 0}
        monkeypatch.setattr(zoom, "_process_recording", lambda *a_, **kw: called.__setitem__("count", called["count"] + 1))

        payload = {
            "event": "recording.completed",
            "payload": {"account_id": "SHARED_ACCT_2",
                       "object": {"uuid": "m1", "topic": "t", "recording_files": [], "start_time": "now"}},
        }
        import json as _json
        req = _FakeZoomRequest(_json.dumps(payload).encode())

        import asyncio
        result = asyncio.run(zoom.zoom_events(req))

        assert called["count"] == 0, "ambiguous account_id must never trigger processing"
        assert result.status_code == 200  # still acknowledges receipt to Zoom
    finally:
        _cleanup(ws_a, ws_b)


def test_zoom_webhook_unambiguous_recording_completed_processes(monkeypatch):
    """Sanity counterpart: exactly one active connection for the account_id
    -> processing proceeds normally."""
    import connector_zoom as zoom
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "zoom")
        bc.supabase.table("connections").update(
            {"status": "active", "external_team_id": "SOLO_ACCT_2",
             "access_token_enc": "", "refresh_token_enc": "", "token_expires_at": None}
        ).eq("id", conn["id"]).execute()

        called = {"count": 0}
        monkeypatch.setattr(zoom, "_process_recording", lambda *a_, **kw: called.__setitem__("count", called["count"] + 1))
        monkeypatch.setattr(zoom, "_valid_access_token", lambda _c: "fake-token")

        payload = {
            "event": "recording.completed",
            "payload": {"account_id": "SOLO_ACCT_2",
                       "object": {"uuid": "m1", "topic": "t", "recording_files": [], "start_time": "now"}},
        }
        import json as _json
        req = _FakeZoomRequest(_json.dumps(payload).encode())

        import asyncio
        asyncio.run(zoom.zoom_events(req))

        assert called["count"] == 1
    finally:
        _cleanup(ws)


# =====================================================================
# OAuth callback idempotency (3.F)
# =====================================================================

def test_google_callback_update_by_id_is_idempotent(monkeypatch):
    """Calling the same connection's finalize-UPDATE twice (e.g. a retried
    callback) converges to the same end state, never creates a second row
    or corrupts config."""
    import connector_google as google
    ws = _ws()
    try:
        conn = bc.create_pending_connection(ws, "google_drive", display_name="Idempotency test")

        monkeypatch.setattr(bc, "encrypt_secret", lambda v: v)  # see note above re: CONNECTOR_ENCRYPTION_KEY
        monkeypatch.setattr(google, "_google_credentials", lambda w: ("cid", "csecret"))
        monkeypatch.setattr(bc, "decode_oauth_state",
                            lambda state: {"w": ws, "u": "user-1", "connection_id": conn["id"], "surfaces": ["drive"]})

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", lambda *a, **kw: SimpleNamespace(
            json=lambda: {"access_token": "tok", "refresh_token": "rt", "expires_in": 3600, "scope": "x"}))
        monkeypatch.setattr(_httpx, "get", lambda *a, **kw: SimpleNamespace(json=lambda: {"email": "idempotent@co.com"}))

        import asyncio
        asyncio.run(google.google_callback(code="c1", state="s1", error=""))
        asyncio.run(google.google_callback(code="c2", state="s1", error=""))

        rows = bc.supabase.table("connections").select("id").eq("workspace_id", ws).execute().data
        assert len(rows) == 1, "repeated callback for the same connection must never create a second row"
        row = bc.supabase.table("connections").select("status,external_team_id").eq("id", conn["id"]).execute().data[0]
        assert row["status"] == "active"
        assert row["external_team_id"] == "idempotent@co.com"
    finally:
        _cleanup(ws)
