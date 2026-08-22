"""
Public self-serve trial: onboarding, the mandatory mobile gate, and HARD tenant
isolation.

WHY THIS SUITE MAKES REAL REQUESTS. Reading a pg_policies row proves a policy
was written, not that it stops anybody. Every isolation assertion below is an
actual HTTP call to PostgREST carrying a real end-user JWT, so what is measured
is what a hostile client would actually get back. The service key is used only
to build and destroy fixtures -- never to make an assertion, because a
service-role request bypasses RLS and would pass no matter how broken the
policies were.

TWO SYNTHETIC TENANTS. User A and User B are created through the Auth admin
API, each provisioned through the real `provision_trial_workspace` RPC, and then
pointed at each other. Everything they own is deleted in `teardown_module`,
including their auth users, so the suite leaves nothing behind in a live
database it shares with real customer data.

NOTHING HERE PRINTS A SECRET. Keys come from the environment, tokens are held in
memory, and no assertion message interpolates a credential or an OTP.
"""
import os
import time
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

APP_URL = (os.getenv("APP_SUPABASE_URL") or "").rstrip("/")
ANON = os.getenv("APP_SUPABASE_ANON_KEY") or ""
SERVICE = os.getenv("APP_SUPABASE_SERVICE_KEY") or ""

pytestmark = pytest.mark.skipif(
    not (APP_URL and ANON and SERVICE),
    reason="APP_SUPABASE_URL / APP_SUPABASE_ANON_KEY / APP_SUPABASE_SERVICE_KEY required",
)

TRIAL_PLAN = "trial_7day"
_PW = "Trial-Isolation-" + uuid.uuid4().hex[:12] + "!aA1"

STATE: dict = {"a": {}, "b": {}}


# ── fixture plumbing (service role: build/destroy only, never assert) ────────

def _svc(method: str, path: str, **kw) -> httpx.Response:
    headers = {"apikey": SERVICE, "Authorization": f"Bearer {SERVICE}",
               "Content-Type": "application/json"}
    headers.update(kw.pop("headers", {}))
    with httpx.Client(timeout=30) as c:
        return c.request(method, f"{APP_URL}{path}", headers=headers, **kw)


def _as_user(token: str, method: str, path: str, **kw) -> httpx.Response:
    """A request exactly as the browser would send it: anon apikey plus the
    end user's access token. This is the surface a real attacker has."""
    headers = {"apikey": ANON, "Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    headers.update(kw.pop("headers", {}))
    with httpx.Client(timeout=30) as c:
        return c.request(method, f"{APP_URL}{path}", headers=headers, **kw)


def _create_user(email: str, *, confirm_phone: bool) -> dict:
    body = {"email": email, "password": _PW, "email_confirm": True}
    if confirm_phone:
        # A real onboarding sets this only via verifyOtp. The admin API is used
        # here purely to reach the post-OTP state without sending an SMS.
        body["phone"] = "+1555" + str(int(time.time() * 1000))[-7:]
        body["phone_confirm"] = True
    r = _svc("POST", "/auth/v1/admin/users", json=body)
    assert r.status_code in (200, 201), f"could not create fixture user: {r.status_code}"
    return r.json()


def _token_for(email: str) -> str:
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{APP_URL}/auth/v1/token?grant_type=password",
                   headers={"apikey": ANON, "Content-Type": "application/json"},
                   json={"email": email, "password": _PW})
    assert r.status_code == 200, f"fixture sign-in failed: {r.status_code}"
    return r.json()["access_token"]


def _rpc(token: str, fn: str, payload: dict) -> httpx.Response:
    return _as_user(token, "POST", f"/rest/v1/rpc/{fn}", json=payload)


def setup_module():
    stamp = uuid.uuid4().hex[:10]
    for key, phone_ok in (("a", True), ("b", True)):
        email = f"trialtest-{key}-{stamp}@knova-isolation.test"
        u = _create_user(email, confirm_phone=phone_ok)
        STATE[key]["email"] = email
        STATE[key]["user_id"] = u["id"]
        STATE[key]["token"] = _token_for(email)

    # A third identity that never verified a mobile, for the gate tests.
    email_c = f"trialtest-c-{stamp}@knova-isolation.test"
    u = _create_user(email_c, confirm_phone=False)
    STATE["c"] = {"email": email_c, "user_id": u["id"], "token": _token_for(email_c)}

    # Provision A and B through the REAL onboarding path.
    for key in ("a", "b"):
        r = _rpc(STATE[key]["token"], "provision_trial_workspace",
                 {"p_full_name": f"Isolation {key.upper()}",
                  "p_workspace_name": f"Isolation {key.upper()} Co"})
        assert r.status_code == 200, f"provisioning {key} failed: {r.status_code} {r.text[:200]}"
        STATE[key]["ws"] = r.json()


def teardown_module():
    for key in ("a", "b", "c"):
        st = STATE.get(key) or {}
        ws = st.get("ws")
        if ws:
            _svc("DELETE", f"/rest/v1/workspace_members?workspace_id=eq.{ws}")
            _svc("DELETE", f"/rest/v1/dashboards?workspace_id=eq.{ws}")
            _svc("DELETE", f"/rest/v1/workspace_settings?workspace_id=eq.{ws}")
            _svc("DELETE", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}")
            _svc("DELETE", f"/rest/v1/departments?workspace_id=eq.{ws}")
            _svc("DELETE", f"/rest/v1/workspaces?id=eq.{ws}")
        uid = st.get("user_id")
        if uid:
            _svc("DELETE", f"/rest/v1/profiles?id=eq.{uid}")
            _svc("DELETE", f"/auth/v1/admin/users/{uid}")


# =====================================================================
# 1-8. Onboarding and the mandatory mobile gate.
# =====================================================================

def test_1_onboarding_creates_a_workspace_for_a_verified_user():
    assert STATE["a"]["ws"], "A should have been provisioned"
    r = _as_user(STATE["a"]["token"], "GET",
                 f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}&select=id,name,plan_id,owner_id")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["owner_id"] == STATE["a"]["user_id"], "the caller must be the owner"


def test_2_unverified_mobile_cannot_provision_a_workspace():
    """The gate. User C authenticated successfully but never verified a mobile,
    so onboarding must refuse -- there is no UI path and no API path around it."""
    r = _rpc(STATE["c"]["token"], "provision_trial_workspace",
             {"p_full_name": "No Phone", "p_workspace_name": "Should Not Exist"})
    assert r.status_code >= 400, "an unverified mobile must not yield a workspace"
    assert "mobile_not_verified" in r.text

    # And nothing was created as a side effect.
    got = _svc("GET", f"/rest/v1/workspaces?owner_id=eq.{STATE['c']['user_id']}&select=id")
    assert got.json() == [], "no workspace may exist for an unverified user"


def test_3_a_client_cannot_mark_its_own_phone_verified():
    """phone_confirmed_at lives in auth.users, which PostgREST does not expose.
    There is no column on a client-writable table that means 'verified'."""
    for path in ("/rest/v1/users", "/rest/v1/auth.users"):
        r = _as_user(STATE["c"]["token"], "PATCH", path + "?id=eq." + STATE["c"]["user_id"],
                     json={"phone_confirmed_at": "2030-01-01T00:00:00Z"})
        assert r.status_code >= 400, f"{path} must not be writable by a client"

    # Nor by pretending on the profile row the user CAN write.
    r = _as_user(STATE["c"]["token"], "PATCH",
                 f"/rest/v1/profiles?id=eq.{STATE['c']['user_id']}",
                 json={"phone_verified": True})
    assert r.status_code >= 400, "there must be no client-writable phone_verified column"


def test_4_provisioning_is_idempotent_under_retry():
    """Double-click, refresh, replayed OAuth callback, network retry."""
    before = STATE["a"]["ws"]
    for _ in range(4):
        r = _rpc(STATE["a"]["token"], "provision_trial_workspace",
                 {"p_full_name": "Isolation A", "p_workspace_name": "Isolation A Co"})
        assert r.status_code == 200
        assert r.json() == before, "a retry must return the SAME workspace"

    got = _svc("GET", f"/rest/v1/workspaces?owner_id=eq.{STATE['a']['user_id']}&select=id")
    assert len(got.json()) == 1, "exactly one workspace per onboarded user"


def test_5_exactly_one_owner_membership():
    r = _svc("GET", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['a']['ws']}&select=user_id,role,status")
    rows = r.json()
    owners = [m for m in rows if m["role"] == "owner"]
    assert len(owners) == 1, f"expected one owner membership, got {len(owners)}"
    assert owners[0]["user_id"] == STATE["a"]["user_id"]
    assert owners[0]["status"] == "active"


def test_6_exactly_one_trial_with_server_side_dates():
    r = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}"
                    "&select=plan_id,plan_started_at,plan_expires_at")
    row = r.json()[0]
    assert row["plan_id"] == TRIAL_PLAN
    assert row["plan_started_at"] and row["plan_expires_at"]

    st = _rpc(STATE["a"]["token"], "workspace_trial_status", {"p_workspace_id": STATE["a"]["ws"]})
    s = st.json()[0]
    assert s["is_trial"] is True
    assert s["expired"] is False
    assert 1 <= s["days_remaining"] <= 7, f"a fresh trial should read 7 days, got {s['days_remaining']}"


def test_7_onboarding_refuses_a_blank_name():
    r = _rpc(STATE["b"]["token"], "provision_trial_workspace",
             {"p_full_name": "   ", "p_workspace_name": "X"})
    assert r.status_code >= 400 and "name_required" in r.text


def test_8_the_rpc_accepts_no_identity_or_entitlement_arguments():
    """Defence by signature: what the function will not accept, a client cannot
    send. Extra keys are rejected outright by PostgREST."""
    for hostile in ({"p_user_id": STATE["b"]["user_id"]},
                    {"p_workspace_id": STATE["b"]["ws"]},
                    {"p_owner_id": STATE["b"]["user_id"]},
                    {"p_role": "owner"},
                    {"p_trial_ends_at": "2099-01-01T00:00:00Z"},
                    {"p_plan_id": "growth_100"}):
        payload = {"p_full_name": "Attacker"}
        payload.update(hostile)
        r = _rpc(STATE["a"]["token"], "provision_trial_workspace", payload)
        assert r.status_code >= 400, f"the RPC must reject unknown argument {list(hostile)[0]}"


# =====================================================================
# 9-20. Hard tenant isolation: A must never reach B.
# =====================================================================

def test_9_a_cannot_read_bs_workspace():
    r = _as_user(STATE["a"]["token"], "GET",
                 f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}&select=id,name")
    assert r.status_code == 200
    assert r.json() == [], "knowing a workspace id must not grant access to it"


def test_10_a_cannot_list_bs_memberships():
    r = _as_user(STATE["a"]["token"], "GET",
                 f"/rest/v1/workspace_members?workspace_id=eq.{STATE['b']['ws']}&select=user_id,role")
    assert r.status_code == 200
    assert r.json() == [], "membership of another tenant must not be enumerable"


def test_11_an_unfiltered_read_returns_only_the_callers_own_tenant():
    """The strongest form: ask for EVERYTHING and check what comes back."""
    r = _as_user(STATE["a"]["token"], "GET", "/rest/v1/workspaces?select=id")
    ids = {row["id"] for row in r.json()}
    assert STATE["b"]["ws"] not in ids, "B's workspace leaked into an unfiltered read"


def test_12_a_cannot_insert_into_bs_workspace():
    r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/dashboards",
                 json={"workspace_id": STATE["b"]["ws"], "user_id": STATE["a"]["user_id"],
                       "name": "planted", "position": 0, "layout": []})
    assert r.status_code >= 400, "INSERT into another tenant must be refused"


def test_13_a_cannot_move_its_own_row_into_bs_workspace():
    """The subtler attack: create legitimately, then re-parent."""
    mk = _as_user(STATE["a"]["token"], "POST", "/rest/v1/dashboards",
                  headers={"Prefer": "return=representation"},
                  json={"workspace_id": STATE["a"]["ws"], "user_id": STATE["a"]["user_id"],
                        "name": "isolation probe", "position": 99, "layout": []})
    assert mk.status_code in (200, 201), f"A should be able to create in its OWN workspace: {mk.text[:200]}"
    did = mk.json()[0]["id"]
    try:
        mv = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/dashboards?id=eq.{did}",
                      json={"workspace_id": STATE["b"]["ws"]})
        after = _svc("GET", f"/rest/v1/dashboards?id=eq.{did}&select=workspace_id").json()
        assert after[0]["workspace_id"] == STATE["a"]["ws"], \
            "a row was re-parented into another tenant"
        assert mv.status_code >= 400 or after[0]["workspace_id"] == STATE["a"]["ws"]
    finally:
        _svc("DELETE", f"/rest/v1/dashboards?id=eq.{did}")


def test_14_a_cannot_delete_bs_data():
    mk = _svc("POST", "/rest/v1/dashboards", headers={"Prefer": "return=representation"},
              json={"workspace_id": STATE["b"]["ws"], "user_id": STATE["b"]["user_id"],
                    "name": "B private", "position": 0, "layout": []})
    did = mk.json()[0]["id"]
    try:
        _as_user(STATE["a"]["token"], "DELETE", f"/rest/v1/dashboards?id=eq.{did}")
        still = _svc("GET", f"/rest/v1/dashboards?id=eq.{did}&select=id").json()
        assert len(still) == 1, "A deleted a dashboard belonging to B"
    finally:
        _svc("DELETE", f"/rest/v1/dashboards?id=eq.{did}")


def test_15_a_dashboard_id_alone_does_not_grant_access():
    mk = _svc("POST", "/rest/v1/dashboards", headers={"Prefer": "return=representation"},
              json={"workspace_id": STATE["b"]["ws"], "user_id": STATE["b"]["user_id"],
                    "name": "B secret board", "position": 1, "layout": []})
    did = mk.json()[0]["id"]
    try:
        r = _as_user(STATE["a"]["token"], "GET", f"/rest/v1/dashboards?id=eq.{did}&select=id,name,layout")
        assert r.json() == [], "possession of a dashboard id must not be authorization"
    finally:
        _svc("DELETE", f"/rest/v1/dashboards?id=eq.{did}")


def test_16_a_cannot_grant_itself_membership_of_bs_workspace():
    for role in ("owner", "admin", "employee"):
        r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/workspace_members",
                     json={"workspace_id": STATE["b"]["ws"], "user_id": STATE["a"]["user_id"],
                           "role": role, "status": "active"})
        assert r.status_code >= 400, f"A inserted itself as {role} into B's workspace"
    got = _svc("GET", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['b']['ws']}"
                      f"&user_id=eq.{STATE['a']['user_id']}&select=id").json()
    assert got == [], "A gained a membership in B's workspace"


def test_17_a_cannot_mutate_its_own_membership_row():
    """Assert the EFFECT, not the status code.

    PostgREST answers 204 for an UPDATE that matched zero rows, and an update
    filtered away by RLS matches zero rows -- so a 204 here is what a correctly
    blocked write looks like, and testing for >=400 would have failed against a
    perfectly secure database. What matters is whether anything actually moved.
    """
    before = _svc("GET", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['a']['ws']}"
                         f"&user_id=eq.{STATE['a']['user_id']}"
                         "&select=role,status,job_title").json()[0]

    _as_user(STATE["a"]["token"], "PATCH",
             f"/rest/v1/workspace_members?workspace_id=eq.{STATE['a']['ws']}"
             f"&user_id=eq.{STATE['a']['user_id']}",
             json={"role": "owner", "status": "active", "job_title": "ESCALATION-MARKER"})

    after = _svc("GET", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['a']['ws']}"
                        f"&user_id=eq.{STATE['a']['user_id']}"
                        "&select=role,status,job_title").json()[0]
    assert after == before, f"a member rewrote its own membership row: {before} -> {after}"


def test_18_a_cannot_become_a_super_admin():
    r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/super_admins",
                 json={"user_id": STATE["a"]["user_id"]})
    assert r.status_code >= 400
    got = _svc("GET", f"/rest/v1/super_admins?user_id=eq.{STATE['a']['user_id']}&select=user_id").json()
    assert got == [], "A escalated itself to super admin"


def test_19_a_cannot_read_bs_profile_row():
    r = _as_user(STATE["a"]["token"], "GET",
                 f"/rest/v1/profiles?id=eq.{STATE['b']['user_id']}&select=id,email,full_name")
    assert r.json() == [], "profiles of an unrelated tenant must not be readable"


def test_20_bs_phone_number_is_not_exposed_to_a():
    """Lead data is the point of this feature, so it is also the thing worth
    stealing. Reading one's OWN phone is fine; reading another tenant's is not."""
    r = _as_user(STATE["a"]["token"], "GET", "/rest/v1/profiles?select=*")
    blob = r.text
    assert STATE["b"]["user_id"] not in blob, "B's profile row reached A"
    b_phone = _svc("GET", f"/rest/v1/profiles?id=eq.{STATE['b']['user_id']}&select=phone").json()
    if b_phone and b_phone[0].get("phone"):
        assert b_phone[0]["phone"] not in blob, "B's phone number reached A"


def test_20b_a_client_cannot_forge_its_own_verified_phone():
    """The one real vulnerability this suite found.

    `profiles_update_self` grants UPDATE on the whole row, so before this was
    fixed a signed-in user could PATCH their own profile with
    phone_verified_at=2030 and a phone number they had never proved, and both
    stuck. The trial gate reads auth.users and was never fooled, but a column
    named phone_verified_at is exactly what a lead export or a later billing
    check would believe.

    profiles.phone / phone_verified_at are now overwritten from auth.users on
    every write, so the forgery is discarded rather than refused -- the client's
    legitimate edits in the same request still apply.
    """
    uid = STATE["a"]["user_id"]
    before = _svc("GET", f"/rest/v1/profiles?id=eq.{uid}&select=phone,phone_verified_at").json()[0]

    _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/profiles?id=eq.{uid}",
             json={"phone": "+19999999999", "phone_verified_at": "2030-01-01T00:00:00Z",
                   "full_name": "Isolation A"})

    after = _svc("GET", f"/rest/v1/profiles?id=eq.{uid}&select=phone,phone_verified_at").json()[0]
    assert after == before, f"a client forged its own verified phone: {before} -> {after}"

    truth = _svc("GET", f"/auth/v1/admin/users/{uid}").json()
    assert after["phone_verified_at"] is not None, "fixture A should be phone-verified"
    assert (after["phone"] or "").lstrip("+") == (truth.get("phone") or "").lstrip("+"),         "profiles.phone must mirror auth.users, not whatever the client last sent"


# =====================================================================
# 21-28. Trial tampering and cross-tenant entitlement.
# =====================================================================

def test_21_a_cannot_extend_its_own_trial():
    r = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}",
                 json={"plan_expires_at": "2099-01-01T00:00:00Z"})
    after = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}&select=plan_expires_at").json()
    assert not after[0]["plan_expires_at"].startswith("2099"), "A extended its own trial"
    assert r.status_code >= 400 or True


def test_22_a_cannot_upgrade_its_own_plan():
    r = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}",
                 json={"plan_id": "growth_100"})
    after = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['a']['ws']}&select=plan_id").json()
    assert after[0]["plan_id"] == TRIAL_PLAN, "A self-upgraded its plan"
    assert r.status_code >= 400 or True


def test_23_a_cannot_alter_bs_trial():
    r = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
                 json={"plan_expires_at": "2099-01-01T00:00:00Z", "plan_id": "growth_100"})
    after = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}"
                        "&select=plan_id,plan_expires_at").json()
    assert after[0]["plan_id"] == TRIAL_PLAN
    assert not after[0]["plan_expires_at"].startswith("2099")
    assert r.status_code >= 400 or True


def test_24_a_cannot_read_bs_trial_status():
    r = _rpc(STATE["a"]["token"], "workspace_trial_status", {"p_workspace_id": STATE["b"]["ws"]})
    assert r.status_code == 200
    assert r.json() == [], "trial status of another tenant must return nothing"


def test_25_my_workspaces_returns_only_the_callers_own():
    for key, other in (("a", "b"), ("b", "a")):
        r = _rpc(STATE[key]["token"], "my_workspaces", {})
        ids = {row["workspace_id"] for row in r.json()}
        assert STATE[key]["ws"] in ids
        assert STATE[other]["ws"] not in ids, f"{other} leaked into {key}'s workspace list"


def test_26_expiry_is_computed_from_the_database_clock():
    """Move the stored date into the past with the service key -- the only actor
    allowed to -- and the SAME read now reports expired, without the client
    being consulted about the time."""
    _svc("PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
         json={"plan_expires_at": "2020-01-01T00:00:00Z"})
    try:
        s = _rpc(STATE["b"]["token"], "workspace_trial_status",
                 {"p_workspace_id": STATE["b"]["ws"]}).json()[0]
        assert s["expired"] is True
        assert s["days_remaining"] == 0, "an elapsed trial must not report days remaining"
    finally:
        _svc("PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
             json={"plan_expires_at": "2099-06-01T00:00:00Z"})
        _svc("PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
             json={"plan_expires_at": None})


def test_27_a_cannot_suspend_or_reactivate_bs_workspace():
    r = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
                 json={"is_suspended": True, "is_active": False})
    after = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}"
                        "&select=is_active,is_suspended").json()
    assert after[0]["is_suspended"] in (False, None), "A suspended another tenant"
    assert r.status_code >= 400 or True


def test_28_a_cannot_rename_or_reassign_bs_workspace():
    r = _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}",
                 json={"name": "seized", "owner_id": STATE["a"]["user_id"]})
    after = _svc("GET", f"/rest/v1/workspaces?id=eq.{STATE['b']['ws']}&select=name,owner_id").json()
    assert after[0]["owner_id"] == STATE["b"]["user_id"], "ownership of B was transferred to A"
    assert after[0]["name"] != "seized"
    assert r.status_code >= 400 or True


# =====================================================================
# 29-34. Sharing boundary and workspace-scoped surfaces.
# =====================================================================

def test_29_a_cannot_share_bs_dashboard_with_itself():
    mk = _svc("POST", "/rest/v1/dashboards", headers={"Prefer": "return=representation"},
              json={"workspace_id": STATE["b"]["ws"], "user_id": STATE["b"]["user_id"],
                    "name": "B board", "position": 2, "layout": []})
    did = mk.json()[0]["id"]
    try:
        r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/dashboard_shares",
                     json={"dashboard_id": did, "workspace_id": STATE["b"]["ws"],
                           "shared_with_user_id": STATE["a"]["user_id"],
                           "shared_by": STATE["a"]["user_id"]})
        assert r.status_code >= 400, "A granted itself a share of B's dashboard"
        seen = _as_user(STATE["a"]["token"], "GET",
                        f"/rest/v1/dashboards?id=eq.{did}&select=id").json()
        assert seen == [], "A can now read B's dashboard"
    finally:
        _svc("DELETE", f"/rest/v1/dashboard_shares?dashboard_id=eq.{did}")
        _svc("DELETE", f"/rest/v1/dashboards?id=eq.{did}")


@pytest.mark.parametrize("table", [
    "knowledge_items", "knowledge_folders", "chatbots", "departments",
    "ai_search_conversations", "saved_presentations", "generated_trainings",
    "workspace_settings", "workspace_usage", "audit_log",
])
def test_30_every_workspace_scoped_table_refuses_a_cross_tenant_read(table):
    """One assertion per tenant-scoped surface, because isolation is only as
    strong as the weakest table -- and a new table added without a policy is
    exactly how a boundary quietly develops a hole."""
    r = _as_user(STATE["a"]["token"], "GET",
                 f"/rest/v1/{table}?workspace_id=eq.{STATE['b']['ws']}&select=workspace_id")
    assert r.status_code in (200, 401, 403), f"unexpected status for {table}: {r.status_code}"
    if r.status_code == 200:
        assert r.json() == [], f"{table} leaked rows from another tenant"


@pytest.mark.parametrize("table", [
    "knowledge_items", "knowledge_folders", "chatbots", "departments",
    "saved_presentations", "workspace_settings",
])
def test_31_every_workspace_scoped_table_refuses_a_cross_tenant_insert(table):
    r = _as_user(STATE["a"]["token"], "POST", f"/rest/v1/{table}",
                 json={"workspace_id": STATE["b"]["ws"], "name": "planted-by-a"})
    assert r.status_code >= 400, f"{table} accepted an INSERT into another tenant"


def test_32_the_anon_key_alone_reaches_nothing():
    """No token at all: the public surface of a tenant-scoped table is empty."""
    with httpx.Client(timeout=30) as c:
        for table in ("workspaces", "workspace_members", "dashboards", "profiles"):
            r = c.get(f"{APP_URL}/rest/v1/{table}?select=id", headers={"apikey": ANON})
            assert r.status_code >= 400 or r.json() == [], \
                f"{table} is readable with only the publishable key"


def test_33_a_forged_workspace_id_in_a_trial_call_is_ignored():
    for hostile in (STATE["b"]["ws"], str(uuid.uuid4())):
        r = _rpc(STATE["a"]["token"], "workspace_trial_status", {"p_workspace_id": hostile})
        assert r.status_code == 200 and r.json() == [], \
            "a workspace id in the request body must never be the authorization"


def test_34_no_synthetic_state_escaped_into_a_real_tenant():
    """Data integrity: the fixtures must not have touched anything real."""
    for key in ("a", "b"):
        rows = _svc("GET", f"/rest/v1/workspaces?owner_id=eq.{STATE[key]['user_id']}&select=id").json()
        assert len(rows) == 1, f"{key} owns {len(rows)} workspaces, expected exactly 1"
    planted = _svc("GET", "/rest/v1/dashboards?name=eq.planted&select=id").json()
    assert planted == [], "a planted row survived"
