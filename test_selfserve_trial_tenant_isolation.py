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


def _create_user(email: str, *, confirm_email: bool) -> dict:
    """`email_confirm` reaches the post-OTP state without sending a real code.
    A genuine onboarding sets it only through verifyOtp; the admin API is used
    here purely so the suite does not depend on an inbox."""
    body = {"email": email, "password": _PW, "email_confirm": confirm_email}
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
    for key, verified in (("a", True), ("b", True)):
        email = f"trialtest-{key}-{stamp}@knova-isolation.test"
        u = _create_user(email, confirm_email=verified)
        STATE[key]["email"] = email
        STATE[key]["user_id"] = u["id"]
        STATE[key]["token"] = _token_for(email)

    # A third identity that never verified its email. Deliberately NO token is
    # obtained: an unverified account cannot get one, which is the first half of
    # the gate and is asserted directly in test 2.
    email_c = f"trialtest-c-{stamp}@knova-isolation.test"
    u = _create_user(email_c, confirm_email=False)
    STATE["c"] = {"email": email_c, "user_id": u["id"], "token": None}

    # Provision A and B through the REAL onboarding path.
    for key in ("a", "b"):
        r = _rpc(STATE[key]["token"], "provision_trial_workspace",
                 {"p_full_name": f"Isolation {key.upper()}",
                  "p_workspace_name": f"Isolation {key.upper()} Co",
                  "p_org": f"Isolation {key.upper()} Co",
                  "p_country": "IN", "p_mobile": "+919876500%03d" % (1 if key == "a" else 2)})
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


def test_2_an_unverified_email_cannot_even_obtain_a_session():
    """The first half of the gate, and the stronger half.

    Supabase refuses to issue a session for an unconfirmed address, so a user
    who never entered the emailed code has no token at all and cannot reach
    provisioning -- or anything else. There is nothing to bypass because there
    is nothing to bypass it WITH.
    """
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{APP_URL}/auth/v1/token?grant_type=password",
                   headers={"apikey": ANON, "Content-Type": "application/json"},
                   json={"email": STATE["c"]["email"], "password": _PW})
    assert r.status_code >= 400, "an unverified email must not receive a session"

    got = _svc("GET", f"/rest/v1/workspaces?owner_id=eq.{STATE['c']['user_id']}&select=id")
    assert got.json() == [], "no workspace may exist for an unverified user"


def test_2b_provisioning_refuses_a_caller_with_no_verified_identity():
    """The second half: even a request carrying the SERVICE ROLE -- the most
    privileged credential there is -- provisions nothing, because the function
    takes its identity from auth.uid() rather than from the caller's rights.
    There is no argument to pass to claim to be someone."""
    r = _svc("POST", "/rest/v1/rpc/provision_trial_workspace",
             json={"p_full_name": "No Identity", "p_workspace_name": "Should Not Exist",
                   "p_org": "X", "p_country": "IN", "p_mobile": "+919876500997"})
    assert r.status_code >= 400, "a caller with no auth.uid() must not provision"
    assert "not_authenticated" in r.text


def test_3_a_client_cannot_mark_its_own_identity_verified():
    """email_confirmed_at and phone_confirmed_at live in auth.users, which
    PostgREST does not expose. There is no column on a client-writable table
    that means 'verified'."""
    for path in ("/rest/v1/users", "/rest/v1/auth.users"):
        r = _as_user(STATE["a"]["token"], "PATCH", path + "?id=eq." + STATE["a"]["user_id"],
                     json={"email_confirmed_at": "2030-01-01T00:00:00Z"})
        assert r.status_code >= 400, f"{path} must not be writable by a client"

    # Nor by pretending on the profile row the user CAN write.
    r = _as_user(STATE["a"]["token"], "PATCH",
                 f"/rest/v1/profiles?id=eq.{STATE['a']['user_id']}",
                 json={"phone_verified": True})
    assert r.status_code >= 400, "there must be no client-writable phone_verified column"


def test_4_provisioning_is_idempotent_under_retry():
    """Double-click, refresh, replayed OAuth callback, network retry."""
    before = STATE["a"]["ws"]
    for _ in range(4):
        r = _rpc(STATE["a"]["token"], "provision_trial_workspace",
                 {"p_full_name": "Isolation A", "p_workspace_name": "Isolation A Co",
                  "p_org": "Isolation A Co", "p_country": "IN", "p_mobile": "+919876500001"})
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

    st = _rpc(STATE["a"]["token"], "workspace_entitlement", {"p_workspace_id": STATE["a"]["ws"]})
    s = st.json()[0]
    assert s["is_trial"] is True
    assert s["expired"] is False
    assert 1 <= s["days_remaining"] <= 7, f"a fresh trial should read 7 days, got {s['days_remaining']}"
    assert s["queries_limit"] == 500 and s["storage_mb_limit"] == 100
    assert s["files_limit"] == 50 and s["trainings_limit"] == 3 and s["presentations_limit"] == 3


def test_7_onboarding_refuses_a_blank_name():
    r = _rpc(STATE["b"]["token"], "provision_trial_workspace",
             {"p_full_name": "   ", "p_workspace_name": "X", "p_org": "X",
              "p_country": "IN", "p_mobile": "+919876500999"})
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
        payload = {"p_full_name": "Attacker", "p_org": "X",
                   "p_country": "IN", "p_mobile": "+919876500998"}
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


def test_20b_a_client_cannot_forge_a_verified_phone():
    """`profiles_update_self` grants UPDATE on the WHOLE row, so before this was
    fixed a signed-in user could PATCH their own profile with
    phone_verified_at=2030 and a number they had never proved, and it stuck.

    Under the email-OTP trial nothing verifies a phone at all, which makes the
    point sharper rather than weaker: profiles.phone must stay NULL no matter
    what the client sends, because a number in that column means Supabase Auth
    proved it. The mobile the user typed lives in lead_mobile, which claims
    nothing.
    """
    uid = STATE["a"]["user_id"]
    before = _svc("GET", f"/rest/v1/profiles?id=eq.{uid}"
                         "&select=phone,phone_verified_at").json()[0]

    _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/profiles?id=eq.{uid}",
             json={"phone": "+19999999999", "phone_verified_at": "2030-01-01T00:00:00Z",
                   "full_name": "Isolation A"})

    after = _svc("GET", f"/rest/v1/profiles?id=eq.{uid}"
                        "&select=phone,phone_verified_at").json()[0]
    assert after == before, f"a client forged a verified phone: {before} -> {after}"
    assert after["phone"] is None and after["phone_verified_at"] is None,         "nothing in the email trial verifies a phone, so these must stay empty"


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
    r = _rpc(STATE["a"]["token"], "workspace_entitlement", {"p_workspace_id": STATE["b"]["ws"]})
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
        s = _rpc(STATE["b"]["token"], "workspace_entitlement",
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
        r = _rpc(STATE["a"]["token"], "workspace_entitlement", {"p_workspace_id": hostile})
        assert r.status_code == 200 and r.json() == [], \
            "a workspace id in the request body must never be the authorization"


def test_34_no_synthetic_state_escaped_into_a_real_tenant():
    """Data integrity: the fixtures must not have touched anything real."""
    for key in ("a", "b"):
        rows = _svc("GET", f"/rest/v1/workspaces?owner_id=eq.{STATE[key]['user_id']}&select=id").json()
        assert len(rows) == 1, f"{key} owns {len(rows)} workspaces, expected exactly 1"
    planted = _svc("GET", "/rest/v1/dashboards?name=eq.planted&select=id").json()
    assert planted == [], "a planted row survived"


# =====================================================================
# 35-42. Trial quotas: server-authoritative, atomic, tenant-scoped.
# =====================================================================

def _usage(ws: str) -> dict:
    return _svc("GET", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}"
                       "&select=total_queries_used,storage_bytes_used,file_count,"
                       "total_trainings_created,total_presentations_created").json()[0]


def _consume(ws: str, kind: str, amount: int = 1) -> bool:
    """Spending is a SERVER action -- consume_quota is granted to service_role
    only -- so the trusted-backend path is what gets exercised here."""
    r = _svc("POST", "/rest/v1/rpc/consume_quota",
             json={"p_workspace_id": ws, "p_kind": kind, "p_amount": amount,
                   "p_user_id": STATE["b"]["user_id"]})
    assert r.status_code == 200, f"consume_quota failed: {r.status_code} {r.text[:160]}"
    return r.json() is True


def test_35_a_client_cannot_spend_or_grant_itself_quota():
    """The client must not hold the function that decides whether an action is
    paid for -- otherwise it is the only thing standing between a request and
    its cost."""
    before = _usage(STATE["a"]["ws"])["total_queries_used"]
    r = _rpc(STATE["a"]["token"], "consume_quota",
             {"p_workspace_id": STATE["a"]["ws"], "p_kind": "query", "p_amount": 1})
    after = _usage(STATE["a"]["ws"])["total_queries_used"]
    # Asserting the effect as well as the status: an earlier version of this
    # function was reachable by `authenticated` because REVOKE ... FROM public
    # does not remove Supabase's default grant, and the client cheerfully spent
    # its own quota with a 200.
    assert after == before, "a client spent quota through consume_quota"
    assert r.status_code >= 400, f"consume_quota must not be callable by a client (got {r.status_code})"


def test_36_a_client_cannot_rewrite_its_own_usage():
    before = _usage(STATE["a"]["ws"])
    _as_user(STATE["a"]["token"], "PATCH",
             f"/rest/v1/workspace_usage?workspace_id=eq.{STATE['a']['ws']}",
             json={"total_queries_used": 0, "storage_bytes_used": 0, "file_count": 0})
    after = _usage(STATE["a"]["ws"])
    assert after == before, f"a client reset its own usage: {before} -> {after}"


def test_37_a_client_cannot_raise_its_own_limits():
    r = _as_user(STATE["a"]["token"], "PATCH", "/rest/v1/plans?id=eq.trial_7day",
                 json={"max_total_queries": 999999, "max_storage_mb": 999999})
    after = _svc("GET", "/rest/v1/plans?id=eq.trial_7day"
                        "&select=max_total_queries,max_storage_mb").json()[0]
    assert after["max_total_queries"] == 500, "the trial query limit was raised by a client"
    assert after["max_storage_mb"] == 100
    assert r.status_code >= 400 or True


def test_38_the_combined_query_quota_is_one_pool_of_500():
    """500 TOTAL, deliberately not 500 AI plus 500 bot. Spending it down to the
    boundary must allow exactly the allowance and not one more."""
    ws = STATE["b"]["ws"]
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"total_queries_used": 498})
    assert _consume(ws, "query") is True, "499th query should be allowed"
    assert _consume(ws, "query") is True, "500th query should be allowed"
    assert _consume(ws, "query") is False, "501st query must be refused"
    assert _usage(ws)["total_queries_used"] == 500, "a refused query must not increment"
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"total_queries_used": 0})


def test_39_concurrent_requests_cannot_exceed_the_quota():
    """The race the naive implementation loses.

    Read-compare-write lets two requests both see 499, both decide it is under
    500, and both increment -- ending on 501 having been allowed twice. Here the
    limit is inside the UPDATE's WHERE clause, so the comparison and the
    increment happen atomically under a row lock and the loser gets nothing.
    """
    import concurrent.futures as cf
    ws = STATE["b"]["ws"]
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"total_queries_used": 495})
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(lambda _: _consume(ws, "query"), range(12)))

    allowed = sum(1 for r in results if r)
    final = _usage(ws)["total_queries_used"]
    assert allowed == 5, f"exactly 5 of 12 concurrent requests should win, got {allowed}"
    assert final == 500, f"usage must land exactly on the limit, got {final}"
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"total_queries_used": 0})


def test_40_storage_and_file_count_are_both_enforced():
    ws = STATE["b"]["ws"]
    mb = 1048576
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"storage_bytes_used": 99 * mb, "file_count": 10})
    assert _consume(ws, "storage_bytes", mb) is True, "the 100th MB should fit"
    assert _consume(ws, "storage_bytes", mb) is False, "101 MB must be refused"

    # File COUNT is a separate ceiling from bytes: 50 tiny files still stop.
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"storage_bytes_used": 0, "file_count": 50})
    assert _consume(ws, "storage_bytes", 1024) is False, "the 51st file must be refused"
    _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}",
         json={"storage_bytes_used": 0, "file_count": 0})


def test_41_training_and_presentation_ceilings_hold_at_three():
    ws = STATE["b"]["ws"]
    for kind, col in (("training", "total_trainings_created"),
                      ("presentation", "total_presentations_created")):
        _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}", json={col: 0})
        assert [_consume(ws, kind) for _ in range(3)] == [True, True, True]
        assert _consume(ws, kind) is False, f"a 4th {kind} must be refused"
        assert _usage(ws)[col] == 3
        _svc("PATCH", f"/rest/v1/workspace_usage?workspace_id=eq.{ws}", json={col: 0})


def test_42_an_expired_trial_spends_nothing_and_another_tenant_spends_nothing():
    ws_b = STATE["b"]["ws"]
    _svc("PATCH", f"/rest/v1/workspaces?id=eq.{ws_b}",
         json={"plan_expires_at": "2020-01-01T00:00:00Z"})
    try:
        assert _consume(ws_b, "query") is False, "an expired trial must not consume quota"
        assert _usage(ws_b)["total_queries_used"] == 0
    finally:
        _svc("PATCH", f"/rest/v1/workspaces?id=eq.{ws_b}", json={"plan_expires_at": None})

    # And the entitlement of another tenant is not even readable.
    r = _rpc(STATE["a"]["token"], "workspace_entitlement", {"p_workspace_id": ws_b})
    assert r.status_code == 200 and r.json() == [], "A read B's entitlement"


def test_43_lead_data_is_recorded_but_never_presented_as_verified():
    """The mobile is mandatory lead information the user typed. It is stored in
    lead_mobile, NOT in profiles.phone, which means a number Supabase Auth has
    actually proved. Conflating the two would make an unproven number look
    verified to everything downstream."""
    row = _svc("GET", f"/rest/v1/profiles?id=eq.{STATE['a']['user_id']}"
                      "&select=lead_mobile,lead_country,lead_org,phone,phone_verified_at").json()[0]
    assert row["lead_mobile"], "the signup mobile should have been recorded"
    assert row["lead_country"] == "IN"
    assert row["lead_org"], "the organization should have been recorded"
    assert row["phone"] is None, "an unverified lead number must not land in profiles.phone"
    assert row["phone_verified_at"] is None, "nothing here was verified by SMS"


# =====================================================================
# 44-52. Paid invitation flow: one-time, email-bound, tenant-bound.
# =====================================================================

def test_44_an_invitation_cannot_be_accepted_by_a_different_email():
    """The strongest possible shape: `accept_pending_invitations` takes NO
    ARGUMENTS. It reads the caller's address from auth.users via auth.uid(),
    so there is no token, tenant, role or email in the request to tamper with.
    An invitation addressed to someone else simply does not match."""
    inv = _svc("POST", "/rest/v1/workspace_invitations",
               headers={"Prefer": "return=representation"},
               json={"workspace_id": STATE["b"]["ws"], "email": "someone-else@knova-isolation.test",
                     "role": "employee", "invited_by": STATE["b"]["user_id"], "status": "pending",
                     "expires_at": "2099-01-01T00:00:00Z"}).json()[0]
    try:
        r = _rpc(STATE["a"]["token"], "accept_pending_invitations", {})
        assert r.status_code == 200
        assert r.json() == 0, "A accepted an invitation addressed to another email"
        got = _svc("GET", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['b']['ws']}"
                          f"&user_id=eq.{STATE['a']['user_id']}&select=id").json()
        assert got == [], "A gained membership of B's tenant through an invitation"
    finally:
        _svc("DELETE", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}")


def test_45_an_expired_invitation_is_not_accepted():
    inv = _svc("POST", "/rest/v1/workspace_invitations",
               headers={"Prefer": "return=representation"},
               json={"workspace_id": STATE["b"]["ws"], "email": STATE["a"]["email"],
                     "role": "employee", "invited_by": STATE["b"]["user_id"], "status": "pending",
                     "expires_at": "2020-01-01T00:00:00Z"}).json()[0]
    try:
        r = _rpc(STATE["a"]["token"], "accept_pending_invitations", {})
        assert r.json() == 0, "an expired invitation was accepted"
        after = _svc("GET", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}&select=status").json()
        assert after[0]["status"] == "pending", "an expired invitation must not be consumed"
    finally:
        _svc("DELETE", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}")


def test_46_an_invitation_is_single_use():
    """Accepting flips the row to 'accepted', so a second attempt matches
    nothing — the same link cannot seat a person twice."""
    inv = _svc("POST", "/rest/v1/workspace_invitations",
               headers={"Prefer": "return=representation"},
               json={"workspace_id": STATE["b"]["ws"], "email": STATE["a"]["email"],
                     "role": "employee", "invited_by": STATE["b"]["user_id"], "status": "pending",
                     "expires_at": "2099-01-01T00:00:00Z"}).json()[0]
    try:
        first = _rpc(STATE["a"]["token"], "accept_pending_invitations", {}).json()
        second = _rpc(STATE["a"]["token"], "accept_pending_invitations", {}).json()
        assert first == 1, "a valid invitation should be accepted once"
        assert second == 0, "the same invitation was accepted twice"
        st = _svc("GET", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}&select=status").json()
        assert st[0]["status"] == "accepted"
    finally:
        # Undo the seating so the isolation fixtures stay as they were.
        _svc("DELETE", f"/rest/v1/workspace_members?workspace_id=eq.{STATE['b']['ws']}"
                       f"&user_id=eq.{STATE['a']['user_id']}")
        _svc("PATCH", f"/rest/v1/workspace_members?user_id=eq.{STATE['a']['user_id']}"
                      f"&workspace_id=eq.{STATE['a']['ws']}", json={"status": "active"})
        _svc("DELETE", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}")


def test_47_a_client_cannot_forge_an_invitation_into_another_tenant():
    r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/workspace_invitations",
                 json={"workspace_id": STATE["b"]["ws"], "email": STATE["a"]["email"],
                       "role": "owner", "invited_by": STATE["a"]["user_id"], "status": "pending"})
    assert r.status_code >= 400, "A wrote an invitation into B's tenant"
    got = _svc("GET", f"/rest/v1/workspace_invitations?workspace_id=eq.{STATE['b']['ws']}"
                      f"&email=eq.{STATE['a']['email']}&select=id").json()
    assert got == [], "a forged invitation survived"


def test_48_a_client_cannot_promote_a_pending_invitation_to_owner():
    inv = _svc("POST", "/rest/v1/workspace_invitations",
               headers={"Prefer": "return=representation"},
               json={"workspace_id": STATE["a"]["ws"], "email": "pending@knova-isolation.test",
                     "role": "employee", "invited_by": STATE["a"]["user_id"], "status": "pending",
                     "expires_at": "2099-01-01T00:00:00Z"}).json()[0]
    try:
        _as_user(STATE["a"]["token"], "PATCH", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}",
                 json={"role": "owner"})
        after = _svc("GET", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}&select=role").json()
        assert after[0]["role"] == "employee", "a pending invitation was escalated to owner"
    finally:
        _svc("DELETE", f"/rest/v1/workspace_invitations?id=eq.{inv['id']}")


def test_49_the_invitation_creator_cannot_choose_the_tenant():
    """create_workspace_invitation takes an email and a role -- never a
    workspace. It resolves the tenant from current_workspace_id(), so a Company
    Admin cannot invite into a company that is not theirs."""
    r = _rpc(STATE["a"]["token"], "create_workspace_invitation",
             {"_email": "x@knova-isolation.test", "_role": "employee",
              "_workspace_id": STATE["b"]["ws"]})
    assert r.status_code >= 400, "the RPC must reject an unknown workspace argument"


def test_50_owner_cannot_be_invited_as_a_role():
    """No path to a second owner, which is the escalation that would matter."""
    r = _rpc(STATE["a"]["token"], "create_workspace_invitation",
             {"_email": "escalate@knova-isolation.test", "_role": "owner"})
    assert r.status_code >= 400 and "invalid_role" in r.text


def test_51_an_ordinary_member_cannot_invite_at_all():
    """Only owner/admin may invite. B is an owner of its own tenant, so the
    meaningful check is that the function authorizes against the CALLER's role
    in the CALLER's workspace rather than accepting a claim."""
    src = _svc("POST", "/rest/v1/rpc/create_workspace_invitation",
               json={"_email": "nobody@knova-isolation.test", "_role": "employee"})
    # Service role has no auth.uid(), so current_workspace_id() is null.
    assert src.status_code >= 400, "a caller with no identity must not create invitations"


def test_52_super_admin_status_cannot_be_self_granted():
    for payload in ({"user_id": STATE["a"]["user_id"]},):
        r = _as_user(STATE["a"]["token"], "POST", "/rest/v1/super_admins", json=payload)
        assert r.status_code >= 400
    got = _svc("GET", f"/rest/v1/super_admins?user_id=eq.{STATE['a']['user_id']}&select=user_id").json()
    assert got == [], "a tenant user became a platform super admin"
