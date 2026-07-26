"""
Tenant authentication and authorisation for the Knova API.

WHY THIS EXISTS
---------------
Every endpoint in this service used to take `workspace_id` straight from the
request body or query string and trust it. There was no authentication at all.
An unauthenticated caller holding any workspace UUID could read that entire
workspace's indexed content, connections and analytics — the app DB's row level
security and the storage policies were both correct, but Railway holds the
service key and simply never asked who was calling, so it bypassed both.

HOW IT WORKS
------------
The frontend now sends the caller's Supabase access token as
`Authorization: Bearer <jwt>`. This module:

  1. Verifies the token signature locally against the app project's public JWKS
     (ES256). No secret is needed for this — JWKS is public — which is why this
     service still does not hold an app-DB service key.
  2. Resolves which workspaces that user actually belongs to, by calling the app
     DB's PostgREST *with the caller's own token*. Row level security answers the
     question for us, so a forged `sub` cannot widen the result.
  3. Fails the request unless the workspace the caller asked for is in that set.

Both lookups are cached briefly so a burst of calls costs one round trip.

STAGED ROLLOUT
--------------
`AUTH_ENFORCE` (default "false") controls whether a missing or invalid token is
fatal. Deploy with it off, confirm real traffic is arriving with tokens by
watching for the `AUTH-SKIP` lines below, then set it to "true". Leaving it off
means the service is still wide open, so it is a launch step, not a setting.
"""

import os
import time
import json
import base64
import threading
from typing import Optional

import httpx
import jwt
from jwt import PyJWKClient
from fastapi import Depends, Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────────
# These point at the APP database project (auth, workspaces, members), which is a
# different Supabase project from SUPABASE_URL (the vector DB this service writes).
# Both values below are public by design: the URL, and the anon/publishable key.
APP_SUPABASE_URL      = (os.getenv("APP_SUPABASE_URL") or "").rstrip("/")
APP_SUPABASE_ANON_KEY = os.getenv("APP_SUPABASE_ANON_KEY") or ""

AUTH_ENFORCE = (os.getenv("AUTH_ENFORCE") or "false").strip().lower() in ("1", "true", "yes")

MEMBERSHIP_TTL = 60      # seconds; short so a revoked membership stops working fast
JWKS_TTL       = 3600    # seconds; signing keys rotate rarely

_membership_cache: dict[str, tuple[float, "AuthContext"]] = {}
_cache_lock = threading.Lock()

_jwk_client: Optional[PyJWKClient] = None
_jwk_client_at = 0.0


# ── Auth context ────────────────────────────────────────────────────────────────
class AuthContext:
    """
    Who the caller is and what they may touch.

    `workspaces` maps workspace_id -> role for every ACTIVE membership.
    `is_super_admin` is a deliberate cross-workspace bypass, matching the
    `is_super_admin()` escape hatch already present in the app DB's RLS policies.
    `enforced` is False only when the caller got through because AUTH_ENFORCE is
    off; nothing should treat such a context as trustworthy.
    """

    def __init__(self, user_id: str, workspaces: dict, is_super_admin: bool = False,
                 enforced: bool = True):
        self.user_id        = user_id
        self.workspaces     = workspaces
        self.is_super_admin = is_super_admin
        self.enforced       = enforced

    def assert_workspace(self, workspace_id: str) -> None:
        """
        Authorises this caller for one workspace. Call it in every endpoint that
        accepts a workspace_id, BEFORE touching any data for that workspace.
        """
        if not self.enforced:
            print(f"AUTH-SKIP: unauthenticated access to workspace {workspace_id} "
                  f"allowed because AUTH_ENFORCE is off")
            return

        if self.is_super_admin:
            return

        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required.")

        if workspace_id not in self.workspaces:
            # Deliberately identical to the "no such workspace" case: never confirm
            # that a workspace exists to someone who is not a member of it.
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this workspace.",
            )

    def role_in(self, workspace_id: str) -> Optional[str]:
        return self.workspaces.get(workspace_id)


# ── Token verification ──────────────────────────────────────────────────────────
def _get_jwk_client() -> PyJWKClient:
    global _jwk_client, _jwk_client_at
    now = time.time()
    if _jwk_client is None or (now - _jwk_client_at) > JWKS_TTL:
        _jwk_client = PyJWKClient(
            f"{APP_SUPABASE_URL}/auth/v1/.well-known/jwks.json",
            cache_keys=True,
        )
        _jwk_client_at = now
    return _jwk_client


def _unverified_claims(token: str) -> dict:
    """Reads the payload without checking the signature. Only used for diagnostics."""
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))
    except Exception:
        return {}


def _verify_token(token: str) -> dict:
    """
    Returns verified JWT claims, or raises 401.

    Supabase projects may issue asymmetric (ES256/RS256, verified via JWKS) or
    legacy symmetric (HS256) tokens. This project's JWKS serves ES256. HS256
    tokens cannot be verified without the project's JWT secret, which this
    service deliberately does not hold — so they are rejected rather than
    silently trusted.
    """
    try:
        header = jwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed access token.")

    alg = header.get("alg", "")
    if alg.startswith("HS"):
        raise HTTPException(
            status_code=401,
            detail="Symmetric (HS256) tokens are not accepted by this API.",
        )

    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"AUTH-FAIL: token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid access token.")


# ── Membership lookup ───────────────────────────────────────────────────────────
def _load_memberships(token: str, user_id: str) -> tuple[dict, bool]:
    """
    Asks the APP DB which workspaces this user belongs to, using the caller's own
    token so row level security does the filtering. That is the whole point: even
    if a caller lied about `sub`, RLS resolves `auth.uid()` from the token itself,
    so the answer is still only ever their own memberships.

    Returns (workspaces, is_super_admin).
    """
    if not APP_SUPABASE_URL or not APP_SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server auth is misconfigured (APP_SUPABASE_URL / "
                   "APP_SUPABASE_ANON_KEY are not set).",
        )

    headers = {
        "apikey":        APP_SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    }

    try:
        with httpx.Client(timeout=10) as client:
            members = client.get(
                f"{APP_SUPABASE_URL}/rest/v1/workspace_members",
                params={
                    "select":  "workspace_id,role,status",
                    "user_id": f"eq.{user_id}",
                    "status":  "eq.active",
                },
                headers=headers,
            )
            if members.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid access token.")
            members.raise_for_status()

            supers = client.get(
                f"{APP_SUPABASE_URL}/rest/v1/super_admins",
                params={"select": "user_id", "user_id": f"eq.{user_id}"},
                headers=headers,
            )
            # A non-super-admin simply gets an empty list here; a failure to answer
            # must NOT be read as "yes", so anything non-200 means not a super admin.
            is_super = supers.status_code == 200 and len(supers.json()) > 0

    except HTTPException:
        raise
    except Exception as e:
        print(f"AUTH-FAIL: membership lookup failed for {user_id}: {e}")
        raise HTTPException(status_code=503, detail="Could not verify workspace access.")

    workspaces = {
        row["workspace_id"]: row.get("role", "member")
        for row in members.json()
        if row.get("workspace_id")
    }
    return workspaces, is_super


# ── FastAPI dependency ──────────────────────────────────────────────────────────
def current_user(authorization: Optional[str] = Header(None)) -> AuthContext:
    """
    FastAPI dependency. Use as:

        @router.post("/thing")
        def thing(body: ThingRequest, auth: AuthContext = Depends(current_user)):
            auth.assert_workspace(body.workspace_id)

    Resolving identity is NOT the same as authorising a workspace — the
    assert_workspace call is what actually closes the hole, so it must appear in
    every endpoint that accepts a workspace_id.
    """
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if not token:
        if AUTH_ENFORCE:
            raise HTTPException(status_code=401, detail="Missing access token.")
        print("AUTH-SKIP: request arrived with no bearer token (AUTH_ENFORCE off)")
        return AuthContext(user_id="", workspaces={}, enforced=False)

    now = time.time()
    with _cache_lock:
        hit = _membership_cache.get(token)
        if hit and (now - hit[0]) < MEMBERSHIP_TTL:
            return hit[1]

    try:
        claims  = _verify_token(token)
        user_id = claims.get("sub") or ""
        if not user_id:
            raise HTTPException(status_code=401, detail="Access token has no subject.")

        workspaces, is_super = _load_memberships(token, user_id)
        ctx = AuthContext(user_id=user_id, workspaces=workspaces, is_super_admin=is_super)

    except HTTPException:
        if AUTH_ENFORCE:
            raise
        claims = _unverified_claims(token)
        print(f"AUTH-SKIP: token present but unusable "
              f"(sub={claims.get('sub')}) — allowed because AUTH_ENFORCE is off")
        return AuthContext(user_id="", workspaces={}, enforced=False)

    with _cache_lock:
        _membership_cache[token] = (now, ctx)
        if len(_membership_cache) > 500:
            cutoff = now - MEMBERSHIP_TTL
            for k in [k for k, v in _membership_cache.items() if v[0] < cutoff]:
                _membership_cache.pop(k, None)

    return ctx


def require_super_admin(auth: AuthContext = Depends(current_user)) -> AuthContext:
    """Dependency for cross-workspace admin endpoints."""
    if not auth.enforced:
        print("AUTH-SKIP: super-admin endpoint reached without auth (AUTH_ENFORCE off)")
        return auth
    if not auth.is_super_admin:
        raise HTTPException(status_code=403, detail="Super admin access required.")
    return auth
