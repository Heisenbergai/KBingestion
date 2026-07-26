import os
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

from auth import AuthContext, current_user

load_dotenv()

router = APIRouter()

# ── Clients ────────────────────────────────────────────────────────────────────
# Uses personal Supabase for vector data
vector_db = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

SUPER_ADMIN_SECRET = os.getenv("SUPER_ADMIN_SECRET", "change-this-secret")


# ── Auth helper ────────────────────────────────────────────────────────────────
def verify_super_admin(x_admin_secret: str = Header(None)):
    if x_admin_secret != SUPER_ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Request shapes ─────────────────────────────────────────────────────────────
class CreateWorkspaceRequest(BaseModel):
    name:           str
    slug:           str
    plan_id:        str
    billing_email:  str
    owner_email:    str        # will be invited as owner
    logo_url:       Optional[str] = None
    primary_color:  Optional[str] = "#1E2761"


class UpdateWorkspacePlanRequest(BaseModel):
    workspace_id:   str
    new_plan_id:    str


class SuspendWorkspaceRequest(BaseModel):
    workspace_id:    str
    suspend:         bool
    reason:          Optional[str] = None


# ── Usage check endpoint (called by Lovable before every AI feature) ───────────
class CheckUsageRequest(BaseModel):
    workspace_id:  str
    feature:       str
    user_id:       Optional[str] = None


@router.post("/check-usage")
async def check_usage(body: CheckUsageRequest,
                      auth: AuthContext = Depends(current_user)):
    """
    DEAD STUB — a Lovable-era passthrough that always answers "allowed" and reads
    nothing. Lovable is gone, no caller in the current frontend references it, and
    quota enforcement was never moved here. It is authorised anyway so no endpoint
    in this service takes an unchecked workspace_id, but the real fix is to delete
    it or implement it against the app DB. See 06_immediate_next_steps.md.

    Features: ai_search | chatbot_internal | chatbot_external |
              presentation | course_generation | video_generation | document_ingestion
    """
    auth.assert_workspace(body.workspace_id)
    return {
        "allowed": True,
        "note": "Stub — no quota enforcement is implemented in this service.",
    }


# ── Super admin endpoints (protected by SUPER_ADMIN_SECRET header) ─────────────

@router.get("/admin/workspaces")
async def list_workspaces(x_admin_secret: str = Header(None)):
    """
    Returns all workspaces with their usage summary.
    Called by your super admin dashboard.
    """
    verify_super_admin(x_admin_secret)
    # This returns structure — actual data fetched by Lovable from its own DB
    return {"message": "Fetch from Lovable DB directly via super admin user"}


@router.post("/admin/workspaces")
async def create_workspace(
    body: CreateWorkspaceRequest,
    x_admin_secret: str = Header(None)
):
    """
    Creates a new workspace + sends owner invite email.
    Called by your super admin panel in Lovable.
    Returns the workspace config to save in Lovable's DB.
    """
    verify_super_admin(x_admin_secret)

    # Validate slug format
    import re
    if not re.match(r'^[a-z0-9-]+$', body.slug):
        raise HTTPException(
            status_code=400,
            detail="Slug must contain only lowercase letters, numbers, and hyphens"
        )

    # Return workspace config — Lovable saves to its own DB
    return {
        "workspace": {
            "name":          body.name,
            "slug":          body.slug,
            "plan_id":       body.plan_id,
            "billing_email": body.billing_email,
            "logo_url":      body.logo_url,
            "primary_color": body.primary_color,
            "is_active":     True,
            "is_suspended":  False,
        },
        "owner_invite": {
            "email": body.owner_email,
            "role":  "owner",
        },
        "message": f"Workspace '{body.name}' config ready. Save to DB and send invite to {body.owner_email}"
    }
