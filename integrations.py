"""
Integrations platform — the admin "connect your tools" layer (Phase 2.5).

This is the registry + generic API that lets the frontend render a standard
SaaS "Integrations" settings page: a grid of provider cards, each Connect/
Manage/Disconnect, without the frontend hardcoding anything provider-specific.

Adding a new integration later is just:
  1. add an entry to PROVIDERS here (flip status to "available")
  2. add a connector_<provider>.py implementing OAuth install/callback
     (for oauth) OR a validate()+fetch path (for api_key)
The admin UI needs ZERO changes — it's driven entirely by GET /integrations.

Connection records + tokens live in the `connections` table (see
brain_connectors.py); tokens are Fernet-encrypted. The captured-knowledge
pipeline (filtration → notes → embeddings) is shared across all providers.
"""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

import brain_connectors as bc

load_dotenv()

router = APIRouter()

APP_REDIRECT_URL = os.getenv("APP_REDIRECT_URL", "https://knova.lovable.app")


# ── Provider registry — the single source of truth for the admin UI ─────────────
# auth_method: "oauth" (popup consent) | "api_key" (paste a token)
# status:      "available" (works now) | "coming_soon" (card shown, disabled)
PROVIDERS: dict[str, dict] = {
    "slack": {
        "name": "Slack", "category": "Communication", "auth_method": "oauth",
        "status": "available", "icon": "slack", "accent": "#4A154B",
        "description": "Capture decisions and knowledge from selected channels.",
        "captures": "Messages from channels you choose",
        "install_path": "/slack/install", "needs_channel_selection": True,
        "setup_hint": "Invite the Knova bot to a channel in Slack, then pick channels here.",
    },
    "google_drive": {
        "name": "Google Drive", "category": "Documents", "auth_method": "oauth",
        "status": "coming_soon", "icon": "google-drive", "accent": "#1FA463",
        "description": "Auto-ingest documents from selected Drive folders.",
        "captures": "Docs, Sheets, Slides, PDFs in chosen folders",
        "install_path": "/google/install", "needs_channel_selection": True,
    },
    "google_meet": {
        "name": "Google Meet", "category": "Meetings", "auth_method": "oauth",
        "status": "coming_soon", "icon": "google-meet", "accent": "#00897B",
        "description": "Turn meeting transcripts into knowledge notes.",
        "captures": "Meet recordings & transcripts", "install_path": "/google/install",
    },
    "google_calendar": {
        "name": "Google Calendar", "category": "Meetings", "auth_method": "oauth",
        "status": "coming_soon", "icon": "google-calendar", "accent": "#4285F4",
        "description": "Add meeting context (titles, attendees) to notes.",
        "captures": "Event titles & attendees", "install_path": "/google/install",
    },
    "zoom": {
        "name": "Zoom", "category": "Meetings", "auth_method": "oauth",
        "status": "coming_soon", "icon": "zoom", "accent": "#2D8CFF",
        "description": "Distill Zoom meeting recordings into decisions & action items.",
        "captures": "Cloud recording transcripts", "install_path": "/zoom/install",
    },
    "microsoft_teams": {
        "name": "Microsoft Teams", "category": "Communication", "auth_method": "oauth",
        "status": "coming_soon", "icon": "teams", "accent": "#6264A7",
        "description": "Capture knowledge from Teams channels and meetings.",
        "captures": "Channel messages & meeting transcripts", "install_path": "/microsoft/install",
    },
    "outlook": {
        "name": "Outlook / Microsoft 365", "category": "Documents", "auth_method": "oauth",
        "status": "coming_soon", "icon": "outlook", "accent": "#0F6CBD",
        "description": "Ingest shared documents from SharePoint / OneDrive.",
        "captures": "SharePoint & OneDrive documents", "install_path": "/microsoft/install",
    },
    "notion": {
        "name": "Notion", "category": "Productivity", "auth_method": "oauth",
        "status": "coming_soon", "icon": "notion", "accent": "#000000",
        "description": "Sync knowledge from selected Notion databases and pages.",
        "captures": "Pages & databases you share", "install_path": "/notion/install",
    },
    "confluence": {
        "name": "Confluence", "category": "Productivity", "auth_method": "api_key",
        "status": "coming_soon", "icon": "confluence", "accent": "#172B4D",
        "description": "Ingest spaces and pages from Atlassian Confluence.",
        "captures": "Spaces & pages", "api_key_label": "Confluence API token",
    },
}

CATEGORY_ORDER = ["Communication", "Meetings", "Documents", "Productivity"]


def _provider_public(pid: str, p: dict) -> dict:
    """Registry fields safe to expose to the frontend."""
    return {
        "id": pid,
        "name": p["name"],
        "category": p["category"],
        "auth_method": p["auth_method"],
        "status": p["status"],
        "icon": p.get("icon", pid),
        "accent": p.get("accent", "#2563EB"),
        "description": p.get("description", ""),
        "captures": p.get("captures", ""),
        "needs_channel_selection": p.get("needs_channel_selection", False),
        "install_path": p.get("install_path"),
        "api_key_label": p.get("api_key_label"),
        "setup_hint": p.get("setup_hint"),
    }


@router.get("/integrations")
async def list_integrations(workspace_id: str):
    """
    Everything the admin Integrations page needs: every provider with its
    live connection status for this workspace. The page renders purely from
    this — no provider-specific frontend code.
    """
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")

    conns = bc.supabase.table("connections").select(
        "id, provider, external_team_name, status, error_detail, config, created_at"
    ).eq("workspace_id", workspace_id).neq("status", "revoked").execute().data or []
    by_provider = {c["provider"]: c for c in conns}

    items = []
    for pid, p in PROVIDERS.items():
        entry = _provider_public(pid, p)
        conn = by_provider.get(pid)
        if conn and p["status"] == "available":
            channels = (conn.get("config") or {}).get("channels", [])
            entry["status"] = "connected"
            entry["connection"] = {
                "id": conn["id"],
                "team": conn.get("external_team_name"),
                "connected_at": conn.get("created_at"),
                "channels_selected": len(channels),
                "error": conn.get("error_detail"),
            }
        else:
            entry["connection"] = None
        items.append(entry)

    return {
        "categories": CATEGORY_ORDER,
        "railway_base": os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app"),
        "integrations": items,
    }


class ApiKeyConnectRequest(BaseModel):
    workspace_id: str
    provider:     str
    api_key:      str
    user_id:      Optional[str] = ""
    config:       Optional[dict] = {}


@router.post("/integrations/connect-key")
async def connect_api_key(body: ApiKeyConnectRequest):
    """
    Connect an API-key-based provider (e.g. Confluence). OAuth providers use
    their own /{provider}/install popup instead. Stored encrypted, same as
    OAuth tokens. Refuses providers that aren't 'available' yet.
    """
    p = PROVIDERS.get(body.provider)
    if not p:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{body.provider}'.")
    if p["auth_method"] != "api_key":
        raise HTTPException(status_code=400, detail=f"{p['name']} connects via OAuth, not an API key.")
    if p["status"] != "available":
        raise HTTPException(status_code=400, detail=f"{p['name']} integration is coming soon.")
    if not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required.")

    row = {
        "workspace_id":     body.workspace_id,
        "provider":         body.provider,
        "external_team_name": p["name"],
        "access_token_enc": bc.encrypt_secret(body.api_key.strip()),
        "status":           "active",
        "connected_by":     body.user_id,
        "config":           body.config or {},
    }
    bc.supabase.table("connections").insert(row).execute()
    return {"success": True, "provider": body.provider}


def oauth_complete_html(provider: str, status: str) -> HTMLResponse:
    """
    Popup-friendly OAuth landing: notifies the opener window and closes
    itself (standard SaaS feel). Falls back to redirecting the whole tab to
    the app if it wasn't opened as a popup.
    """
    ok = status == "connected"
    title = "Connected" if ok else "Connection failed"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;display:flex;height:100vh;margin:0;
align-items:center;justify-content:center;background:#F8FAFC;color:#0F172A}}
.card{{text-align:center}}.dot{{font-size:48px}}</style></head>
<body><div class="card"><div class="dot">{'✅' if ok else '⚠️'}</div>
<h2>{title}</h2><p>{'You can close this window.' if ok else 'Please try again.'}</p></div>
<script>
  try {{
    if (window.opener) {{
      window.opener.postMessage({{type:'integration_result', provider:'{provider}', status:'{status}'}}, '*');
      setTimeout(function(){{ window.close(); }}, 800);
    }} else {{
      setTimeout(function(){{ window.location = '{APP_REDIRECT_URL}?integration={provider}&status={status}'; }}, 1200);
    }}
  }} catch (e) {{ window.location = '{APP_REDIRECT_URL}'; }}
</script></body></html>"""
    return HTMLResponse(html)
