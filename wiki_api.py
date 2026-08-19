"""
Phase 6G -- the minimal HTTP surface a frontend needs to actually show a
Wiki page. Nothing before this phase exposed wiki_projection/wiki_navigation/
wiki_generation to the frontend at all -- every prior Wiki phase was proven
entirely through direct Python calls and pytest. This file adds no new
business logic of its own; every endpoint is a thin composition of the three
existing Wiki modules, following this codebase's own established router-
per-feature-file convention (see query.py's /document-tables for the exact
auth/sensitivity pattern this mirrors).

AUTH: identical to every other authenticated endpoint in this service --
AuthContext via auth.current_user, auth.assert_workspace() before touching
any workspace data, role/is_super_admin resolved into the same sensitivity
ladder graph_query.py's own tests already use
(graph_query.resolve_allowed_sensitivities). No new auth mechanism.
"""
import dataclasses
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends

from auth import AuthContext, current_user
import graph_query
import wiki_projection
import wiki_navigation
import wiki_generation

router = APIRouter()


def _parse_as_of(as_of: Optional[str]) -> Optional[datetime]:
    if not as_of:
        return None
    try:
        dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be a valid ISO 8601 datetime.")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _allowed_sensitivities(auth: AuthContext, workspace_id: str) -> list[str]:
    role = auth.role_in(workspace_id)
    return graph_query.resolve_allowed_sensitivities(role, auth.is_super_admin)


@router.get("/wiki/pages")
async def list_wiki_pages(workspace_id: str, as_of: Optional[str] = None,
                          auth: AuthContext = Depends(current_user)):
    """The Wiki's own minimal entry point -- every real page this caller's
    workspace currently has, so the frontend has somewhere to start from
    without already knowing a page id. NOT a visibility guarantee by itself
    (see wiki_projection.list_available_pages's own docstring); the detail
    endpoint below re-checks independently."""
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(workspace_id)
    allowed = _allowed_sensitivities(auth, workspace_id)
    as_of_dt = _parse_as_of(as_of)
    pages = wiki_projection.list_available_pages(workspace_id, allowed, as_of_dt)
    return {"pages": pages, "workspace_id": workspace_id}


@router.get("/wiki/{page_type}/{object_id}")
async def get_wiki_page(page_type: str, object_id: str, workspace_id: str,
                        as_of: Optional[str] = None, hops: int = 1,
                        include_prose: bool = True,
                        auth: AuthContext = Depends(current_user)):
    """The one endpoint a Wiki detail page needs: the deterministic page
    model, its bounded navigation neighborhood, and (unless disabled) LLM
    prose over the same claims -- all built from data already gated by this
    caller's real sensitivity ceiling before any of it left wiki_projection.

    include_prose=False lets the frontend fetch structure fast and prose
    separately/later if it wants to (Part 20 asks for both, not that they
    must be one round trip) -- default true keeps this a single request for
    the common case.
    """
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    if hops not in (1, 2):
        raise HTTPException(status_code=400, detail="hops must be 1 or 2.")
    auth.assert_workspace(workspace_id)
    allowed = _allowed_sensitivities(auth, workspace_id)
    as_of_dt = _parse_as_of(as_of)

    page = wiki_projection.build_page(page_type, object_id, workspace_id, allowed, as_of_dt)
    if page is None:
        # Deliberately identical whether the id doesn't exist, belongs to
        # another workspace, or is real but invisible to this caller --
        # never confirms existence to someone who can't see it (Part 15,
        # matching auth.assert_workspace's own identical-403 convention).
        raise HTTPException(status_code=404, detail="Page not found.")

    nav = wiki_navigation.get_navigation_context(page, workspace_id, allowed, as_of_dt, hops=hops)

    rendered = None
    if include_prose:
        rendered = wiki_generation.generate_wiki_page(page, user_id=auth.user_id or None)

    return {
        "page": dataclasses.asdict(page),
        "navigation": dataclasses.asdict(nav),
        "rendered": dataclasses.asdict(rendered) if rendered else None,
    }
