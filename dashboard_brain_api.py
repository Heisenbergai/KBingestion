"""
Phase 8B -- the authenticated HTTP bridge from the product to the Brain.

WHY THIS FILE EXISTS. The browser talks to the App DB with the user's own
JWT and Supabase RLS. Brain data lives in a different Postgres project the
browser holds no credentials for, and must never be given any. Before this
file, the entire Phase 7 intelligence stack (company_state, change_detection,
proactive_intelligence, organizational_learning, impact_analysis) had no HTTP
surface at all -- it was reachable only from Python. This router is that
surface, and it is the ONLY place where Brain data crosses into the product.

AUTH: no new mechanism. AuthContext via auth.current_user, then
auth.assert_workspace() before any workspace data is touched, then the role
resolved into a sensitivity ceiling by graph_query.resolve_allowed_sensitivities
-- the identical pattern wiki_api.py already uses.

WHAT A CLIENT MAY AND MAY NOT SEND (Part 2). A client may name a workspace,
a dataset, fields, filters, a group_by, an aggregation, a temporal mode and
an as_of. A client may NEVER send its effective authorization: role,
is_super_admin, or an allowed-sensitivity list are derived server-side from
the verified token on every single request and are not request parameters at
all -- there is no field to send them in, which is stronger than validating
them away. Everything a client does send is looked up in the semantic
registry and rejected if unknown; nothing is interpolated into a query.

CROSS-DATABASE (Part 9). Department names live in the App DB. They are
resolved server-side by forwarding THE CALLER'S OWN token to the App DB REST
API with the anon key, exactly as query.py and chatbot.py already do -- so
App-DB RLS decides what comes back. The service key is never used for this
path, because a service-key read would bypass RLS and hand a caller rows
their own session could not see. If the lookup cannot be performed, the
Brain-side result is returned with an explicit not_established note rather
than a fabricated join.
"""
import dataclasses
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import auth as auth_mod
from auth import AuthContext, current_user
import graph_query
import change_detection
import impact_analysis
import memory_retrieval
import semantic_datasets as sd
import dashboard_detail
import dashboard_ai
import brain_connectors as bc

router = APIRouter()

MAX_ROWS = 1000


def _allowed_sensitivities(auth: AuthContext, workspace_id: str) -> list[str]:
    """Server-derived, always. The caller's role comes from the verified
    token's memberships, never from the request body."""
    role = auth.role_in(workspace_id)
    return graph_query.resolve_allowed_sensitivities(role, auth.is_super_admin)


def _parse_as_of(as_of: Optional[str]) -> Optional[datetime]:
    if not as_of:
        return None
    try:
        dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be a valid ISO 8601 datetime.")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _bearer(authorization: Optional[str]) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _resolve_app_departments(workspace_id: str, token: str) -> Optional[dict]:
    """Cross-database department lookup, performed as the CALLER.

    Returns None (not {}) when the lookup could not be performed, so the
    resolver can tell "no departments" apart from "couldn't ask" and report
    the difference honestly instead of silently showing blank names."""
    if not token or not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return None
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get(
                f"{auth_mod.APP_SUPABASE_URL}/rest/v1/departments",
                params={"select": "id,name,parent_id",
                         "workspace_id": f"eq.{workspace_id}"},
                headers={"apikey": auth_mod.APP_SUPABASE_ANON_KEY,
                          "Authorization": f"Bearer {token}",
                          "Accept": "application/json"},
            )
            res.raise_for_status()
            rows = res.json()
    except Exception as e:
        # Fail closed to "unknown", never to a guess.
        print(f"DASHBOARD: app-DB department lookup failed (reporting as unresolved): {e}")
        return None
    by_id = {r["id"]: r for r in rows}
    return {r["id"]: {"name": r.get("name"),
                       "parent_name": (by_id.get(r.get("parent_id")) or {}).get("name")}
            for r in rows}


def _resolve_caller_emails(token: str) -> Optional[list]:
    """The caller's OWN email addresses, read as the caller.

    Used only by the `calendar` dataset, where authorization is participation
    rather than a sensitivity level. Two properties make this safe:

    1. It is derived from the VERIFIED TOKEN, never from the request body.
       There is no field a client could set to claim someone else's calendar.
    2. It is read with the caller's own JWT against the App DB, so RLS decides
       what comes back — a service-key read would happily return any profile.

    Returns None (not []) when the lookup cannot be performed, so the resolver
    can tell "nobody" from "couldn't ask". Both end up showing no events, which
    is the correct failure direction for a calendar.
    """
    if not token or not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return None
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get(
                f"{auth_mod.APP_SUPABASE_URL}/auth/v1/user",
                headers={"apikey": auth_mod.APP_SUPABASE_ANON_KEY,
                          "Authorization": f"Bearer {token}",
                          "Accept": "application/json"},
            )
            res.raise_for_status()
            body = res.json()
    except Exception as e:
        print(f"DASHBOARD: caller email lookup failed (calendar will show nothing): {e}")
        return None

    emails = []
    primary = body.get("email")
    if isinstance(primary, str) and primary.strip():
        emails.append(primary.strip())
    # Some providers carry a second verified address; include only VERIFIED
    # ones, since an unverified address is not proof of identity.
    for ident in (body.get("identities") or []):
        data = (ident or {}).get("identity_data") or {}
        em = data.get("email")
        if isinstance(em, str) and em.strip() and data.get("email_verified") is not False:
            if em.strip().lower() not in {e.lower() for e in emails}:
                emails.append(em.strip())
    return emails or None


class FilterSpec(BaseModel):
    field: str
    op: str = "eq"
    value: object = None


class DatasetQueryRequest(BaseModel):
    workspace_id: str
    dataset: str
    fields: Optional[list[str]] = None
    filters: Optional[list[FilterSpec]] = None
    group_by: Optional[str] = None
    group_bucket: Optional[str] = None
    aggregation: Optional[str] = None
    value_field: Optional[str] = None
    # Phase 8E analytical options. Every one is validated against the
    # registry exactly like group_by/aggregation already are -- richer
    # analysis, same authoritative boundary.
    series_by: Optional[str] = None
    series_bucket: Optional[str] = None
    top_n: Optional[int] = None
    top_direction: str = "top"
    percent: bool = False
    temporal_mode: str = sd.MODE_CURRENT
    as_of: Optional[str] = None
    window_days: Optional[int] = None
    window_offset_days: Optional[int] = None
    limit: int = Field(default=MAX_ROWS, ge=1, le=MAX_ROWS)
    # NOTE: there is deliberately no role / is_super_admin / sensitivity
    # field here. Authorization is derived from the token, not requested.


class DrillRequest(BaseModel):
    workspace_id: str
    dataset: str
    object_kind: str
    object_id: str
    as_of: Optional[str] = None
    filters: Optional[list[FilterSpec]] = None
    max_hops: int = 1


@router.get("/dashboard/datasets")
async def list_datasets(auth: AuthContext = Depends(current_user)):
    """The registry itself -- what a widget builder may offer. Contains only
    schema, no workspace data, so it needs authentication but no workspace
    authorization."""
    return {"datasets": sd.list_datasets(),
             "allowed_aggregations": sorted(sd.ALLOWED_AGGREGATIONS),
             "filter_operators": sorted(sd.FILTER_OPERATORS),
             "change_markers": sorted(sd.VALID_MARKERS),
             "temporal_meanings": sorted(sd.TEMPORAL_MEANINGS)}


@router.post("/dashboard/query")
async def query_dataset(body: DatasetQueryRequest,
                        auth: AuthContext = Depends(current_user),
                        authorization: Optional[str] = Header(None)):
    if not body.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(body.workspace_id)
    allowed = _allowed_sensitivities(auth, body.workspace_id)
    as_of = _parse_as_of(body.as_of)

    app_departments = None
    if body.dataset == "departments":
        app_departments = _resolve_app_departments(body.workspace_id, _bearer(authorization))

    # Resolved only for the one dataset that needs it, and only from the
    # caller's own verified token.
    caller_emails = None
    if body.dataset == "calendar":
        caller_emails = _resolve_caller_emails(_bearer(authorization))

    try:
        result = sd.run_query(
            dataset=body.dataset, workspace_id=body.workspace_id,
            allowed_sensitivities=allowed,
            caller_emails=caller_emails,
            fields=body.fields,
            filters=[f.model_dump() for f in (body.filters or [])],
            group_by=body.group_by, aggregation=body.aggregation,
            value_field=body.value_field, group_bucket=body.group_bucket,
            series_by=body.series_by, series_bucket=body.series_bucket,
            top_n=body.top_n, top_direction=body.top_direction,
            percent=body.percent,
            temporal_mode=body.temporal_mode, as_of=as_of,
            window_days=body.window_days,
            window_offset_days=body.window_offset_days,
            app_departments=app_departments,
        )
    except sd.DatasetError as e:
        # Registry rejections are the caller's mistake, not a server fault.
        raise HTTPException(status_code=400, detail=str(e))

    if body.dataset == "departments" and app_departments is None:
        result.not_established = list(result.not_established) + [
            "Workspace department records could not be resolved for this request; "
            "Brain-side department data is shown without workspace names."]

    payload = dataclasses.asdict(result)
    if len(payload["rows"]) > body.limit:
        payload["rows"] = payload["rows"][:body.limit]
        payload["notes"] = list(payload["notes"]) + [
            f"Truncated to {body.limit} rows; aggregates above cover the full result."]
    payload["security"] = {
        "workspace_id": body.workspace_id,
        "sensitivity_ceiling_applied": True,
        "filtered_before_aggregation": True,
    }
    return payload


@router.post("/dashboard/drilldown")
async def drilldown(body: DrillRequest, auth: AuthContext = Depends(current_user)):
    """The one drill-down endpoint. Phase 8D moved the per-object-type
    resolution into dashboard_detail.build_detail so this stays a thin,
    auditable authorization wrapper: authenticate, authorize the workspace,
    derive the ceiling from the token, then resolve.

    Authorization is re-established on EVERY call. An object id arriving from
    the frontend is treated as untrusted input, and no authorization from the
    dashboard request that produced the id is carried over (Part 19)."""
    if not body.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    if body.max_hops not in (1, 2):
        raise HTTPException(status_code=400, detail="max_hops must be 1 or 2.")
    auth.assert_workspace(body.workspace_id)
    allowed = _allowed_sensitivities(auth, body.workspace_id)
    as_of = _parse_as_of(body.as_of)

    try:
        sd.get_dataset(body.dataset)
    except sd.DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if body.object_kind not in dashboard_detail.SUPPORTED_OBJECT_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported object_kind {body.object_kind!r}.")

    try:
        return dashboard_detail.build_detail(
            body.object_kind, body.object_id, body.workspace_id, allowed,
            as_of=as_of, max_hops=body.max_hops)
    except dashboard_detail.DetailNotFound:
        # Identical for "no such object", "another workspace", and "exists but
        # you may not see it" — a probing client learns nothing either way.
        raise HTTPException(status_code=404, detail="Not found.")


# ── Phase 8G: AI proposal and grounded explanation ────────────────────────────
#
# SECURITY ORDER (8G Part 15), enforced here and nowhere else:
#
#     authenticate -> workspace -> sensitivity ceiling -> Brain query
#     -> visible result -> MODEL
#
# The model sits at the END of that chain, downstream of every check. Note what
# `/dashboard/ai/explain` deliberately does NOT accept: a result blob. Letting a
# client post the numbers to be explained would mean the explanation described
# whatever the client claimed rather than what the caller may actually see, and
# a hostile client could hand the model figures no query ever produced. So the
# endpoint takes the WIDGET CONFIG, re-runs the query itself under the ceiling
# derived from this request's token, and explains its own result.


class AIGenerateRequest(BaseModel):
    workspace_id: str
    request: str
    # No dataset allow-list, no role, no sensitivities: generation reads only
    # the registry SCHEMA, never workspace data, so there is nothing here to
    # authorize beyond membership.


class AIExplainRequest(BaseModel):
    """The widget to explain -- its QUESTION, never its answer."""
    workspace_id: str
    dataset: str
    title: Optional[str] = None
    fields: Optional[list[str]] = None
    filters: Optional[list[FilterSpec]] = None
    group_by: Optional[str] = None
    group_bucket: Optional[str] = None
    series_by: Optional[str] = None
    series_bucket: Optional[str] = None
    aggregation: Optional[str] = None
    value_field: Optional[str] = None
    top_n: Optional[int] = None
    # These two mirror DatasetQueryRequest's defaults EXACTLY. They are not
    # Optional-with-None: `run_query` treats an explicit None as a request for
    # temporal mode "None", which no dataset supports, so a widget that simply
    # never set a temporal mode would 400 on Explain while rendering fine.
    top_direction: str = "top"
    percent: bool = False
    temporal_mode: str = sd.MODE_CURRENT
    as_of: Optional[str] = None
    window_days: Optional[int] = None
    compare: bool = False


@router.post("/dashboard/ai/build")
async def ai_build(body: AIGenerateRequest, auth: AuthContext = Depends(current_user)):
    """Natural language -> a validated DRAFT. Creates nothing.

    The response is a proposal for a human to review and apply. It writes no
    dashboard, grants no share, and changes no permission -- applying it is a
    separate, explicit user action through the existing dashboard write path."""
    if not body.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    if not (body.request or "").strip():
        raise HTTPException(status_code=400, detail="request is required.")
    auth.assert_workspace(body.workspace_id)

    out = dashboard_ai.generate_dashboard(
        body.request, workspace_id=body.workspace_id, user_id=auth.user_id)
    out["security"] = {
        "model_saw_workspace_data": False,   # schema only -- see _registry_summary
        "validated_against_registry": True,
        "creates_nothing": True,
    }
    return out


@router.post("/dashboard/ai/explain")
async def ai_explain(body: AIExplainRequest, auth: AuthContext = Depends(current_user),
                     authorization: Optional[str] = Header(None)):
    if not body.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(body.workspace_id)
    allowed = _allowed_sensitivities(auth, body.workspace_id)
    as_of = _parse_as_of(body.as_of)

    app_departments = None
    if body.dataset == "departments":
        app_departments = _resolve_app_departments(body.workspace_id, _bearer(authorization))

    def _run(offset: int = 0):
        return sd.run_query(
            dataset=body.dataset, workspace_id=body.workspace_id,
            allowed_sensitivities=allowed,
            fields=body.fields,
            filters=[f.model_dump() for f in (body.filters or [])],
            group_by=body.group_by, aggregation=body.aggregation,
            value_field=body.value_field, group_bucket=body.group_bucket,
            series_by=body.series_by, series_bucket=body.series_bucket,
            top_n=body.top_n, top_direction=body.top_direction,
            percent=body.percent,
            temporal_mode=body.temporal_mode, as_of=as_of,
            window_days=body.window_days, window_offset_days=offset,
            app_departments=app_departments)

    try:
        result = _run()
        comparison = None
        if body.compare and body.temporal_mode == sd.MODE_WINDOW and body.window_days:
            # The previous window, resolved by the SAME query under the SAME
            # ceiling -- so any delta the model states is a real difference
            # between two real results, not arithmetic the model performed.
            prior = _run(offset=body.window_days)
            comparison = {"label": f"previous {body.window_days} days",
                           "value": prior.row_count}
    except sd.DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = dataclasses.asdict(result)
    response = {
        "dataset": payload["dataset"],
        "row_count": payload["row_count"],
        "temporal_context": payload["temporal_context"],
        "not_established": payload["not_established"],
        "aggregation": payload["aggregation"],
    }

    explanation = dashboard_ai.explain_widget(
        response, {"title": body.title}, comparison=comparison,
        workspace_id=body.workspace_id, user_id=auth.user_id)

    # EVIDENCE = the rows this number is literally built from, taken from the
    # SAME already-ceiling-filtered result. Not a second query, not a wider
    # one: if the ceiling excluded a row from the count, it is absent here too,
    # so the evidence list can never be richer than the number it explains.
    # Each carries its object identity so the UI can hand it to the EXISTING
    # Phase 8D drill-down for the full evidence chain, rather than growing a
    # second evidence system here.
    contributing = []
    for row in (payload["rows"] or [])[:5]:
        values = row.get("values") or {}
        label = next((values[k] for k in
                       ("statement", "label", "subject_label", "item_label", "name")
                       if isinstance(values.get(k), str) and values[k].strip()), None)
        contributing.append({
            "object_kind": row.get("object_kind"),
            "object_id": row.get("object_id"),
            "label": label,
        })

    return {
        "explanation": dataclasses.asdict(explanation),
        # Returned so the UI shows the SAME numbers the explanation describes.
        "result": response,
        "comparison": comparison,
        "contributing": contributing,
        "evidence_available": payload.get("evidence_available", False),
        "drilldown_target": payload.get("drilldown_target"),
        "security": {
            "sensitivity_ceiling_applied": True,
            "filtered_before_aggregation": True,
            "model_saw_only_visible_result": True,
        },
    }
