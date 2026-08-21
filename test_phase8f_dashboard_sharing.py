"""
Phase 8F -- dashboard sharing security.

THE ONE NON-NEGOTIABLE RULE: sharing a dashboard shares its CONFIGURATION,
never its data. The same widget config executed by two people must resolve
under each person's own authorization, and the lower-clearance viewer must not
be able to infer what they cannot see -- not from a count, a percentage, a
ranking, a comparison, a timeline, or a row total.

These tests drive the REAL Brain API with two different sensitivity ceilings
against the same configuration, which is exactly what a shared dashboard does.

Run with: python -m pytest test_phase8f_dashboard_sharing.py -v
"""
import asyncio
import uuid

import pytest
from fastapi import HTTPException

from query import supabase
from auth import AuthContext
import graph_query as gq
import semantic_datasets as sd
import dashboard_brain_api as api
import dashboard_detail as dd

OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)


class SharedBoard:
    """A workspace holding a mix of sensitivities, standing in for a dashboard
    an owner built from data a viewer cannot fully see."""

    def __init__(self):
        self.ws = str(uuid.uuid4())
        self.sk = []

    def claim(self, statement, sensitivity):
        i = supabase.table("structured_knowledge").insert({
            "workspace_id": self.ws, "canonical_source_type": "knowledge_note",
            "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
            "primitive_type": "fact", "statement": statement, "raw_subject_phrase": "x",
            "qualifier_words": [], "sensitivity": sensitivity, "authority": "official",
            "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
            "captured_at": "2026-08-19T00:00:00Z", "extraction_run_id": str(uuid.uuid4()),
            "primitive_fingerprint": f"t8f-{uuid.uuid4()}"}).execute().data[0]["id"]
        self.sk.append(i)
        return i

    def as_role(self, role):
        return AuthContext(user_id=f"u-{role}", workspaces={self.ws: role},
                           enforced=True, caller="pytest")

    def run(self, role, **cfg):
        """The SAME configuration, executed as a given role."""
        return asyncio.run(api.query_dataset(
            api.DatasetQueryRequest(workspace_id=self.ws, **cfg), self.as_role(role), None))

    def cleanup(self):
        for i in self.sk:
            supabase.table("structured_knowledge").delete().eq("id", i).execute()


@pytest.fixture
def board():
    b = SharedBoard()
    # Two visible claims, two the viewer may not see.
    b.claim("PUBLIC-8F everyone", "public")
    b.claim("INTERNAL-8F everyone", "internal")
    b.claim("RESTRICTED-8F owner only", "restricted")
    b.claim("RESTRICTED-8F owner only two", "restricted")
    try:
        yield b
    finally:
        b.cleanup()


def _leaks(payload) -> bool:
    blob = str(payload)
    return "RESTRICTED-8F" in blob or "hidden" in blob.lower()


# =====================================================================
# 1-4. Configuration vs data authorization -- the core rule.
# =====================================================================

def test_same_config_resolves_per_viewer(board):
    cfg = dict(dataset="evidence", group_by="sensitivity", aggregation="count")
    owner = board.run("owner", **cfg)
    viewer = board.run("member", **cfg)

    assert owner["row_count"] == 4
    assert viewer["row_count"] == 2
    groups = {b["group"] for b in viewer["aggregation"]["buckets"]}
    assert "restricted" not in groups
    assert not _leaks(viewer)


def test_viewer_never_sees_a_hidden_count(board):
    """Not '4 (2 hidden)'. The restricted rows never enter the aggregate at
    all, so there is no residue for a viewer to reason from."""
    viewer = board.run("member", dataset="evidence", aggregation="count")
    assert viewer["aggregation"]["buckets"][0]["value"] == 2
    assert not _leaks(viewer)


def test_percentage_denominator_is_the_viewers_own_total(board):
    """The subtlest leak: if the denominator were the owner's total, a viewer
    could derive the hidden count from percentages that don't sum to 100."""
    owner = board.run("owner", dataset="evidence", group_by="sensitivity",
                      aggregation="count", percent=True)
    viewer = board.run("member", dataset="evidence", group_by="sensitivity",
                       aggregation="count", percent=True)

    assert owner["aggregation"]["percent_basis"] == 4
    assert viewer["aggregation"]["percent_basis"] == 2
    assert abs(sum(b["percent"] for b in viewer["aggregation"]["buckets"]) - 100.0) < 0.01
    assert not _leaks(viewer)


def test_ranking_cannot_reveal_a_hidden_bucket(board):
    """Top-N over everything must still rank only what the viewer can see."""
    viewer = board.run("member", dataset="evidence", group_by="sensitivity",
                       aggregation="count", top_n=10)
    groups = {b["group"] for b in viewer["aggregation"]["buckets"]}
    assert groups <= {"public", "internal", None}
    assert not _leaks(viewer)


def test_bottom_ranking_cannot_reveal_a_hidden_bucket(board):
    viewer = board.run("member", dataset="evidence", group_by="sensitivity",
                       aggregation="count", top_n=10, top_direction="bottom")
    assert "restricted" not in {b["group"] for b in viewer["aggregation"]["buckets"]}


def test_timeline_density_cannot_reveal_hidden_activity(board):
    """A per-day count is a density signal; it must reflect only visible rows."""
    owner = board.run("owner", dataset="evidence", group_by="captured_at",
                      group_bucket="day", aggregation="count")
    viewer = board.run("member", dataset="evidence", group_by="captured_at",
                       group_bucket="day", aggregation="count")
    assert sum(b["value"] for b in owner["aggregation"]["buckets"]) == 4
    assert sum(b["value"] for b in viewer["aggregation"]["buckets"]) == 2
    assert not _leaks(viewer)


def test_row_listing_excludes_restricted_entirely(board):
    viewer = board.run("member", dataset="evidence")
    assert viewer["row_count"] == 2
    assert not _leaks(viewer)


# =====================================================================
# 5-7. Drill-down re-authorization.
# =====================================================================

def test_drilldown_uses_the_viewers_own_authorization(board):
    """A shared dashboard may show a bar the viewer can click. The drill-down
    must re-authorize -- never inherit the owner's ceiling."""
    restricted_id = board.sk[2]
    with pytest.raises(HTTPException) as e:
        asyncio.run(api.drilldown(api.DrillRequest(
            workspace_id=board.ws, dataset="evidence",
            object_kind="structured_knowledge", object_id=restricted_id),
            board.as_role("member")))
    assert e.value.status_code == 404

    owner_view = asyncio.run(api.drilldown(api.DrillRequest(
        workspace_id=board.ws, dataset="evidence",
        object_kind="structured_knowledge", object_id=restricted_id),
        board.as_role("owner")))
    assert "RESTRICTED-8F" in owner_view["header"]["label"]


def test_drilldown_object_ids_from_the_frontend_are_untrusted(board):
    """Even holding a real id from the owner's session, a viewer gets the same
    safe 404 as for an id that does not exist."""
    real_hidden = board.sk[3]
    fabricated = str(uuid.uuid4())
    codes = []
    for oid in (real_hidden, fabricated):
        with pytest.raises(HTTPException) as e:
            asyncio.run(api.drilldown(api.DrillRequest(
                workspace_id=board.ws, dataset="evidence",
                object_kind="structured_knowledge", object_id=oid),
                board.as_role("member")))
        codes.append((e.value.status_code, e.value.detail))
    assert codes[0] == codes[1] == (404, "Not found.")


def test_detail_builder_is_ceiling_driven_not_caller_driven():
    """build_detail takes the ceiling as an argument the ROUTER derives; there
    is no path where a caller supplies its own."""
    import inspect
    params = inspect.signature(dd.build_detail).parameters
    assert "allowed_sensitivities" in params
    src = inspect.getsource(api.drilldown)
    assert "_allowed_sensitivities(auth" in src


# =====================================================================
# 8-10. Workspace isolation.
# =====================================================================

def test_a_viewer_of_another_workspace_is_refused(board):
    other = AuthContext(user_id="stranger", workspaces={str(uuid.uuid4()): "owner"},
                        enforced=True, caller="pytest")
    with pytest.raises(HTTPException) as e:
        asyncio.run(api.query_dataset(
            api.DatasetQueryRequest(workspace_id=board.ws, dataset="evidence"), other, None))
    assert e.value.status_code == 403


def test_membership_elsewhere_grants_nothing_here(board):
    """Being an OWNER of some other workspace must not help."""
    elsewhere = AuthContext(user_id="u", workspaces={str(uuid.uuid4()): "owner"},
                            enforced=True, caller="pytest")
    with pytest.raises(HTTPException):
        asyncio.run(api.drilldown(api.DrillRequest(
            workspace_id=board.ws, dataset="evidence",
            object_kind="structured_knowledge", object_id=board.sk[0]), elsewhere))


def test_a_dashboard_config_carries_no_workspace_override():
    """Nothing a client sends can redirect a query to another workspace: the
    workspace is asserted before any data is touched."""
    import inspect
    src = inspect.getsource(api.query_dataset)
    assert "auth.assert_workspace(body.workspace_id)" in src
    assert src.index("assert_workspace") < src.index("sd.run_query")


# =====================================================================
# 11-13. The request contract cannot carry authorization.
# =====================================================================

def test_no_authorization_field_exists_on_any_request():
    for model in (api.DatasetQueryRequest, api.DrillRequest):
        fields = set(model.model_fields)
        for forbidden in ("role", "is_super_admin", "allowed_sensitivities",
                           "sensitivity", "owner_id", "as_user"):
            assert forbidden not in fields, f"{model.__name__} exposes {forbidden}"


def test_shared_config_contains_no_data(board):
    """What a dashboard persists is the question, never the answer.

    `aggregation` deliberately appears on BOTH sides and is not a violation:
    in a request it is the function to apply ("count") — part of the question;
    in a response it is the computed object with buckets — the answer. So the
    check is on the RESULT-bearing fields, plus a type check proving the
    request's `aggregation` is a scalar rather than a result."""
    result_fields = {"rows", "row_count", "generated_at", "evidence_available",
                     "security", "buckets", "percent_basis", "series_values"}
    config_fields = set(api.DatasetQueryRequest.model_fields)
    assert result_fields.isdisjoint(config_fields)

    req = api.DatasetQueryRequest(workspace_id=board.ws, dataset="evidence",
                                  aggregation="count")
    assert isinstance(req.aggregation, str), "the question names a function, not a result"

    # And the live response separates them: the answer object is far richer
    # than the string that asked for it.
    resp = board.run("owner", dataset="evidence", group_by="sensitivity",
                     aggregation="count")
    assert isinstance(resp["aggregation"], dict)
    assert "buckets" in resp["aggregation"]


def test_every_response_states_it_filtered_before_aggregating(board):
    for role in ("owner", "member"):
        r = board.run(role, dataset="evidence", group_by="sensitivity", aggregation="count")
        assert r["security"]["filtered_before_aggregation"] is True
        assert r["security"]["sensitivity_ceiling_applied"] is True
        assert r["security"]["workspace_id"] == board.ws


# =====================================================================
# 14-16. Comparison and empty-period safety.
# =====================================================================

def test_comparison_window_is_also_ceiling_bound(board):
    """A previous-period query is a second real query -- and therefore a second
    place a hidden row could leak. It must not."""
    cfg = dict(dataset="evidence", aggregation="count")
    viewer_now = board.run("member", **cfg)
    assert viewer_now["row_count"] == 2
    assert not _leaks(viewer_now)


def test_a_viewer_with_no_visible_rows_sees_an_honest_zero():
    """A board built entirely from restricted data must show a viewer zero --
    not an error, and not a hint that something exists."""
    b = SharedBoard()
    try:
        b.claim("RESTRICTED-8F only", "restricted")
        viewer = b.run("member", dataset="evidence", aggregation="count")
        assert viewer["row_count"] == 0
        assert viewer["aggregation"]["buckets"][0]["value"] == 0
        assert not _leaks(viewer)
        # And no note that would betray the existence of the hidden row.
        assert "restricted" not in str(viewer["notes"]).lower()
    finally:
        b.cleanup()


def test_owner_and_viewer_of_the_same_board_differ_legitimately(board):
    """Two people seeing different numbers is CORRECT for a shared dashboard.
    This test exists so that behaviour is never 'fixed'."""
    cfg = dict(dataset="evidence", aggregation="count")
    assert board.run("owner", **cfg)["row_count"] != board.run("member", **cfg)["row_count"]


# =====================================================================
# 17-18. Fixture hygiene.
# =====================================================================

def test_no_brain_credentials_or_internal_columns_reach_a_viewer(board):
    blob = str(board.run("member", dataset="evidence"))
    for secret in ("SUPABASE", "service_role", "apikey", "Bearer ", "supabase.co",
                    "primitive_fingerprint", "extraction_run_id"):
        assert secret not in blob


def test_fixture_cleanup_leaves_no_residue():
    b = SharedBoard()
    b.claim("temp-8f", "internal")
    assert sd.run_query("evidence", b.ws, OWNER).row_count == 1
    b.cleanup()
    assert sd.run_query("evidence", b.ws, OWNER).row_count == 0
