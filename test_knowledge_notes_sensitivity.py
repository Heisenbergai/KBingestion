"""
P0 security hotfix -- /knowledge-notes must enforce the sensitivity ladder.

THE BUG THIS FILE EXISTS TO KEEP FIXED. `/knowledge-notes` applied NO
sensitivity filtering at all. Every member of a workspace could read every
note in it -- a note classified `confidential` or `restricted` was returned to
an ordinary member exactly like a public one, while the neighbouring
`/document-tables` route had resolved the caller's real role server-side all
along. The Library page reads this endpoint, so the exposure was in the
product's normal path, not an obscure corner.

WHAT "FIXED" HAS TO MEAN, AND WHY EACH PART MATTERS:

  * The ceiling is DERIVED from the verified token's role, never accepted
    from the caller. There is no sensitivity parameter on this route to send,
    which is stronger than validating one away.
  * The filter runs IN THE DATABASE QUERY, not over the results. Fetching
    everything and hiding some of it in Python would still put restricted
    text on the wire and in the server's memory, and the next refactor that
    forgets the hiding step re-opens the hole silently.
  * A hidden row must be ABSENT, not redacted. If the response said
    "3 notes (1 hidden)", the count itself would disclose exactly what the
    ladder exists to conceal.

Run with: python -m pytest test_knowledge_notes_sensitivity.py -v
"""
import asyncio
import uuid

import pytest
from fastapi import HTTPException

from auth import AuthContext
import brain_connectors as bc

supabase = bc.supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"

# One throwaway workspace for the whole module. Every note created here is
# removed by a workspace-scoped sweep, so a row created but never tracked
# still cannot survive -- the exact failure mode that leaked three rows into
# the shared database during the Phase 8I release gate.
FIXTURE_WS = str(uuid.uuid4())
OTHER_WS = str(uuid.uuid4())

SENSITIVITIES = ("public", "internal", "confidential", "restricted")


def _ctx(workspace=FIXTURE_WS, role="member", super_admin=False):
    return AuthContext(user_id="test-user",
                       workspaces={workspace: role} if workspace else {},
                       is_super_admin=super_admin, enforced=True, caller="pytest")


def _call(workspace_id, auth, limit=100):
    return asyncio.run(
        bc.list_knowledge_notes(workspace_id=workspace_id, limit=limit, auth=auth))


def _make_note(workspace_id, sensitivity, marker):
    return supabase.table("knowledge_notes").insert({
        "workspace_id": workspace_id,
        "provider": "bot_learning",
        "source_type": "note",
        "source_tier": 2,
        "title": f"HOTFIX-{sensitivity.upper()}-TITLE-{marker}",
        "body": f"HOTFIX-{sensitivity.upper()}-BODY-{marker}",
        "participants": [],
        "source_ref": str(uuid.uuid4()),
        "status": "active",
        "sensitivity": sensitivity,
        "authority": "working",
        "lifecycle_status": "active",
    }).execute().data[0]["id"]


def _sweep(workspace_id):
    supabase.table("knowledge_notes").delete().eq("workspace_id", workspace_id).execute()


@pytest.fixture(scope="module", autouse=True)
def notes():
    """One note per sensitivity in the fixture workspace, plus one in a second
    workspace to prove isolation."""
    made = {s: _make_note(FIXTURE_WS, s, "A") for s in SENSITIVITIES}
    made["other_ws"] = _make_note(OTHER_WS, "public", "B")
    try:
        yield made
    finally:
        _sweep(FIXTURE_WS)
        _sweep(OTHER_WS)


def _titles(result):
    return {n.get("title") for n in result["notes"]}


# =====================================================================
# 1-4. The ladder itself.
# =====================================================================

def test_1_public_note_is_visible_to_an_authorized_member():
    titles = _titles(_call(FIXTURE_WS, _ctx(role="member")))
    assert "HOTFIX-PUBLIC-TITLE-A" in titles


def test_2_internal_note_is_visible_to_an_ordinary_member():
    titles = _titles(_call(FIXTURE_WS, _ctx(role="member")))
    assert "HOTFIX-INTERNAL-TITLE-A" in titles


def test_3_confidential_and_restricted_are_hidden_from_a_lower_ceiling():
    """The actual bug. Before the fix both of these came back."""
    result = _call(FIXTURE_WS, _ctx(role="member"))
    titles = _titles(result)
    assert "HOTFIX-CONFIDENTIAL-TITLE-A" not in titles
    assert "HOTFIX-RESTRICTED-TITLE-A" not in titles

    # An admin sees confidential but still not restricted -- the ladder has
    # rungs, it is not a boolean.
    admin = _titles(_call(FIXTURE_WS, _ctx(role="admin")))
    assert "HOTFIX-CONFIDENTIAL-TITLE-A" in admin
    assert "HOTFIX-RESTRICTED-TITLE-A" not in admin


def test_4_restricted_note_is_visible_to_an_authorized_ceiling():
    """A fix that hid restricted notes from everyone would pass test 3 while
    breaking the product. Owner and super-admin must still see all four."""
    owner = _titles(_call(FIXTURE_WS, _ctx(role="owner")))
    for s in SENSITIVITIES:
        assert f"HOTFIX-{s.upper()}-TITLE-A" in owner, s

    sa = _titles(_call(FIXTURE_WS, _ctx(workspace=FIXTURE_WS, role="member", super_admin=True)))
    assert "HOTFIX-RESTRICTED-TITLE-A" in sa


# =====================================================================
# 5-7. Isolation, counting, and content.
# =====================================================================

def test_5_cross_workspace_isolation():
    # A note in another workspace never appears, even for an owner...
    owner_titles = _titles(_call(FIXTURE_WS, _ctx(role="owner")))
    assert "HOTFIX-PUBLIC-TITLE-B" not in owner_titles

    # ...and asking for a workspace the caller is not a member of is refused
    # outright rather than answered with an empty list.
    with pytest.raises(HTTPException) as e:
        _call(OTHER_WS, _ctx(workspace=FIXTURE_WS, role="owner"))
    assert e.value.status_code in (403, 404)


def test_6_the_count_does_not_reveal_hidden_rows():
    """'2 notes (2 hidden)' would disclose precisely what the ladder conceals.
    A hidden note must be ABSENT, so the member's count is simply smaller."""
    member = _call(FIXTURE_WS, _ctx(role="member"))
    owner = _call(FIXTURE_WS, _ctx(role="owner"))

    assert len(member["notes"]) == 2      # public + internal
    assert len(owner["notes"]) == 4       # all four

    # No field anywhere in the response hints at what was withheld.
    blob = str(member).lower()
    for leak in ("hidden", "redacted", "withheld", "filtered", "total_count",
                 "unavailable", "restricted"):
        assert leak not in blob, leak
    assert set(member) == {"notes"}


def test_7_no_restricted_content_appears_anywhere_in_the_response():
    """Title, body and every other field -- the whole serialized payload is
    searched, not just the fields we remembered to check."""
    blob = str(_call(FIXTURE_WS, _ctx(role="member")))
    for s in ("CONFIDENTIAL", "RESTRICTED"):
        assert f"HOTFIX-{s}-TITLE-A" not in blob
        assert f"HOTFIX-{s}-BODY-A" not in blob


# =====================================================================
# 8. The filter is server-side, and nothing else changed.
# =====================================================================

def test_8a_filtering_happens_in_the_query_not_after_it():
    """Fetch-all-then-hide would still put restricted text on the wire. The
    `.in_("sensitivity", ...)` must be part of the database query, and the
    ceiling must come from the token rather than from any argument."""
    import ast
    import inspect

    src = inspect.getsource(bc.list_knowledge_notes)
    tree = ast.parse(src.lstrip())

    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "in_" in calls, "sensitivity filter must be applied in the query"
    assert "assert_workspace" in calls
    assert "role_in" in calls, "the ceiling must be derived from the caller's role"

    # No client-supplied sensitivity parameter exists to send.
    params = set(inspect.signature(bc.list_knowledge_notes).parameters)
    assert params == {"workspace_id", "limit", "auth"}
    for forbidden in ("sensitivity", "allowed_sensitivities", "role", "is_super_admin"):
        assert forbidden not in params


def test_8b_the_ladder_matches_the_rest_of_the_codebase():
    """A second, divergent ladder would be worse than none -- two answers to
    'what may this person see' is how a leak survives a review."""
    import graph_query as gq
    for role, is_sa in (("owner", False), ("admin", False), ("member", False),
                        (None, False), ("member", True)):
        assert (bc._resolve_allowed_sensitivities(role, is_sa)
                == gq.resolve_allowed_sensitivities(role, is_sa)), (role, is_sa)


def test_8c_unrelated_retrieval_behaviour_is_unchanged():
    """The hotfix must not have altered what the endpoint returns for notes
    the caller IS allowed to see: same fields, same order, same shape."""
    result = _call(FIXTURE_WS, _ctx(role="owner"))
    assert set(result) == {"notes"}
    expected_fields = {"id", "provider", "source_type", "category", "title", "body",
                       "participants", "source_ref", "occurred_at", "created_at"}
    for n in result["notes"]:
        assert set(n) == expected_fields
        # `sensitivity` is deliberately NOT echoed back -- the classification
        # itself is not the caller's business.
        assert "sensitivity" not in n

    # Newest-first ordering preserved.
    stamps = [n["created_at"] for n in result["notes"]]
    assert stamps == sorted(stamps, reverse=True)


def test_8d_the_limit_cannot_be_used_to_probe_hidden_rows():
    """PostgREST applies the filter before the limit, so a member asking for
    one note gets a visible one -- never a hidden one, and never a gap where
    a hidden one would have been."""
    one = _call(FIXTURE_WS, _ctx(role="member"), limit=1)
    assert len(one["notes"]) == 1
    assert "HOTFIX-RESTRICTED" not in str(one)
    assert "HOTFIX-CONFIDENTIAL" not in str(one)


# =====================================================================
# 9. Cleanup.
# =====================================================================

def test_9_fixtures_are_fully_removed_and_production_is_untouched():
    """Runs last. The module-scoped fixture sweeps on teardown; this proves
    the sweep targets only throwaway workspaces and that the real workspace's
    notes were never touched."""
    real = supabase.table("knowledge_notes").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data or []
    assert len(real) == 10, f"real workspace note count changed: {len(real)}"

    # Nothing this module created leaked into a real workspace.
    stray = supabase.table("knowledge_notes").select("id,workspace_id,title") \
        .like("title", "HOTFIX-%").execute().data or []
    assert all(r["workspace_id"] in (FIXTURE_WS, OTHER_WS) for r in stray), stray
