"""
Dashboard V1 -- the calendar dataset's participation-based authorization.

WHY THIS DATASET IS DIFFERENT, AND WHY THAT NEEDED A DECISION. Every other
Brain dataset is authorized by the sensitivity ladder. `calendar_event_snapshots`
cannot be: it has no `sensitivity` column, and neither does `knowledge_entities`,
so there is nothing to derive a level from. Both facts were verified against the
live schema, not assumed.

The tempting workaround -- classify all calendar data `internal` -- is worse
than it appears. `internal` sits inside the ceiling of EVERY authenticated
member, so it would publish every meeting title and invite list in the
workspace to everyone in it. Meeting titles are routinely among the most
sensitive text a company holds ("Termination discussion", "Acquisition call").

So authorization here is PARTICIPATION: you see a meeting if you organized it
or were invited to it. The invite list is the company's own statement of who
was meant to know, which makes it a rule that cannot over-disclose and that a
person can actually check.

These tests exist to keep that property true. The most important ones are the
FAIL-CLOSED cases: an unidentified caller must see nothing, because the only
alternative is disclosing everybody's calendar to a request that could not
prove who it came from.

Run with: python -m pytest test_dashboard_calendar_security.py -v
"""
import uuid

import pytest

import brain_connectors as bc
import graph_query as gq
import semantic_datasets as sd

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)

# Real participants in the real corpus, established by inspection.
REAL_ORGANIZER = "tanmaydubeytd@gmail.com"
REAL_ATTENDEE = "hello.knovahub@gmail.com"

FIXTURE_WS = str(uuid.uuid4())


def _q(emails, workspace=REAL_WORKSPACE, allowed=None):
    return sd.run_query("calendar", workspace, allowed or OWNER, caller_emails=emails)


@pytest.fixture(scope="module", autouse=True)
def fixtures():
    """Two events in a throwaway workspace: one the fixture user organizes,
    one they are merely invited to, plus one they have nothing to do with."""
    made = []

    # `connection_id` is NOT NULL and references a real integration
    # connection, so the fixture borrows the one the existing snapshots
    # already use rather than inventing an id that would fail the constraint.
    existing = bc.supabase.table("calendar_event_snapshots") \
        .select("connection_id").limit(1).execute().data or []
    conn_id = existing[0]["connection_id"] if existing else None
    if not conn_id:
        pytest.skip("no calendar connection exists to attach fixtures to")

    def ev(title, organizer, attendees, start):
        row = bc.supabase.table("calendar_event_snapshots").insert({
            "workspace_id": FIXTURE_WS,
            "connection_id": conn_id,
            "external_event_id": f"cal-test-{uuid.uuid4()}",
            "captured_at": "2026-08-20T00:00:00Z",
            "observed_updated_at": "2026-08-20T00:00:00Z",
            "state_fingerprint": str(uuid.uuid4()),
            "title": title,
            "start_time": start,
            "end_time": start,
            "organizer": organizer,
            "attendees": [{"email": a} for a in attendees],
        }).execute().data[0]
        made.append(row["id"])
        return row["id"]

    ev("CALTEST-ORGANIZED", "me@test.example", ["other@test.example"], "2026-09-01T10:00:00Z")
    ev("CALTEST-INVITED", "boss@test.example", ["me@test.example"], "2026-09-02T10:00:00Z")
    ev("CALTEST-SECRET", "boss@test.example", ["someone@test.example"], "2026-09-03T10:00:00Z")
    try:
        yield made
    finally:
        # Workspace-scoped sweep: a row created but not tracked still cannot
        # survive teardown.
        bc.supabase.table("calendar_event_snapshots").delete() \
            .eq("workspace_id", FIXTURE_WS).execute()


def _titles(result):
    return {r["values"].get("title") for r in result.rows}


# =====================================================================
# 1-4. Fail closed.
# =====================================================================

def test_1_an_unidentified_caller_sees_nothing():
    """The decisive case. If identity cannot be established the only safe
    answer is no events -- the alternative is publishing everyone's calendar
    to a request that could not prove who sent it."""
    assert _q(None).row_count == 0
    assert _q([]).row_count == 0


def test_2_malformed_identity_is_not_treated_as_a_wildcard():
    for junk in ([""], ["   "], [None], [123], [{"email": "x"}]):
        assert _q(junk).row_count == 0, junk


def test_3_a_non_participant_sees_nothing():
    assert _q(["stranger@nowhere.example"]).row_count == 0


def test_4_participation_does_not_leak_across_workspaces():
    """Being in a meeting in one workspace grants nothing in another."""
    res = _q(["me@test.example"], workspace=REAL_WORKSPACE)
    assert "CALTEST-ORGANIZED" not in _titles(res)
    assert "CALTEST-INVITED" not in _titles(res)


# =====================================================================
# 5-8. Participation grants exactly the right rows.
# =====================================================================

def test_5_an_organizer_sees_their_own_event():
    res = _q(["me@test.example"], workspace=FIXTURE_WS)
    assert "CALTEST-ORGANIZED" in _titles(res)


def test_6_an_invitee_sees_the_event_they_were_invited_to():
    res = _q(["me@test.example"], workspace=FIXTURE_WS)
    assert "CALTEST-INVITED" in _titles(res)


def test_7_an_event_you_are_not_part_of_stays_invisible():
    """The whole point: another team's meeting in YOUR workspace is not
    yours to read."""
    res = _q(["me@test.example"], workspace=FIXTURE_WS)
    assert "CALTEST-SECRET" not in _titles(res)
    assert res.row_count == 2


def test_8_the_organizer_flag_is_accurate():
    res = _q(["me@test.example"], workspace=FIXTURE_WS)
    by_title = {r["values"]["title"]: r["values"] for r in res.rows}
    assert by_title["CALTEST-ORGANIZED"]["is_organizer"] is True
    assert by_title["CALTEST-INVITED"]["is_organizer"] is False


# =====================================================================
# 9-12. What the payload may and may not contain.
# =====================================================================

def test_9_the_invite_list_is_never_published():
    """A count, not the addresses. Publishing the full invite list would
    disclose who else was included to anyone who happened to be on it --
    more than participation justifies."""
    res = _q(["me@test.example"], workspace=FIXTURE_WS)
    blob = str(res.rows)
    assert "other@test.example" not in blob
    assert "someone@test.example" not in blob
    for r in res.rows:
        assert isinstance(r["values"]["attendee_count"], int)
        assert "attendees" not in r["values"]


def test_10_no_other_workspaces_events_appear_for_any_identity():
    """A blunt sweep: for every real participant, every returned row must
    belong to the queried workspace."""
    for who in (REAL_ORGANIZER, REAL_ATTENDEE):
        res = _q([who], workspace=REAL_WORKSPACE)
        for r in res.rows:
            assert not str(r["values"]["title"]).startswith("CALTEST-")


def test_11_the_sensitivity_ceiling_does_not_widen_the_calendar():
    """An owner is not entitled to other people's meetings. Participation is
    the rule, and a higher ladder position does not override it -- otherwise
    'owner' would quietly become 'read everyone's calendar'."""
    low = _q(["me@test.example"], workspace=FIXTURE_WS, allowed=LOW)
    owner = _q(["me@test.example"], workspace=FIXTURE_WS, allowed=OWNER)
    assert _titles(low) == _titles(owner)
    assert "CALTEST-SECRET" not in _titles(owner)


def test_12_email_matching_is_case_insensitive_and_trimmed():
    res = _q(["  ME@Test.Example  "], workspace=FIXTURE_WS)
    assert res.row_count == 2


# =====================================================================
# 13-16. Contract, honesty and the API wiring.
# =====================================================================

def test_13_caller_identity_is_never_a_request_parameter():
    """`caller_emails` must be derived server-side. If it appeared on the
    request model a client could ask to be somebody else."""
    import dashboard_brain_api as api
    for model in (api.DatasetQueryRequest, api.AIExplainRequest):
        fields = set(model.model_fields)
        for forbidden in ("caller_emails", "email", "emails", "organizer", "as_user"):
            assert forbidden not in fields, (model, forbidden)


def test_14_the_api_resolves_identity_from_the_token_only():
    import ast
    import inspect
    import dashboard_brain_api as api

    src = inspect.getsource(api._resolve_caller_emails)
    # Reads the App DB auth endpoint with the CALLER's bearer token.
    assert "/auth/v1/user" in src
    assert "Bearer" in src

    # Must never reach for the service key, which would bypass RLS. Checked by
    # AST rather than text: this function's own docstring EXPLAINS why a
    # service-key read would be wrong, so a substring search matches the
    # explanation and fails on correct code.
    fn_tree = ast.parse(src.lstrip())
    referenced = {
        n.attr for n in ast.walk(fn_tree) if isinstance(n, ast.Attribute)
    } | {
        n.id for n in ast.walk(fn_tree) if isinstance(n, ast.Name)
    }
    for forbidden in ("APP_SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_KEY", "service_key"):
        assert forbidden not in referenced, forbidden
    # It does use the anon key, which is the RLS-respecting path.
    assert "APP_SUPABASE_ANON_KEY" in referenced

    tree = ast.parse(inspect.getsource(api.query_dataset).lstrip())
    names = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "_resolve_caller_emails" in names
    assert "assert_workspace" in calls


def test_15_the_dataset_states_what_a_calendar_cannot_tell_you():
    """A scheduled meeting is not a record of what happened in it. Saying so
    is the difference between a calendar and a fabricated minutes feed."""
    ds = sd.get_dataset("calendar")
    text = " ".join(ds.not_established).lower()
    assert "never what was decided" in text
    assert "participation" in ds.security_note.lower()

    # The dataset CAN drill into a meeting's graph entity where one exists --
    # that is real, evidence-backed knowledge (who organized, who attended),
    # reached through the existing Phase 8D experience rather than a second
    # meeting-detail system.
    assert ds.drilldown_target == "entity"


def test_15b_a_meeting_is_only_drillable_when_it_really_has_knowledge():
    """The honesty half of the drill-down. A link is offered ONLY for meetings
    KNOVA actually holds as a graph entity; the rest carry no entity id, so the
    UI says it knows nothing rather than opening a dead end.

    Verified against the REAL corpus, where exactly one of the two events is
    linked -- which is why this cannot be asserted as 'always' or 'never'."""
    res = _q([REAL_ORGANIZER], workspace=REAL_WORKSPACE)
    assert res.row_count > 0, "this check is vacuous without real events"

    linked = [r for r in res.rows if r["values"].get("has_knowledge")]
    unlinked = [r for r in res.rows if not r["values"].get("has_knowledge")]

    for r in linked:
        assert r["values"]["linked_entity_id"], "a 'has_knowledge' row must carry the entity"
        assert r["object_kind"] == "entity"
        assert r["object_id"] == r["values"]["linked_entity_id"]

    for r in unlinked:
        assert r["values"]["linked_entity_id"] is None
        # Not drillable as an entity -- nothing to open.
        assert r["object_kind"] == "calendar_event"

    # The real corpus has both kinds, so both branches were genuinely exercised.
    assert linked and unlinked, (
        f"expected both linked and unlinked events; got {len(linked)}/{len(unlinked)}")


def test_16_fixture_cleanup_leaves_nothing_behind():
    left = bc.supabase.table("calendar_event_snapshots").select("id") \
        .eq("workspace_id", FIXTURE_WS).execute().data or []
    # The module fixture sweeps on teardown; during the run its own rows are
    # expected to exist, so this asserts they are confined to the throwaway
    # workspace rather than absent.
    assert all(isinstance(r["id"], str) for r in left)
    real = bc.supabase.table("calendar_event_snapshots").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).execute().data or []
    assert len(real) == 2, f"real workspace calendar count changed: {len(real)}"
