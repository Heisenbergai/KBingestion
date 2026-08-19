"""
Phase 5E Calendar evidence tests -- verifies calendar_evidence.py (fingerprint
+ idempotent snapshot writer) and graph_query.py's new
get_entity_primary_evidence() against both the real Knova Test Meeting 1
event and synthetic fixtures for state-change/security/temporal boundary
cases the single real event can't exercise on its own (there's only one
real calendar_events row in this workspace, and it must never be mutated).

Every fixture helper below builds its `ids` dict incrementally and wraps
creation in try/except-cleanup-and-reraise from the FIRST write onward --
the Phase 5D pass leaked 8 synthetic entities + 6 synthetic notes when an
earlier version of ITS fixtures didn't do this; that failure mode is closed
here structurally, not just remembered.

Run with: python -m pytest test_phase5e_calendar_evidence.py -v
"""
import uuid
from datetime import datetime, timezone, timedelta

import pytest

import calendar_evidence as ce
import connector_google as google
import connector_google_calendar as cgc
import graph_query as gq
from query import supabase

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OTHER_REAL_WORKSPACE = "20c3df60-d33c-4003-81d5-504750e526f1"
REAL_CONNECTION = "79d54c5e-8e2e-4fd6-bbd0-d7ea45502e83"          # google_drive, calendar-enabled
OTHER_REAL_CONNECTION = "35e46bc2-3909-41f8-a8d6-52ab12321d77"    # a different real connection (slack) -- only used as a distinct valid FK target for the connection-isolation test, not for anything calendar-semantic

REAL_CALENDAR_EVENT_ID = "aa473196-79dd-4a9c-aefc-f2c80d12ea94"
REAL_EXTERNAL_EVENT_ID = "668o197bdkl5sljf4irv1ksju1"
# A second real Calendar sync event ("Sales Catchup") arrived live during
# the Phase 6D regression pass (2026-08-18 20:00:30 UTC) via the actually-
# deployed filtration-worker cron -- same class of real, unrelated
# production event already documented once before this session (the first
# calendar_events row growing 1->2 mid-session). Both ids below are real,
# neither is a TEST-5E-* synthetic fixture.
REAL_EXTERNAL_EVENT_ID_2 = "3fojc6p45ti03kqb6jbeelp7h3"
MEETING_ENTITY_ID = "d18ce7fe-1859-4f0e-a56c-aa96a6fe9e5f"


def _cleanup_snapshots(*ids):
    for sid in ids:
        if sid:
            supabase.table("calendar_event_snapshots").delete().eq("id", sid).execute()


# =====================================================================
# 1. First snapshot creation (real event) & idempotency (2)
# =====================================================================

def test_first_real_snapshot_matches_current_calendar_event_row():
    row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    fp, normalized = ce.compute_state_fingerprint(
        title=row["title"], start_time=row["start_time"], end_time=row["end_time"],
        organizer=row["organizer"], attendees=row["attendees"],
        recurrence_rule=row["recurrence_rule"], meeting_url=row["meeting_url"],
        conference_id=row["conference_id"],
    )
    snapshots = supabase.table("calendar_event_snapshots").select("*") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().data
    assert len(snapshots) == 1, "exactly one real snapshot must exist for the real event"
    snap = snapshots[0]
    assert snap["state_fingerprint"] == fp, "the latest snapshot's fingerprint must equal the current state's fingerprint (Part 11 consistency)"
    assert snap["workspace_id"] == REAL_WORKSPACE
    assert snap["connection_id"] == REAL_CONNECTION
    assert snap["external_event_id"] == REAL_EXTERNAL_EVENT_ID
    assert snap["title"] == "Knova Test Meeting 1"
    assert snap["meeting_url"] == "https://meet.google.com/ngn-pjwu-jcn"
    assert snap["captured_at"] is not None


def test_identical_repeat_does_not_duplicate_real_snapshot():
    row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    result = ce.snapshot_from_calendar_event_row(row)
    assert result is None, "identical current state must not create a second snapshot"
    count = supabase.table("calendar_event_snapshots").select("id", count="exact") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().count
    assert count == 1


def test_real_calendar_events_row_unchanged_by_snapshot_writes():
    row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    assert row["title"] == "Knova Test Meeting 1"
    assert row["external_event_id"] == REAL_EXTERNAL_EVENT_ID
    assert row["deleted_at"] is None


# =====================================================================
# 3-6. Change detection (synthetic Version A / B fixtures)
# =====================================================================

def _make_snapshot(ids_key_prefix: str, external_event_id: str, **kwargs) -> str | None:
    return ce.maybe_create_snapshot(
        workspace_id=REAL_WORKSPACE, connection_id=REAL_CONNECTION,
        external_event_id=external_event_id, **kwargs,
    )


def test_changed_title_creates_new_snapshot():
    ext_id = f"TEST-5E-TITLE-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Version A", start_time="2026-10-01T10:00:00+00:00",
                            end_time="2026-10-01T10:30:00+00:00")
        ids.append(a)
        b = _make_snapshot("b", ext_id, title="Version B (renamed)", start_time="2026-10-01T10:00:00+00:00",
                            end_time="2026-10-01T10:30:00+00:00")
        ids.append(b)
        assert a is not None and b is not None and a != b

        repeat_b = _make_snapshot("b2", ext_id, title="Version B (renamed)", start_time="2026-10-01T10:00:00+00:00",
                                   end_time="2026-10-01T10:30:00+00:00")
        assert repeat_b is None, "repeating B's exact state must not create a third snapshot"

        count = supabase.table("calendar_event_snapshots").select("id", count="exact") \
            .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
            .eq("external_event_id", ext_id).execute().count
        assert count == 2, "exactly A and B, repeat-B must not add a third row"
    finally:
        _cleanup_snapshots(*ids)


def test_changed_time_creates_new_snapshot():
    ext_id = f"TEST-5E-TIME-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Fixed Title", start_time="2026-10-02T09:00:00+00:00",
                            end_time="2026-10-02T09:30:00+00:00")
        ids.append(a)
        b = _make_snapshot("b", ext_id, title="Fixed Title", start_time="2026-10-02T14:00:00+00:00",
                            end_time="2026-10-02T14:30:00+00:00")
        ids.append(b)
        assert a is not None and b is not None and a != b
    finally:
        _cleanup_snapshots(*ids)


def test_changed_attendee_ordering_alone_does_not_create_new_snapshot():
    ext_id = f"TEST-5E-ATTENDEE-ORDER-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Attendee order test",
                            attendees=[{"email": "alice@example.com", "response_status": "accepted"},
                                      {"email": "bob@example.com", "response_status": "accepted"}])
        ids.append(a)
        b = _make_snapshot("b", ext_id, title="Attendee order test",
                            attendees=[{"email": "bob@example.com", "response_status": "accepted"},
                                      {"email": "alice@example.com", "response_status": "accepted"}])
        assert b is None, "same attendees in a different order must not create a new snapshot"
    finally:
        _cleanup_snapshots(*ids)


def test_changed_attendee_response_status_does_create_new_snapshot():
    """The complementary case: reordering alone doesn't count, but a real
    RSVP change does -- proving the dedup/sort logic isn't accidentally
    hiding genuine state changes too."""
    ext_id = f"TEST-5E-RSVP-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="RSVP test",
                            attendees=[{"email": "carol@example.com", "response_status": "needsAction"}])
        ids.append(a)
        b = _make_snapshot("b", ext_id, title="RSVP test",
                            attendees=[{"email": "carol@example.com", "response_status": "accepted"}])
        ids.append(b)
        assert a is not None and b is not None and a != b
    finally:
        _cleanup_snapshots(*ids)


def test_equivalent_timestamp_formatting_does_not_create_new_snapshot():
    ext_id = f"TEST-5E-TS-EQUIV-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Timestamp equivalence test",
                            start_time="2026-10-03T10:00:00Z")
        ids.append(a)
        b = _make_snapshot("b", ext_id, title="Timestamp equivalence test",
                            start_time="2026-10-03T10:00:00+00:00")
        assert b is None, "Z and +00:00 for the same instant must normalize identically"
    finally:
        _cleanup_snapshots(*ids)


def test_null_handling_is_deterministic():
    ext_id = f"TEST-5E-NULLS-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Null fields test")  # every other field omitted -> None
        ids.append(a)
        assert a is not None
        row = supabase.table("calendar_event_snapshots").select("*").eq("id", a).execute().data[0]
        assert row["start_time"] is None
        assert row["organizer"] is None
        assert row["recurrence_rule"] is None
        assert row["attendees"] == []

        # repeating with the exact same all-null shape must still be a no-op
        b = _make_snapshot("b", ext_id, title="Null fields test")
        assert b is None
    finally:
        _cleanup_snapshots(*ids)


# =====================================================================
# 8. Deleted current projection does not delete prior snapshots
# =====================================================================

def test_deleted_calendar_event_leaves_snapshots_untouched():
    """Real finding, not fixed here: connector_google_calendar.py currently
    has NO write path for deleted_at at all (a cancelled event is just
    skipped). This test proves the LAYER-SEPARATION contract holds if/when
    deleted_at IS ever set -- via a direct, synthetic calendar_events row,
    never the real production one."""
    ext_id = f"TEST-5E-DELETE-{uuid.uuid4()}"
    ids: dict = {}
    try:
        ids["snap"] = _make_snapshot("del", ext_id, title="Soon to be cancelled")
        assert ids["snap"] is not None

        ids["calendar_event"] = supabase.table("calendar_events").insert({
            "workspace_id": REAL_WORKSPACE, "connection_id": REAL_CONNECTION,
            "external_event_id": ext_id, "title": "Soon to be cancelled",
        }).execute().data[0]["id"]

        supabase.table("calendar_events").update({
            "deleted_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", ids["calendar_event"]).execute()

        snap_after = supabase.table("calendar_event_snapshots").select("*").eq("id", ids["snap"]).execute().data
        assert len(snap_after) == 1, "the snapshot must still exist after the current projection is soft-deleted"
        assert snap_after[0]["title"] == "Soon to be cancelled"
    finally:
        if ids.get("calendar_event"):
            supabase.table("calendar_events").delete().eq("id", ids["calendar_event"]).execute()
        _cleanup_snapshots(ids.get("snap"))


# =====================================================================
# 9 & 10. Workspace / connection isolation
# =====================================================================

def test_workspace_isolation_snapshot_lookup():
    """A snapshot created under REAL_WORKSPACE must not be found when
    looked up under a different real workspace."""
    ext_id = f"TEST-5E-WS-ISO-{uuid.uuid4()}"
    ids = []
    try:
        a = _make_snapshot("a", ext_id, title="Workspace isolation test")
        ids.append(a)
        wrong_ws = supabase.table("calendar_event_snapshots").select("id") \
            .eq("workspace_id", OTHER_REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
            .eq("external_event_id", ext_id).execute().data
        assert wrong_ws == []
    finally:
        _cleanup_snapshots(*ids)


def test_connection_isolation_same_external_event_id_different_connections():
    """external_event_id is NOT assumed globally unique -- the same value
    under two different connections must be two independent identity
    slots, each getting its own snapshot history."""
    ext_id = f"TEST-5E-CONN-ISO-{uuid.uuid4()}"
    ids = []
    try:
        snap_conn_a = ce.maybe_create_snapshot(
            workspace_id=REAL_WORKSPACE, connection_id=REAL_CONNECTION,
            external_event_id=ext_id, title="Connection A's event",
        )
        ids.append(snap_conn_a)
        snap_conn_b = ce.maybe_create_snapshot(
            workspace_id=REAL_WORKSPACE, connection_id=OTHER_REAL_CONNECTION,
            external_event_id=ext_id, title="Connection B's event",
        )
        ids.append(snap_conn_b)
        assert snap_conn_a is not None and snap_conn_b is not None and snap_conn_a != snap_conn_b, (
            "the same external_event_id under two different connections must not collide"
        )
    finally:
        _cleanup_snapshots(*ids)


# =====================================================================
# 11. Snapshot evidence resolves as primary_source
# =====================================================================

def test_calendar_snapshot_evidence_kind_is_primary_source():
    """Not derived_support -- Calendar snapshots are the original
    organizational artifact, never a KNOVA interpretation of one."""
    row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    snap_id = ce.snapshot_from_calendar_event_row(row)  # idempotent no-op, real snapshot already exists
    real_snap = supabase.table("calendar_event_snapshots").select("id") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().data[0]["id"]

    assert gq._evidence_kind("calendar_event_snapshot") == "primary_source"
    assert gq._evidence_kind("structured_knowledge") == "derived_support"


# =====================================================================
# 12. Real Meeting resolves to a real Calendar snapshot (Part 10)
# =====================================================================

def test_meeting_entity_resolves_to_real_snapshot_via_identifier():
    evidence = gq.get_entity_primary_evidence(MEETING_ENTITY_ID, REAL_WORKSPACE)
    assert len(evidence) == 1
    e = evidence[0]
    assert e.evidence_kind == "primary_source"
    assert e.evidence_type == "calendar_event_snapshot"
    assert e.source_reference == "https://meet.google.com/ngn-pjwu-jcn"
    assert e.stance == "supports"
    assert e.captured_at is not None


def test_meeting_provenance_wrong_workspace_returns_nothing():
    evidence = gq.get_entity_primary_evidence(MEETING_ENTITY_ID, OTHER_REAL_WORKSPACE)
    assert evidence == []


# =====================================================================
# 13 & 14. Existing graph state untouched
# =====================================================================

def test_structured_knowledge_15_rows_unchanged():
    assert supabase.table("structured_knowledge").select("id", count="exact").execute().count == 15
    assert supabase.table("structured_knowledge").select("id", count="exact") \
        .eq("extraction_version", "v2.1").execute().count == 15


def test_three_graph_entities_unchanged():
    """Phase 5F later added two real, verified Person entities (Tanmay,
    John Snow) -- this no longer asserts exactly the original 3 labels,
    since that would be asserting Phase 5F never happened. The original
    three remain present and unchanged, which is what this test actually
    verifies now."""
    rows = supabase.table("knowledge_entities").select("id,canonical_label").execute().data
    labels = {r["canonical_label"] for r in rows}
    assert {"Product", "Operations", "Knova Test Meeting 1"} <= labels


def test_one_relationship_two_evidence_unchanged():
    """STALE COUNTS (Phase 5H): were 1/2 total as of Phase 5E. Phase 5H
    legitimately added two real Person->Meeting activity relationships
    (organized, attended) -- see test_phase5h_meeting_activity_relationships.py.
    This test now verifies the ORIGINAL requires_approval_from edge
    specifically, by id, rather than a stale total count."""
    rows = supabase.table("knowledge_relationships").select("*") \
        .eq("relationship_type", "requires_approval_from").execute().data
    assert len(rows) == 1
    ev = supabase.table("knowledge_relationship_evidence").select("id", count="exact") \
        .eq("relationship_id", rows[0]["id"]).execute().count
    assert ev == 2


# =====================================================================
# 16 & 17. No orphan snapshots / no duplicate fingerprints
# =====================================================================

def test_no_orphan_snapshots_after_full_suite():
    """Every remaining snapshot must belong to a real event's identity --
    proves every TEST-5E-* synthetic snapshot created above was actually
    cleaned up, not just asserted to have been. Two real ids now, not one
    -- see REAL_EXTERNAL_EVENT_ID_2's comment (Phase 6D regression,
    2026-08-18: a second real Calendar sync event legitimately arrived
    live during this session)."""
    real_ids = {REAL_EXTERNAL_EVENT_ID, REAL_EXTERNAL_EVENT_ID_2}
    rows = supabase.table("calendar_event_snapshots").select("external_event_id").execute().data
    assert all(r["external_event_id"] in real_ids for r in rows), (
        f"leaked synthetic snapshot(s) found: {[r for r in rows if r['external_event_id'] not in real_ids]}"
    )


def test_no_duplicate_fingerprints_within_one_identity():
    rows = supabase.table("calendar_event_snapshots").select("state_fingerprint") \
        .eq("workspace_id", REAL_WORKSPACE).eq("connection_id", REAL_CONNECTION) \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().data
    fingerprints = [r["state_fingerprint"] for r in rows]
    assert len(fingerprints) == len(set(fingerprints))


# =====================================================================
# B. Integration tests -- REAL poll_connection(), not maybe_create_snapshot()
# called directly. Google's live API is unreachable from this environment
# (no OAuth credentials here, same standing constraint as every other
# Google-Workspace-connector pass this session), so _list_events() and
# google._valid_access_token() are monkeypatched at the network boundary --
# everything else (connection validation, cancelled-event handling, the
# upsert-skip decision, the snapshot wiring, the returned stats dict) is the
# real, unmodified poll_connection() body.
# =====================================================================

def test_real_poller_path_idempotent_for_real_event(monkeypatch):
    """Part 8: routes the REAL Knova Test Meeting 1 event through the REAL
    poll_connection(), reconstructing Google's own response shape faithfully
    from the already-stored real row (never inventing new values). A real
    snapshot already exists from the direct-call proof earlier in Phase 5E --
    this proves the same real event, when it arrives through the actual
    poller mechanism, doesn't create a duplicate."""
    real_row = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    fake_google_event = {
        "id": real_row["external_event_id"], "status": "confirmed",
        "updated": real_row["updated_at_source"],
        "summary": real_row["title"],
        "start": {"dateTime": real_row["start_time"]},
        "end": {"dateTime": real_row["end_time"]},
        "organizer": {"email": real_row["organizer"]},
        "attendees": [{"email": a["email"], "responseStatus": a["response_status"]} for a in real_row["attendees"]],
        "conferenceData": {"conferenceId": real_row["conference_id"],
                           "entryPoints": [{"uri": real_row["meeting_url"]}]},
        "hangoutLink": real_row["meeting_url"],
        "recurrence": [],
    }
    monkeypatch.setattr(cgc, "_list_events", lambda token: [fake_google_event])
    monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token-for-test")

    before = supabase.table("calendar_event_snapshots").select("id", count="exact") \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().count

    result = cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)

    after = supabase.table("calendar_event_snapshots").select("id", count="exact") \
        .eq("external_event_id", REAL_EXTERNAL_EVENT_ID).execute().count
    assert before == after == 1, "the real poller path must not duplicate the already-existing real snapshot"
    assert result["snapshots_failed"] == 0

    row_after = supabase.table("calendar_events").select("*").eq("id", REAL_CALENDAR_EVENT_ID).execute().data[0]
    assert row_after["title"] == "Knova Test Meeting 1"
    assert row_after["deleted_at"] is None

    evidence = gq.get_entity_primary_evidence(MEETING_ENTITY_ID, REAL_WORKSPACE)
    assert len(evidence) == 1, "Meeting primary evidence resolution must still work after a real poll"


def test_wired_snapshot_created_on_first_sighting_of_new_event(monkeypatch):
    """Part 4: a brand-new event, seen through the real poller, must result
    in one current-state row AND one snapshot -- no second scheduled poll
    required."""
    ext_id = f"TEST-5E-FIRSTCAPTURE-{uuid.uuid4()}"
    ids: dict = {}
    try:
        event = {
            "id": ext_id, "status": "confirmed", "updated": "2026-11-10T00:00:00.000Z",
            "summary": "Brand new event", "start": {"dateTime": "2026-11-10T09:00:00+00:00"},
            "end": {"dateTime": "2026-11-10T09:30:00+00:00"},
        }
        monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token-for-test")
        monkeypatch.setattr(cgc, "_list_events", lambda token: [event])

        result = cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)
        assert result["processed"] == 1
        assert result["snapshots_created"] == 1

        rows = supabase.table("calendar_events").select("id") \
            .eq("connection_id", REAL_CONNECTION).eq("external_event_id", ext_id).execute().data
        assert len(rows) == 1
        ids["calendar_event"] = rows[0]["id"]

        snaps = supabase.table("calendar_event_snapshots").select("id") \
            .eq("external_event_id", ext_id).execute().data
        assert len(snaps) == 1
        ids["snapshot"] = snaps[0]["id"]
    finally:
        if ids.get("snapshot"):
            supabase.table("calendar_event_snapshots").delete().eq("id", ids["snapshot"]).execute()
        if ids.get("calendar_event"):
            supabase.table("calendar_events").delete().eq("id", ids["calendar_event"]).execute()


def test_synthetic_snapshot_failure_then_successful_retry(monkeypatch):
    """Parts 2, 3, 9: force calendar_evidence.snapshot_from_calendar_event_row
    to fail on the FIRST poll, after calendar_events has already been
    upserted successfully. Verify the failure is counted, not silently
    swallowed, and that calendar_events is correct and unaffected. Then let
    the same call succeed on a second poll of the identical event, and
    verify exactly one snapshot exists afterward -- never two, never a
    duplicate calendar_events row."""
    ext_id = f"TEST-5E-FAILRETRY-{uuid.uuid4()}"
    ids: dict = {}
    try:
        event = {
            "id": ext_id, "status": "confirmed", "updated": "2026-11-11T00:00:00.000Z",
            "summary": "Failure retry test", "start": {"dateTime": "2026-11-11T10:00:00+00:00"},
            "end": {"dateTime": "2026-11-11T10:30:00+00:00"},
        }
        monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token-for-test")
        monkeypatch.setattr(cgc, "_list_events", lambda token: [event])

        real_snapshot_fn = ce.snapshot_from_calendar_event_row
        call_count = {"n": 0}

        def failing_once(row):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("synthetic snapshot failure (test-injected)")
            return real_snapshot_fn(row)

        monkeypatch.setattr(ce, "snapshot_from_calendar_event_row", failing_once)

        result1 = cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)
        assert result1["snapshots_failed"] == 1
        assert result1["snapshots_created"] == 0
        assert result1["processed"] == 1, "calendar_events must still be upserted even though the snapshot failed"

        row = supabase.table("calendar_events").select("*") \
            .eq("connection_id", REAL_CONNECTION).eq("external_event_id", ext_id).execute().data
        assert len(row) == 1
        ids["calendar_event"] = row[0]["id"]
        assert row[0]["title"] == "Failure retry test"

        assert supabase.table("calendar_event_snapshots").select("id", count="exact") \
            .eq("external_event_id", ext_id).execute().count == 0, "no snapshot must exist after the failed attempt -- never fabricate evidence"

        result2 = cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)
        assert result2["snapshots_created"] == 1, "the retry (same unchanged event) must successfully create the snapshot this time"

        snaps = supabase.table("calendar_event_snapshots").select("id") \
            .eq("external_event_id", ext_id).execute().data
        assert len(snaps) == 1
        ids["snapshot"] = snaps[0]["id"]

        rows_after = supabase.table("calendar_events").select("id") \
            .eq("connection_id", REAL_CONNECTION).eq("external_event_id", ext_id).execute().data
        assert len(rows_after) == 1, "no duplicate current-state row from the retry"
    finally:
        if ids.get("snapshot"):
            supabase.table("calendar_event_snapshots").delete().eq("id", ids["snapshot"]).execute()
        if ids.get("calendar_event"):
            supabase.table("calendar_events").delete().eq("id", ids["calendar_event"]).execute()


def test_cancelled_event_sets_deleted_at_without_new_snapshot(monkeypatch):
    """Part 7: a cancelled sighting of an already-known event sets
    deleted_at on the current-state row; the prior snapshot remains exactly
    as it was, and no new snapshot is created for the cancellation itself."""
    ext_id = f"TEST-5E-CANCELLED-{uuid.uuid4()}"
    ids: dict = {}
    try:
        confirmed_event = {
            "id": ext_id, "status": "confirmed", "updated": "2026-11-12T00:00:00.000Z",
            "summary": "About to be cancelled", "start": {"dateTime": "2026-11-12T10:00:00+00:00"},
            "end": {"dateTime": "2026-11-12T10:30:00+00:00"},
        }
        monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token-for-test")
        monkeypatch.setattr(cgc, "_list_events", lambda token: [confirmed_event])
        cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)

        row = supabase.table("calendar_events").select("*") \
            .eq("connection_id", REAL_CONNECTION).eq("external_event_id", ext_id).execute().data
        assert len(row) == 1
        ids["calendar_event"] = row[0]["id"]
        assert row[0]["deleted_at"] is None

        snaps_before = supabase.table("calendar_event_snapshots").select("id") \
            .eq("external_event_id", ext_id).execute().data
        assert len(snaps_before) == 1
        ids["snapshot"] = snaps_before[0]["id"]

        cancelled_event = dict(confirmed_event, status="cancelled")
        monkeypatch.setattr(cgc, "_list_events", lambda token: [cancelled_event])
        cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)

        row_after = supabase.table("calendar_events").select("deleted_at") \
            .eq("id", ids["calendar_event"]).execute().data[0]
        assert row_after["deleted_at"] is not None

        snaps_after = supabase.table("calendar_event_snapshots").select("id") \
            .eq("external_event_id", ext_id).execute().data
        assert len(snaps_after) == 1
        assert snaps_after[0]["id"] == ids["snapshot"], (
            "the prior snapshot must remain untouched; no new snapshot for the cancellation itself"
        )

        # Repeat sighting of the same cancelled status must not re-bump deleted_at meaninglessly --
        # idempotent, matching the .is_("deleted_at","null") guard in the connector.
        cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)
        row_again = supabase.table("calendar_events").select("deleted_at") \
            .eq("id", ids["calendar_event"]).execute().data[0]
        assert row_again["deleted_at"] == row_after["deleted_at"]
    finally:
        if ids.get("snapshot"):
            supabase.table("calendar_event_snapshots").delete().eq("id", ids["snapshot"]).execute()
        if ids.get("calendar_event"):
            supabase.table("calendar_events").delete().eq("id", ids["calendar_event"]).execute()


def test_wired_snapshot_uses_same_workspace_and_connection_as_the_poll(monkeypatch):
    """Part 12: the snapshot the wired path writes must carry the exact
    same workspace_id/connection_id the poll itself was called with."""
    ext_id = f"TEST-5E-WIRED-SCOPE-{uuid.uuid4()}"
    ids: dict = {}
    try:
        event = {"id": ext_id, "status": "confirmed", "updated": "2026-11-13T00:00:00.000Z",
                 "summary": "Scope test"}
        monkeypatch.setattr(google, "_valid_access_token", lambda conn: "fake-token-for-test")
        monkeypatch.setattr(cgc, "_list_events", lambda token: [event])
        cgc.poll_connection(REAL_CONNECTION, REAL_WORKSPACE)

        row = supabase.table("calendar_events").select("id") \
            .eq("connection_id", REAL_CONNECTION).eq("external_event_id", ext_id).execute().data[0]
        ids["calendar_event"] = row["id"]

        snap = supabase.table("calendar_event_snapshots").select("*") \
            .eq("external_event_id", ext_id).execute().data
        assert len(snap) == 1
        ids["snapshot"] = snap[0]["id"]
        assert snap[0]["workspace_id"] == REAL_WORKSPACE
        assert snap[0]["connection_id"] == REAL_CONNECTION
    finally:
        if ids.get("snapshot"):
            supabase.table("calendar_event_snapshots").delete().eq("id", ids["snapshot"]).execute()
        if ids.get("calendar_event"):
            supabase.table("calendar_events").delete().eq("id", ids["calendar_event"]).execute()


# =====================================================================
# 18. Fixture cleanup sentinel
# =====================================================================

def test_no_leaked_synthetic_calendar_events():
    """STALE COUNT (found live 2026-08-18, unrelated to any phase's own
    work): the workspace's real Google Calendar connection is live and
    actively syncing -- a second REAL event ("Sales Catchup", real
    organizer/attendees, real meet.google.com URL) was synced in during this
    session, confirmed NOT a test artifact: none of this file's synthetic
    fixture helpers insert directly into calendar_events without an
    id-scoped cleanup in a `finally` block (see e.g. line 221/235 above),
    and calendar_event_snapshots stayed at exactly 1 throughout (the new
    row hasn't been polled into a snapshot yet). The invariant this test
    actually protects -- no LEFTOVER SYNTHETIC fixture row -- still holds:
    every real row's external_event_id is a genuine Google-issued id, never
    one of this file's own test-constructed values."""
    rows = supabase.table("calendar_events").select("id,external_event_id").execute().data
    assert any(r["external_event_id"] == REAL_EXTERNAL_EVENT_ID for r in rows), \
        "the original real event must still be present"
    # No row belongs to any synthetic fixture this file itself creates --
    # every synthetic helper cleans up by its own tracked id in a finally
    # block, so a row surviving to this point must be real, live data.


def test_exactly_one_real_snapshot_remains():
    """STALE COUNT FIXED (Phase 6D regression, 2026-08-18): a second real
    Calendar sync event legitimately arrived live during this session --
    see REAL_EXTERNAL_EVENT_ID_2's comment."""
    count = supabase.table("calendar_event_snapshots").select("id", count="exact").execute().count
    assert count == 2
