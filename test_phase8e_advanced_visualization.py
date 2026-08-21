"""
Phase 8E -- advanced analytical configuration.

WHAT THIS COVERS. Multi-series grouping, the five temporal buckets, top-N,
server-computed percentages, and the comparison window that makes a KPI delta
real. All of it runs through the SAME registry validation and the SAME
already-visibility-filtered row list as Phase 8B, so widening the analysis did
not widen the data.

Run with: python -m pytest test_phase8e_advanced_visualization.py -v
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

REAL_WORKSPACE = "f7aab311-c7b5-49c8-a8e4-36c89fa0b25d"
OWNER = gq.resolve_allowed_sensitivities("owner", False)
LOW = gq.resolve_allowed_sensitivities(None, False)


def _q(**kw):
    kw.setdefault("workspace_id", REAL_WORKSPACE)
    kw.setdefault("allowed_sensitivities", OWNER)
    return sd.run_query(**kw)


def _api(**kw):
    kw.setdefault("workspace_id", REAL_WORKSPACE)
    auth = AuthContext(user_id="u", workspaces={REAL_WORKSPACE: "owner"},
                       enforced=True, caller="pytest")
    return asyncio.run(api.query_dataset(api.DatasetQueryRequest(**kw), auth, None))


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# 1-4. Multi-series.
# =====================================================================

def test_multi_series_splits_a_group_into_real_series():
    r = _q(dataset="changes", group_by="occurred_at", group_bucket="month",
           series_by="change_type", aggregation="count",
           temporal_mode="window", window_days=90)
    agg = r.aggregation
    assert agg["series_by"] == "change_type"
    assert len(agg["series_values"]) > 1, "real corpus has several change types"
    for b in agg["buckets"]:
        assert b["series"] in agg["series_values"]
    # The series split must PARTITION the rows, never duplicate them.
    assert sum(b["row_count"] for b in agg["buckets"]) == r.row_count


def test_series_must_be_groupable_like_any_group():
    """A second dimension is held to exactly the same registry rule as the
    first -- a series cannot be looser than a group."""
    with pytest.raises(sd.DatasetError, match="not groupable"):
        _q(dataset="memories", group_by="memory_type", series_by="statement",
           aggregation="count")


def test_series_requires_a_group_and_must_differ_from_it():
    with pytest.raises(sd.DatasetError, match="requires group_by"):
        _q(dataset="memories", series_by="memory_type", aggregation="count")
    with pytest.raises(sd.DatasetError, match="must differ"):
        _q(dataset="memories", group_by="memory_type", series_by="memory_type",
           aggregation="count")


def test_single_series_result_carries_no_series_values():
    r = _q(dataset="memories", group_by="memory_type", aggregation="count")
    assert r.aggregation["series_values"] == []
    assert all(b["series"] is None for b in r.aggregation["buckets"])


# =====================================================================
# 5-8. Temporal bucketing.
# =====================================================================

@pytest.mark.parametrize("bucket,shape", [
    ("day", 10), ("month", 7), ("year", 4),
])
def test_prefix_buckets_have_the_right_shape(bucket, shape):
    r = _q(dataset="evidence", group_by="captured_at", group_bucket=bucket,
           aggregation="count")
    for b in r.aggregation["buckets"]:
        assert len(b["group"]) == shape


def test_week_and_quarter_are_derived_from_the_real_date():
    week = _q(dataset="evidence", group_by="captured_at", group_bucket="week",
              aggregation="count")
    quarter = _q(dataset="evidence", group_by="captured_at", group_bucket="quarter",
                 aggregation="count")
    assert all("-W" in b["group"] for b in week.aggregation["buckets"])
    assert all("-Q" in b["group"] for b in quarter.aggregation["buckets"])


def test_a_date_group_still_requires_an_explicit_bucket():
    with pytest.raises(sd.DatasetError, match="requires group_bucket"):
        _q(dataset="evidence", group_by="captured_at", aggregation="count")


def test_bucket_is_rejected_on_a_non_date_field():
    with pytest.raises(sd.DatasetError, match="only to date fields"):
        _q(dataset="memories", group_by="memory_type", group_bucket="month",
           aggregation="count")


def test_temporal_meaning_survives_bucketing():
    """Bucketing reads ONE field; it must never merge two temporal concepts."""
    r = _q(dataset="memories", group_by="created_at", group_bucket="month",
           aggregation="count")
    f = next(x for x in r.fields if x["key"] == "created_at")
    assert f["temporal_meaning"] == sd.TEMPORAL_AVAILABILITY
    assert r.aggregation["group_bucket"] == "month"


# =====================================================================
# 9-11. Top-N.
# =====================================================================

def test_top_n_keeps_only_the_highest_groups():
    full = _q(dataset="changes", group_by="change_type", aggregation="count",
              temporal_mode="window", window_days=90)
    top = _q(dataset="changes", group_by="change_type", aggregation="count",
             top_n=2, temporal_mode="window", window_days=90)
    assert len(top.aggregation["buckets"]) == 2
    assert len(top.aggregation["buckets"]) < len(full.aggregation["buckets"])
    kept = {b["value"] for b in top.aggregation["buckets"]}
    dropped = {b["value"] for b in full.aggregation["buckets"]} - kept
    assert min(kept) >= max(dropped)


def test_bottom_n_is_the_mirror_image():
    bottom = _q(dataset="changes", group_by="change_type", aggregation="count",
                top_n=2, top_direction="bottom",
                temporal_mode="window", window_days=90)
    assert bottom.aggregation["top_direction"] == "bottom"
    assert len(bottom.aggregation["buckets"]) == 2


def test_top_n_is_validated():
    for bad in (0, 101, -1):
        with pytest.raises(sd.DatasetError, match="top_n must be"):
            _q(dataset="changes", group_by="change_type", aggregation="count", top_n=bad)
    with pytest.raises(sd.DatasetError, match="requires group_by"):
        _q(dataset="changes", aggregation="count", top_n=3)
    with pytest.raises(sd.DatasetError, match="top_direction"):
        _q(dataset="changes", group_by="change_type", aggregation="count",
           top_n=2, top_direction="sideways")


# =====================================================================
# 12-14. Percentages.
# =====================================================================

def test_percentages_are_server_computed_and_sum_to_the_whole():
    r = _q(dataset="changes", group_by="change_type", aggregation="count",
           percent=True, temporal_mode="window", window_days=90)
    agg = r.aggregation
    assert agg["percent"] is True
    assert agg["percent_basis"] == sum(b["value"] for b in agg["buckets"])
    assert abs(sum(b["percent"] for b in agg["buckets"]) - 100.0) < 0.5


def test_percent_is_refused_where_it_would_be_meaningless():
    """A percentage of an earliest-date is not a quantity."""
    with pytest.raises(sd.DatasetError, match="not a quantity"):
        _q(dataset="memories", group_by="memory_type", aggregation="max",
           value_field="created_at", percent=True)


def test_percent_basis_is_the_visible_total_only():
    """The denominator is the rows the caller can see -- never a larger
    population, which would disclose the size of what is hidden."""
    r = _q(dataset="evidence", group_by="sensitivity", aggregation="count", percent=True)
    assert r.aggregation["percent_basis"] == r.row_count


# =====================================================================
# 15-17. KPI comparison window.
# =====================================================================

def test_comparison_window_is_a_real_second_query():
    cur = _q(dataset="changes", aggregation="count",
             temporal_mode="window", window_days=30)
    prev = _q(dataset="changes", aggregation="count",
              temporal_mode="window", window_days=30, window_offset_days=30)
    assert cur.temporal_context != prev.temporal_context
    assert cur.aggregation["buckets"][0]["value"] is not None
    assert prev.aggregation["buckets"][0]["value"] is not None


def test_an_empty_period_still_reports_zero_not_nothing():
    """A period with no events is a real answer. Without a bucket the caller
    could not tell 'nothing happened' from 'no result'."""
    prev = _q(dataset="changes", aggregation="count",
              temporal_mode="window", window_days=1, window_offset_days=3000)
    assert prev.row_count == 0
    assert prev.aggregation["buckets"] == [
        {"group": "__all__", "series": None, "value": 0, "row_count": 0}]


def test_window_offset_is_validated():
    with pytest.raises(sd.DatasetError, match="window_offset_days"):
        _q(dataset="changes", aggregation="count", temporal_mode="window",
           window_days=30, window_offset_days=-5)


# =====================================================================
# 18-22. Security: widening analysis must not widen data.
# =====================================================================

def test_aggregation_still_runs_only_over_visible_rows():
    """`security` is stamped by the API layer, so it is asserted there. At the
    registry level the equivalent invariant is structural: every bucket is
    built from the same visible row list, so bucket row_counts must account
    for exactly the rows that were returned -- no more, no fewer."""
    for kw in (
        dict(group_by="memory_type", aggregation="count"),
        dict(group_by="memory_type", aggregation="count", percent=True),
    ):
        r = _q(dataset="memories", **kw)
        assert sum(b["row_count"] for b in r.aggregation["buckets"]) == r.row_count

    # Top-N deliberately drops groups, so it accounts for a SUBSET.
    topped = _q(dataset="memories", group_by="memory_type", aggregation="count", top_n=1)
    assert 0 < sum(b["row_count"] for b in topped.aggregation["buckets"]) <= topped.row_count

    # And the API stamps the guarantee explicitly.
    assert _api(dataset="memories", group_by="memory_type", aggregation="count",
                percent=True)["security"]["filtered_before_aggregation"] is True


def test_no_advanced_option_can_leak_a_restricted_row(tmp_path=None):
    """The whole point: series/top-N/percent are computed from the same
    already-filtered list, so a restricted row cannot influence any of them."""
    ws, ids = str(uuid.uuid4()), []
    try:
        for sens in ("restricted", "restricted"):
            i = supabase.table("structured_knowledge").insert({
                "workspace_id": ws, "canonical_source_type": "knowledge_note",
                "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
                "primitive_type": "fact", "statement": "SECRET-8E claim",
                "raw_subject_phrase": "x", "qualifier_words": [], "sensitivity": sens,
                "authority": "official", "source_tier": 2, "lifecycle_status": "active",
                "extraction_version": "v2.1", "captured_at": _now_iso(),
                "extraction_run_id": str(uuid.uuid4()),
                "primitive_fingerprint": f"t8e-{uuid.uuid4()}"}).execute().data[0]["id"]
            ids.append(i)

        hi = sd.run_query("evidence", ws, OWNER, group_by="sensitivity",
                          aggregation="count", percent=True)
        lo = sd.run_query("evidence", ws, LOW, group_by="sensitivity",
                          aggregation="count", percent=True)

        assert hi.row_count == 2
        # A low-clearance caller must not learn that 2 restricted claims exist
        # -- not through a count, not through a percentage, not through a basis.
        assert lo.row_count == 0
        assert lo.aggregation["percent_basis"] in (0, None)
        assert lo.aggregation["buckets"] == []
        assert "SECRET-8E" not in str(lo.rows)
    finally:
        for i in ids:
            supabase.table("structured_knowledge").delete().eq("id", i).execute()


def test_top_n_cannot_reveal_a_hidden_ranking():
    """Top-N ranks only what the caller can already see."""
    r_low = sd.run_query("evidence", REAL_WORKSPACE, LOW, group_by="sensitivity",
                         aggregation="count", top_n=5)
    for b in r_low.aggregation["buckets"]:
        assert b["group"] in (None, "public", "internal"), \
            "a low caller must never see a confidential/restricted bucket"


def test_advanced_options_reach_the_api_and_are_validated_there():
    ok = _api(dataset="changes", group_by="change_type", aggregation="count",
              top_n=2, percent=True, temporal_mode="window", window_days=90)
    assert ok["aggregation"]["top_n"] == 2
    assert ok["aggregation"]["percent"] is True

    with pytest.raises(HTTPException) as e:
        _api(dataset="memories", group_by="memory_type", series_by="statement",
             aggregation="count")
    assert e.value.status_code == 400


def test_client_still_cannot_send_authorization():
    fields = set(api.DatasetQueryRequest.model_fields)
    for forbidden in ("role", "is_super_admin", "allowed_sensitivities", "sensitivity"):
        assert forbidden not in fields
    # The new analytical fields exist and are plain data.
    for added in ("series_by", "series_bucket", "top_n", "top_direction",
                   "percent", "window_offset_days"):
        assert added in fields


# =====================================================================
# 23-26. Contract, isolation, no fabrication.
# =====================================================================

def test_workspace_isolation_holds_for_advanced_queries():
    leak = sd.run_query("memories", "892e3fc6-04a3-4421-a729-f83ed8c92ea3", OWNER,
                        group_by="memory_type", aggregation="count", percent=True)
    blob = str(leak.rows).lower()
    for term in ("credential", "procurement", "tanmay"):
        assert term not in blob


def test_no_projects_dataset_and_no_arbitrary_expression():
    assert "projects" not in sd.DATASETS
    with pytest.raises(sd.DatasetError):
        _q(dataset="memories", group_by="memory_type", aggregation="count(*) OR 1=1")
    with pytest.raises(sd.DatasetError):
        _q(dataset="memories", group_by="memory_type; DROP TABLE", aggregation="count")


def test_aggregation_vocabulary_is_unchanged():
    """8E widened HOW you slice, never WHAT you may compute."""
    assert sd.ALLOWED_AGGREGATIONS == {"count", "count_distinct", "min", "max"}
    assert sd.PERCENTABLE_AGGREGATIONS == {"count", "count_distinct"}
    assert sd.GROUP_BUCKETS == {"day", "week", "month", "quarter", "year"}


def test_fixture_cleanup_leaves_no_residue():
    ws = str(uuid.uuid4())
    i = supabase.table("structured_knowledge").insert({
        "workspace_id": ws, "canonical_source_type": "knowledge_note",
        "canonical_id": str(uuid.uuid4()), "provider": "google_chat",
        "primitive_type": "fact", "statement": "temp-8e", "raw_subject_phrase": "x",
        "qualifier_words": [], "sensitivity": "internal", "authority": "official",
        "source_tier": 2, "lifecycle_status": "active", "extraction_version": "v2.1",
        "captured_at": _now_iso(), "extraction_run_id": str(uuid.uuid4()),
        "primitive_fingerprint": f"t8e-{uuid.uuid4()}"}).execute().data[0]["id"]
    supabase.table("structured_knowledge").delete().eq("id", i).execute()
    assert (supabase.table("structured_knowledge").select("id")
            .eq("workspace_id", ws).execute().data or []) == []
