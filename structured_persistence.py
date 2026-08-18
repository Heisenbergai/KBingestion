"""
Phase 4 -- connects the FROZEN extraction contract (structured_extraction.py,
never imported for its prompt/logic here beyond the ExtractedPrimitive shape
-- this module does not touch EXTRACT_SYSTEM or the extraction function at
all) to the live structured_knowledge persistence layer.

Single-canonical-item, controlled writes only. No bulk processing, no batch
worker, no Phase 5 concepts -- matches this pass's explicit safety scope.

Pipeline this module implements the LAST step of:
  CanonicalKnowledge -> extract_primitives_from_canonical() [unchanged]
    -> V2.1 validation / deterministic qualifier guard [unchanged]
    -> compute_primitive_fingerprint() [new, this module]
    -> persist_extracted_primitives() [new, this module] -> structured_knowledge
"""
import hashlib
import json
import re
from typing import Optional

import brain_connectors as bc
from structured_extraction import ExtractedPrimitive

# Same ladder brain_connectors._SENSITIVITY_RANK already uses -- not
# reinvented, just a local copy to avoid a cross-module coupling for one
# small dict (matches this codebase's own stated convention, see
# query.py's docstring on "small per-file helpers over shared coupling").
_SENSITIVITY_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

# "knowledge_item" is deliberately excluded -- documents remain unavailable
# through the current canonical read path (no service-role read path for
# the app-DB knowledge_items table -- see canonical.py's own module
# docstring and get_canonical_knowledge()'s "document" handling). Claiming
# support here would be pretending a capability that doesn't exist.
VALID_CANONICAL_SOURCE_TYPES = {"knowledge_note", "calendar_event"}


def _normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def compute_primitive_fingerprint(primitive: ExtractedPrimitive) -> str:
    """Deterministic, code-computed identity -- exactly the algorithm from
    the approved Phase 4 persistence schema design. Never LLM-generated;
    a pure function of already-validated ExtractedPrimitive fields."""
    normalized = {
        "primitive_type": primitive.type,
        "statement": _normalize_text(primitive.statement),
        "requirement_kind": primitive.requirement_kind or "",
        "effective_from": primitive.effective_from or "",
        "effective_until": primitive.effective_until or "",
        "event_time": primitive.event_time or "",
        "event_start": primitive.event_start or "",
        "event_end": primitive.event_end or "",
        "recurrence_text": _normalize_text(primitive.recurrence_text or ""),
        "qualifier_words": sorted({q.strip().lower() for q in primitive.qualifier_words}),
    }
    canonical_string = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()


def _fetch_canonical_parent(canonical_source_type: str, canonical_id: str) -> Optional[dict]:
    """Real, live lookup of the canonical parent's current usability and
    real sensitivity -- never cached, never trusted from the caller. See
    module docstring on why only these two source types are supported."""
    if canonical_source_type == "knowledge_note":
        rows = bc.supabase.table("knowledge_notes").select("id, workspace_id, status, sensitivity") \
            .eq("id", canonical_id).execute().data
        if not rows:
            return None
        row = rows[0]
        return {"workspace_id": row["workspace_id"], "usable": row.get("status") == "active",
                "sensitivity": row.get("sensitivity")}
    if canonical_source_type == "calendar_event":
        rows = bc.supabase.table("calendar_events").select("id, workspace_id, deleted_at") \
            .eq("id", canonical_id).execute().data
        if not rows:
            return None
        row = rows[0]
        # Calendar carries no sensitivity concept at all (Google Workspace
        # scope lock -- deliberately never classified, see canonical.py) --
        # None here means "nothing to cap against", not "most permissive".
        return {"workspace_id": row["workspace_id"], "usable": row.get("deleted_at") is None,
                "sensitivity": None}
    return None


def persist_extracted_primitives(
    workspace_id: str,
    canonical_source_type: str,
    canonical_id: str,
    provider: str,
    extraction_version: str,
    extraction_run_id: str,
    primitives: list[ExtractedPrimitive],
) -> dict:
    """
    The single controlled persistence entry point for ONE canonical item's
    already-extracted, already-validated primitives. Never bulk -- one call
    per canonical item, by design, matching this pass's explicit safety
    scope ("do NOT bulk-process the real corpus immediately").

    Fails SAFE at the canonical-item level (persists nothing at all) for:
      - an unsupported canonical_source_type
      - a canonical parent that doesn't exist
      - workspace_id that doesn't match the parent's REAL workspace
        (re-verified here server-side, never trusted from the caller)
      - a canonical parent that is not currently usable (deleted/inactive)
      - an extraction_version that isn't a registered contract version

    Rejects INDIVIDUAL primitives (skips, does not abort the rest of the
    batch) when a primitive's sensitivity exceeds the canonical parent's --
    one bad item must not cost the others, matching classify_batch's own
    established per-item failure contract elsewhere in this codebase. The
    same contract also covers unanticipated database-level rejections
    (e.g. a constraint this function's own validation didn't already
    check for) -- caught per-primitive, recorded in `rejected`, never
    allowed to abort the rest of the batch.

    Idempotent via the same ON CONFLICT DO NOTHING pattern save_ingest_items
    already uses elsewhere -- re-running this with identical primitives
    against the same canonical_id/extraction_version inserts zero
    duplicates, proven by primitive_fingerprint alone (no LLM-generated
    ids anywhere).

    Never writes `confidence` or `record_status` -- neither is a field on
    ExtractedPrimitive nor a column on structured_knowledge; there is
    nothing here that could persist them even by accident.

    Returns {"inserted": int, "skipped_duplicates": int,
             "rejected": [{"statement", "reason"}, ...],
             "canonical_rejected_reason": Optional[str]}.
    """
    result = {"inserted": 0, "skipped_duplicates": 0, "rejected": [], "canonical_rejected_reason": None}

    if canonical_source_type not in VALID_CANONICAL_SOURCE_TYPES:
        result["canonical_rejected_reason"] = (
            f"canonical_source_type '{canonical_source_type}' is not currently supported for "
            f"extraction persistence -- only {sorted(VALID_CANONICAL_SOURCE_TYPES)} are readable "
            f"through the current canonical layer (documents remain unavailable)."
        )
        return result

    parent = _fetch_canonical_parent(canonical_source_type, canonical_id)
    if parent is None:
        result["canonical_rejected_reason"] = "canonical parent not found."
        return result

    if parent["workspace_id"] != workspace_id:
        result["canonical_rejected_reason"] = "workspace_id does not match the canonical parent's real workspace."
        return result

    if not parent["usable"]:
        result["canonical_rejected_reason"] = "canonical parent is not currently usable (deleted/inactive)."
        return result

    version_rows = bc.supabase.table("extraction_contract_versions").select("version") \
        .eq("version", extraction_version).execute().data
    if not version_rows:
        result["canonical_rejected_reason"] = (
            f"extraction_version '{extraction_version}' is not a registered contract version."
        )
        return result

    parent_sensitivity_rank = _SENSITIVITY_RANK.get(parent["sensitivity"]) if parent["sensitivity"] else None

    for primitive in primitives:
        if parent_sensitivity_rank is not None:
            primitive_rank = _SENSITIVITY_RANK.get(primitive.sensitivity)
            if primitive_rank is None or primitive_rank > parent_sensitivity_rank:
                result["rejected"].append({
                    "statement": primitive.statement,
                    "reason": f"sensitivity '{primitive.sensitivity}' exceeds canonical parent's '{parent['sensitivity']}'",
                })
                continue

        row = {
            "workspace_id": workspace_id,
            "canonical_source_type": canonical_source_type,
            "canonical_id": canonical_id,
            "provider": provider,
            "primitive_type": primitive.type,
            "requirement_kind": primitive.requirement_kind,
            "statement": primitive.statement,
            "raw_subject_phrase": primitive.raw_subject_phrase,
            "qualifier_words": primitive.qualifier_words,
            "sensitivity": primitive.sensitivity,
            "authority": primitive.authority,
            "source_tier": primitive.source_tier,
            "lifecycle_status": primitive.lifecycle_status,
            "captured_at": primitive.captured_at,
            "event_time": primitive.event_time,
            "event_start": primitive.event_start,
            "event_end": primitive.event_end,
            "effective_from": primitive.effective_from,
            "effective_until": primitive.effective_until,
            "recurrence_text": primitive.recurrence_text,
            "extraction_version": extraction_version,
            "extraction_run_id": extraction_run_id,
            "primitive_fingerprint": compute_primitive_fingerprint(primitive),
        }

        # A DB-level rejection for ONE primitive (a constraint this
        # function didn't already anticipate) must not abort the rest of
        # the batch -- found live: an unanticipated NOT NULL violation on
        # a Calendar-derived primitive silently prevented every remaining
        # primitive in the same run from ever being attempted. Same
        # non-fatal, one-bad-item-doesn't-cost-the-others contract
        # classify_batch already uses elsewhere in this codebase.
        try:
            inserted_rows = bc.supabase.table("structured_knowledge").upsert(
                row,
                on_conflict="canonical_source_type,canonical_id,extraction_version,primitive_fingerprint",
                ignore_duplicates=True,
            ).execute().data
        except Exception as e:
            result["rejected"].append({"statement": primitive.statement, "reason": f"database rejected insert: {e}"})
            continue

        if inserted_rows:
            result["inserted"] += 1
        else:
            result["skipped_duplicates"] += 1

    return result
