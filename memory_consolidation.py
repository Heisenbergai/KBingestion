"""
Phase 6C -- Sleep / Memory Consolidation Engine.

READ-ORIENTED consolidation layer over structured_knowledge, knowledge_
relationships, evidence, org_memory, and memory_review_queue. This module
NEVER mutates structured_knowledge or knowledge_relationships -- it only
ever writes to org_memory, memory_evidence, memory_review_queue, and
memory_consolidation_runs (all through the existing atomic RPCs where a
multi-step validated write is involved, plain updates for single-field
lifecycle bumps, same convention worker.py already uses for sync_runs).

THE deterministic callable service is run_consolidation(workspace_id) --
one workspace per call, matching persist_extracted_primitives()'s and
run_filtration()'s established per-unit call shape elsewhere in this
codebase (see structured_persistence.py, brain_connectors.py). It never
enumerates workspaces itself. See sleep_cycle.py for the scheduler entry
point that loops over workspaces and calls this once each.

ARCHITECTURAL ROLE (not a generic cron job):
    continuous ingestion -> working knowledge -> sleep -> consolidated
    organizational memory.
run_consolidation() answers, for one workspace, one call:
    "What changed since I last slept?"       -- the incremental boundary
    "What deserves durable memory?"          -- the 4 frozen promotion bases
    "What needs human attention?"            -- memory_review_queue
    "What old memory is no longer current?"  -- revalidation
    "Did anything contradict an existing durable belief?" -- contradiction
                                                              pre-filter + LLM
"""
import re
from datetime import datetime, timezone
from typing import Optional

import ai
import brain_connectors as bc

supabase = bc.supabase

# The 4 frozen promotion bases, in deterministic priority order (first match
# wins when more than one could theoretically apply -- in practice a given
# candidate's shape only ever satisfies one of the requirement_kind-gated
# bases, priority only matters between a requirement_kind-gated basis and
# cross_source_corroboration/explicit_user_keep).
PROMOTION_BASES = (
    "authoritative_policy",
    "recurring_durable_process",
    "cross_source_corroboration",
    "explicit_user_keep",
)

_MEMORY_TYPE_BY_BASIS = {
    "authoritative_policy": "policy",
    "recurring_durable_process": "process",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _db_now_iso() -> str:
    """The database server's OWN clock, via a trivial `SELECT now()` RPC --
    never the calling process's local clock. Found live: local vs. server
    clock skew (observed ~100-200ms) can make the boundary's upper bound
    (`until`, compared against structured_knowledge.created_at, which the
    DB itself stamps) appear to precede a row that was genuinely inserted
    before the boundary was captured -- silently and PERMANENTLY excluding
    it from candidate discovery, since created_at never changes and no
    future run would re-check it either. `since` needs no separate fix: it
    is always read back from a previously stored `until` value, which is
    now itself DB-clock-sourced."""
    return supabase.rpc("consolidation_clock", {}).execute().data


def _normalize_statement(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")


# =====================================================================
# Part 2 -- run boundary
# =====================================================================

def _last_completed_boundary(workspace_id: str) -> Optional[str]:
    """The upper bound of the most recent successfully COMPLETED engine run
    for this workspace -- a 'failed' or 'running' run is never a candidate,
    so a failure never advances the cursor past what it actually finished
    (Part 13). input_boundary_until IS NULL is excluded deliberately: NULL
    there means "this run predates the cursor mechanism" (the one legacy
    Phase 6B manual run), not a real checkpoint -- a run that never used
    this cursor logic must not silently become the window's lower bound,
    or the engine's true first run would skip re-evaluating the real corpus
    entirely instead of proving itself against it (found live during this
    pass's own real safety run, see the phase6c_reopen_legacy_run_boundary_
    to_null migration)."""
    rows = supabase.table("memory_consolidation_runs").select("input_boundary_until") \
        .eq("workspace_id", workspace_id).eq("status", "completed") \
        .not_.is_("input_boundary_until", "null") \
        .order("input_boundary_until", desc=True).limit(1).execute().data
    return rows[0]["input_boundary_until"] if rows else None


def _start_run(workspace_id: str, since: Optional[str], until: str, started_at_dt: datetime) -> str:
    row = supabase.table("memory_consolidation_runs").insert({
        "workspace_id": workspace_id,
        "started_at": _iso(started_at_dt),
        "status": "running",
        "input_boundary_since": since,
        "input_boundary_until": until,
    }).execute().data
    return row[0]["id"]


def _finish_run(run_id: str, status: str, stats: dict) -> None:
    supabase.table("memory_consolidation_runs").update({
        "status": status,
        "completed_at": _iso(_now()),
        "stats": stats,
    }).eq("id", run_id).execute()


# =====================================================================
# Part 1/3 -- incremental, workspace-scoped candidate discovery
# =====================================================================

def _fetch_candidates(workspace_id: str, since: Optional[str], until: str) -> list[dict]:
    """structured_knowledge rows created in (since, until] for THIS
    workspace only -- never a global scan (Part 3). lifecycle_status='active'
    only: 'draft'/NULL rows are not yet a finalized statement and a
    consolidation pass is not a second ingestion system that decides that
    for them."""
    query = supabase.table("structured_knowledge").select("*") \
        .eq("workspace_id", workspace_id) \
        .eq("lifecycle_status", "active") \
        .lte("created_at", until)
    if since:
        query = query.gt("created_at", since)
    return query.execute().data or []


# =====================================================================
# Part 4/5 -- classification
# =====================================================================

def _grounding_fingerprint_single(sk_id: str) -> str:
    """Same algorithm create_memory_with_evidence computes server-side
    (sorted, deduped 'evidence_type:evidence_id' pairs) -- for the V1
    single-evidence-item shape every real promotion uses, that's just one
    pair. Computed here too so classification can check ALREADY_DURABLE
    without a wasted RPC round-trip."""
    return f"structured_knowledge:{sk_id}"


def _already_durable(workspace_id: str, fingerprint: str) -> Optional[dict]:
    rows = supabase.table("org_memory").select("id, promotion_basis, lifecycle_status") \
        .eq("workspace_id", workspace_id).eq("grounding_fingerprint", fingerprint) \
        .in_("lifecycle_status", ["active", "dormant"]).execute().data
    return rows[0] if rows else None


def _check_explicit_user_keep(candidate: dict) -> bool:
    """
    Part 7 -- extension point ONLY, not a real signal yet. No field
    upstream of structured_knowledge currently records "a human explicitly
    said keep this" -- there is no UI for it, and this pass does not invent
    one. The intended future contract, when a real UI/service writes it:
    a nullable `structured_knowledge.user_marked_keep_at timestamptz`
    (or a small separate `structured_knowledge_user_signals` table, if
    touching the frozen structured_knowledge schema is undesirable at that
    point), set by whatever service the eventual "keep this" UI action
    calls. This function is the ONE seam classify_candidate() calls for
    that signal -- wiring in the real column/table later is a one-line
    change here, not a redesign. Always False today: there is nothing to
    read yet, and adding a column nothing writes to would be pretending a
    capability that doesn't exist.
    """
    return False


def _cross_source_corroboration(workspace_id: str, candidate: dict) -> bool:
    """Strict same-claim test: the candidate's normalized statement text
    matches another active structured_knowledge row's normalized statement
    from a DISTINCT provider. Exact normalized-text match only -- no fuzzy/
    semantic matching (deterministic-only design discipline)."""
    norm = _normalize_statement(candidate.get("statement"))
    if not norm:
        return False
    rows = supabase.table("structured_knowledge").select("provider, statement") \
        .eq("workspace_id", workspace_id).eq("lifecycle_status", "active").execute().data or []
    providers = {r["provider"] for r in rows if _normalize_statement(r.get("statement")) == norm}
    return len(providers) >= 2


def _is_graph_connected(workspace_id: str, sk_id: str) -> bool:
    src = supabase.table("knowledge_relationships").select("id") \
        .eq("workspace_id", workspace_id).eq("status", "active") \
        .eq("source_object_type", "structured_knowledge").eq("source_object_id", sk_id) \
        .limit(1).execute().data
    if src:
        return True
    tgt = supabase.table("knowledge_relationships").select("id") \
        .eq("workspace_id", workspace_id).eq("status", "active") \
        .eq("target_object_type", "structured_knowledge").eq("target_object_id", sk_id) \
        .limit(1).execute().data
    return bool(tgt)


def classify_candidate(workspace_id: str, candidate: dict) -> tuple[str, object]:
    """
    Deterministic classification into exactly one of (Part 4):
      "already_durable" -- detail = the existing org_memory row (dict)
      "promote"          -- detail = the promotion_basis (str)
      "review"           -- detail = the reason (str)
      "reject"           -- detail = the reason (str)
    Only the 4 frozen promotion bases are ever consulted for PROMOTE. No
    graph centrality, retrieval frequency, or sensitivity-as-durability-
    signal. Graph-connectedness is used ONLY as a REVIEW-escalation signal
    (not a promotion path) -- the same deterministic criterion the real
    Phase 6B Q4-launch review item already used.
    """
    fingerprint = _grounding_fingerprint_single(candidate["id"])
    existing = _already_durable(workspace_id, fingerprint)
    if existing:
        return "already_durable", existing

    if candidate.get("sensitivity") is None:
        return "reject", (
            "structured_knowledge has NULL sensitivity -- cannot ground durable memory "
            "(no classification concept exists for this source); deliberate rejection, "
            "not an inferred default"
        )

    basis = None
    if candidate.get("authority") == "official" and candidate.get("requirement_kind") == "policy":
        basis = "authoritative_policy"
    elif candidate.get("requirement_kind") == "process_step" and candidate.get("recurrence_text"):
        basis = "recurring_durable_process"
    elif _cross_source_corroboration(workspace_id, candidate):
        basis = "cross_source_corroboration"
    elif _check_explicit_user_keep(candidate):
        basis = "explicit_user_keep"

    if basis:
        return "promote", basis

    if candidate.get("authority") == "official" and _is_graph_connected(workspace_id, candidate["id"]):
        return "review", (
            f"Official authority, graph-connected, but fails all four frozen automatic "
            f"promotion paths (requirement_kind={candidate.get('requirement_kind')!r}, "
            f"recurrence_text={candidate.get('recurrence_text')!r}, no cross-source "
            f"corroboration, no explicit user keep marker). Graph centrality alone is not "
            f"a promotion path. Recommended for human review, not automatic promotion."
        )

    return "reject", "fails all four frozen promotion paths and is not graph-connected+official"


def _memory_type_for(candidate: dict, basis: str) -> str:
    if basis in _MEMORY_TYPE_BY_BASIS:
        return _MEMORY_TYPE_BY_BASIS[basis]
    if candidate.get("requirement_kind") == "policy":
        return "policy"
    if candidate.get("requirement_kind") == "process_step":
        return "process"
    return "decision"


def _resolve_valid_from(candidate: dict) -> Optional[str]:
    """Frozen Part 10/6B.1 rule: effective_from (a resolved real-world
    validity date) if present; else a genuine event-start (only when the
    source IS an event whose own start time is the claim's real start);
    else NULL. Never captured_at/event_time as a substitute -- both are
    observation time, not validity time."""
    effective_from = candidate.get("effective_from")
    if effective_from:
        return f"{effective_from}T00:00:00+00:00"
    if candidate.get("primitive_type") == "event" and candidate.get("event_start"):
        return candidate["event_start"]
    return None


# =====================================================================
# Part 8 -- contradiction pre-filter + narrow LLM classification
# =====================================================================

def _temporal_overlap(a: dict, b: dict) -> bool:
    a_start, a_end = a.get("effective_from"), a.get("effective_until")
    b_start, b_end = b.get("effective_from"), b.get("effective_until")
    if a_end and b_start and a_end < b_start:
        return False
    if b_end and a_start and b_end < a_start:
        return False
    return True


def _find_contradiction_candidate(workspace_id: str, candidate: dict, memory_type: str) -> Optional[dict]:
    """Cheap, bounded pre-filter (Part 8/11) -- only the currently ACTIVE
    memories of the SAME memory_type are ever consulted, never the full
    historical corpus. Returns {"memory_id", "sk"} for the first related-
    but-different-statement match, or None."""
    existing_memories = supabase.table("org_memory").select("id") \
        .eq("workspace_id", workspace_id).eq("memory_type", memory_type) \
        .eq("lifecycle_status", "active").execute().data or []
    if not existing_memories:
        return None
    mem_ids = [m["id"] for m in existing_memories]
    evidence_rows = supabase.table("memory_evidence").select("memory_id, evidence_id") \
        .in_("memory_id", mem_ids).execute().data or []
    sk_ids = list({e["evidence_id"] for e in evidence_rows})
    if not sk_ids:
        return None
    sk_rows = {r["id"]: r for r in supabase.table("structured_knowledge").select(
        "id, requirement_kind, raw_subject_phrase, qualifier_words, statement, "
        "effective_from, effective_until"
    ).in_("id", sk_ids).execute().data or []}

    cand_subject = (candidate.get("raw_subject_phrase") or "").strip().lower()
    cand_quals = {q.strip().lower() for q in (candidate.get("qualifier_words") or [])}
    cand_norm_statement = _normalize_statement(candidate.get("statement"))
    cand_req_kind = candidate.get("requirement_kind")

    for e in evidence_rows:
        sk = sk_rows.get(e["evidence_id"])
        if not sk or sk.get("requirement_kind") != cand_req_kind:
            continue
        subject = (sk.get("raw_subject_phrase") or "").strip().lower()
        quals = {q.strip().lower() for q in (sk.get("qualifier_words") or [])}
        related = (subject and subject == cand_subject) or bool(quals & cand_quals)
        if not related:
            continue
        if _normalize_statement(sk.get("statement")) == cand_norm_statement:
            continue  # identical claim -- ALREADY_DURABLE/dedup territory, not a contradiction
        if not _temporal_overlap(sk, candidate):
            continue
        return {"memory_id": e["memory_id"], "sk": sk}
    return None


_CONTRADICTION_SYSTEM_PROMPT = (
    "You are a narrow classification function inside KNOVA's memory consolidation "
    "engine. You are given exactly two organizational statements that a deterministic "
    "pre-filter has already identified as covering the same subject area. Your ONLY "
    "job is to classify their relationship. You do NOT create memories, you do NOT "
    "choose which statement is better, you do NOT rewrite either statement. Respond "
    "with ONLY a JSON object: "
    '{"verdict": "unresolved" | "resolved_supersession", "rationale": "<one sentence>"}. '
    'Use "resolved_supersession" ONLY if statement B is unambiguously a later, '
    "authoritative update that explicitly replaces statement A's claim. Use "
    '"unresolved" for anything else, including any ambiguity, partial overlap, or a '
    "case where both statements could plausibly coexist. When genuinely unsure, "
    'always choose "unresolved" -- a false "unresolved" costs one human review; a '
    'false "resolved_supersession" silently discards organizational knowledge.'
)


def _llm_classify_contradiction(existing_statement: str, new_statement: str, workspace_id: str) -> tuple[str, str]:
    """Fails safe: any parse failure, unexpected verdict, or unavailable
    model (e.g. no live credentials) defaults to 'unresolved' -- never
    silently supersedes on an LLM error."""
    try:
        result = ai.chat_json(
            messages=[{"role": "user", "content":
                f"Statement A (existing): {existing_statement}\nStatement B (new): {new_statement}"}],
            system=_CONTRADICTION_SYSTEM_PROMPT,
            max_tokens=300, temperature=0.0,
            workspace_id=workspace_id, feature="memory_contradiction_classification",
        )
        verdict = result.get("verdict") if isinstance(result, dict) else None
        if verdict not in ("unresolved", "resolved_supersession"):
            return "unresolved", "LLM returned an unrecognized verdict -- defaulting to the safe outcome"
        return verdict, (result.get("rationale") or "")
    except Exception as e:
        return "unresolved", f"LLM classification unavailable ({e}) -- defaulting to the safe outcome"


# =====================================================================
# Part 6 -- review queue integration
# =====================================================================

def _upsert_review(workspace_id: str, sk_id: str, reason: str, run_id: str) -> dict:
    res = supabase.rpc("upsert_review_candidate", {
        "p_workspace_id": workspace_id,
        "p_structured_knowledge_id": sk_id,
        "p_reason": reason,
        "p_consolidation_run_id": run_id,
    }).execute().data
    return res[0] if res else {"was_new": False, "was_updated": False}


def _resolve_pending_review_if_any(workspace_id: str, sk_id: str, resolution_text: str, run_id: str) -> None:
    rows = supabase.table("memory_review_queue").select("id") \
        .eq("workspace_id", workspace_id).eq("structured_knowledge_id", sk_id) \
        .eq("status", "pending").execute().data or []
    for r in rows:
        supabase.table("memory_review_queue").update({
            "status": "promoted", "resolved_at": _iso(_now()),
            "resolution": resolution_text, "consolidation_run_id": run_id,
        }).eq("id", r["id"]).execute()


# =====================================================================
# Part 9/10 -- promotion + supersession
# =====================================================================

def _promote(workspace_id: str, run_id: str, candidate: dict, basis: str, supersedes_id: Optional[str]) -> str:
    memory_type = _memory_type_for(candidate, basis)
    valid_from = _resolve_valid_from(candidate)
    evidence = [{
        "evidence_type": "structured_knowledge", "evidence_id": candidate["id"],
        "stance": "supports", "captured_at": candidate["captured_at"],
    }]
    memory_id = supabase.rpc("create_memory_with_evidence", {
        "p_workspace_id": workspace_id,
        "p_memory_type": memory_type,
        "p_promotion_basis": basis,
        "p_valid_from": valid_from,
        "p_valid_until": None,
        "p_supersedes_memory_id": supersedes_id,
        "p_consolidation_run_id": run_id,
        "p_evidence": evidence,
    }).execute().data
    return memory_id


# =====================================================================
# Per-candidate processing
# =====================================================================

def _process_candidate(workspace_id: str, run_id: str, candidate: dict, stats: dict) -> None:
    bucket, detail = classify_candidate(workspace_id, candidate)

    if bucket == "already_durable":
        stats["already_durable"] += 1
        return

    if bucket == "reject":
        stats["rejected"] += 1
        return

    if bucket == "review":
        result = _upsert_review(workspace_id, candidate["id"], detail, run_id)
        if result.get("was_new"):
            stats["review_candidates"] += 1
        return

    # bucket == "promote"
    basis = detail
    memory_type = _memory_type_for(candidate, basis)
    contradiction = _find_contradiction_candidate(workspace_id, candidate, memory_type)
    supersedes_id = None

    if contradiction:
        verdict, rationale = _llm_classify_contradiction(
            contradiction["sk"]["statement"], candidate["statement"], workspace_id)
        if verdict == "unresolved":
            reason = (
                f"Deterministic pre-filter found a same-subject, same-{memory_type}-type conflict "
                f"with existing memory {contradiction['memory_id']} "
                f"(structured_knowledge {contradiction['sk']['id']}); LLM classification: "
                f"{verdict} -- {rationale}. Not auto-promoted; both existing memory and this "
                f"candidate are preserved as-is; routed to review."
            )
            result = _upsert_review(workspace_id, candidate["id"], reason, run_id)
            if result.get("was_new"):
                stats["review_candidates"] += 1
            stats["contradiction_flagged"] += 1
            return
        supersedes_id = contradiction["memory_id"]

    _promote(workspace_id, run_id, candidate, basis, supersedes_id)
    stats["promoted"] += 1
    if supersedes_id:
        stats["superseded"] += 1
    _resolve_pending_review_if_any(
        workspace_id, candidate["id"],
        f"Promoted via {basis} in consolidation run {run_id}.", run_id,
    )


# =====================================================================
# Part 11 -- revalidation (cheap, bounded -- never a full historical
# contradiction sweep; only an evidence-existence check every run)
# =====================================================================

def _revalidate(workspace_id: str, stats: dict) -> None:
    memories = supabase.table("org_memory").select("id, lifecycle_status") \
        .eq("workspace_id", workspace_id).in_("lifecycle_status", ["active", "dormant"]).execute().data or []
    if not memories:
        stats["revalidated"] = 0
        stats["dormant_transitions"] = 0
        return

    mem_ids = [m["id"] for m in memories]
    evidence_rows = supabase.table("memory_evidence").select("memory_id, evidence_id") \
        .in_("memory_id", mem_ids).execute().data or []
    sk_ids = list({e["evidence_id"] for e in evidence_rows})
    sk_status = {}
    if sk_ids:
        sk_status = {r["id"]: r["lifecycle_status"] for r in
                     supabase.table("structured_knowledge").select("id, lifecycle_status")
                     .in_("id", sk_ids).execute().data or []}

    evidence_by_memory: dict[str, list[str]] = {}
    for e in evidence_rows:
        evidence_by_memory.setdefault(e["memory_id"], []).append(e["evidence_id"])

    revalidated = dormant_transitions = 0
    now_iso = _iso(_now())
    for m in memories:
        ev_ids = evidence_by_memory.get(m["id"], [])
        healthy = bool(ev_ids) and all(sk_status.get(eid) == "active" for eid in ev_ids)
        if healthy:
            supabase.table("org_memory").update({"last_confirmed_at": now_iso}).eq("id", m["id"]).execute()
            revalidated += 1
        elif m["lifecycle_status"] == "active":
            # Evidence vanished/deactivated -- never observed in this real
            # corpus (nothing deletes structured_knowledge), but the check
            # is cheap and honest. Never deleted, never silently archived --
            # only demoted one step (active -> dormant) for a human to see.
            supabase.table("org_memory").update({"lifecycle_status": "dormant"}).eq("id", m["id"]).execute()
            dormant_transitions += 1

    stats["revalidated"] = revalidated
    stats["dormant_transitions"] = dormant_transitions


# =====================================================================
# Main entry point (Part 15 -- the deterministic callable service)
# =====================================================================

def run_consolidation(workspace_id: str) -> dict:
    """
    One workspace, one run. Safe to retry (Part 12/13): a failed run never
    advances the boundary (only status='completed' rows are ever chosen as
    a future run's `since`), so the next call -- whether a manual retry or
    the next scheduled tick -- naturally re-derives a candidate window that
    still includes everything the failed run didn't finish, and idempotent
    RPCs guarantee whatever DID already succeed is never duplicated.

    Returns {"run_id", "workspace_id", "status", "stats"}.
    """
    run_start_dt = _now()
    since = _last_completed_boundary(workspace_id)
    until = _db_now_iso()
    run_id = _start_run(workspace_id, since, until, run_start_dt)

    stats = {
        "evaluated": 0, "promoted": 0, "review_candidates": 0, "rejected": 0,
        "already_durable": 0, "superseded": 0, "contradiction_flagged": 0,
        "failed": 0, "skipped": 0, "revalidated": 0, "dormant_transitions": 0,
    }
    errors: list[dict] = []
    run_level_error = None

    try:
        candidates = _fetch_candidates(workspace_id, since, until)
        stats["evaluated"] = len(candidates)
        for candidate in candidates:
            try:
                _process_candidate(workspace_id, run_id, candidate, stats)
            except Exception as e:
                stats["failed"] += 1
                errors.append({"structured_knowledge_id": candidate.get("id"), "error": str(e)})

        _revalidate(workspace_id, stats)
    except Exception as e:
        run_level_error = str(e)

    stats["duration_seconds"] = round((_now() - run_start_dt).total_seconds(), 3)
    if errors:
        stats["errors"] = errors
    if run_level_error:
        stats["run_level_error"] = run_level_error

    status = "failed" if (errors or run_level_error) else "completed"
    _finish_run(run_id, status, stats)
    return {"run_id": run_id, "workspace_id": workspace_id, "status": status, "stats": stats}


def list_workspaces_with_memory_activity() -> list[str]:
    """
    Workspace discovery for the SCHEDULER LOOP only -- run_consolidation()
    itself always takes an explicit workspace_id and never enumerates on its
    own (Part 15: "a deterministic callable service", one workspace per
    call). Deliberately reads only tables this module already reads
    (structured_knowledge, org_memory) rather than adding a new app-DB
    dependency -- see drive_app_db.py's module docstring on why
    APP_SUPABASE_SERVICE_KEY is scoped to exactly that one module and
    nothing else should read the app DB's `workspaces` table directly. A
    workspace with zero structured_knowledge and zero existing memories has
    nothing for this engine to do regardless, so this is an honest, minimal
    signal, not a coverage gap in practice.
    """
    sk_rows = supabase.table("structured_knowledge").select("workspace_id").execute().data or []
    mem_rows = supabase.table("org_memory").select("workspace_id").execute().data or []
    return sorted({r["workspace_id"] for r in sk_rows} | {r["workspace_id"] for r in mem_rows})
