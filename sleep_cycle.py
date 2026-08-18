"""
Phase 6C -- sleep cycle scheduler entry point.

Railway CRON JOB, same pattern as worker.py (see that file's own setup
notes at the bottom, and its module docstring on why this shape was chosen:
one process, does its work, exits, no persistent state beyond what it writes
to a DB-backed run table). This is a SECOND, separate Cron Job -- it does
not change worker.py's own schedule. Filtration/polling and memory
consolidation are different concerns on different natural cadences.

Loops over every workspace with real memory-relevant activity and runs one
consolidation pass each; one workspace's failure never blocks another's
(same per-unit isolation worker.py already uses for connections).
"""
import sys
import traceback

import memory_consolidation as mc


def main() -> int:
    workspaces = mc.list_workspaces_with_memory_activity()
    overall_ok = True
    summary = {}

    for workspace_id in workspaces:
        try:
            result = mc.run_consolidation(workspace_id)
            summary[workspace_id] = result["status"]
            if result["status"] != "completed":
                overall_ok = False
            print(f"[sleep_cycle] workspace={workspace_id} status={result['status']} stats={result['stats']}")
        except Exception as e:
            overall_ok = False
            summary[workspace_id] = "failed"
            print(f"[sleep_cycle] workspace={workspace_id} FAILED before a run row could be created: {e}")
            print(traceback.format_exc())

    print(f"[sleep_cycle] run complete: {summary}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())


# ── Railway Cron Job setup (Tanmay, one-time) ───────────────────────────────────
# Same project/pattern as worker.py's cron job -- a SECOND Cron Job service,
# not a change to worker.py's own schedule.
#
# 1. Railway dashboard -> your project -> "+ New" -> "Cron Job"
# 2. Source: same GitHub repo, same branch (main)
# 3. Schedule: daily default per the architecture target --
#    "0 3 * * *" (03:00 UTC, off-peak). This is one global schedule for
#    every workspace list_workspaces_with_memory_activity() discovers --
#    true per-workspace-configurable cadence needs a real settings surface
#    that does not exist yet; wiring that in later only changes the loop in
#    main() (skip a workspace whose configured next-run hasn't arrived), not
#    run_consolidation() itself.
# 4. Start command: python sleep_cycle.py
# 5. Environment: same as worker.py -- SUPABASE_URL, SUPABASE_SERVICE_KEY.
#    This file (and memory_consolidation.py) never touches
#    APP_SUPABASE_SERVICE_KEY -- see list_workspaces_with_memory_activity()'s
#    docstring for why.
# 6. First run: trigger manually from the Railway dashboard once, then check
#    memory_consolidation_runs (SQL, or ask Claude) for one row per workspace
#    with status='completed'.
