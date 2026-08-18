"""
PHASE 4 — REAL EXTRACTION BENCHMARK — standalone runner (V2 prompt).

V2 changes since the last real run: qualifier rule strengthened + enforced
in code (not just the prompt), anti-over-extraction rule added, requirement
definition broadened (polite instructions, scope exclusions), simple-fact
confidence-calibration guidance added, deterministic relative-date
convention added (with a code-level guard for "next week"/"next month"),
recurrence_text field added for recurring-time patterns. Re-run against the
SAME 10-item corpus as the V1 run -- nothing added, nothing removed.

Run this in an environment that has REAL AWS Bedrock credentials (this
local dev environment does not -- confirmed via `boto3.Session()
.get_credentials()` returning None here). Nothing else about the codebase
needs to differ: this script imports the exact, unmodified
canonical.py / structured_extraction.py already in this repo and calls the
real extract_primitives_from_canonical() -- the same function the test
suite already exercises with fixture responses. This run is the first time
it would ever hit a real model.

READ-ONLY. No writes to any table, no Phase 4 persistence (there is no
Phase 4 table). One real Bedrock call per corpus item, nothing else.

Usage:
    python run_phase4_benchmark.py

Then paste the full terminal output back for write-up. The script prints a
human-readable block per item AND a final machine-readable JSON dump (so
nothing gets lost in reformatting when you paste it back).
"""
import json

import canonical
import structured_extraction as se

# The exact real corpus from the approved benchmark request -- 9 named
# items + 1 additional real note (Q4 roadmap priorities, an ambiguous
# announcement/Fact/Requirement boundary case).
CORPUS = [
    ("note", "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1", "Q4 Chat launch-gate (Requirement/policy expected)"),
    ("note", "011a2f6e-6f9c-48fd-b1f9-36a799ad3981", "Slack 'target September 12' (hedged, no committed date expected)"),
    ("note", "e3626652-c791-453a-8d49-a9d337b874d7", "Slack suggestion (must NOT become Decision)"),
    ("note", "660c2c81-b567-4bdc-bc9e-05ced7b02439", "Credential-change policy, 'Starting next week' (relative date)"),
    ("note", "433bb17d-3327-4a23-967b-36b7f3961bf2", "BOM revision-control policy (no date expected)"),
    ("note", "ff5972e5-fda3-4e70-ae45-bb0025420cc4", "Monday capacity-review process (recurring time, not a single date)"),
    ("note", "da347d97-0512-43de-9fe1-987872cd1b9e", "Office-maintenance event, 'tomorrow morning' (relative date, Event expected)"),
    ("calendar", "aa473196-79dd-4a9c-aefc-f2c80d12ea94", "Real Calendar event (Event with interval expected)"),
    ("note", "7c25e10c-da91-4239-b7dd-9f8426c37d6c", "bot_learning HR contact fact (simplest case, no date/qualifier)"),
    ("note", "51ef25a4-af79-4638-b668-50cdde145cbd", "Q4 roadmap priorities (ambiguous Fact/Requirement boundary)"),
]


def _fetch(kind: str, item_id: str):
    if kind == "calendar":
        return canonical.project_calendar_event(item_id)
    return canonical.project_knowledge_note(item_id)


def _primitive_to_dict(p) -> dict:
    return {
        "type": p.type,
        "requirement_kind": p.requirement_kind,
        "statement": p.statement,
        "raw_subject_phrase": p.raw_subject_phrase,
        "effective_from": p.effective_from,
        "effective_until": p.effective_until,
        "qualifier_words": p.qualifier_words,
        "recurrence_text": p.recurrence_text,
        "event_time": p.event_time,
        "event_start": p.event_start,
        "event_end": p.event_end,
        "captured_at": p.captured_at,
        "sensitivity": p.sensitivity,
        "authority": p.authority,
        "source_tier": p.source_tier,
        "source_evidence_link": p.source_evidence_link,
    }


def main():
    all_results = []
    for kind, item_id, label in CORPUS:
        print("=" * 100)
        print(f"CANONICAL_ID: {item_id}")
        print(f"LABEL: {label}")

        item = _fetch(kind, item_id)
        if item is None:
            print("STATUS: FIXTURE MISSING -- this real row no longer exists, skipped.")
            all_results.append({"canonical_id": item_id, "label": label, "status": "missing"})
            continue

        print(f"SOURCE: {item.source}")
        print(f"ORIGINAL TEXT:\n{item.content}")
        print(f"event_time={item.event_time!r} event_start={item.event_start!r} "
              f"event_end={item.event_end!r} captured_at={item.captured_at!r}")

        try:
            extracted = se.extract_primitives_from_canonical(item)
        except Exception as e:
            print(f"STATUS: EXTRACTION CALL RAISED (should never happen -- report this): {e}")
            all_results.append({"canonical_id": item_id, "label": label, "status": "error", "error": str(e)})
            continue

        if not extracted:
            print("STATUS: DROPPED -- zero primitives returned (either the model found nothing "
                  "worth extracting, or every candidate was below min_confidence).")
            all_results.append({"canonical_id": item_id, "label": label, "status": "dropped", "primitives": []})
        else:
            print(f"STATUS: {len(extracted)} primitive(s) extracted")
            for i, p in enumerate(extracted):
                print(f"  [{i}] {json.dumps(_primitive_to_dict(p), indent=2)}")
            all_results.append({
                "canonical_id": item_id, "label": label, "status": "extracted",
                "primitives": [_primitive_to_dict(p) for p in extracted],
            })
        print()

    print("=" * 100)
    print("FULL JSON DUMP (paste this whole block back too):")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
