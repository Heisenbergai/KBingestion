"""
PHASE 4 — CONTROLLED LIVE PERSISTENCE TEST — standalone runner.

Run this in an environment with REAL AWS Bedrock credentials (this local
dev environment does not have them -- confirmed via `boto3.Session()
.get_credentials()` returning None). Same pattern as
run_phase4_benchmark.py: nothing in this script re-implements or modifies
the frozen extraction contract (structured_extraction.py) or the
persistence path (structured_persistence.py) -- it only calls them, exactly
as a caller would, for exactly the 9 real canonical items named in the
approved controlled-test scope. Nothing else is processed -- no bulk
corpus, no scan.

ONE extraction_run_id is generated for this single controlled execution
(per the approved scope: no extraction_runs table yet, just the column).

Usage:
    python run_phase4_controlled_persist.py

Then paste the full terminal output back. Every persisted row is real,
written under the real 'v2.1' contract version, with server-side
workspace/usability/sensitivity validation applied exactly as
persist_extracted_primitives() enforces for any other caller.
"""
import json
import uuid

import canonical
import structured_extraction as se
import structured_persistence as sp

EXTRACTION_VERSION = "v2.1"  # the real, already-frozen contract -- never invented here

# Exactly the 9 real canonical items named in the approved controlled-test
# scope. (kind, canonical_id, canonical_source_type, provider, label)
CORPUS = [
    ("note", "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1", "knowledge_note", "google_chat",
     "Q4 Chat launch-gate requirement"),
    ("note", "011a2f6e-6f9c-48fd-b1f9-36a799ad3981", "knowledge_note", "slack",
     "Slack target-September-12 item"),
    ("note", "e3626652-c791-453a-8d49-a9d337b874d7", "knowledge_note", "slack",
     "Slack suggestion"),
    ("note", "660c2c81-b567-4bdc-bc9e-05ced7b02439", "knowledge_note", "slack",
     "Credential-change policy"),
    ("note", "ff5972e5-fda3-4e70-ae45-bb0025420cc4", "knowledge_note", "slack",
     "Monday capacity process"),
    ("note", "da347d97-0512-43de-9fe1-987872cd1b9e", "knowledge_note", "slack",
     "Office-maintenance event"),
    ("calendar", "aa473196-79dd-4a9c-aefc-f2c80d12ea94", "calendar_event", "calendar",
     "Calendar event"),
    ("note", "7c25e10c-da91-4239-b7dd-9f8426c37d6c", "knowledge_note", "bot_learning",
     "HR contact fact"),
    ("note", "51ef25a4-af79-4638-b668-50cdde145cbd", "knowledge_note", "slack",
     "Q4 roadmap"),
]


def _fetch(kind: str, item_id: str):
    if kind == "calendar":
        return canonical.project_calendar_event(item_id)
    return canonical.project_knowledge_note(item_id)


def _primitive_summary(p) -> dict:
    return {
        "type": p.type, "requirement_kind": p.requirement_kind, "statement": p.statement,
        "effective_from": p.effective_from, "effective_until": p.effective_until,
        "qualifier_words": p.qualifier_words, "recurrence_text": p.recurrence_text,
        "event_start": p.event_start, "event_end": p.event_end,
    }


def main():
    run_id = str(uuid.uuid4())
    print(f"EXTRACTION_RUN_ID: {run_id}")
    print(f"EXTRACTION_VERSION: {EXTRACTION_VERSION}")
    print()

    all_results = []
    for kind, item_id, canonical_source_type, provider, label in CORPUS:
        print("=" * 100)
        print(f"CANONICAL_ID: {item_id}  ({label})")

        item = _fetch(kind, item_id)
        if item is None:
            print("STATUS: FIXTURE MISSING -- skipped, nothing persisted.")
            all_results.append({"canonical_id": item_id, "label": label, "status": "missing"})
            continue

        print(f"ORIGINAL TEXT:\n{item.content}")

        try:
            extracted = se.extract_primitives_from_canonical(item)
        except Exception as e:
            print(f"STATUS: EXTRACTION CALL RAISED (report this): {e}")
            all_results.append({"canonical_id": item_id, "label": label, "status": "extraction_error", "error": str(e)})
            continue

        if not extracted:
            print("STATUS: DROPPED at extraction -- zero primitives, nothing to persist.")
            all_results.append({"canonical_id": item_id, "label": label, "status": "extraction_empty"})
            continue

        print(f"EXTRACTED: {len(extracted)} primitive(s)")
        for i, p in enumerate(extracted):
            print(f"  [{i}] {json.dumps(_primitive_summary(p), indent=2)}")

        # A persistence-layer exception for ONE item must never abort the
        # whole controlled run -- the prior run hit exactly this (an
        # uncaught schema error on the Calendar item silently skipped the
        # remaining 2 real items entirely). Caught and reported per-item,
        # matching the same non-fatal contract the extraction call above
        # already has.
        try:
            persist_result = sp.persist_extracted_primitives(
                workspace_id=item.workspace_id,
                canonical_source_type=canonical_source_type,
                canonical_id=item_id,
                provider=provider,
                extraction_version=EXTRACTION_VERSION,
                extraction_run_id=run_id,
                primitives=extracted,
            )
        except Exception as e:
            print(f"STATUS: PERSIST CALL RAISED (report this): {e}")
            all_results.append({
                "canonical_id": item_id, "label": label, "status": "persist_error", "error": str(e),
                "extracted": [_primitive_summary(p) for p in extracted],
            })
            continue
        print(f"PERSIST RESULT: {json.dumps(persist_result, indent=2)}")
        all_results.append({
            "canonical_id": item_id, "label": label, "status": "persisted",
            "extracted": [_primitive_summary(p) for p in extracted],
            "persist_result": persist_result,
        })
        print()

    print("=" * 100)
    print("FULL JSON DUMP (paste this whole block back too):")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
