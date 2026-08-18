"""
Phase 4 extraction PROTOTYPE tests.

IMPORTANT, stated once here rather than repeated everywhere: this local
environment has no AWS/Bedrock credentials (confirmed -- boto3.Session()
.get_credentials() returns None here, the same gap already established for
CONNECTOR_ENCRYPTION_KEY/Google token decryption earlier this project).
`ai.chat_json` is therefore monkeypatched in every test below to a FIXED
FIXTURE response -- these tests prove the deterministic CODE around the LLM
call (validation, filtering, propagation, never-invent guards) is correct.
They do NOT prove the LLM itself extracts these primitives correctly from
real text -- that requires a real Bedrock call this environment cannot make.
See the final report's "Results"/"Failure modes" sections for the explicit,
honest boundary between what's proven here and what isn't.

The INPUT side is real wherever possible: every fixture below is fetched
from the real live vector DB via canonical.project_knowledge_note()/
project_calendar_event() (Phase 3's own orchestration functions -- no new
read path), using real notes already present from this project's earlier
phases (the real Q4 Chat launch-gate note, real Slack process/policy/
suggestion/announcement notes, a real Calendar event). The "model response"
half of each test is a hand-written fixture representing a plausible
well-behaved extraction for that real text, explicitly labeled as such.

Run with: python -m pytest test_phase4_extraction.py -v
"""
import pytest

import canonical
import structured_extraction as se

# Real canonical ids used throughout (see the report's "Real canonical test
# cases" section for the full real text of each).
REAL_Q4_NOTE_ID = "7a9eaa34-21b4-4ed4-b171-2ebc52cdb3a1"                 # Requirement/policy, real "Starting September 15"
REAL_HEDGED_RELEASE_NOTE_ID = "011a2f6e-6f9c-48fd-b1f9-36a799ad3981"     # real "target" hedge language
REAL_SUGGESTION_NOTE_ID = "e3626652-c791-453a-8d49-a9d337b874d7"        # real suggestion, must NOT become Decision
REAL_CREDENTIAL_POLICY_NOTE_ID = "660c2c81-b567-4bdc-bc9e-05ced7b02439"  # real "Starting next week" relative date
REAL_CALENDAR_EVENT_ID = "aa473196-79dd-4a9c-aefc-f2c80d12ea94"         # real Calendar event
REAL_HEDGED_RELEASE_NOTE_ID_2 = "3f835c28-1fd5-4d2c-9fd1-af53b4d2dcbc"  # real second near-duplicate "target Sept 12" note, DIFFERENT workspace


def _fetch_real_note(note_id: str) -> canonical.CanonicalKnowledge:
    item = canonical.project_knowledge_note(note_id)
    assert item is not None, f"fixture note {note_id} no longer exists -- update the fixture"
    return item


def _fetch_real_calendar_event(event_id: str) -> canonical.CanonicalKnowledge:
    item = canonical.project_calendar_event(event_id)
    assert item is not None, f"fixture calendar event {event_id} no longer exists -- update the fixture"
    return item


# =====================================================================
# Real input + well-behaved fixture verdict -- proves the pipeline
# correctly SHAPES a good model response into ExtractedPrimitive objects.
# =====================================================================

def test_real_q4_note_extracts_requirement_with_correct_effective_from(monkeypatch):
    item = _fetch_real_note(REAL_Q4_NOTE_ID)
    assert "September 15" in item.content  # sanity check: real text unchanged

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "policy",
        "statement": "Starting September 15, the Q4 smart-switch launch requires Product and QA "
                     "approval of the Matter stability checklist before production.",
        "raw_subject_phrase": "Product and QA",
        "effective_from": "2026-09-15", "effective_until": None,
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})

    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    r = results[0]
    assert r.type == "requirement"
    assert r.requirement_kind == "policy"
    assert r.effective_from == "2026-09-15"
    assert r.raw_subject_phrase == "Product and QA"
    assert r.canonical_id == REAL_Q4_NOTE_ID
    # Propagation, not reclassification -- direct copies from the real canonical item.
    assert r.sensitivity == item.sensitivity == "internal"
    assert r.authority == item.authority == "official"
    assert r.source_tier == item.source_tier == 2
    assert r.captured_at == item.captured_at


def test_real_hedged_release_note_does_not_get_effective_date(monkeypatch):
    """Real text: 'The release target is September 12.' -- 'target' is
    exactly the hedge word the extraction contract lists. A well-behaved
    model response must leave effective_from null and record the qualifier
    -- this test proves the pipeline PRESERVES that null rather than
    somehow filling it in itself."""
    item = _fetch_real_note(REAL_HEDGED_RELEASE_NOTE_ID)
    assert "target" in item.content.lower()

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact",
        "statement": "The next smart-switch firmware release targets September 12, prioritizing "
                     "Matter stability and offline recovery, with no new features after September 5.",
        "raw_subject_phrase": "the next smart-switch firmware release",
        "effective_from": None, "effective_until": None,
        "has_qualifier": True, "qualifier_words": ["target"], "confidence": "high",
    }]})

    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].effective_from is None, "a hedged target date must never become a committed effective_from"
    assert "target" in results[0].qualifier_words


def test_real_suggestion_note_never_becomes_a_decision(monkeypatch):
    """Real text: 'John Snow suggests moving the current production review...'
    -- a suggestion, not a decision. A well-behaved model response
    correctly returns [] (nothing worth structuring as a settled choice) --
    the pipeline must not force a decision out of it."""
    item = _fetch_real_note(REAL_SUGGESTION_NOTE_ID)
    assert "suggests" in item.content.lower()

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": []})

    results = se.extract_primitives_from_canonical(item)
    assert results == []


def test_real_credential_policy_passes_real_reference_date_into_prompt(monkeypatch):
    """Real text: 'Starting next week, all production credential changes
    must be recorded...'. This test proves the pipeline passes a real
    REFERENCE_DATE into the prompt (not blank) -- the actual resolvability
    of 'next week' is covered separately (see the ambiguous-relative-phrase
    guard test below): under the V2 deterministic convention, 'next week'
    ALONE (no specific day named) is never resolved to an exact date, so a
    well-behaved model response correctly returns effective_from=null here,
    which the pipeline must preserve as null, not silently fill in."""
    item = _fetch_real_note(REAL_CREDENTIAL_POLICY_NOTE_ID)
    assert "starting next week" in item.content.lower()
    assert item.event_time is not None, "sanity check: a real reference date exists to resolve against"

    captured_prompt = {}
    def _fake_chat_json(**kwargs):
        captured_prompt["content"] = kwargs["messages"][0]["content"]
        return {"primitives": [{
            "type": "requirement", "requirement_kind": "policy",
            "statement": "Starting next week, all production credential changes must be recorded in the "
                         "security log and reviewed by another administrator.",
            "raw_subject_phrase": "production credential changes",
            "effective_from": None, "effective_until": None,
            "has_qualifier": False, "qualifier_words": [], "confidence": "medium",
        }]}
    monkeypatch.setattr(se.ai, "chat_json", _fake_chat_json)

    results = se.extract_primitives_from_canonical(item)
    assert "REFERENCE_DATE:" in captured_prompt["content"]
    assert item.event_time[:10] in captured_prompt["content"], \
        "the real occurred_at date must actually be passed into the prompt, not blank"
    assert len(results) == 1
    assert results[0].effective_from is None


def test_ambiguous_next_week_phrase_forces_null_date_even_if_model_returns_one(monkeypatch):
    """V2 code-level guard, real-benchmark finding #5: even if the model
    misbehaves and returns a specific date for the real 'Starting next
    week...' text, the pipeline must discard it -- this is the defense in
    depth that doesn't depend on the model getting the prompt's convention
    right every time."""
    item = _fetch_real_note(REAL_CREDENTIAL_POLICY_NOTE_ID)
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "policy",
        "statement": "Credential changes must be recorded starting next week.",
        "effective_from": "2026-08-22",  # the model misbehaving -- should be discarded
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].effective_from is None, \
        "'next week' anywhere in the source text must force effective_from to null, regardless of model output"


def test_real_calendar_event_extracts_as_event_with_interval(monkeypatch):
    item = _fetch_real_calendar_event(REAL_CALENDAR_EVENT_ID)

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "event",
        "statement": f"{item.title} is scheduled from {item.event_start} to {item.event_end}.",
        "raw_subject_phrase": item.title,
        "effective_from": None, "effective_until": None,
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})

    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    r = results[0]
    assert r.type == "event"
    assert r.event_start == item.event_start
    assert r.event_end == item.event_end
    assert r.sensitivity is None, "Calendar items carry no classification -- must stay None, never defaulted"


# =====================================================================
# Adversarial fixtures -- proves the pipeline's safety/validation logic,
# not just its happy path. Synthetic canonical items here (not real DB
# rows) since these test malformed/adversarial MODEL OUTPUT, not real text.
# =====================================================================

def _synthetic_item(content="Some real-shaped test content.", **overrides) -> canonical.CanonicalKnowledge:
    base = dict(
        id="synthetic-1", workspace_id="ws-1", connection_id=None, source="slack",
        source_type="slack", title="x", content=content, category=None,
        sensitivity="internal", authority="working", source_tier=3, doc_class=None,
        lifecycle_status="active", record_status="active", processing_status=None,
        effective_from=None, valid_until=None, superseded_by=None,
        captured_at="2026-08-01T00:00:00Z", event_time="2026-08-01T00:00:00Z",
        event_start=None, event_end=None, source_updated_at=None,
    )
    base.update(overrides)
    return canonical.CanonicalKnowledge(**base)


def test_malformed_date_string_is_never_passed_through(monkeypatch):
    """The model returning a non-ISO date ('next month', a hallucination-
    shaped guess) must be rejected, not silently trusted."""
    item = _synthetic_item()
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "statement": "Something must happen soon.",
        "effective_from": "next month", "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].effective_from is None, "a malformed date string must never be passed through as real"


def test_low_confidence_extraction_is_dropped(monkeypatch):
    item = _synthetic_item()
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "A vague inference.", "confidence": "low",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert results == [], "low-confidence extractions must be dropped, not returned"


def test_unrecognized_type_is_dropped_not_guessed(monkeypatch):
    item = _synthetic_item()
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "opinion",  # not one of the four valid types
        "statement": "Something.", "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert results == []


def test_invalid_requirement_kind_kept_as_none_not_guessed(monkeypatch):
    item = _synthetic_item()
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "guideline",  # not a valid kind
        "statement": "Something must happen.", "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].requirement_kind is None, \
        "an unrecognized requirement_kind must be dropped to None, never guessed into a valid one"


def test_empty_content_never_calls_the_model(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: calls.__setitem__("n", calls["n"] + 1) or {"primitives": []})
    item = _synthetic_item(content="")
    results = se.extract_primitives_from_canonical(item)
    assert results == []
    assert calls["n"] == 0, "no content means nothing to extract -- must not spend an LLM call on it"


def test_model_returning_empty_list_is_respected(monkeypatch):
    """The model itself deciding nothing is worth extracting must produce
    zero results -- never a forced fallback extraction."""
    item = _synthetic_item(content="lol ok thanks")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": []})
    results = se.extract_primitives_from_canonical(item)
    assert results == []


def test_malformed_llm_response_shape_fails_closed(monkeypatch):
    """A non-dict / missing-'primitives' response must not raise and must
    not fabricate a result."""
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: "not a dict at all")
    assert se.extract_primitives_from_canonical(_synthetic_item()) == []

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"unexpected_key": []})
    assert se.extract_primitives_from_canonical(_synthetic_item()) == []


def test_llm_call_exception_fails_closed_not_raised(monkeypatch):
    def _raise(**k):
        raise Exception("simulated Bedrock outage")
    monkeypatch.setattr(se.ai, "chat_json", _raise)
    results = se.extract_primitives_from_canonical(_synthetic_item())
    assert results == []


# =====================================================================
# Conflict / non-merging behavior -- real data (two separate, real
# near-duplicate "target September 12" notes in two different real
# workspaces), each extracted independently.
# =====================================================================

def test_two_real_competing_notes_extracted_independently_never_merged(monkeypatch):
    item_a = _fetch_real_note(REAL_HEDGED_RELEASE_NOTE_ID)
    item_b = _fetch_real_note(REAL_HEDGED_RELEASE_NOTE_ID_2)
    assert item_a.workspace_id != item_b.workspace_id, \
        "sanity check: these are genuinely two separate real workspaces/sources"

    def _fake_extraction(**kwargs):
        text = kwargs["messages"][0]["content"]
        return {"primitives": [{
            "type": "fact", "statement": "Firmware release targeted for September 12.",
            "raw_subject_phrase": "the firmware release", "effective_from": None,
            "has_qualifier": True, "qualifier_words": ["target"], "confidence": "high",
        }]}
    monkeypatch.setattr(se.ai, "chat_json", _fake_extraction)

    results_a = se.extract_primitives_from_canonical(item_a)
    results_b = se.extract_primitives_from_canonical(item_b)

    assert len(results_a) == 1 and len(results_b) == 1
    assert results_a[0].canonical_id == REAL_HEDGED_RELEASE_NOTE_ID
    assert results_b[0].canonical_id == REAL_HEDGED_RELEASE_NOTE_ID_2
    assert results_a[0].canonical_id != results_b[0].canonical_id, \
        "two separate canonical sources must never collapse into one extracted item"
    # No supersedes/merge field exists anywhere on ExtractedPrimitive at all --
    # structurally impossible for this prototype to link them, by design.
    assert not hasattr(results_a[0], "supersedes")


def test_no_supersedes_field_exists_on_extracted_primitive():
    """Structural proof, not just behavioral: the dataclass itself has no
    supersedes/merge-target field in this prototype -- conflict resolution
    is entirely out of scope, not just unused."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(se.ExtractedPrimitive)}
    assert "supersedes" not in field_names
    assert "merged_from" not in field_names


def test_no_confidence_field_persisted_on_extracted_primitive():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(se.ExtractedPrimitive)}
    assert "confidence" not in field_names
    assert "extraction_confidence" not in field_names


def test_sensitivity_propagation_is_direct_copy_never_reclassified(monkeypatch):
    """Section 8 of the audit's acceptance criteria: propagation is a
    straight copy, so it structurally cannot lower or raise anything --
    proven by checking it against several different real sensitivity
    values, not just 'internal'."""
    for sens in ("public", "internal", "confidential", "restricted"):
        item = _synthetic_item(sensitivity=sens)
        monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [
            {"type": "fact", "statement": "x", "confidence": "high"}
        ]})
        results = se.extract_primitives_from_canonical(item)
        assert results[0].sensitivity == sens


# =====================================================================
# V2 -- real-benchmark-driven fixes (2026-08-17)
# =====================================================================

def test_v2_qualifier_defense_in_depth_nulls_date_even_if_model_contradicts_itself(monkeypatch):
    """Finding #1: even if the model returns has_qualifier=true AND a date
    for the same primitive (a self-contradiction), the code-level guard
    must null the date -- never trust the date half of a contradictory
    response."""
    item = _synthetic_item(content="The release target is September 12.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "The release target is September 12.",
        "effective_from": "2026-09-12",  # contradicts has_qualifier below
        "has_qualifier": True, "qualifier_words": ["target"], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].effective_from is None
    assert "target" in results[0].qualifier_words


def test_v2_qualifier_words_alone_also_trigger_the_guard(monkeypatch):
    """Same guard, triggered by a non-empty qualifier_words list even if
    has_qualifier itself was omitted/false -- belt and suspenders against
    the model being internally inconsistent in either direction."""
    item = _synthetic_item()
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "x", "effective_from": "2026-09-12",
        "has_qualifier": False, "qualifier_words": ["target"], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert results[0].effective_from is None


def test_v2_recurrence_text_captured_and_never_produces_an_effective_date(monkeypatch):
    """Finding #6: 'every Monday by 11 AM' -- recurrence_text preserves the
    phrase verbatim; no new primitive type invented; effective_from stays
    null even if the model attaches a date to a recurring item."""
    item = _synthetic_item(content="Procurement, warehouse, assembly, and QA must submit "
                                   "their capacity by 11 AM every Monday.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "process_step",
        "statement": "Procurement, warehouse, assembly, and QA must submit their capacity by 11 AM every Monday.",
        "raw_subject_phrase": "Procurement, warehouse, assembly, and QA",
        "recurrence_text": "every Monday by 11 AM",
        "effective_from": "2026-08-24",  # model misbehaving -- must be discarded
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].recurrence_text == "every Monday by 11 AM"
    assert results[0].effective_from is None
    assert results[0].type == "requirement"  # no new primitive type invented for recurrence


def test_v2_requirement_broadened_definition_covers_polite_instructions(monkeypatch):
    """Finding #3: 'Please keep the kitchen area clear after 6 PM' is a
    real instruction, not a neutral fact -- this test proves the pipeline
    correctly SHAPES a requirement-typed response for this real text (the
    prompt content itself is checked separately below; this proves nothing
    in the code blocks a requirement classification for polite-imperative
    phrasing)."""
    item = _synthetic_item(content="The office maintenance team will be visiting tomorrow morning. "
                                   "Please keep the kitchen area clear after 6 PM.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [
        {"type": "event", "statement": "The office maintenance team will visit tomorrow morning.",
         "confidence": "high", "has_qualifier": False, "qualifier_words": []},
        {"type": "requirement", "requirement_kind": "process_step",
         "statement": "The kitchen area must be kept clear after 6 PM.",
         "confidence": "high", "has_qualifier": False, "qualifier_words": []},
    ]})
    results = se.extract_primitives_from_canonical(item)
    types = {r.type for r in results}
    assert "requirement" in types
    assert "event" in types


def test_v2_prompt_contains_all_required_hardening_rules():
    """Structural proof that every requirement (A-I) from the V2 request
    actually landed in the live prompt text, not just in this module's
    intentions. Cheap, deterministic, and catches prompt regressions a
    fixture-based behavioral test cannot."""
    prompt = se.EXTRACT_SYSTEM
    checks = {
        "A. qualifier-aware extraction": "QUALIFIER RULE",
        "B. Requirement semantic detection": "explicitly EXCLUDED",
        "C. anti-over-extraction rule": "ANTI-OVER-EXTRACTION",
        "D. simple-fact retention": "does NOT make something low-confidence",
        "E. deterministic relative-date instructions": "RELATIVE DATE RESOLUTION",
        "F. recurring-time handling": "RECURRING TIME PATTERNS",
        "G. no date invention": "NEVER invent a date",
        "H. no entity resolution": "NEVER invent, resolve, or normalize a subject",
        "I. no cross-source merging": "NEVER merge or compare this text against any other source",
    }
    for label, must_contain in checks.items():
        assert must_contain in prompt, f"missing V2 prompt requirement {label}: expected {must_contain!r}"


def test_v2_prompt_still_contains_v1_anti_patterns():
    """The V1 anti-patterns (suggestion != decision) must survive into V2,
    not be lost while adding new rules."""
    prompt = se.EXTRACT_SYSTEM
    assert "maybe we" in prompt
    assert "NEVER decisions" in prompt


def test_v2_ambiguous_relative_phrase_regex_matches_real_source_text():
    """Direct proof the regex actually matches the real note's real
    wording, not just a hand-crafted synthetic string."""
    item = _fetch_real_note(REAL_CREDENTIAL_POLICY_NOTE_ID)
    assert se._AMBIGUOUS_RELATIVE_RE.search(item.content) is not None


def test_v2_ambiguous_relative_phrase_regex_does_not_false_positive_on_unrelated_text():
    assert se._AMBIGUOUS_RELATIVE_RE.search("The next release will focus on stability.") is None
    assert se._AMBIGUOUS_RELATIVE_RE.search("Starting September 15, approval is required.") is None


# =====================================================================
# V2.1 -- deterministic source-text qualifier extraction (real-benchmark
# finding: the model correctly classified "John Snow suggests..." as
# fact/not-decision, but silently returned qualifier_words=[] even though
# "suggests" is literally in the source text)
# =====================================================================

def test_v21_real_suggestion_note_qualifier_recovered_even_when_model_returns_empty(monkeypatch):
    """The exact real failure case: model gets the TYPE right (fact, not
    decision) but returns qualifier_words=[] despite 'suggests' being
    literally present in the real source text. The deterministic helper
    must recover it without the model's help."""
    item = _fetch_real_note(REAL_SUGGESTION_NOTE_ID)
    assert "suggests" in item.content.lower()

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact",
        "statement": "John Snow suggests moving the current production review earlier in the week.",
        "raw_subject_phrase": "John Snow",
        "effective_from": None, "effective_until": None,
        "has_qualifier": False, "qualifier_words": [],  # the real observed model failure
        "confidence": "medium",
    }]})

    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert results[0].type == "fact", "semantic classification was already correct -- must not change"
    assert "suggests" in results[0].qualifier_words


@pytest.mark.parametrize("word,text", [
    ("suggests", "The team suggests we revisit this next quarter."),
    ("suggested", "It was suggested that we revisit this."),
    ("suggestion", "The suggestion is to revisit this."),
    ("proposes", "Ops proposes a new schedule."),
    ("proposed", "A new schedule was proposed."),
    ("may", "The date may change."),
    ("might", "The date might change."),
    ("target", "The target date is unclear."),
    ("targeted", "This is targeted for later this year."),
    ("planned", "This is planned for later this year."),
    ("tentative", "This is a tentative plan."),
    ("expected", "This is expected to happen soon."),
    ("considering", "We are considering a change."),
    ("potential", "There is a potential change coming."),
])
def test_v21_deterministic_qualifier_detected_for_each_vocabulary_term(word, text):
    detected = se._extract_source_qualifiers(text)
    assert word in detected


def test_v21_no_qualifier_in_text_yields_empty_list():
    clean_text = "Only the revision stored in the controlled BOM folder is considered valid for production."
    # sanity: this real-shaped sentence has none of the vocabulary terms
    assert se._extract_source_qualifiers(clean_text) == []


def test_v21_deterministic_helper_never_removes_a_model_reported_qualifier(monkeypatch):
    """Rule 3: never remove a qualifier the model already returned, even
    one outside the fixed vocabulary (e.g. 'hopefully')."""
    item = _synthetic_item(content="A plain statement with no listed qualifier words.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "x", "qualifier_words": ["hopefully"],
        "has_qualifier": True, "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert "hopefully" in results[0].qualifier_words


def test_v21_deterministic_helper_deduplicates_case_insensitively(monkeypatch):
    """Rule 4: deduplicate -- the model reporting 'Target' and the source
    text containing 'target' must not produce two entries."""
    item = _synthetic_item(content="The target date is September 12.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "x", "qualifier_words": ["Target"],
        "has_qualifier": True, "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    lowered = [q.lower() for q in results[0].qualifier_words]
    assert lowered.count("target") == 1


def test_v21_helper_does_not_infer_qualifiers_not_literally_present():
    """Rule 5: no inference -- a synonym or implied hedge not in the exact
    vocabulary must not be detected."""
    text = "This is somewhat uncertain and could change, we think."
    detected = se._extract_source_qualifiers(text)
    assert "could" not in detected, "not in the fixed vocabulary -- must not be inferred"
    assert "somewhat" not in detected


def test_v21_helper_never_changes_primitive_type(monkeypatch):
    """Rule 6: this helper only touches qualifier_words/has_qualifier
    (and, via the pre-existing qualifier-suppresses-date guard, dates) --
    never the type the model assigned."""
    item = _synthetic_item(content="This is a suggested approach.")
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "process_step",
        "statement": "x", "qualifier_words": [], "has_qualifier": False, "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert results[0].type == "requirement", "type must be untouched even though a qualifier was recovered"
    assert "suggested" in results[0].qualifier_words


def test_v21_existing_date_defense_behavior_unchanged_for_real_q4_note(monkeypatch):
    """Regression check: the real Q4 note's text contains none of the
    qualifier vocabulary terms, so V2.1 must not affect it at all -- its
    committed effective_from must still come through exactly as before."""
    item = _fetch_real_note(REAL_Q4_NOTE_ID)
    assert se._extract_source_qualifiers(item.content) == [], \
        "fixture sanity check: this real text has no vocabulary terms present"

    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "requirement", "requirement_kind": "policy",
        "statement": "Starting September 15, Product and QA must approve the Matter checklist.",
        "raw_subject_phrase": "Product and QA",
        "effective_from": "2026-09-15", "effective_until": None,
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert results[0].effective_from == "2026-09-15"
    assert results[0].qualifier_words == []


def test_v21_date_still_suppressed_when_only_the_deterministic_helper_finds_a_qualifier(monkeypatch):
    """The chained defense-in-depth effect: if the model reports NO
    qualifier and NO has_qualifier, but the deterministic scan finds one in
    the real source text, the existing qualifier-suppresses-date guard
    must still fire -- proving V2's date guard and V2.1's qualifier
    recovery compose correctly together."""
    item = _fetch_real_note(REAL_HEDGED_RELEASE_NOTE_ID)  # real "target" text
    monkeypatch.setattr(se.ai, "chat_json", lambda **k: {"primitives": [{
        "type": "fact", "statement": "The release target is September 12.",
        "effective_from": "2026-09-12",  # model misbehaving: didn't self-report the hedge
        "has_qualifier": False, "qualifier_words": [], "confidence": "high",
    }]})
    results = se.extract_primitives_from_canonical(item)
    assert len(results) == 1
    assert "target" in results[0].qualifier_words
    assert results[0].effective_from is None, \
        "the deterministically-recovered qualifier must still trigger the existing date-suppression guard"
