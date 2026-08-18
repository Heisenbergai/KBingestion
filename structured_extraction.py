"""
Phase 4 PROTOTYPE -- structured knowledge EXTRACTION only. In-memory results,
NO persistence, NO schema, NO migration. Consumes canonical.get_canonical_knowledge()
exclusively -- never reads knowledge_notes/document_chunks/calendar_events/
any connector table directly, matching the Phase 4 audit's own "input
contract" boundary.

This is the single new capability the Phase 4 audit identified as genuinely
unbuilt: nothing upstream (Phase 2's classify_batch/classify_document, or
Phase 3's projection layer) parses a stated effective date, distinguishes a
settled decision from a hedged suggestion, or restates content into one of
the four approved primitive shapes (Fact/Decision/Requirement/Event). This
module is that step, and only that step.

Four primitives, matching the approved vocabulary exactly:
  - "decision": an explicit choice that was actually made/settled.
  - "requirement": a policy/process_step/commitment -- something that must
    or will happen, is expected of people, or is explicitly excluded/out of
    scope, distinguished from a mere recommendation.
  - "event": something that occurs or is scheduled, distinct from mere
    discussion ABOUT an event.
  - "fact": a declarative statement not primarily a decision/requirement/event
    -- including terse, explicit, durable facts (a contact number, a name).

Confidence is used ONLY to filter which extractions are returned (see
_MIN_CONFIDENCE below) -- it is never a field on ExtractedPrimitive and
never returned to the caller, per the approved contract ("Do NOT persist
confidence yet").

No entity resolution anywhere: raw_subject_phrase is always the literal
text the source used, never resolved to a person/team/product identity.

V2 (2026-08-17): prompt and code hardened against 6 real failure modes
found in the V1 real Bedrock benchmark -- see EXTRACT_SYSTEM's
QUALIFIER RULE, ANTI-OVER-EXTRACTION RULE, broadened requirement
definition, confidence-calibration guidance, deterministic relative-date
convention, and recurrence handling. The qualifier-suppresses-date and
recurrence-suppresses-date rules are additionally enforced in CODE (not
just the prompt) as defense in depth -- see
extract_primitives_from_canonical()'s post-processing.

V2.1 (2026-08-17): the real V2 benchmark showed the model can still SILENTLY
DROP a qualifier that is literally present in the source text (real case:
"John Snow suggests..." was correctly classified as "fact", not "decision",
but qualifier_words came back empty even though "suggests" is right there in
the text). Rather than trust the model to self-report every qualifier, a
small DETERMINISTIC, rule-based scan of the ORIGINAL source text (never the
model's paraphrased statement) now runs after every extraction and merges
any known qualifier term it finds into qualifier_words -- see
_extract_source_qualifiers(). This never removes a qualifier the model
already reported, never changes primitive type, and never invents a date --
it only ever ADDS qualifier words that are literally present in the text.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import ai
import canonical

VALID_TYPES = {"fact", "decision", "requirement", "event"}
VALID_REQUIREMENT_KINDS = {"policy", "process_step", "commitment"}
_MIN_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ExtractedPrimitive:
    """One structured item extracted from ONE canonical item. In-memory
    only -- no table, no id of its own is persisted anywhere by this
    module (a future persistence layer would assign one)."""
    type: str                              # fact | decision | requirement | event
    canonical_id: str                      # parent CanonicalKnowledge.id
    workspace_id: str
    statement: str                         # the structured content/restatement
    source_evidence_link: str              # a real permalink if the caller opted into
                                            # provenance, else the canonical_id itself --
                                            # never fabricated, never a guessed URL
    sensitivity: Optional[str]             # direct passthrough from the canonical item
    authority: Optional[str]               # direct passthrough
    source_tier: Optional[int]             # direct passthrough
    lifecycle_status: Optional[str]        # initialized from the canonical item
    captured_at: Optional[str]             # direct passthrough
    event_time: Optional[str]              # direct passthrough (message/event time)
    event_start: Optional[str]             # direct passthrough (Calendar interval)
    event_end: Optional[str]               # direct passthrough (Calendar interval)
    effective_from: Optional[str] = None   # ONLY if explicitly, unhedgedly stated
    effective_until: Optional[str] = None  # same rule
    raw_subject_phrase: Optional[str] = None  # opaque text, NEVER an entity id
    requirement_kind: Optional[str] = None    # policy | process_step | commitment
                                               # -- only meaningful when type == "requirement"
    qualifier_words: list = field(default_factory=list)  # hedge words found, if any
    recurrence_text: Optional[str] = None  # V2: raw recurrence phrase ("every Monday by
                                            # 11 AM"), preserved verbatim -- NOT parsed into
                                            # a structured recurrence rule (no schema for
                                            # that exists yet) and NEVER used to populate
                                            # effective_from/effective_until (a recurring
                                            # rule has no single effective date)


EXTRACT_SYSTEM = """You extract STRUCTURED KNOWLEDGE PRIMITIVES from one piece of company
knowledge. You will see REFERENCE_DATE (when this was captured/said, ISO date) and TEXT (the
knowledge content).

Extract ZERO, ONE, or MULTIPLE primitives. Each primitive is exactly one of:

  "decision" -- an explicit choice that was ACTUALLY MADE / settled. NEVER an opinion,
    suggestion, question, possibility, or speculation. "I think we should...", "maybe we
    could...", "X suggests...", "what if we..." are NEVER decisions -- if that's all the
    text contains, do not produce a decision item for it (a "fact" item describing that a
    suggestion was made is fine instead, if worth keeping at all).

  "requirement" -- something that MUST or WILL happen, is EXPECTED of people, or is
    explicitly EXCLUDED/OUT OF SCOPE. This is broader than formal policy language --
    it includes:
      * policy: a standing rule ("all changes must be logged")
      * process_step: an operational instruction or expected behavior, INCLUDING polite
        imperatives ("please keep the kitchen area clear after 6 PM" IS a requirement --
        an instruction directed at people, not a plain fact about the world)
      * commitment: a promise that something will happen
    A stated boundary or exclusion ("X is explicitly out of scope", "X will not be
    included", "X is not permitted") is ALSO a requirement (policy/constraint) -- it
    constrains what may happen, it is not a neutral fact.
    Distinguish a requirement from a mere recommendation ("should consider", "could",
    "might want to") -- a recommendation alone is not a requirement.
    requirement_kind must be exactly one of "policy" | "process_step" | "commitment".

  "event" -- something that occurs or is scheduled to occur, at a point in time or over an
    interval. Distinguish the event itself from mere discussion ABOUT an event.

  "fact" -- a declarative statement that isn't primarily a decision/requirement/event.
    Use this for simple, explicit, durable information too (a contact number, a name, a
    stated value) -- terse or informal phrasing does NOT make something low-confidence or
    unworthy of extraction. "88994448877 this is HR contact" is a clear, explicit,
    HIGH-confidence fact despite its terse phrasing -- confidence reflects how EXPLICITLY
    the text states something, not how formal or grammatically complete the sentence is.

ANTI-OVER-EXTRACTION RULE (mandatory): do not split ONE underlying claim into multiple
primitives that restate the same thing at different granularity. A single sentence
mentioning one plan/date once must produce AT MOST ONE primitive about it -- e.g. "the
release target is September 12" is ONE primitive (a fact, since it's hedged -- see below),
never both a "decision" and an "event" for the same date. Only produce multiple primitives
when the text contains genuinely SEPARATE, independently-true claims (e.g. two different
decisions, or a decision plus an unrelated fact elsewhere in the text).

QUALIFIER RULE (mandatory, the most common failure mode -- read carefully): if the text
uses ANY hedging/uncertainty word about a date or a choice -- "target", "targeted",
"proposed", "tentative", "expected", "planned", "aim", "aiming", "hope", "may", "might",
"could", "roughly", "approximately", "around" -- then:
  1. effective_from/effective_until for that primitive MUST be null. A "target" date is
     NOT a committed effective date, ever, even though the text does mention a date.
  2. has_qualifier MUST be true, and qualifier_words MUST include the actual hedge word(s).
  3. Prefer extracting it as "fact" (a statement that a target/plan exists) rather than
     "decision" or a hard-dated "requirement" -- a hedged target is not a settled decision.
Example: "The release target is September 12." -> ONE "fact" primitive, statement mentions
September 12, effective_from=null, has_qualifier=true, qualifier_words=["target"]. This is
NOT a decision and NOT an event with a committed date.

RELATIVE DATE RESOLUTION (deterministic convention -- follow exactly, do not improvise):
  - "today" -> REFERENCE_DATE's own calendar date.
  - "tomorrow" -> REFERENCE_DATE + 1 calendar day. This is always resolvable.
  - A specific weekday with no other qualifier ("Monday", "next Monday") -> the next
    occurrence of that weekday strictly after REFERENCE_DATE. Resolvable.
  - A month + day with no year ("September 15") -> the next occurrence of that date on or
    after REFERENCE_DATE (use next year only if that month/day has already passed this
    year relative to REFERENCE_DATE). Resolvable, provided it is not ALSO hedged (see
    QUALIFIER RULE above -- a hedged date stays null regardless of this rule).
  - "next week" or "next month" ALONE, with NO specific day/date also named -> this is NOT
    resolvable to one exact date under this convention. effective_from MUST be null. Do
    NOT guess a specific date (never assume "+7 days" or any other arithmetic shortcut) --
    instead keep the phrase itself in the statement text so the information isn't lost,
    just not falsely precise.
  - Any other vague or open-ended relative phrase -> same treatment as "next week": null
    date, phrase preserved in the statement.

RECURRING TIME PATTERNS ("every Monday", "every Monday by 11 AM", "weekly"): these describe
a RECURRENCE, not a single effective date. Do NOT invent a single effective_from for a
recurring pattern. Instead: set "recurrence_text" to the exact recurring-time phrase as
stated (e.g. "every Monday by 11 AM"), and leave effective_from/effective_until null unless
the text ALSO separately states a specific one-time start date for when the recurrence
itself begins.

For each primitive produce exactly these fields:
  "type": one of the four above
  "requirement_kind": ONLY when type == "requirement", else omit or null
  "statement": a clean, standalone, third-person restatement of the primitive -- preserve
    qualifier words and any unresolved relative-date phrases VERBATIM
  "raw_subject_phrase": the literal text naming who/what this is about -- copy the phrase AS
    STATED. NEVER resolve, guess, or normalize this into a specific person/team/product
    identity.
  "effective_from" / "effective_until": ISO date (YYYY-MM-DD) ONLY under the rules above.
    Null in every other case -- absence, hedging, or unresolvable relative phrasing.
  "has_qualifier": true if ANY hedging/uncertainty language appears in the source about THIS
    specific primitive.
  "qualifier_words": the exact hedging words found (list of strings), empty list if none.
  "recurrence_text": the exact recurring-time phrase if this primitive describes a
    recurring pattern, else null.
  "confidence": "high" | "medium" | "low" -- how EXPLICITLY the source text supports this
    exact extraction. Terse/informal phrasing of an explicit fact is HIGH confidence, not
    low. Use "low" only when you are inferring beyond what's explicitly stated.

Rules, all mandatory:
- If nothing in the text is a genuine decision/requirement/event/fact worth structuring,
  return an empty list. Never force an extraction to fill a type slot.
- NEVER invent a date. Absence, hedging, or unresolvable relative phrasing always means
  null, never a guess -- including never applying arithmetic shortcuts like "next week"
  = "+7 days".
- NEVER invent, resolve, or normalize a subject/entity. raw_subject_phrase is always the raw
  text, never an ID, never a canonicalized name.
- NEVER merge or compare this text against any other source -- you only ever see ONE piece
  of knowledge at a time; do not assume or reference other messages/conflicting information.
- Do not split one underlying claim into multiple primitives (see ANTI-OVER-EXTRACTION RULE).

Respond ONLY with valid JSON, no markdown fences:
{"primitives": [ { ... }, ... ]}
If nothing worth extracting: {"primitives": []}"""


def _valid_iso_date(value) -> Optional[str]:
    """VALIDATION, not trust -- mirrors classify_batch's own index-bounds
    validation pattern. A malformed or non-ISO date from the model is
    discarded (null), never passed through as if it were real."""
    if isinstance(value, str) and _ISO_DATE_RE.match(value):
        return value
    return None


# V2 real-benchmark finding #5: "next week"/"next month" resolution must
# never rely on an implicit model arithmetic shortcut (the V1 benchmark
# produced an unexplained, seemingly arbitrary date for this exact phrase).
# This is a CODE-level backstop, not just a prompt instruction: if the
# source text itself contains one of these inherently ambiguous phrases,
# any date the model attributes to that primitive is discarded regardless
# of what the model returned -- the prompt's deterministic convention
# already tells the model to leave these null itself, this just makes that
# guarantee real rather than hoped-for.
_AMBIGUOUS_RELATIVE_RE = re.compile(r"\bnext\s+(week|month)\b", re.IGNORECASE)


# V2.1 real-benchmark finding: the model can classify a primitive correctly
# (e.g. "fact", not "decision") while still silently failing to report a
# qualifier that is literally present in the source text. This vocabulary
# and regex are the deterministic backstop -- exact terms only, never a
# fuzzy/semantic match, so this can never claim a qualifier is present that
# isn't literally in the text.
_QUALIFIER_VOCABULARY = [
    "suggests", "suggested", "suggestion",
    "proposes", "proposed",
    "may", "might",
    "target", "targeted",
    "planned", "tentative", "expected",
    "considering", "potential",
]
_SOURCE_QUALIFIER_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _QUALIFIER_VOCABULARY) + r")\b",
    re.IGNORECASE,
)


def _extract_source_qualifiers(text: str) -> list[str]:
    """Deterministic, rule-based qualifier detection against the ORIGINAL
    canonical source text -- never the model's paraphrased statement, and
    never an inference. Exact vocabulary, word-boundary matched,
    case-insensitive; returns lowercase tokens, de-duplicated, in the order
    first encountered. Never used to change primitive type, never used to
    invent a date -- see extract_primitives_from_canonical() for how this
    is merged (additively only) into the model's own qualifier_words."""
    if not text:
        return []
    seen: list[str] = []
    for match in _SOURCE_QUALIFIER_RE.findall(text):
        low = match.lower()
        if low not in seen:
            seen.append(low)
    return seen


def extract_primitives_from_canonical(
    item: canonical.CanonicalKnowledge,
    min_confidence: str = "medium",
) -> list[ExtractedPrimitive]:
    """
    The one real LLM call this module makes. Returns [] on any failure,
    malformed response, or when the model itself finds nothing worth
    extracting -- never guesses, never raises into the caller.

    min_confidence: items below this confidence are dropped BEFORE being
    returned -- this is "return NO extraction when evidence is
    insufficient" enforced as code. Confidence itself is never attached to
    the returned ExtractedPrimitive (see module docstring).
    """
    if not item.content or not item.content.strip():
        return []

    # event_time (when the source content was actually SAID/happened) is the
    # correct anchor for resolving relative dates in the text -- "starting
    # next week" means next week relative to when it was said, not relative
    # to whenever KNOVA happened to capture/store it. captured_at is only a
    # fallback for sources with no real event_time (e.g. bot_learning).
    reference_date = item.event_time or item.captured_at or ""
    try:
        verdict = ai.chat_json(
            messages=[{"role": "user",
                       "content": f"REFERENCE_DATE: {reference_date}\n\nTEXT:\n{item.content}"}],
            system=EXTRACT_SYSTEM, max_tokens=1200, temperature=0.1,
            workspace_id=item.workspace_id, feature="phase4_extraction_prototype",
        )
    except Exception as e:
        print(f"[structured_extraction] extraction call failed (non-fatal, returns []): {e}")
        return []

    if not isinstance(verdict, dict):
        return []
    raw_items = verdict.get("primitives")
    if not isinstance(raw_items, list):
        return []

    ambiguous_relative_phrase = bool(_AMBIGUOUS_RELATIVE_RE.search(item.content))
    source_qualifiers = _extract_source_qualifiers(item.content)

    min_rank = _MIN_CONFIDENCE_RANK.get(min_confidence, 1)
    evidence_link = (item.provenance[0].permalink if item.provenance else None) or item.id

    results: list[ExtractedPrimitive] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        ptype = raw.get("type")
        if ptype not in VALID_TYPES:
            continue  # unrecognized type -- drop this item, never guess a type

        statement = raw.get("statement")
        if not statement or not isinstance(statement, str):
            continue  # no genuine content -- nothing to structure

        confidence = raw.get("confidence")
        if _MIN_CONFIDENCE_RANK.get(confidence, -1) < min_rank:
            continue  # insufficient evidence -- dropped, never returned

        requirement_kind = None
        if ptype == "requirement":
            candidate_kind = raw.get("requirement_kind")
            if candidate_kind in VALID_REQUIREMENT_KINDS:
                requirement_kind = candidate_kind
            # else: leave None rather than invent/guess a sub-type the
            # model didn't clearly assert -- still a valid requirement,
            # just with an unresolved kind.

        model_qualifiers = raw.get("qualifier_words")
        model_qualifiers = [str(q) for q in model_qualifiers] if isinstance(model_qualifiers, list) else []
        # V2.1: merge in deterministic source-text qualifiers -- ADDITIVE
        # only, never removes anything the model reported (rule 3), never
        # changes `ptype`/statement (rule 6). Case-insensitive de-dupe so
        # the model's own casing wins if it already reported the same word.
        qualifiers = list(model_qualifiers)
        existing_lower = {q.lower() for q in model_qualifiers}
        for term in source_qualifiers:
            if term not in existing_lower:
                qualifiers.append(term)
                existing_lower.add(term)
        has_qualifier = bool(raw.get("has_qualifier")) or bool(qualifiers)

        effective_from = _valid_iso_date(raw.get("effective_from"))
        effective_until = _valid_iso_date(raw.get("effective_until"))
        # V2 DEFENSE IN DEPTH (real benchmark finding #1): the model itself
        # flagging has_qualifier=true (or listing qualifier words) while
        # ALSO returning a date is a self-contradiction -- never trust the
        # date in that case, regardless of what the prompt asked for. A
        # code-level guard is strictly more reliable than a prompt
        # instruction alone.
        if has_qualifier:
            effective_from = None
            effective_until = None

        recurrence_text = raw.get("recurrence_text")
        recurrence_text = recurrence_text if isinstance(recurrence_text, str) and recurrence_text.strip() else None
        if recurrence_text:
            # A recurring pattern has no single effective date by
            # definition -- same defense-in-depth reasoning as above.
            effective_from = None
            effective_until = None

        if ambiguous_relative_phrase:
            # See _AMBIGUOUS_RELATIVE_RE's docstring -- "next week"/"next
            # month" with no further specificity is never resolved to an
            # exact date, regardless of what the model returned.
            effective_from = None
            effective_until = None

        results.append(ExtractedPrimitive(
            type=ptype,
            canonical_id=item.id,
            workspace_id=item.workspace_id,
            statement=statement,
            source_evidence_link=evidence_link,
            sensitivity=item.sensitivity,
            authority=item.authority,
            source_tier=item.source_tier,
            lifecycle_status=item.lifecycle_status,
            captured_at=item.captured_at,
            event_time=item.event_time,
            event_start=item.event_start,
            event_end=item.event_end,
            effective_from=effective_from,
            effective_until=effective_until,
            raw_subject_phrase=raw.get("raw_subject_phrase") if isinstance(raw.get("raw_subject_phrase"), str) else None,
            requirement_kind=requirement_kind,
            qualifier_words=qualifiers,
            recurrence_text=recurrence_text,
        ))
    return results


def extract_primitives_for_workspace(
    workspace_id: str,
    sensitivity_ceiling: list[str],
    sources: Optional[list[str]] = None,
    limit: int = 100,
) -> dict:
    """
    Convenience batch entry point: runs extract_primitives_from_canonical()
    over every item get_canonical_knowledge() returns for this workspace.
    Fetches WITH provenance (include_provenance=True) so
    source_evidence_link can be a real permalink where one exists, not just
    the canonical_id fallback.

    Returns {"primitives": [...], "unavailable_sources": {...}} -- the
    latter passed straight through from get_canonical_knowledge() so a
    caller can see e.g. "document" was never even attempted.
    """
    result = canonical.get_canonical_knowledge(
        workspace_id=workspace_id, sensitivity_ceiling=sensitivity_ceiling,
        sources=sources, limit=limit, include_provenance=True,
    )
    primitives: list[ExtractedPrimitive] = []
    for item in result.items:
        primitives.extend(extract_primitives_from_canonical(item))
    return {"primitives": primitives, "unavailable_sources": result.unavailable_sources}
