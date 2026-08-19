"""
Phase 6F -- Company Wiki Prose Generation: an evidence-bounded LLM RENDERER
over Phase 6E's WikiPageModel. No new ontology, no second source of truth,
no Wiki persistence table.

Pipeline (Part 12), strictly one-directional:

    WikiPageModel (already built by wiki_projection.build_page -- NOT called
                    from this module; the caller builds it first)
        -> build_claim_inventory()   deterministic, template-rendered, no LLM
        -> generate_wiki_page()      calls the LLM with ONLY the claims above
        -> validate_rendered_output() deterministic, rejects on any violation
        -> WikiRenderedPage           either the validated LLM prose, or a
                                       deterministic fallback rendering

This module NEVER imports brain_connectors and NEVER calls Supabase, the
graph, memory, or structured_knowledge directly (Part 2) -- wiki_projection
is imported for its dataclasses only (type references), never for build_page
or any other DB-touching function. The WikiPageModel handed in is the ONLY
source of factual input. If a fact isn't already on that page, this module
has no way to look it up, by construction -- not by discipline alone.

THE LLM IS A RENDERER, NOT A RESOLVER. It never decides what a person, a
relationship, or a memory IS -- wiki_projection already decided that,
deterministically, before this module ever runs. This module's only job is
to (a) hand the LLM a fixed, numbered list of pre-verified claims, (b) make
it structurally impossible to cite anything not on that list without being
caught, and (c) fall back to safe, deterministic prose the moment anything
about the model's output can't be verified.

MODEL: reuses ai.chat_json() (AWS Bedrock, Amazon Nova Lite) exactly as
memory_consolidation.py's _llm_classify_contradiction already established --
same fail-safe wrapping convention (catch broad Exception, never raise to
the caller, always produce a working result). No second AI stack. Per
explicit instruction, this pass does NOT touch ANTHROPIC_API_KEY or switch
providers -- see the final report's Model Availability section for why (that
key carries zero credits and is flagged P0-17 pending rotation; it plays no
role here regardless).

CLAIM BOUNDARY ENFORCEMENT, precisely: every claim the LLM may draw from is
ITSELF deterministic template-rendered text (never LLM output) built
straight from a WikiPageModel section item or link -- the LLM never invents
a claim's factual content, only its phrasing/organization. The validator
then checks, all deterministically: (1) every claim_id cited actually exists
in the inventory handed to the model; (2) every paragraph cites at least one
claim_id (the citation contract, Part 5); (3) no forbidden-vocabulary word
(ownership/employment/causality/project-membership) appears in rendered text
unless that same word already appears in one of ITS OWN cited claims' real
text (so honest quotation of real content is never penalized, but genuinely
new vocabulary is); (4) no multi-word Title-Case phrase appears that isn't
already a real label somewhere on the page (a heuristic closed-world entity
check -- documented as imperfect, not a full NER guarantee, in the final
report); (5) the model's echoed temporal_context matches the page's real
one exactly; (6) a paragraph's tokenized content must be MAJORITY grounded
in the union of its own cited claims' text (_content_words/
_MAX_NOVEL_CONTENT_RATIO). Check (6) was added in Phase 6H after an
adversarial validator battery found that (1)-(5) alone let three attacks
through: an invented date, a fabricated-but-plausible policy requirement,
and a fabricated restricted-sounding fact -- each cited a REAL claim_id and
used no forbidden word and no unrecognized proper noun, yet stated content
entirely absent from what that claim actually says. This exact gap had
already been named, but left open, in the Phase 6F report's "anything not
verified" section; Phase 6H's battery made it concrete rather than
theoretical, so it was closed rather than re-deferred. Any single violation
rejects the WHOLE page's LLM output and falls back to deterministic
rendering -- never a partial accept, never a
silent pass-through of unsupported prose (Part 12).
"""
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import ai
from wiki_projection import WikiPageModel, WikiSection, WikiLink

_ENTITY_PAGE_TYPES = ("person", "department", "meeting")
_MEMORY_PAGE_TYPES = ("policy", "process", "decision")

# The 4 frozen promotion bases (memory_consolidation.PROMOTION_BASES),
# translated into restrained human language ONLY because the PageModel's
# own identity item already supplies promotion_basis literally (Part 8) --
# this dict never adds a fact, it only relabels one that's already there.
_PROMOTION_BASIS_PHRASES = {
    "authoritative_policy": "an officially classified policy",
    "recurring_durable_process": "a recurring, durable operational process",
    "cross_source_corroboration": "corroborated by multiple independent sources",
    "explicit_user_keep": "explicitly marked for retention by a user",
}

# The real, frozen V1 relationship_type vocabulary (Phase 5C) gets slightly
# nicer phrasing; anything else (a future relationship_type this module has
# never seen) falls back to a safe, literal underscore-to-space rendering --
# never blocks on an unrecognized real relationship_type.
_RELATIONSHIP_PHRASES = {
    "attended": "attended",
    "organized": "organized",
    "requires_approval_from": "requires approval from",
}

# Forbidden semantics (Part 3's NO list): ownership, employment, causality,
# project-membership. Word-boundary regexes, not substrings -- a naive
# substring check on "own " would false-positive inside "shown ". A match is
# only a violation if the SAME word does not already appear in the specific
# claims that paragraph cited (see _forbidden_vocab_violations) -- honest
# quotation of real claim text is never penalized.
_FORBIDDEN_VOCAB_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\bowns\b", r"\bown\b", r"\bowned\b", r"\bownership\b",
    r"\bemploys\b", r"\bemployee\b", r"\bemployed\b", r"\bemployment\b", r"\bworks for\b", r"\bhired\b",
    r"\bmember of\b", r"\bmembership\b",
    r"\bmanages\b", r"\bmanager of\b", r"\breports to\b", r"\bsupervises\b",
    r"\bbecause of\b", r"\bcaused\b", r"\bcauses\b", r"\bdue to\b", r"\bled to\b", r"\bresulted in\b",
    r"\bworks on\b", r"\bassigned to\b", r"\bproject member\b",
)]

_CAP_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+(?:\s+\d+)?\b")

# Phase 6H Part 6 -- found via the adversarial validator battery: citing a
# REAL claim_id was previously sufficient to pass, regardless of whether the
# paragraph's actual content had anything to do with that claim's text (a
# gap already named, but left open, in the Phase 6F report's own "anything
# not verified" section -- now concretely demonstrated, not theoretical).
# "All credentials must be rotated every 30 days" cites a real claim_id and
# contains no forbidden word and no unrecognized proper noun, yet states an
# entirely fabricated policy requirement. _content_words/_MAX_NOVEL_CONTENT_
# RATIO close this: a paragraph's tokenized content must be MAJORITY
# grounded in the union of its own cited claims' text. Legitimate
# summarizing/combining naturally reuses most of the same content words
# (verified against the compliant-mock 9-real-page benchmark and a
# multi-claim-combination case); a fabricated fact does not.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "of", "in", "on",
    "for", "and", "or", "to", "with", "by", "from", "at", "also", "which",
    "who", "has", "have", "had", "not",
})
_MAX_NOVEL_CONTENT_RATIO = 0.5


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}

_SYSTEM_PROMPT = """You are a restrained, factual renderer for an internal Company Wiki. You turn a fixed list of pre-verified claims into readable prose. You are a RENDERER, not a research assistant -- you never add information, you only reorganize and smooth the claims you are given.

You MAY:
- summarize and combine the given claims;
- improve readability and organize claims into paragraphs;
- write a concise introduction using only the given claims.

You MUST NOT:
- add a new person, entity, or relationship not in the given claims;
- infer ownership, employment, causality, or project membership;
- infer facts from general world knowledge;
- resolve an ambiguous entity;
- state a date or temporal frame other than what a claim already gives you;
- invent a citation.

Respond with ONLY a JSON object of this exact shape:
{
  "temporal_context_echo": "<echo the temporal_context value you were given, exactly, character for character>",
  "paragraphs": [
    {"text": "<one or more sentences>", "claim_ids": ["<a claim_id from the list you were given>", "..."]}
  ]
}

Every paragraph's claim_ids must be non-empty and must only contain claim_ids that were given to you. Do not include markdown fences, do not include any text outside the JSON object."""


# =====================================================================
# Part 4/17 -- claim inventory and output contracts.
# =====================================================================

@dataclass
class Claim:
    claim_id: str
    claim_type: str            # 'identity' | 'relationship' | 'relationship_absence'
    text: str                  # deterministic, template-rendered -- never LLM output
    evidence_refs: list[str]
    sensitivity: Optional[str]
    temporal_context: str


@dataclass
class WikiRenderedPage:
    page_id: str
    page_type: str
    title: str
    rendered_content: str
    sections: list                 # unchanged pass-through of the source WikiPageModel's sections
    citations: list[dict]          # [{claim_id, claim_type, evidence_refs}] for every claim actually used
    links: list                    # unchanged pass-through
    temporal_context: str
    content_hash: str              # the SOURCE WikiPageModel's own content_hash, unchanged -- rendering
                                    # doesn't change the underlying facts, so no second hash concept is
                                    # introduced. NOTE: rendered_content itself is NOT guaranteed stable
                                    # across calls (LLM output varies); content_hash certifies the FACTS
                                    # are unchanged, never the prose.
    generation_metadata: dict = field(default_factory=dict)


def _format_as_of(temporal_context: str) -> str:
    """Deterministic date formatting -- the LLM never formats a date itself,
    it only relays a string this module already computed. strftime's '%-d'/
    '%#d' (no leading zero) are platform-specific (glibc/MSVC only) and
    would raise on other runtimes, so the day is interpolated manually."""
    if temporal_context == "current":
        return "now"
    dt = datetime.fromisoformat(temporal_context.replace("Z", "+00:00"))
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def _relationship_phrase(relationship_type: str) -> str:
    return _RELATIONSHIP_PHRASES.get(relationship_type, relationship_type.replace("_", " "))


def _identity_claim(item: dict, page: WikiPageModel) -> Claim:
    if page.page_type in _ENTITY_PAGE_TYPES:
        text = f"{item['canonical_label']} is a {item['entity_type']} recorded in the knowledge graph (status: {item['status']})."
        refs = [f"entity:{item['id']}"]
    else:
        basis_phrase = _PROMOTION_BASIS_PHRASES.get(item["promotion_basis"], item["promotion_basis"].replace("_", " "))
        lifecycle_note = "" if item["lifecycle_status"] == "active" else f" Its current status is {item['lifecycle_status']}."
        text = (f"This {item['memory_type']} was retained as durable organizational knowledge "
                f"because it is {basis_phrase}.{lifecycle_note}")
        refs = [f"memory:{item['id']}"]
    return Claim(claim_id="identity:0", claim_type="identity", text=text,
                 evidence_refs=refs, sensitivity=item.get("sensitivity"),
                 temporal_context=page.temporal_context)


def _relationship_claim(index: int, item: dict, page: WikiPageModel, subject_label: str) -> Claim:
    phrase = _relationship_phrase(item["relationship_type"])
    if item["direction"] == "outbound":
        source_label, target_label = subject_label, item["counterpart_label"]
    else:
        source_label, target_label = item["counterpart_label"], subject_label
    fact = f"{source_label} {phrase} {target_label}."
    # Part 7's own historical example prefixes the fact with the as-of point
    # ("As of September 16, 2026, Product was required to..."); current-page
    # facts state directly, matching the CURRENT example's plain phrasing.
    text = fact if page.temporal_context == "current" else f"As of {_format_as_of(page.temporal_context)}, {fact}"
    return Claim(claim_id=f"relationship:{index}", claim_type="relationship", text=text,
                 evidence_refs=[f"relationship:{item['relationship_id']}"],
                 sensitivity=None, temporal_context=page.temporal_context)


def _statement_claim(item: dict, page: WikiPageModel) -> Claim:
    """Memory pages only. page.title IS the real, specific grounding
    statement (Phase 6E's own title-collision fix made it prefer the
    memory's own statement over its parent note's more generic title) --
    surfacing it as its own claim is what lets the renderer produce Part 6's
    POLICY example ('Production credential changes must be recorded...'),
    not just the meta-level promotion-basis framing _identity_claim already
    covers. No new DB access -- page.title was already real, already
    visibility-checked, already on the PageModel."""
    return Claim(claim_id="statement:0", claim_type="statement", text=page.title,
                 evidence_refs=[f"memory:{item['id']}"], sensitivity=item.get("sensitivity"),
                 temporal_context=page.temporal_context)


def _relationship_absence_claim(page: WikiPageModel, subject_label: str) -> Claim:
    when = "currently" if page.temporal_context == "current" else f"as of {_format_as_of(page.temporal_context)}"
    text = f"{subject_label} {when} has no recorded relationships in the knowledge graph."
    return Claim(claim_id="relationships_empty:0", claim_type="relationship_absence", text=text,
                 evidence_refs=[], sensitivity=None, temporal_context=page.temporal_context)


def build_claim_inventory(page: WikiPageModel) -> list[Claim]:
    """The ONLY factual material the LLM will ever see. Every claim's text
    is built directly from a real WikiPageModel section item -- deterministic
    template rendering, zero LLM involvement, zero DB access (this function
    reads only the `page` object already passed in). Evidence-section items
    are not separately narrated in V1 (no example in the spec narrates raw
    evidence directly; identity+relationship claims already carry the
    substantive content) -- they remain visible in WikiRenderedPage.sections
    unchanged, just not turned into their own prose sentences.

    subject_label is what relationship claims use as "this page's own
    subject" -- page.title itself for entity pages (a clean canonical_label
    like "Operations"), but "This policy"/"This process"/"This decision" for
    memory pages, since a memory page's title IS its full real grounding
    statement (Phase 6E's own title fix) and using that whole sentence as a
    grammatical subject reads badly ("<long statement>. currently has no
    recorded relationships...") -- the statement itself is still fully
    preserved, verbatim, in its own separate claim (_statement_claim)."""
    sections_by_type = {s.section_type: s for s in page.sections}
    claims: list[Claim] = []
    subject_label = page.title

    identity_section = sections_by_type.get("identity")
    if identity_section and identity_section.items:
        item = identity_section.items[0]
        claims.append(_identity_claim(item, page))
        if page.page_type in _MEMORY_PAGE_TYPES:
            claims.append(_statement_claim(item, page))
            subject_label = f"This {item['memory_type']}"

    rel_section = sections_by_type.get("relationships")
    if rel_section is not None:
        if rel_section.items:
            for i, item in enumerate(rel_section.items):
                claims.append(_relationship_claim(i, item, page, subject_label))
        else:
            claims.append(_relationship_absence_claim(page, subject_label))

    return claims


# =====================================================================
# Part 5/6 -- rendering. The LLM receives ONLY the claim texts above.
# =====================================================================

def _build_user_prompt(page: WikiPageModel, claims: list[Claim]) -> str:
    lines = [f"Page type: {page.page_type}", f"Title: {page.title}",
             f"temporal_context: {page.temporal_context}", "", "Claims (cite by claim_id):"]
    for c in claims:
        lines.append(f"[{c.claim_id}] {c.text}")
    lines.append("")
    lines.append(f'Echo "temporal_context_echo": {page.temporal_context!r} exactly in your JSON output.')
    return "\n".join(lines)


def _call_renderer(page: WikiPageModel, claims: list[Claim], user_id: Optional[str], chat_json_fn) -> dict:
    return chat_json_fn(
        messages=[{"role": "user", "content": _build_user_prompt(page, claims)}],
        system=_SYSTEM_PROMPT, max_tokens=800, temperature=0.2,
        workspace_id=page.workspace_id, user_id=user_id, feature="wiki_prose_generation",
    )


# =====================================================================
# Part 12 -- deterministic validation. Any single violation rejects the
# WHOLE page's LLM output -- never a partial accept.
# =====================================================================

def _collect_allowed_labels(page: WikiPageModel) -> set[str]:
    labels = {page.title}
    for section in page.sections:
        for item in section.items:
            for key in ("counterpart_label", "canonical_label"):
                val = item.get(key)
                if val:
                    labels.add(val)
    for link in page.links:
        labels.add(link.label)
    return labels


def validate_rendered_output(raw, claims: list[Claim], page: WikiPageModel) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return False, ["model output is not a JSON object"]

    if raw.get("temporal_context_echo") != page.temporal_context:
        errors.append(f"temporal_context_echo mismatch: expected {page.temporal_context!r}, got {raw.get('temporal_context_echo')!r}")

    paragraphs = raw.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        return False, errors + ["paragraphs missing or empty"]

    claim_by_id = {c.claim_id: c for c in claims}
    allowed_labels = _collect_allowed_labels(page)

    for i, p in enumerate(paragraphs):
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            errors.append(f"paragraph {i}: missing or empty text")
            continue
        claim_ids = p.get("claim_ids")
        if not isinstance(claim_ids, list) or not claim_ids:
            errors.append(f"paragraph {i}: no claim_ids cited (citation contract violation)")
            continue

        cited_claims = []
        for cid in claim_ids:
            if cid not in claim_by_id:
                errors.append(f"paragraph {i}: cites unknown claim_id {cid!r}")
            else:
                cited_claims.append(claim_by_id[cid])
        if not cited_claims:
            continue

        text = p["text"]
        cited_text_blob = " ".join(c.text for c in cited_claims)
        for pattern in _FORBIDDEN_VOCAB_PATTERNS:
            if pattern.search(text) and not pattern.search(cited_text_blob):
                errors.append(f"paragraph {i}: forbidden term {pattern.pattern!r} not present in its cited claims")

        for phrase in _CAP_PHRASE_RE.findall(text):
            if phrase not in allowed_labels:
                errors.append(f"paragraph {i}: unrecognized proper-noun phrase {phrase!r} not on this page")

        paragraph_words = _content_words(text)
        if paragraph_words:
            novel_words = paragraph_words - _content_words(cited_text_blob)
            novel_ratio = len(novel_words) / len(paragraph_words)
            if novel_ratio > _MAX_NOVEL_CONTENT_RATIO:
                errors.append(f"paragraph {i}: content not grounded in its cited claims "
                              f"({novel_ratio:.0%} novel: {sorted(novel_words)})")

    return (len(errors) == 0), errors


# =====================================================================
# Part 13 -- deterministic fallback. Always available, zero LLM
# involvement, reuses the exact same claim texts the LLM would have used.
# =====================================================================

_FALLBACK_ORDER = {"identity": 0, "statement": 1, "relationship": 2, "relationship_absence": 2}


def _deterministic_fallback_content(claims: list[Claim]) -> str:
    ordered = sorted(claims, key=lambda c: _FALLBACK_ORDER.get(c.claim_type, 2))
    return " ".join(c.text for c in ordered)


def _citations_for(claims: list[Claim], claim_ids: Optional[set[str]] = None) -> list[dict]:
    used = claims if claim_ids is None else [c for c in claims if c.claim_id in claim_ids]
    return [{"claim_id": c.claim_id, "claim_type": c.claim_type, "evidence_refs": c.evidence_refs} for c in used]


def _assemble(page: WikiPageModel, claims: list[Claim], rendered_content: str,
              citations: list[dict], rendered_by: str, metadata_extra: dict) -> WikiRenderedPage:
    return WikiRenderedPage(
        page_id=page.page_id, page_type=page.page_type, title=page.title,
        rendered_content=rendered_content, sections=page.sections, citations=citations,
        links=page.links, temporal_context=page.temporal_context, content_hash=page.content_hash,
        generation_metadata={"rendered_by": rendered_by, **metadata_extra},
    )


# =====================================================================
# Part 12 -- the single public entry point.
# =====================================================================

def generate_wiki_page(page: WikiPageModel, user_id: Optional[str] = None,
                        chat_json_fn=ai.chat_json) -> WikiRenderedPage:
    """build_page() -> WikiPageModel -> [this function]. Never touches
    Supabase/graph/memory/structured_knowledge -- `page` is the only source
    of factual input. `chat_json_fn` defaults to the real ai.chat_json
    (Bedrock) and is overridable for tests, matching Part 14's explicit
    instruction to validate with deterministic/mock responses rather than
    fabricating live output when a real model isn't reachable."""
    t0 = time.perf_counter()
    claims = build_claim_inventory(page)
    t_claims_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        raw = _call_renderer(page, claims, user_id, chat_json_fn)
    except Exception as e:
        t_llm_ms = (time.perf_counter() - t1) * 1000
        content = _deterministic_fallback_content(claims)
        return _assemble(page, claims, content, _citations_for(claims), "fallback", {
            "reason": f"llm_unavailable: {type(e).__name__}: {e}",
            "claim_inventory_ms": round(t_claims_ms, 2), "llm_ms": round(t_llm_ms, 2),
            "validation_ms": 0.0, "validation_errors": [],
        })
    t_llm_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    ok, errors = validate_rendered_output(raw, claims, page)
    t_validation_ms = (time.perf_counter() - t2) * 1000

    if not ok:
        content = _deterministic_fallback_content(claims)
        return _assemble(page, claims, content, _citations_for(claims), "fallback", {
            "reason": "validation_failed", "claim_inventory_ms": round(t_claims_ms, 2),
            "llm_ms": round(t_llm_ms, 2), "validation_ms": round(t_validation_ms, 2),
            "validation_errors": errors,
        })

    paragraphs = raw["paragraphs"]
    content = "\n\n".join(p["text"] for p in paragraphs)
    used_claim_ids = {cid for p in paragraphs for cid in p["claim_ids"] if cid in {c.claim_id for c in claims}}
    return _assemble(page, claims, content, _citations_for(claims, used_claim_ids), "llm", {
        "reason": "ok", "claim_inventory_ms": round(t_claims_ms, 2),
        "llm_ms": round(t_llm_ms, 2), "validation_ms": round(t_validation_ms, 2),
        "validation_errors": [],
    })
