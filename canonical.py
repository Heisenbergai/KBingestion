"""
Phase 3 -- Multi-Source Knowledge Normalization: a READ-SIDE canonical
projection over the six existing physical models. This module creates NO new
physical table, performs NO writes, and does NOT change any connector or
retrieval code path -- see the approved Phase 3 design (this session) for the
full rationale behind every decision below.

Physical models this projects FROM (all remain authoritative, unmodified):
  - knowledge_notes          (vector DB) -- Slack / Google Chat / Google Meet / bot_learning
  - knowledge_items          (app DB)    -- uploaded documents
  - calendar_events          (vector DB) -- structured metadata, deliberately unclassified
  - knowledge_note_sources   (vector DB) -- per-message provenance for notes
  - external_references      (vector DB) -- Drive reference pointers (NOT knowledge)
  - document_chunks          (vector DB) -- retrieval substrate only (NOT projected here at all)

CanonicalKnowledge deliberately excludes `confidence`: classify_document()
computes it transiently (ingest.py's CLASSIFY_SYSTEM prompt + parsing) but
NEITHER knowledge_notes NOR knowledge_items persists it, so a read-side
projection over stored rows has nothing real to return for it. Exposing it
anyway would mean inventing a field with no data behind it.

Field-by-field grounding, temporal model, authority/source_tier split, and
lifecycle handling are documented on each function below rather than
repeated here -- see each function's own docstring for the specific
NOT AVAILABLE / DERIVED / SOURCE-SPECIFIC reasoning behind that source's
mapping.

Orchestration note: the note/calendar projections below have thin,
read-only fetch wrappers (project_knowledge_note / project_calendar_event)
because brain_connectors.supabase is a service-role client the rest of this
codebase already trusts for exactly this kind of read. Documents
(knowledge_items) do NOT get an equivalent fetch wrapper here: the only
established read path into the app DB from Python is query.py's
RLS-governed, forwarded-user-token pattern (see query.py's
_fetch_my_restricted_grants) -- there is no service-role read client for
knowledge_items anywhere in this codebase today, and building one is a real
credentials/architecture decision outside "implement the read-side
contract." knowledge_item_to_canonical() itself is still fully implemented
and pure -- any caller that already holds a fetched row (e.g. a future
RLS-scoped API route) can use it directly.
"""
from dataclasses import dataclass, field
from typing import Optional

import brain_connectors as bc


@dataclass
class ProvenanceEvidence:
    """Source-agnostic provenance shape. Physical `knowledge_note_sources`
    columns keep their current Slack-shaped names (channel_id/thread_ts/
    message_ts) -- this dataclass is where the relabeling happens, not the
    table. `participant` is None for every note-sourced row today:
    knowledge_note_sources has no participant/user column at all (confirmed
    against the live schema) -- NOT AVAILABLE, never guessed from raw
    ingest_items text that this layer doesn't have access to."""
    container_ref: Optional[str] = None
    thread_ref: Optional[str] = None
    item_ref: Optional[str] = None
    permalink: Optional[str] = None
    occurred_at: Optional[str] = None
    participant: Optional[str] = None


@dataclass
class ExternalReferenceEvidence:
    """Projection of one external_references row. NOT a CanonicalKnowledge
    instance -- external_references has no content/classification/embedding
    (confirmed against the live schema), it is a pointer, not knowledge."""
    file_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    modified_time: Optional[str] = None
    linked_object_type: str = "knowledge_note"
    linked_object_id: str = ""


@dataclass
class CanonicalKnowledge:
    """The approved Phase 3 contract. `confidence` is deliberately absent
    (see module docstring). `conference_id`/`calendar_event_id` and the
    lifecycle-supersession trio (`effective_from`/`valid_until`/
    `superseded_by`) are additive fields required by this same design pass's
    MEET<->CALENDAR and LIFECYCLE sections -- not present in the initial
    bullet list, included here because both sections explicitly require
    them to be exposed."""
    id: str
    workspace_id: str
    connection_id: Optional[str]
    source: str
    source_type: Optional[str]
    title: str
    content: str
    category: Optional[str]
    sensitivity: Optional[str]
    authority: Optional[str]
    source_tier: Optional[int]
    doc_class: Optional[str]
    lifecycle_status: Optional[str]
    record_status: Optional[str]
    processing_status: Optional[str]
    effective_from: Optional[str]
    valid_until: Optional[str]
    superseded_by: Optional[str]
    provenance: list = field(default_factory=list)
    captured_at: Optional[str] = None
    event_time: Optional[str] = None
    event_start: Optional[str] = None
    event_end: Optional[str] = None
    source_updated_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    external_references: list = field(default_factory=list)
    conference_id: Optional[str] = None
    calendar_event_id: Optional[str] = None


# ── Provenance / reference helpers (pure) ───────────────────────────────────

def _note_source_to_provenance(row: dict) -> ProvenanceEvidence:
    return ProvenanceEvidence(
        container_ref=row.get("channel_id"),
        thread_ref=row.get("thread_ts"),
        item_ref=row.get("message_ts"),
        permalink=row.get("source_ref"),
        occurred_at=row.get("occurred_at"),
        participant=None,
    )


def _document_provenance_anchor(item: dict) -> ProvenanceEvidence:
    """Documents have no knowledge_note_sources-equivalent child table --
    one synthesized anchor from the file's own real identity, not a
    fabricated one. `permalink` stays None: an internal upload has no
    public URL to point to."""
    return ProvenanceEvidence(
        container_ref=item.get("folder_id"),
        thread_ref=None,
        item_ref=item.get("file_name"),
        permalink=None,
        occurred_at=None,
        participant=item.get("uploader_name"),
    )


def _record_status_from_deleted_at(row: dict) -> str:
    """Shared derivation for the two sources (knowledge_items,
    calendar_events) whose real record-usability signal is a `deleted_at`
    timestamp rather than a status string -- knowledge_notes.status is the
    only source with a literal string to pass through instead. This is an
    honest boolean-to-enum derivation from a real column, not an invented
    value: `deleted_at IS NULL` unambiguously means "not deleted"."""
    return "deleted" if row.get("deleted_at") else "active"


def _external_reference_to_evidence(row: dict) -> ExternalReferenceEvidence:
    return ExternalReferenceEvidence(
        file_id=row["external_file_id"],
        title=row.get("title"),
        url=row.get("url"),
        modified_time=row.get("modified_time"),
        linked_object_type=row.get("linked_object_type", "knowledge_note"),
        linked_object_id=row.get("linked_object_id", ""),
    )


# ── Source projections (pure -- take already-fetched rows, no I/O) ─────────

def knowledge_note_to_canonical(note: dict, sources: Optional[list[dict]] = None,
                                external_refs: Optional[list[dict]] = None,
                                calendar_event_id: Optional[str] = None) -> CanonicalKnowledge:
    """knowledge_notes -> CanonicalKnowledge. Covers Slack, Google Chat,
    Google Meet, AND bot_learning -- they share one physical table and one
    write path (brain_connectors.create_note_and_embed), so one projection
    function is correct; there is no per-provider schema difference to
    branch on except how `sources` happens to be populated:
      - Slack/Chat: sources are real per-message rows (channel/thread/ts as
        literally captured).
      - Meet: sources are real per-transcript-entry rows, but channel_id/
        thread_ts/message_ts carry conference_id/transcript_id/entry_id
        instead -- still real data, just relabeled by this function into
        source-agnostic names.
      - bot_learning: create_note_and_embed is called with sources=None
        (see its own docstring), so `sources` is genuinely [] here --
        provenance=[] is the honest result, not a bug, and this function
        does not synthesize a fake anchor from the legacy source_ref column.

    Temporal fields: `captured_at`/`created_at` both come from the note's
    own created_at (there is only one KNOVA-side timestamp on this table).
    `event_time` = the note's own `occurred_at` column, whose real meaning
    varies by provider (message time for Slack/Chat, conference start time
    for Meet, the admin-answer instant for bot_learning) -- never
    substituted from created_at.

    Lifecycle: `effective_from`/`valid_until`/`superseded_by` are always
    None here -- knowledge_notes has no such columns (confirmed against the
    live schema). Never derived from document_chunks' copy of these
    concepts, which belongs to a different feature (document supersession).

    `record_status` (2026-08-17 semantics correction): knowledge_notes' own
    real `status` column, passed through directly -- a row-level usability
    flag (only "active" has ever been observed live, and GET
    /knowledge-notes filters on it directly). Functionally inert today:
    delete_note() hard-deletes rows rather than ever writing a different
    status value, so this column has never been observed to hold anything
    but "active" -- still real, still passed through as-is, never
    fabricated into something more dynamic than it currently is.

    `lifecycle_status` is a SEPARATE content-classification axis (draft/
    active/under_review/superseded/archived) -- conflating it with record
    usability was the exact mistake corrected this pass; they are kept as
    two fully independent fields.

    `processing_status`: always None for notes -- there is no async
    ingestion pipeline here (embedding happens synchronously inside
    create_note_and_embed), so there is nothing for this field to mean for
    this source. Never confused with `record_status`.

    source_tier: passed through as-is (2 or 3 today) -- a real,
    connector-assigned int, independent of `authority` (see this design
    pass's DECISION 2 -- they are never combined here).
    """
    sources = sources or []
    external_refs = external_refs or []
    provider = note.get("provider")

    conference_id = None
    if provider == "google_meet" and sources:
        # See connector_google_meet.py's _process_one_conference: channel_id
        # IS the conference record name for this provider specifically.
        conference_id = sources[0].get("channel_id")

    return CanonicalKnowledge(
        id=note["id"],
        workspace_id=note["workspace_id"],
        connection_id=note.get("connection_id"),
        source=provider,
        source_type=note.get("source_type"),
        title=note.get("title") or "",
        content=note.get("body") or "",
        category=note.get("category"),
        sensitivity=note.get("sensitivity"),
        authority=note.get("authority"),
        source_tier=note.get("source_tier"),
        doc_class=note.get("doc_class"),
        lifecycle_status=note.get("lifecycle_status"),
        record_status=note.get("status"),
        processing_status=None,
        effective_from=None,
        valid_until=None,
        superseded_by=None,
        provenance=[_note_source_to_provenance(s) for s in sources],
        captured_at=note.get("created_at"),
        event_time=note.get("occurred_at"),
        event_start=None,
        event_end=None,
        source_updated_at=None,
        created_at=note.get("created_at"),
        updated_at=None,
        external_references=[_external_reference_to_evidence(r) for r in external_refs],
        conference_id=conference_id,
        calendar_event_id=calendar_event_id if provider == "google_meet" else None,
    )


def knowledge_item_to_canonical(item: dict, external_refs: Optional[list[dict]] = None) -> CanonicalKnowledge:
    """knowledge_items (uploaded documents) -> CanonicalKnowledge.

    `content`: knowledge_items has no body/text column (the real text lives
    across N document_chunks rows, deliberately not rolled up here -- see
    module docstring on why document_chunks stays retrieval-only). Falls
    back through description -> title -> file_name, all real columns, no
    invented prose.

    `connection_id`: always None -- knowledge_items has no connection_id
    column at all (documents are never connector-sourced).

    `source_tier`: always None -- knowledge_items has no such column.

    `category`: always None -- no category concept exists for documents.

    Lifecycle: unlike notes, knowledge_items DOES have the full
    effective_from/valid_until/superseded_by/updated_at set -- passed
    through directly, real data.

    `record_status` (2026-08-17 semantics correction): knowledge_items has
    no `status` column, but it DOES have a real, actively-enforced
    soft-delete signal -- `deleted_at` (GET /document-tables already
    filters on `.is_("deleted_at", "null")`). Derived here as
    `"deleted"` if `deleted_at` is set, else `"active"` -- an honest
    boolean-to-string derivation from a real timestamp, not an invented
    value, and the correct cross-source analog to knowledge_notes.status
    (both answer "is this record currently usable"). The earlier version
    of this function returned `processing_status` here instead, which was
    wrong -- that's an ingestion-pipeline concept, not a record-usability
    one. See `processing_status` below for where that value now lives.

    `processing_status`: knowledge_items' own real `processing_status`
    column (e.g. "completed"/"processing"/"failed") -- an ingestion-
    pipeline state, source-specific, deliberately NOT conflated with
    `record_status` above.

    Temporal: `event_time`/`event_start`/`event_end`/`source_updated_at`
    are all None -- there is no "when did this happen" concept for a static
    upload, and no field anywhere captures the original file's own real
    last-modified time (confirmed: no such field exists in ingest.py).
    `captured_at` = created_at (upload time genuinely IS the capture
    moment for this source -- not a substitution, this is what
    captured_at means).
    """
    external_refs = external_refs or []
    content = item.get("description") or item.get("title") or item.get("file_name") or ""

    return CanonicalKnowledge(
        id=item["id"],
        workspace_id=item.get("workspace_id"),
        connection_id=None,
        source="document",
        source_type=item.get("file_type"),
        title=item.get("title") or "",
        content=content,
        category=None,
        sensitivity=item.get("sensitivity"),
        authority=item.get("authority"),
        source_tier=None,
        doc_class=item.get("doc_class"),
        lifecycle_status=item.get("lifecycle_status"),
        record_status=_record_status_from_deleted_at(item),
        processing_status=item.get("processing_status"),
        effective_from=item.get("effective_from"),
        valid_until=item.get("valid_until"),
        superseded_by=item.get("superseded_by"),
        provenance=[_document_provenance_anchor(item)],
        captured_at=item.get("created_at"),
        event_time=None,
        event_start=None,
        event_end=None,
        source_updated_at=None,
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        external_references=[_external_reference_to_evidence(r) for r in external_refs],
    )


def _calendar_content(event: dict) -> str:
    """calendar_events has no body/description text column at all --
    synthesized purely from real structured fields (title/start/end/
    organizer), never invented prose. The honest alternative to either
    duplicating `title` verbatim as `content` or fabricating a summary."""
    parts = [event.get("title") or "Untitled calendar event"]
    if event.get("start_time"):
        window = event["start_time"]
        if event.get("end_time"):
            window = f"{window} to {event['end_time']}"
        parts.append(window)
    if event.get("organizer"):
        parts.append(f"organized by {event['organizer']}")
    return " — ".join(parts)


def calendar_event_to_canonical(event: dict) -> CanonicalKnowledge:
    """calendar_events -> CanonicalKnowledge.

    sensitivity/authority/doc_class are None per the approved contract
    (Calendar was deliberately kept outside classification -- Google
    Workspace scope lock -- and calendar_events genuinely has no such
    columns). This function extends the same no-fabrication rule to
    `lifecycle_status` too, even though only the first three were named
    explicitly: calendar_events has no lifecycle column either, so
    returning anything but None there would be the same kind of
    fabrication the instruction was written to prevent.

    event_time is None here (not the point-in-time concept) --
    event_start/event_end carry the real interval instead, the one
    genuinely required departure from a single occurred_at field.

    source_updated_at = updated_at_source: the ONE source in this whole
    contract that actually has this field populated. `updated_at` (the
    canonical-row-edit concept) stays None -- calendar_events has no such
    column (only created_at/updated_at_source/deleted_at).

    provenance is always [] -- there is no knowledge_note_sources-equivalent
    child table for Calendar; never fabricated to match other sources'
    shape.

    `record_status` (2026-08-17 semantics correction): calendar_events has
    no `status`/`lifecycle_status` column, but it DOES have a real
    `deleted_at` column (confirmed on the live schema) -- the same
    soft-delete signal knowledge_items has. Derived the same way:
    `"deleted"` if `deleted_at` is set, else `"active"`. This corrects the
    earlier (wrong) assumption that Calendar has no record-usability
    signal at all -- it does, it just wasn't recognized as one until this
    semantics review.

    `processing_status`: always None -- Calendar has no ingestion pipeline
    concept, same reasoning as notes.
    """
    return CanonicalKnowledge(
        id=event["id"],
        workspace_id=event["workspace_id"],
        connection_id=event.get("connection_id"),
        source="calendar",
        source_type="event",
        title=event.get("title") or "",
        content=_calendar_content(event),
        category=None,
        sensitivity=None,
        authority=None,
        source_tier=None,
        doc_class=None,
        lifecycle_status=None,
        record_status=_record_status_from_deleted_at(event),
        processing_status=None,
        effective_from=None,
        valid_until=None,
        superseded_by=None,
        provenance=[],
        captured_at=event.get("created_at"),
        event_time=None,
        event_start=event.get("start_time"),
        event_end=event.get("end_time"),
        source_updated_at=event.get("updated_at_source"),
        created_at=event.get("created_at"),
        updated_at=None,
        external_references=[],
    )


# ── Meet <-> Calendar normalization (exact match only) ──────────────────────

def resolve_meet_calendar_event_id(workspace_id: str, conference_id: Optional[str]) -> Optional[str]:
    """Exact string match on calendar_events.conference_id, scoped to the
    claimed workspace -- no fuzzy matching, no entity resolution, no
    semantic linking. Returns None on a missing conference_id, no match, or
    any lookup failure -- fails safe, never guesses."""
    if not conference_id:
        return None
    try:
        rows = bc.supabase.table("calendar_events").select("id") \
            .eq("workspace_id", workspace_id).eq("conference_id", conference_id) \
            .limit(1).execute().data or []
    except Exception as e:
        print(f"[canonical] calendar_event_id lookup failed (non-fatal, returns None): {e}")
        return None
    return rows[0]["id"] if rows else None


# ── Read-only orchestration wrappers (vector DB only -- see module docstring
#    on why documents deliberately don't get an equivalent wrapper here) ───

def project_knowledge_note(note_id: str) -> Optional[CanonicalKnowledge]:
    """Fetches a real knowledge_notes row plus its real child rows
    (knowledge_note_sources, external_references) and projects it. Returns
    None if the note doesn't exist -- a projection over nothing is nothing,
    not an error."""
    rows = bc.supabase.table("knowledge_notes").select("*").eq("id", note_id).execute().data
    if not rows:
        return None
    note = rows[0]
    sources = bc.supabase.table("knowledge_note_sources").select("*") \
        .eq("note_id", note_id).execute().data or []
    external_refs = bc.supabase.table("external_references").select("*") \
        .eq("linked_object_type", "knowledge_note").eq("linked_object_id", note_id).execute().data or []

    calendar_event_id = None
    if note.get("provider") == "google_meet" and sources:
        conference_id = sources[0].get("channel_id")
        calendar_event_id = resolve_meet_calendar_event_id(note["workspace_id"], conference_id)

    return knowledge_note_to_canonical(note, sources, external_refs, calendar_event_id)


def project_calendar_event(event_id: str) -> Optional[CanonicalKnowledge]:
    """Fetches a real calendar_events row and projects it."""
    rows = bc.supabase.table("calendar_events").select("*").eq("id", event_id).execute().data
    if not rows:
        return None
    return calendar_event_to_canonical(rows[0])


# ── The Phase 3 read-side consumer interface ────────────────────────────────

# Every source get_canonical_knowledge knows how to name. "document" is
# recognized but always reported unavailable today -- see the "document"
# section below and the module docstring on why no service-role read path
# exists for it.
_NOTE_PROVIDER_SOURCES = {"slack", "google_chat", "google_meet", "bot_learning"}
VALID_CANONICAL_SOURCES = _NOTE_PROVIDER_SOURCES | {"calendar", "document"}


@dataclass
class CanonicalKnowledgeResult:
    """Return shape for get_canonical_knowledge(). `unavailable_sources` maps
    a requested source name to WHY it produced no results from a physical
    limitation (never silently dropped, never manufactured as an empty
    result indistinguishable from "this workspace just has none")."""
    items: list = field(default_factory=list)
    unavailable_sources: dict = field(default_factory=dict)


def get_canonical_knowledge(
    workspace_id: str,
    sensitivity_ceiling: list[str],
    sources: Optional[list[str]] = None,
    limit: int = 100,
    include_provenance: bool = False,
    include_external_references: bool = False,
) -> CanonicalKnowledgeResult:
    """
    The Phase 3 read-side consumer interface: normalized CanonicalKnowledge
    items across every source this module can currently project, without the
    caller needing to know which physical table backs any of them.

    SECURITY -- this function enforces (1) and (4) below itself; (2)/(3) are
    a CONTRACT ON THE CALLER, not something this function can verify on its
    own, since it has no auth context of its own:
      1. workspace_id is a REQUIRED, non-empty parameter -- there is no
         "give me everything" mode. Every query below is scoped to it.
      2. sensitivity_ceiling is REQUIRED and non-empty -- there is no
         default/bypass value. Raises ValueError if omitted or empty,
         exactly like workspace_id.
      3. sensitivity_ceiling MUST be resolved by the CALLER from the real
         caller's role/is_super_admin via the SAME
         _resolve_allowed_sensitivities() pattern GET /knowledge-notes and
         GET /document-tables already use -- this function has no signature
         path for a client-supplied override and trusts whatever list it is
         given, exactly as those two routes trust their own server-computed
         ladder. A future HTTP wrapper around this function MUST compute
         sensitivity_ceiling server-side from AuthContext, never read it
         from the request body/query string.
      4. record_status is enforced server-side, at the query level, using
         each source's real underlying signal: notes are filtered
         `status == "active"`, calendar events are filtered
         `deleted_at IS NULL` -- both equivalent to filtering on the
         projected `record_status == "active"`, but applied before the row
         is even fetched rather than discarded after.
      5. RLS: every query here goes through brain_connectors.supabase (the
         vector-DB service-role client already trusted throughout this
         codebase for these exact tables -- see project_knowledge_note/
         project_calendar_event above). No new client, no new credential.
      6. No service-role document read path is added -- see "document"
         handling below.

    sources=None means "every source this function can currently read"
    (VALID_CANONICAL_SOURCES minus "document", which is always reported
    unavailable -- see below). An explicit sources=[] means "read nothing"
    (a caller who explicitly asks for zero sources gets zero results, not
    silently reinterpreted as "give me everything"). Unknown source names
    are reported in unavailable_sources with reason "not a recognized
    source" -- never silently dropped.

    Calendar has no sensitivity concept at all (Google Workspace scope lock
    -- deliberately never classified). Calendar items are therefore NEVER
    filtered by sensitivity_ceiling; they pass through unconditionally once
    workspace-scoped and non-deleted. This is a real, deliberate
    consequence of Calendar's existing design, not something invented by
    this function -- worth knowing before wiring this up to anything that
    assumes every canonical item is confidentiality-gated the same way.

    "document" is always returned in unavailable_sources: there is no
    service-role read path for knowledge_items in this codebase today
    (only query.py's RLS-governed, forwarded-user-token pattern, which
    requires a live caller token this function has no way to obtain on its
    own) -- adding one is explicitly out of scope for this pass. Reported
    honestly rather than silently returning zero document results
    indistinguishable from "this workspace has no documents."

    Provenance/external references are opt-in and fetched per matching note
    (N+1 queries when enabled) -- deliberately simple for a first
    implementation with zero real consumers yet; batching is a legitimate
    future optimization once something actually needs the scale.

    Embeddings are never touched -- document_chunks (the retrieval
    substrate) is not queried anywhere in this function. A caller that
    needs a canonical item's retrieval evidence joins on the item's own
    `id` == document_chunks.document_id itself, exactly as
    create_note_and_embed already establishes that relationship at write
    time -- this function does not duplicate or replace that.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required.")
    if not sensitivity_ceiling:
        raise ValueError(
            "sensitivity_ceiling is required and must be non-empty -- there is no "
            "default/bypass. Resolve it server-side from the caller's real role "
            "via the same _resolve_allowed_sensitivities() pattern GET "
            "/knowledge-notes and GET /document-tables already use."
        )

    requested = set(sources) if sources is not None else set(VALID_CANONICAL_SOURCES)
    unknown = requested - VALID_CANONICAL_SOURCES
    unavailable: dict = {name: "not a recognized source" for name in unknown}
    requested = requested & VALID_CANONICAL_SOURCES

    items: list = []

    note_providers = requested & _NOTE_PROVIDER_SOURCES
    if note_providers:
        note_rows = bc.supabase.table("knowledge_notes").select("*") \
            .eq("workspace_id", workspace_id) \
            .eq("status", "active") \
            .in_("sensitivity", sensitivity_ceiling) \
            .in_("provider", list(note_providers)) \
            .order("created_at", desc=True).limit(limit).execute().data or []

        for note in note_rows:
            note_sources: list[dict] = []
            external_refs: list[dict] = []
            if include_provenance or include_external_references:
                if include_provenance:
                    note_sources = bc.supabase.table("knowledge_note_sources").select("*") \
                        .eq("note_id", note["id"]).execute().data or []
                if include_external_references:
                    external_refs = bc.supabase.table("external_references").select("*") \
                        .eq("linked_object_type", "knowledge_note").eq("linked_object_id", note["id"]) \
                        .execute().data or []

            canonical_item = knowledge_note_to_canonical(note, note_sources, external_refs)
            if note.get("provider") == "google_meet":
                # conference_id/calendar_event_id are CORE fields, not part
                # of the opt-in provenance[] list -- they must be resolved
                # (and set directly on the result) regardless of
                # include_provenance, since knowledge_note_to_canonical's
                # own internal derivation only sees `sources` when
                # include_provenance actually fetched them.
                conf_rows = note_sources or (bc.supabase.table("knowledge_note_sources")
                    .select("channel_id").eq("note_id", note["id"]).limit(1).execute().data or [])
                conference_id = conf_rows[0].get("channel_id") if conf_rows else None
                canonical_item.conference_id = conference_id
                canonical_item.calendar_event_id = resolve_meet_calendar_event_id(
                    workspace_id, conference_id) if conference_id else None

            items.append(canonical_item)

    if "calendar" in requested:
        cal_rows = bc.supabase.table("calendar_events").select("*") \
            .eq("workspace_id", workspace_id) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=True).limit(limit).execute().data or []
        for event in cal_rows:
            items.append(calendar_event_to_canonical(event))

    if "document" in requested:
        unavailable["document"] = (
            "No service-role read path exists for knowledge_items from this "
            "consumer context -- only query.py's RLS-governed, forwarded-"
            "user-token pattern exists today, and this function has no live "
            "caller token to forward. Not fabricated as an empty result; "
            "adding new credentials is explicitly out of scope for this pass."
        )

    items.sort(key=lambda it: it.created_at or "", reverse=True)
    return CanonicalKnowledgeResult(items=items[:limit], unavailable_sources=unavailable)
