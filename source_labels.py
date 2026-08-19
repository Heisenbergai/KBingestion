"""
Shared human-readable presentation labels for `source_type` values that flow
through retrieval citations (document_chunks.source_type) and knowledge
notes (knowledge_notes.source_type).

2026-08-17 fix: query.py and chatbot.py each maintained their OWN inline
label dict, independently, with drifted wording ("company document" vs
"official document") and both missing "google_chat"/"google_meet" entirely
-- so a real Chat/Meet-sourced citation showed the raw internal string
("google_chat") as its label instead of a human-readable one. This module
is the single source of truth both call sites now use; there is exactly one
place to add a new source_type's label going forward.

This is presentation only -- it never touches the canonical/stored
source_type value itself, which remains the machine-readable identifier
used for filtering, classification, and joins everywhere else in the app.
"""
from typing import Optional

SOURCE_TYPE_LABELS: dict[str, str] = {
    "document":     "Company document",
    "meeting":      "Meeting note",
    "slack":        "Team chat",
    "google_chat":  "Google Chat",
    "google_meet":  "Google Meet",
    "note":         "Curated note",
    "bot_learning": "Curated knowledge",
    # Phase 5J: a graph_retrieval-produced candidate -- a real relationship
    # plus its real evidence, never a document of its own. Labeled distinctly
    # so a citation is honest about where it came from (Part 9: "the graph is
    # never the final source authority").
    "graph_relationship": "Knowledge graph",
    # Phase 6D: a memory_retrieval-produced candidate -- a durable
    # organizational memory plus its real grounding evidence, never a
    # document of its own (org_memory stores no statement, see
    # memory_retrieval.py's module docstring). Labeled distinctly so a
    # citation is honest that this is KNOVA's own durable interpretation,
    # not a fresh document -- the underlying evidence remains the source
    # of truth, cited inside the candidate's own content.
    "org_memory": "Durable memory",
}


def source_type_label(source_type: Optional[str]) -> str:
    """Falls back to the raw source_type string itself for any value not in
    the map (e.g. a future source_type added elsewhere before this map is
    updated) -- never raises, never returns a blank label. Callers are
    still responsible for resolving which field actually holds the
    source_type value on their own chunk/row shape (e.g. `source_type` vs
    `metadata.source_type` vs a "document" default) -- that resolution is
    call-site-specific and stays at each call site, not here."""
    if not source_type:
        return SOURCE_TYPE_LABELS["document"]
    return SOURCE_TYPE_LABELS.get(source_type, source_type)
