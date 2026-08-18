"""
Signal collection (R-D) — the raw material a future personalisation phase
would learn from. DELIBERATELY DARK: nothing reads this table back yet, no
"personalised for you" surface exists anywhere in the app. This phase is
schema and write paths only, exactly as scoped and locked with Tanmay.

WHY NOW, EVEN THOUGH NOTHING CONSUMES IT YET (risk R6). 30 usage events and no
connector has ever touched a real account — there is genuinely nothing to
learn a pattern FROM today. Building the collection now means real signal
accumulates from this point forward rather than needing a second retrofit
later; reading it is deferred until there is real volume to read.

HONESTY ABOUT SIGNAL STRENGTH — the labels matter. 'source_cited' (AI Search)
and 'source_used_in_context' (bots) are NOT the same strength of signal and
must never be merged under one name: a citation is the model choosing to
attribute a claim to a document; "used in context" only means the chunk was
RETRIEVED and shown to the model, which is a much weaker proxy for relevance.
Mislabeling a passive, automatic signal as if it were genuine user feedback
would corrupt anything built on top of this table later — the R1 constraint
this whole thread is built around ("readable only by the person it's about")
matters precisely because this data will eventually describe how someone
works; it has to describe that HONESTLY from day one, not retroactively.

FORWARD OBLIGATION, not yet built: the eventual read path for this data (R-E)
MUST enforce "only the user themselves, never owner or God tier" — see R1 in
reasoning-and-health.md. This table's RLS (enabled, zero policies) means
nobody can read it via PostgREST today regardless; that is a byproduct of the
same posture every sibling telemetry table uses, not itself the R1 guarantee.
Whoever builds the read endpoint must add that check explicitly.
"""
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client
import os

from auth import AuthContext, current_user

router = APIRouter()

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE,
)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


def log_signal(workspace_id: str, user_id: Optional[str], feature: str, signal_type: str,
               question: Optional[str] = None, document_id: Optional[str] = None,
               metadata: Optional[dict] = None) -> None:
    """
    Best-effort, fail-safe — same convention as ai._log_usage(): a logging
    failure must NEVER break the actual query/chat response that triggered it.
    Every exception is swallowed here, never raised to the caller.

    user_id may be None (e.g. an anonymous widget visitor) — logged anyway
    with a NULL user_id rather than dropped, matching ai._log_usage()'s same
    reasoning for workspace_id: the signal still happened, it just won't
    attribute to any one person.
    """
    try:
        supabase.table("user_signals").insert({
            "workspace_id": workspace_id,
            "user_id":      user_id,
            "feature":      feature,
            "signal_type":  signal_type,
            "question":     (question or None) and question[:500],
            "document_id":  document_id,
            "metadata":     metadata or {},
        }).execute()
    except Exception as e:
        print(f"[signals] log_signal failed, non-fatal ({signal_type}): {e}")


def log_scope_used(workspace_id: str, user_id: Optional[str], feature: str,
                   question: str, document_ids: list[str]) -> None:
    """A query was scoped to specific documents rather than the whole workspace."""
    if not document_ids:
        return
    log_signal(workspace_id, user_id, feature, "scope_used", question=question,
               metadata={"document_ids": document_ids})


def log_sources_cited(workspace_id: str, user_id: Optional[str], feature: str,
                      question: str, document_ids: list[str], confidence: str) -> None:
    """AI Search's model chose to attribute a claim to these documents.

    Phase 5J/5K: a graph_retrieval-produced candidate carries a pseudo id
    like "graph_relationship:<uuid>" in its document_id field (it isn't a
    real document -- it's a relationship, deliberately namespaced so it can
    never collide with a real document_id, see graph_retrieval.py). This
    table's document_id column is a real uuid-typed column, so passing that
    string through used to fail on every write (found live via the Phase
    5K benchmark) -- silently, since log_signal is fail-safe by design.
    Skipped here rather than logged wrong or crashing: a graph relationship
    isn't a document, so "not logged as a cited document" is the honest
    behavior, not a bug to paper over with a schema change."""
    for doc_id in dict.fromkeys(document_ids):  # de-dupe, preserve order
        if doc_id and _UUID_RE.match(doc_id):
            log_signal(workspace_id, user_id, feature, "source_cited", question=question,
                       document_id=doc_id, metadata={"confidence": confidence})


def log_sources_used_in_context(workspace_id: str, user_id: Optional[str], feature: str,
                                question: str, document_ids: list[str], confidence: str) -> None:
    """
    A bot's context included these documents' chunks — weaker than a citation
    (see the module docstring). Never call this the same signal_type as
    log_sources_cited.

    Same non-UUID skip as log_sources_cited above, same reason (graph
    candidates are not documents).
    """
    for doc_id in dict.fromkeys(document_ids):
        if doc_id and _UUID_RE.match(doc_id):
            log_signal(workspace_id, user_id, feature, "source_used_in_context",
                       question=question, document_id=doc_id, metadata={"confidence": confidence})


class FeedbackRequest(BaseModel):
    workspace_id: str
    feature:      str
    question:     Optional[str] = None
    document_id:  Optional[str] = None
    helpful:      bool


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, auth: AuthContext = Depends(current_user)):
    """
    Explicit user feedback — the ONLY genuinely active signal in this phase,
    everything else above is passive/automatic. Exists and is fully testable
    today; nothing in the frontend calls it yet, deliberately. Per the P0 rule,
    the user-visible surface (whatever "mark this helpful" actually looks like
    — thumbs, a star, something else) ships LAST and alone, once its own
    design is decided, not guessed at inside a backend-only phase.
    """
    if not body.workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required.")
    auth.assert_workspace(body.workspace_id)

    log_signal(
        body.workspace_id, auth.user_id, body.feature,
        "feedback_helpful" if body.helpful else "feedback_not_helpful",
        question=body.question, document_id=body.document_id,
    )
    return {"success": True}
