"""
Bot learning — the escalation queue an admin works from when a bot can't
answer something. chatbot.py's run_rag_query/log_usage_event decide WHEN to
escalate (confidence "none"/"low", reusing query.py's exact thresholds); this
file is what an admin DOES about an escalated question once it's in the queue.

Flow:
  1. GET  /bot-unanswered-questions            — list pending/answered/
     transferred questions for a workspace (optionally filtered by bot/status).
  2. POST /bot-unanswered-questions/{id}/answer   — admin types an answer. It
     becomes a real knowledge_note via brain_connectors.create_note_and_embed,
     embedded and immediately searchable by the SAME bot that asked — even if
     that bot is folder-scoped, because extra_metadata={"learned_for_bot_id"}
     is exactly what chatbot.py's folder filter special-cases (see its
     run_rag_query comment). This is the literal "the bot learns" mechanism.
  3. POST /bot-unanswered-questions/{id}/resolve  — used after the admin
     answers by uploading a document instead, through the Library's EXISTING
     upload-to-folder flow (frontend locks the folder picker to the bot's own
     linked folder so it's automatically in-scope once uploaded). No ingestion
     logic is duplicated here — this just records which document answered it.
  4. POST /bot-unanswered-questions/{id}/transfer — assigns the question to a
     specific workspace member, and appends a row to bot_question_assignments
     (append-only history — transferred_to_user_id on the question itself is
     only ever a fast "current assignee" cache, so reassigning never destroys
     who had it before). No email/notification: there is no SMTP in this
     project yet (see 12_go_live_plan.md). The assignee finds it via an
     "assigned to me" filter against the same list endpoint, and can now
     actually answer it (see _require_admin_or_assignee) — RBAC rollout Phase
     3 fixed a real gap where transferring a question to a regular employee
     gave them no endpoint that let them act on it.
  5. GET /user-activity/{user_id} — RBAC rollout Phase 5's User Activity
     Profile, vector-DB half: this specific user's bot query counts (from
     bot_usage_events) and their unanswered-question assignment history (from
     bot_question_assignments, joined against bot_unanswered_questions for
     current status). Admin/owner-only. The app-DB half (training assignment/
     progress, presentation shares) is queried directly by the frontend.

All mutate workspace-scoped tables with no permissive RLS policies (see the
bot_learning_question_logging_and_escalation migration and
add_bot_question_assignments_history) — only this service-role client can
touch them, same posture as bot_usage_events/provider_credentials/connections.
"""
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

from auth import AuthContext, current_user
import brain_connectors as bc

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


def _require_admin(auth: AuthContext, workspace_id: str) -> None:
    """
    Same bar as integrations.py's _require_admin: owner/admin only, super
    admins bypass. Answering/transferring on the workspace's behalf is an
    admin action, not something every member should be able to do.
    """
    if auth.is_super_admin:
        return
    if auth.role_in(workspace_id) not in ("owner", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Only workspace owners/admins can manage the unanswered-questions queue.",
        )


def _require_admin_or_assignee(auth: AuthContext, workspace_id: str, question: dict) -> None:
    """
    Relaxed bar for /answer only (RBAC rollout Phase 3): an admin can still
    answer anything, but a question actually TRANSFERRED to this caller is now
    also answerable by them — the real gap the audit found. Before this, a
    regular employee handed a question via /transfer had no endpoint that let
    them act on it at all, despite the frontend offering a "transfer to
    teammate" flow that implied they could. /resolve and /transfer stay
    admin-only (_require_admin) — this relaxation is deliberately narrow.
    """
    if auth.is_super_admin:
        return
    if auth.role_in(workspace_id) in ("owner", "admin"):
        return
    if question.get("status") == "transferred" and question.get("transferred_to_user_id") == auth.user_id:
        return
    raise HTTPException(
        status_code=403,
        detail="Only workspace owners/admins, or the person this question was transferred to, can answer it.",
    )


@router.get("/bot-unanswered-questions")
async def list_unanswered_questions(
    workspace_id: str,
    status: Optional[str] = None,
    bot_id: Optional[str] = None,
    auth: AuthContext = Depends(current_user),
):
    auth.assert_workspace(workspace_id)
    q = supabase.table("bot_unanswered_questions").select("*").eq("workspace_id", workspace_id)
    if status:
        q = q.eq("status", status)
    if bot_id:
        q = q.eq("bot_id", bot_id)
    rows = q.order("created_at", desc=True).execute().data or []
    return {"questions": rows}


def _get_question(question_id: str, workspace_id: str) -> dict:
    rows = supabase.table("bot_unanswered_questions").select("*") \
        .eq("id", question_id).eq("workspace_id", workspace_id).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Question not found.")
    return rows[0]


class AnswerRequest(BaseModel):
    workspace_id: str
    answer_text:  str


@router.post("/bot-unanswered-questions/{question_id}/answer")
async def answer_question(question_id: str, body: AnswerRequest,
                          auth: AuthContext = Depends(current_user)):
    """
    Types an answer -> becomes a tier-2 knowledge_note, embedded and tagged
    so the asking bot can find it next time, then marks the question answered.
    """
    auth.assert_workspace(body.workspace_id)

    if not body.answer_text.strip():
        raise HTTPException(status_code=400, detail="answer_text is required.")

    question = _get_question(question_id, body.workspace_id)
    _require_admin_or_assignee(auth, body.workspace_id, question)

    note_id = bc.create_note_and_embed(
        body.workspace_id, connection_id=None, provider="bot_learning",
        note={"title": question["question"][:200], "body": body.answer_text.strip()},
        source_type="note", source_tier=2, source_ref=question_id,
        extra_metadata={"learned_for_bot_id": question["bot_id"]},
    )

    supabase.table("bot_unanswered_questions").update({
        "status":         "answered",
        "answer_text":    body.answer_text.strip(),
        "answer_note_id": note_id,
        "answered_by":    auth.user_id,
        "answered_at":    datetime.now(timezone.utc).isoformat(),
    }).eq("id", question_id).execute()

    return {"success": True, "note_id": note_id}


class ResolveRequest(BaseModel):
    workspace_id: str
    document_id:  str


@router.post("/bot-unanswered-questions/{question_id}/resolve")
async def resolve_via_document(question_id: str, body: ResolveRequest,
                               auth: AuthContext = Depends(current_user)):
    """
    Marks a question resolved after the admin answered it by uploading a
    document through the Library's existing upload flow (see module
    docstring) — no ingestion here, just recording what answered it.
    """
    auth.assert_workspace(body.workspace_id)
    _require_admin(auth, body.workspace_id)
    _get_question(question_id, body.workspace_id)

    supabase.table("bot_unanswered_questions").update({
        "status":         "answered",
        "answer_note_id": body.document_id,
        "answered_by":    auth.user_id,
        "answered_at":    datetime.now(timezone.utc).isoformat(),
    }).eq("id", question_id).execute()
    return {"success": True}


@router.get("/user-activity/{user_id}")
async def user_activity(user_id: str, workspace_id: str,
                        auth: AuthContext = Depends(current_user)):
    """
    The vector-DB half of the User Activity Profile (RBAC rollout Phase 5) --
    mirrors visuals.py's /workspace-token-usage shape (simple dict, aggregated
    in Python, no new RPC). The app-DB half (training assigned+progress,
    presentations shared with them) is queried directly from the frontend
    against RLS already extended for this in Phases 1-2; this endpoint only
    covers what lives here: bot usage and unanswered-question assignment
    history. Admin/owner-only -- this is one specific person's activity, a
    tighter bar than the workspace-wide token usage this mirrors.
    """
    auth.assert_workspace(workspace_id)
    _require_admin(auth, workspace_id)

    try:
        events = supabase.table("bot_usage_events") \
            .select("bot_id").eq("workspace_id", workspace_id).eq("user_id", user_id) \
            .execute().data or []
        by_bot: dict[str, int] = {}
        for e in events:
            by_bot[e["bot_id"]] = by_bot.get(e["bot_id"], 0) + 1

        assignments = supabase.table("bot_question_assignments") \
            .select("*").eq("workspace_id", workspace_id).eq("assigned_to_user_id", user_id) \
            .order("assigned_at", desc=True).limit(50).execute().data or []

        question_ids = [a["question_id"] for a in assignments]
        questions_by_id: dict = {}
        if question_ids:
            q_rows = supabase.table("bot_unanswered_questions") \
                .select("id, bot_id, question, status, answered_at, answered_by") \
                .in_("id", question_ids).execute().data or []
            questions_by_id = {q["id"]: q for q in q_rows}

        question_assignments = []
        for a in assignments:
            q = questions_by_id.get(a["question_id"], {})
            question_assignments.append({
                "question_id":  a["question_id"],
                "bot_id":       q.get("bot_id"),
                "question":     q.get("question"),
                "status":       q.get("status"),
                "assigned_at":  a["assigned_at"],
                "assigned_by":  a["assigned_by"],
                "answered_at":  q.get("answered_at"),
                "answered_by":  q.get("answered_by"),
            })

        return {
            "user_id":              user_id,
            "workspace_id":         workspace_id,
            "bot_usage": {
                "total_queries": sum(by_bot.values()),
                "by_bot": [{"bot_id": bid, "count": c} for bid, c in
                           sorted(by_bot.items(), key=lambda kv: -kv[1])],
            },
            "question_assignments": question_assignments,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"USER-ACTIVITY ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"User activity lookup failed: {e}")


class TransferRequest(BaseModel):
    workspace_id:           str
    transferred_to_user_id: str


@router.post("/bot-unanswered-questions/{question_id}/transfer")
async def transfer_question(question_id: str, body: TransferRequest,
                            auth: AuthContext = Depends(current_user)):
    """Hands the question to a specific workspace member. No email — see
    module docstring; the assignee checks the same list filtered to themselves.

    Also appends a row to bot_question_assignments — transferred_to_user_id
    on the question itself is only ever a fast "current assignee" cache, and
    a reassignment used to silently overwrite it with no record of who had it
    before. The history table is append-only so that never happens again."""
    auth.assert_workspace(body.workspace_id)
    _require_admin(auth, body.workspace_id)
    _get_question(question_id, body.workspace_id)

    now = datetime.now(timezone.utc).isoformat()
    supabase.table("bot_unanswered_questions").update({
        "status":                 "transferred",
        "transferred_to_user_id": body.transferred_to_user_id,
        "transferred_at":         now,
    }).eq("id", question_id).execute()

    supabase.table("bot_question_assignments").insert({
        "question_id":          question_id,
        "workspace_id":         body.workspace_id,
        "assigned_to_user_id":  body.transferred_to_user_id,
        "assigned_by":          auth.user_id,
        "assigned_at":          now,
    }).execute()

    return {"success": True}
