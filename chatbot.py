import os
import ai
import httpx
import auth as auth_mod
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from auth import AuthContext, current_user
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Chat model comes from ai.py (AWS Bedrock — Amazon Nova Lite by default,
# override with BEDROCK_CHAT_MODEL env var on Railway).

# How much conversation memory to send to the LLM per request
MAX_HISTORY_MESSAGES  = 10
MAX_MESSAGE_CHARS     = 2000


class ChatMessage(BaseModel):
    role:    str   # "user" or "assistant"
    content: str


class BotConfig(BaseModel):
    id:                str
    name:              str
    workspace_id:      str           # ← REQUIRED — isolates search to this workspace
    system_prompt:     Optional[str] = ""
    greeting_message:  Optional[str] = "Hi! How can I help you today?"
    primary_color:     Optional[str] = "#1E2761"
    avatar_url:        Optional[str] = None
    # IMPORTANT: these must be document_id / asset_id values, NOT folder IDs.
    # Railway has no access to Lovable's knowledge_folders table (see
    # 02_infrastructure.md — Railway cannot query the Lovable-managed DB),
    # so it cannot resolve "which documents live in folder X" itself.
    # Lovable MUST resolve the bot's linked folders to their contained
    # document/asset IDs before calling /internal-query or /widget-query.
    linked_folder_ids: Optional[list[str]] = []
    public_token:      Optional[str] = None
    allowed_domains:   Optional[list[str]] = []


class WidgetQueryRequest(BaseModel):
    question:             str
    session_id:           str
    bot_config:           BotConfig
    conversation_id:      Optional[str] = None
    token:                str
    # Last messages of this conversation, oldest first. The widget keeps them
    # client-side; Lovable keeps them in bot_messages. Without this the bot
    # has no memory and every follow-up question falls flat.
    conversation_history: Optional[list[ChatMessage]] = []


class InternalQueryRequest(BaseModel):
    question:             str
    bot_config:           BotConfig
    user_id:              str
    conversation_id:      Optional[str] = None
    conversation_history: Optional[list[ChatMessage]] = []


def _clean_history(history: Optional[list[ChatMessage]]) -> list[dict]:
    """
    Validates and trims conversation history into Groq message dicts.
    Only user/assistant roles pass through (a client can never inject a
    system message), each message is length-capped, and only the most
    recent MAX_HISTORY_MESSAGES are kept.
    """
    if not history:
        return []
    cleaned = []
    for msg in history:
        role = (msg.role or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = (msg.content or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:MAX_MESSAGE_CHARS]})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def _retrieval_text(question: str, history: list[dict]) -> str:
    """
    Short follow-ups ("what about the second one?", "why?") embed terribly
    on their own — they match nothing. Prepending the previous user turn
    gives the vector search enough context to find the right chunks.
    """
    if len(question) >= 60 or not history:
        return question
    prev_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    if prev_user:
        return f"{prev_user}\n{question}"
    return question


def run_rag_query(
    question: str,
    bot: BotConfig,
    history: Optional[list[ChatMessage]] = None,
    strict_folders: bool = False,
    user_id: Optional[str] = None,
    feature: str = "chatbot_internal",
) -> tuple[str, list[str], str]:
    """
    Searches ONLY document chunks belonging to the bot's workspace.
    workspace_id in bot_config is the single source of truth for isolation.

    strict_folders controls what happens when linked_folder_ids matches no
    chunks: external (widget) bots fail CLOSED — a public bot must never
    answer from documents outside its intended scope. Internal bots fall
    back to the whole workspace (with a loud log), since everyone in the
    workspace can already see those documents anyway.

    user_id/feature are for the token-usage dashboards. `feature` matches the
    same 'chatbot_internal'/'chatbot_external' vocabulary check_and_increment_usage
    already uses, so a workspace's quota checks and its token spend can be
    correlated by feature without translating between two naming schemes.

    Returns (answer, sources, confidence). confidence reuses query.py's exact
    scheme (high/medium/low/none, from top chunk similarity) rather than
    inventing a second one, so the caller can escalate a low/none answer into
    the admin's unanswered-questions queue.
    """
    if not bot.workspace_id:
        raise HTTPException(
            status_code=400,
            detail="workspace_id is required in bot_config. Data isolation cannot be guaranteed without it."
        )

    bot_name      = bot.name or "Assistant"
    custom_prompt = (bot.system_prompt or "").strip()

    if custom_prompt:
        base_personality = custom_prompt
    else:
        base_personality = (
            f"You are {bot_name}, a friendly and knowledgeable AI assistant. "
            f"You help users by answering their questions clearly and warmly. "
            f"You are professional yet approachable."
        )

    history_messages = _clean_history(history)

    # Search ONLY this workspace's chunks
    chunks = []
    context_block = ""

    try:
        search_text = _retrieval_text(question.strip(), history_messages)
        question_embedding = ai.embed_texts(
            [search_text], workspace_id=bot.workspace_id, user_id=user_id, feature=feature,
        )[0]

        # Hybrid retrieval (vector + keyword + tier/freshness boosts), still
        # workspace-isolated. Falls back to the old vector-only RPC if the
        # hybrid function isn't deployed yet.
        try:
            search_result = supabase.rpc("match_chunks_hybrid", {
                "query_text":          search_text,
                "query_embedding":     question_embedding,
                "match_count":         8,
                "filter_workspace_id": bot.workspace_id,  # ← workspace isolation
                "filter_asset_id":     None,
            }).execute()
        except Exception as e:
            print(f"[chatbot] hybrid search unavailable, vector-only fallback: {e}")
            search_result = supabase.rpc("match_chunks_workspace", {
                "query_embedding":     question_embedding,
                "match_count":         8,
                "filter_asset_id":     None,
                "filter_workspace_id": bot.workspace_id,
            }).execute()

        all_chunks = search_result.data or []

        # Additionally filter by linked documents if specified.
        # See the comment on BotConfig.linked_folder_ids — this only works
        # if Lovable resolved folder IDs to document/asset IDs first.
        # A chunk from a note this SPECIFIC bot learned (bot_learning.py,
        # via create_note_and_embed's extra_metadata) always stays in scope —
        # it has no knowledge_folders placement to filter by, and it was
        # taught to this bot directly in response to its own escalated
        # question, so folder-scoping shouldn't hide it from that bot.
        if bot.linked_folder_ids and all_chunks:
            filtered = [
                c for c in all_chunks
                if c.get("document_id") in bot.linked_folder_ids
                or c.get("asset_id") in bot.linked_folder_ids
                or c.get("metadata", {}).get("learned_for_bot_id") == bot.id
            ]
            if filtered:
                chunks = filtered
            elif strict_folders:
                # External/public bot: fail CLOSED. Better to say "I don't
                # know" than to answer a public visitor from internal
                # documents the bot was never scoped to.
                print(
                    f"[chatbot] WARNING: external bot '{bot.name}' (id={bot.id}) "
                    f"linked_folder_ids={bot.linked_folder_ids} matched none of "
                    f"the {len(all_chunks)} retrieved chunks. Failing closed "
                    f"(no context). If this bot should have knowledge, check "
                    f"that Lovable resolves folder IDs to document/asset IDs."
                )
                chunks = []
            else:
                # Internal bot: fall back to unfiltered workspace search —
                # workspace members can see these documents anyway. Log
                # loudly so the folder-scoping bug is easy to spot.
                print(
                    f"[chatbot] WARNING: bot '{bot.name}' (id={bot.id}) has "
                    f"linked_folder_ids={bot.linked_folder_ids} but none of "
                    f"the {len(all_chunks)} matched chunks belong to those IDs. "
                    f"Falling back to unfiltered workspace search. "
                    f"This usually means Lovable sent folder IDs instead of "
                    f"resolved document/asset IDs — check the bot's folder "
                    f"scoping logic in Lovable."
                )
                chunks = all_chunks
        else:
            chunks = all_chunks

        if chunks:
            context_parts = []
            for chunk in chunks:
                file_name = chunk.get("metadata", {}).get("file_name", "Company document")
                stype = chunk.get("source_type") or chunk.get("metadata", {}).get("source_type") or "document"
                label = {"document": "official document", "meeting": "meeting note",
                         "slack": "team chat", "note": "curated note"}.get(stype, stype)
                context_parts.append(f"[{file_name} — {label}]\n{chunk['content']}")
            context_block = "\n\n---\n\n".join(context_parts)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[chatbot] RAG search error: {str(e)}")

    # Confidence mirrors query.py's exact thresholds (query.py:158-160) — reused
    # rather than reinvented, so "answered well" means the same thing across AI
    # Search and chatbots. No chunks at all is "none"; otherwise it's driven by
    # the best-matching chunk's semantic similarity.
    #
    # similarity is normally a float, but PostgREST serialises NaN/Infinity as
    # the STRING "NaN"/"Infinity" (JSON has no such literals) — confirmed while
    # testing this locally with a degenerate query embedding. Coerced safely
    # here so a NaN similarity degrades to "no confidence" instead of crashing
    # run_rag_query with an unhandled TypeError and breaking the chat response.
    def _safe_similarity(c: dict) -> float:
        try:
            v = float(c.get("similarity") or 0)
            return v if v == v else 0.0  # NaN != NaN
        except (TypeError, ValueError):
            return 0.0

    top_sim = max((_safe_similarity(c) for c in chunks), default=0.0)
    confidence = (
        "none" if not chunks else
        "high" if top_sim >= 0.45 else
        "medium" if top_sim >= 0.3 else
        "low"
    )

    if context_block:
        system_content = f"""{base_personality}

You have access to the following company knowledge base documents.
Prioritise information from these documents when relevant.
If the answer is clearly in the documents, use it.
If it is a general conversational question, answer naturally.
Never say you cannot find information if the question is conversational.
Use the conversation history to resolve follow-up questions and references
like "it", "that one", or "the second option".

Documents:
{context_block}"""
    else:
        system_content = f"""{base_personality}

No specific documents match this query right now.
Respond naturally and helpfully based on your role.
For greetings and general questions respond warmly and in character.
For specific company questions you don't have data for, politely let the user know
and suggest they contact the relevant team if urgent.
Never give a cold or robotic response.
Use the conversation history to stay consistent with what was already discussed."""

    # Bedrock takes the system prompt separately — never inside messages.
    # ai.chat also normalizes ordering (must start with user, must alternate).
    answer = ai.chat(
        messages=history_messages + [{"role": "user", "content": question}],
        system=system_content,
        max_tokens=600,
        temperature=0.5,
        workspace_id=bot.workspace_id, user_id=user_id, feature=feature,
    )

    sources = list(set([
        c.get("metadata", {}).get("file_name", "")
        for c in chunks
        if c.get("metadata", {}).get("file_name")
    ]))

    return answer, sources, confidence


def log_usage_event(bot: BotConfig, source: str, question: str, confidence: str,
                    domain: str = None, user_id: str = None, session_id: str = None):
    """
    Records one analytics row per bot query (powers GET /bot-analytics: chats
    per bot, internal vs external, which domains the widget runs on, top
    questions asked, and per-bot answer-rate). The question TEXT is logged for
    every query, not just failures — confirmed with Tanmay: this is the bot
    owner's own data about their own bot, and enables "top questions asked" +
    future training-content mining, not just failure triage.

    When confidence is "none" or "low", the same question is also escalated
    into bot_unanswered_questions — the admin-facing learning queue (answer,
    upload a document, or transfer to a teammate). Both writes are strictly
    best-effort: analytics/escalation must never break a chat response.
    """
    try:
        supabase.table("bot_usage_events").insert({
            "bot_id":       bot.id,
            "workspace_id": bot.workspace_id,
            "source":       source,
            "domain":       (domain or None),
            "user_id":      user_id,
            "session_id":   session_id,
            "question":     question[:MAX_MESSAGE_CHARS],
            "confidence":   confidence,
        }).execute()
    except Exception as e:
        print(f"[chatbot] usage event logging failed (non-fatal): {e}")

    if confidence in ("none", "low"):
        try:
            supabase.table("bot_unanswered_questions").insert({
                "workspace_id": bot.workspace_id,
                "bot_id":       bot.id,
                "question":     question[:MAX_MESSAGE_CHARS],
                "source":       source,
                "session_id":   session_id,
                "user_id":      user_id,
                "confidence":   confidence,
            }).execute()
        except Exception as e:
            print(f"[chatbot] escalation logging failed (non-fatal): {e}")


def _fetch_my_chatbot_ids(token: str, workspace_id: str, user_id: str) -> set[str]:
    """
    Railway has no service-role client for the APP DB (chatbots lives there,
    not here) -- so this forwards the CALLER'S OWN bearer token to the app
    DB's PostgREST API, same pattern auth.py's _load_memberships already uses
    for workspace_members/super_admins. The created_by=eq.{user_id} filter
    does the real work regardless of chatbots' own RLS shape: PostgREST
    applies it as a hard filter on top of whatever RLS allows, so this can
    only ever return bots this exact caller created, never anyone else's.
    """
    if not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        return set()
    try:
        with httpx.Client(timeout=10) as client:
            res = client.get(
                f"{auth_mod.APP_SUPABASE_URL}/rest/v1/chatbots",
                params={
                    "select":       "id",
                    "workspace_id": f"eq.{workspace_id}",
                    "created_by":   f"eq.{user_id}",
                },
                headers={
                    "apikey":        auth_mod.APP_SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {token}",
                    "Accept":        "application/json",
                },
            )
            res.raise_for_status()
            return {row["id"] for row in res.json()}
    except Exception as e:
        print(f"BOT-ANALYTICS: failed to resolve caller's own bots (failing closed): {e}")
        return set()


@router.get("/bot-analytics")
async def bot_analytics(workspace_id: str, days: int = 30,
                        auth: AuthContext = Depends(current_user),
                        authorization: Optional[str] = Header(None)):
    """
    Usage analytics for all bots in a workspace, keyed by bot_id
    (Lovable joins display names from its own chatbots table):
    {
      bots: { "<bot_id>": { total_chats, internal_chats, widget_chats,
                            domains: {"example.com": n}, last_used,
                            top_questions: [{question, count}],   # last `days`
                            confidence_breakdown: {high, medium, low, none},
                            unanswered_count, answer_rate } },
      daily: [ {bot_id, day, chats} ]   # last `days` days, for charts
    }

    top_questions/confidence_breakdown/unanswered_count/answer_rate are
    computed directly from bot_usage_events (not a new RPC — cheap at current
    volume, and keeps bot_usage_summary/daily untouched). answer_rate is
    1 - unanswered/total, over queries that have a confidence value recorded
    (older rows predating this column are excluded, not counted as failures).

    RBAC rollout Phase 4: an `admin`-tier caller (not owner/super admin) only
    ever sees analytics for bots THEY created, not the full tenant — owner and
    admin used to see the identical full-workspace view, which contradicted
    "Admin cannot see overall tenant plan credits" once tier boundaries became
    a real thing this session. Enforced server-side via _fetch_my_chatbot_ids,
    not just a frontend filter — an admin cannot see past this by editing the
    request, since the id set comes from the app DB with their own token.
    """
    try:
        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required.")

        auth.assert_workspace(workspace_id)

        summary = supabase.rpc("bot_usage_summary",
                               {"filter_workspace_id": workspace_id}).execute().data or []
        daily = supabase.rpc("bot_usage_daily",
                             {"filter_workspace_id": workspace_id, "days": days}).execute().data or []

        bots: dict = {}
        for row in summary:
            b = bots.setdefault(row["bot_id"], {
                "total_chats": 0, "internal_chats": 0, "widget_chats": 0,
                "domains": {}, "last_used": None,
            })
            chats = row["chats"]
            b["total_chats"] += chats
            if row["source"] == "widget":
                b["widget_chats"] += chats
                if row["domain"]:
                    b["domains"][row["domain"]] = b["domains"].get(row["domain"], 0) + chats
            else:
                b["internal_chats"] += chats
            if b["last_used"] is None or row["last_used"] > b["last_used"]:
                b["last_used"] = row["last_used"]

        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        events = supabase.table("bot_usage_events") \
            .select("bot_id, question, confidence") \
            .eq("workspace_id", workspace_id).gte("created_at", since) \
            .execute().data or []

        for bot_id, b in bots.items():
            bot_events = [e for e in events if e["bot_id"] == bot_id]

            confidence_breakdown: dict[str, int] = {}
            question_groups: dict[str, dict] = {}
            for e in bot_events:
                conf = e.get("confidence")
                if conf:
                    confidence_breakdown[conf] = confidence_breakdown.get(conf, 0) + 1
                qtext = (e.get("question") or "").strip()
                if qtext:
                    key = qtext.lower()
                    g = question_groups.setdefault(key, {"question": qtext, "count": 0})
                    g["count"] += 1

            total_with_confidence = sum(confidence_breakdown.values())
            unanswered = confidence_breakdown.get("none", 0) + confidence_breakdown.get("low", 0)

            b["confidence_breakdown"] = confidence_breakdown
            b["unanswered_count"] = unanswered
            b["answer_rate"] = (
                round(1 - unanswered / total_with_confidence, 3)
                if total_with_confidence else None
            )
            b["top_questions"] = sorted(
                question_groups.values(), key=lambda g: g["count"], reverse=True
            )[:5]

        if not auth.is_super_admin and auth.role_in(workspace_id) == "admin":
            token = ""
            if authorization and authorization.lower().startswith("bearer "):
                token = authorization[7:].strip()
            my_bot_ids = _fetch_my_chatbot_ids(token, workspace_id, auth.user_id)
            bots = {bid: b for bid, b in bots.items() if bid in my_bot_ids}
            daily = [row for row in daily if row.get("bot_id") in my_bot_ids]

        return {"workspace_id": workspace_id, "bots": bots, "daily": daily}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"BOT-ANALYTICS ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analytics failed: {e}")


def verify_domain(bot: BotConfig, request: Request):
    allowed = bot.allowed_domains or []
    if not allowed:
        return
    origin  = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    source  = origin or referer
    if not any(domain in source for domain in allowed):
        raise HTTPException(status_code=403, detail="Domain not allowed.")


def resolve_public_bot(token: str) -> dict:
    """
    Turns a widget's public_token into the REAL bot record from the app DB.

    This exists because /widget-query is genuinely public — there is no user
    session to authenticate — so the workspace it reads must come from the server,
    never from the embedding page. Previously bot_config arrived from the browser,
    which meant anyone could POST an arbitrary workspace_id and read that whole
    workspace. The token is the bearer credential; get_public_bot() matches it
    exactly and only returns active bots.
    """
    if not token or len(token) < 16:
        raise HTTPException(status_code=401, detail="Invalid bot token.")

    if not auth_mod.APP_SUPABASE_URL or not auth_mod.APP_SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server auth is misconfigured (APP_SUPABASE_URL / APP_SUPABASE_ANON_KEY).",
        )

    try:
        with httpx.Client(timeout=10) as client:
            res = client.post(
                f"{auth_mod.APP_SUPABASE_URL}/rest/v1/rpc/get_public_bot",
                json={"_token": token},
                headers={
                    "apikey":       auth_mod.APP_SUPABASE_ANON_KEY,
                    "Content-Type": "application/json",
                },
            )
            res.raise_for_status()
            rows = res.json()
    except Exception as e:
        print(f"WIDGET-BOT-RESOLVE ERROR: {e}")
        raise HTTPException(status_code=503, detail="Could not verify this bot.")

    if not rows:
        raise HTTPException(status_code=401, detail="Invalid bot token.")
    return rows[0]


@router.post("/widget-query")
async def widget_query(request: Request, body: WidgetQueryRequest):
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Server-resolved identity wins over anything the page sent. Folder scoping
        # is now ALSO fully server-resolved: get_public_bot's RPC (app DB) resolves
        # linked_folder_ids (folder IDs) against knowledge_items itself and returns
        # resolved_document_ids, since it lives in the same Postgres instance as
        # knowledge_folders/knowledge_items and Railway does not. The client-sent
        # bot_config.linked_folder_ids is no longer trusted at all for the widget
        # path — this closes the gap where a folder-scoped external bot previously
        # returned zero results (raw folder UUIDs never matched document/asset IDs).
        record = resolve_public_bot(body.token)
        bot = BotConfig(
            id                = str(record["id"]),
            name              = record.get("name") or "Assistant",
            workspace_id      = str(record["workspace_id"]),
            system_prompt     = record.get("system_prompt") or "",
            greeting_message  = record.get("greeting_message") or "",
            allowed_domains   = record.get("allowed_domains") or [],
            linked_folder_ids = [str(i) for i in (record.get("resolved_document_ids") or [])],
            public_token      = None,
        )

        verify_domain(bot, request)
        answer, sources, confidence = run_rag_query(
            body.question,
            bot,
            history=body.conversation_history,
            strict_folders=True,   # public bots never fall back outside their scope
            feature="chatbot_external",
        )
        origin = request.headers.get("origin") or request.headers.get("referer") or ""
        domain = origin.split("//")[-1].split("/")[0] if origin else None
        log_usage_event(bot, "widget", body.question.strip(), confidence,
                        domain=domain, session_id=body.session_id)
        return {
            "answer":          answer,
            "sources":         sources,
            "bot_name":        bot.name,
            "conversation_id": body.conversation_id,
            "session_id":      body.session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"WIDGET-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/internal-query")
async def internal_query(body: InternalQueryRequest,
                         auth: AuthContext = Depends(current_user)):
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # bot_config is client-supplied, so its workspace_id is a claim, not a fact.
        # This is the check whose absence let an unauthenticated caller read any
        # workspace's documents by naming its UUID.
        auth.assert_workspace(body.bot_config.workspace_id)

        answer, sources, confidence = run_rag_query(
            body.question,
            body.bot_config,
            history=body.conversation_history,
            strict_folders=False,  # internal users can see workspace docs anyway
            user_id=body.user_id, feature="chatbot_internal",
        )
        log_usage_event(body.bot_config, "internal", body.question.strip(), confidence,
                        user_id=body.user_id)
        return {
            "answer":          answer,
            "sources":         sources,
            "bot_name":        body.bot_config.name,
            "conversation_id": body.conversation_id,
            "user_id":         body.user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"INTERNAL-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/widget.js")
async def serve_widget():
    widget_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "widget_template.js"
    )
    if not os.path.exists(widget_path):
        raise HTTPException(status_code=404, detail="Widget file not found")
    return FileResponse(
        widget_path,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=3600"}
    )
