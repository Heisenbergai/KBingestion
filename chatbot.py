import os
import ai
from fastapi import APIRouter, HTTPException, Request
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
) -> tuple[str, list[str]]:
    """
    Searches ONLY document chunks belonging to the bot's workspace.
    workspace_id in bot_config is the single source of truth for isolation.

    strict_folders controls what happens when linked_folder_ids matches no
    chunks: external (widget) bots fail CLOSED — a public bot must never
    answer from documents outside its intended scope. Internal bots fall
    back to the whole workspace (with a loud log), since everyone in the
    workspace can already see those documents anyway.
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
        question_embedding = ai.embed_texts([search_text])[0]

        # Use workspace-scoped RPC function
        search_result = supabase.rpc("match_chunks_workspace", {
            "query_embedding":     question_embedding,
            "match_count":         8,
            "filter_asset_id":     None,
            "filter_workspace_id": bot.workspace_id,  # ← workspace isolation
        }).execute()

        all_chunks = search_result.data or []

        # Additionally filter by linked documents if specified.
        # See the comment on BotConfig.linked_folder_ids — this only works
        # if Lovable resolved folder IDs to document/asset IDs first.
        if bot.linked_folder_ids and all_chunks:
            filtered = [
                c for c in all_chunks
                if c.get("document_id") in bot.linked_folder_ids
                or c.get("asset_id") in bot.linked_folder_ids
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
                context_parts.append(f"[{file_name}]\n{chunk['content']}")
            context_block = "\n\n---\n\n".join(context_parts)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[chatbot] RAG search error: {str(e)}")

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
    )

    sources = list(set([
        c.get("metadata", {}).get("file_name", "")
        for c in chunks
        if c.get("metadata", {}).get("file_name")
    ]))

    return answer, sources


def verify_domain(bot: BotConfig, request: Request):
    allowed = bot.allowed_domains or []
    if not allowed:
        return
    origin  = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    source  = origin or referer
    if not any(domain in source for domain in allowed):
        raise HTTPException(status_code=403, detail="Domain not allowed.")


@router.post("/widget-query")
async def widget_query(request: Request, body: WidgetQueryRequest):
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        verify_domain(body.bot_config, request)
        answer, sources = run_rag_query(
            body.question,
            body.bot_config,
            history=body.conversation_history,
            strict_folders=True,   # public bots never fall back outside their scope
        )
        return {
            "answer":          answer,
            "sources":         sources,
            "bot_name":        body.bot_config.name,
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
async def internal_query(body: InternalQueryRequest):
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        answer, sources = run_rag_query(
            body.question,
            body.bot_config,
            history=body.conversation_history,
            strict_folders=False,  # internal users can see workspace docs anyway
        )
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
