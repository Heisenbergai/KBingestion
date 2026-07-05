import os
import json
import secrets
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import voyageai
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Clients ────────────────────────────────────────────────────────────────────
# Uses personal Supabase (vector DB) — NOT Lovable's Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

voyage  = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq    = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Lovable Supabase client (read-only for bot config + save messages) ─────────
# This reads bot config + saves conversations to Lovable's DB
# Uses the anon key — Lovable's RLS policies handle access control
lovable_supabase = create_client(
    os.getenv("LOVABLE_SUPABASE_URL"),
    os.getenv("LOVABLE_SUPABASE_ANON_KEY")
)


# ── Request shapes ─────────────────────────────────────────────────────────────
class WidgetQueryRequest(BaseModel):
    bot_id:         str
    token:          str            # public_token from chatbots table
    question:       str
    session_id:     str            # random UUID from browser localStorage
    conversation_id: Optional[str] = None  # if continuing existing conversation


class InternalQueryRequest(BaseModel):
    bot_id:          str
    question:        str
    conversation_id: Optional[str] = None
    user_id:         str           # authenticated user's ID


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_bot_config(bot_id: str) -> dict:
    """Loads bot config from Lovable's Supabase chatbots table."""
    result = lovable_supabase.table("chatbots") \
        .select("*") \
        .eq("id", bot_id) \
        .eq("is_active", True) \
        .single() \
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Bot not found or inactive")
    return result.data


def verify_token(bot: dict, token: str):
    """Verifies the public token matches the bot's token."""
    if bot.get("public_token") != token:
        raise HTTPException(status_code=401, detail="Invalid bot token")


def verify_domain(bot: dict, request: Request):
    """
    If the bot has allowed_domains set, verifies the request origin matches.
    Empty allowed_domains = allow all origins (useful during development).
    """
    allowed = bot.get("allowed_domains", [])
    if not allowed:
        return  # no restriction

    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    source = origin or referer

    if not any(domain in source for domain in allowed):
        raise HTTPException(
            status_code=403,
            detail=f"Domain not allowed. Configure allowed domains in bot settings."
        )


def get_conversation_history(conversation_id: str, limit: int = 6) -> list[dict]:
    """
    Loads the last N messages from a conversation for context.
    Returns as Groq-compatible message list.
    """
    if not conversation_id:
        return []

    result = lovable_supabase.table("bot_messages") \
        .select("role, content") \
        .eq("conversation_id", conversation_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()

    if not result.data:
        return []

    # Reverse so oldest first
    messages = list(reversed(result.data))
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def save_conversation(bot_id: str, user_id: Optional[str],
                      session_id: Optional[str],
                      conversation_id: Optional[str]) -> str:
    """
    Gets or creates a conversation record.
    Returns the conversation_id to use for saving messages.
    """
    if conversation_id:
        return conversation_id

    result = lovable_supabase.table("bot_conversations").insert({
        "bot_id":              bot_id,
        "user_id":             user_id,
        "external_session_id": session_id,
    }).select("id").single().execute()

    return result.data["id"]


def save_messages(conversation_id: str, question: str,
                  answer: str, sources: list):
    """Saves the user question + assistant answer to bot_messages."""
    lovable_supabase.table("bot_messages").insert([
        {"conversation_id": conversation_id, "role": "user",      "content": question, "sources": []},
        {"conversation_id": conversation_id, "role": "assistant",  "content": answer,   "sources": sources},
    ]).execute()


def run_rag_query(question: str, bot: dict) -> tuple[str, list]:
    """
    Core RAG pipeline for chatbot queries.

    1. Embed the question (Voyage AI)
    2. Search document_chunks filtered by the bot's linked folders
    3. Build context from retrieved chunks
    4. Generate answer via Groq using bot's custom system prompt
    """

    # Step 1: Embed the question
    embed_result = voyage.embed([question], model="voyage-3", input_type="query")
    question_embedding = embed_result.embeddings[0]

    # Step 2: Search vectors — filtered by linked folder documents
    # The bot's linked_folder_ids are stored in chatbots.linked_departments
    # We use asset_id filtering in document_chunks (asset_id = knowledge_item_id)
    # To filter by folder: get knowledge_item IDs in the linked folder first
    linked_folders = bot.get("linked_departments", [])  # these are folder IDs

    # Get knowledge_item IDs that belong to the linked folders
    filter_doc_ids = None
    if linked_folders:
        items_result = lovable_supabase.table("knowledge_items") \
            .select("id") \
            .in_("folder_id", linked_folders) \
            .is_("deleted_at", None) \
            .execute()

        if items_result.data:
            filter_doc_ids = [item["id"] for item in items_result.data]

    # Search Supabase pgvector
    search_result = supabase.rpc("match_chunks", {
        "query_embedding":  question_embedding,
        "match_count":      6,
        "filter_asset_id":  None   # we filter by doc IDs below if needed
    }).execute()

    chunks = search_result.data or []

    # If we have folder filters, apply them post-search
    if filter_doc_ids and chunks:
        chunks = [c for c in chunks if c.get("document_id") in filter_doc_ids]

    if not chunks:
        return ("I couldn't find any relevant information in the linked documents. "
                "Please try rephrasing your question or contact your administrator "
                "to ensure the relevant documents have been uploaded and processed."), []

    # Step 3: Build context
    context_parts = []
    for chunk in chunks:
        file_name = chunk.get("metadata", {}).get("file_name", "Company document")
        context_parts.append(f"[{file_name}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer using bot's custom system prompt
    bot_name    = bot.get("name", "Assistant")
    bot_prompt  = bot.get("system_prompt") or ""
    greeting    = bot.get("greeting_message", "")

    default_instructions = f"""You are {bot_name}, an AI assistant.
Answer questions using ONLY the provided document context.
Be helpful, clear, and concise.
If the answer isn't in the documents, say so honestly.
Never make up information."""

    system_content = bot_prompt if bot_prompt.strip() else default_instructions

    # Add grounding instruction to whatever custom prompt they set
    system_content += """

IMPORTANT: Answer ONLY from the provided document context below.
If the information isn't available, say: "I don't have that information in my current knowledge base."
Keep answers focused and actionable."""

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": f"Document context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer  = response.choices[0].message.content
    sources = list(set([
        c.get("metadata", {}).get("file_name", "")
        for c in chunks if c.get("metadata", {}).get("file_name")
    ]))

    return answer, sources


# ── Public endpoint — used by external widget ──────────────────────────────────
@router.post("/widget-query")
async def widget_query(request: Request, body: WidgetQueryRequest):
    """
    Public endpoint called by the embeddable widget.
    Auth: public_token (not user JWT).
    No login required — anyone with the embed code can use this.
    """
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Load and verify bot
        bot = load_bot_config(body.bot_id)
        verify_token(bot, body.token)
        verify_domain(bot, request)

        if not bot.get("is_external"):
            raise HTTPException(status_code=403, detail="This bot is not enabled for external use")

        # Load conversation history for context
        history = get_conversation_history(body.conversation_id)

        # Run RAG
        answer, sources = run_rag_query(body.question, bot)

        # Save conversation
        conv_id = save_conversation(body.bot_id, None, body.session_id, body.conversation_id)
        save_messages(conv_id, body.question, answer, sources)

        return {
            "answer":          answer,
            "sources":         sources,
            "conversation_id": conv_id,
            "bot_name":        bot.get("name"),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"WIDGET-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Internal endpoint — used by authenticated users inside the app ─────────────
@router.post("/internal-query")
async def internal_query(body: InternalQueryRequest):
    """
    Internal endpoint called by the in-app chat bubble.
    Auth: user_id passed from Lovable (Lovable handles JWT verification).
    """
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        bot = load_bot_config(body.bot_id)

        # Check user has access to this bot
        allowed_users = bot.get("allowed_user_ids", [])
        allowed_roles = bot.get("allowed_role_keys", [])

        # If neither list is set — bot is accessible to all users
        if allowed_users or allowed_roles:
            if body.user_id not in allowed_users:
                # Check role-based access
                role_result = lovable_supabase.table("user_roles") \
                    .select("role") \
                    .eq("user_id", body.user_id) \
                    .execute()
                user_roles = [r["role"] for r in (role_result.data or [])]

                if not any(r in allowed_roles for r in user_roles):
                    raise HTTPException(status_code=403, detail="Access denied to this bot")

        # Load conversation history
        history = get_conversation_history(body.conversation_id)

        # Run RAG
        answer, sources = run_rag_query(body.question, bot)

        # Save conversation
        conv_id = save_conversation(body.bot_id, body.user_id, None, body.conversation_id)
        save_messages(conv_id, body.question, answer, sources)

        return {
            "answer":          answer,
            "sources":         sources,
            "conversation_id": conv_id,
            "bot_name":        bot.get("name"),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"INTERNAL-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Bot config endpoint — used by widget.js on load ───────────────────────────
@router.get("/widget-config/{bot_id}")
async def widget_config(bot_id: str, token: str):
    """
    Called by widget.js when it first loads on the external website.
    Returns only safe public config — no system prompt exposed.
    """
    try:
        bot = load_bot_config(bot_id)
        verify_token(bot, {"public_token": token} if False else bot)

        if bot.get("public_token") != token:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {
            "name":             bot.get("name"),
            "avatar_url":       bot.get("avatar_url"),
            "primary_color":    bot.get("primary_color", "#1E2761"),
            "greeting_message": bot.get("greeting_message", "Hi! How can I help?"),
            "is_active":        bot.get("is_active"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve widget.js ────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse, PlainTextResponse

@router.get("/widget.js")
async def serve_widget():
    """
    Serves the injectable widget JavaScript file.
    Companies paste a <script src="...widget.js"> tag in their website.
    """
    widget_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "widget_template.js")
    if not os.path.exists(widget_path):
        raise HTTPException(status_code=404, detail="Widget file not found")
    return FileResponse(
        widget_path,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=3600"}
    )


# ── List bots for internal use ─────────────────────────────────────────────────
@router.get("/bots-for-user/{user_id}")
async def bots_for_user(user_id: str):
    """
    Called by Lovable on app load to get all bots a user can access.
    Used to render the floating chat bubbles.
    """
    try:
        # Get user's roles
        role_result = lovable_supabase.table("user_roles") \
            .select("role") \
            .eq("user_id", user_id) \
            .execute()
        user_roles = [r["role"] for r in (role_result.data or [])]

        # Get all active bots
        bots_result = lovable_supabase.table("chatbots") \
            .select("id, name, avatar_url, primary_color, greeting_message, allowed_user_ids, allowed_role_keys") \
            .eq("is_active", True) \
            .execute()

        accessible = []
        for bot in (bots_result.data or []):
            allowed_users = bot.get("allowed_user_ids") or []
            allowed_roles = bot.get("allowed_role_keys") or []

            # No restrictions = accessible to all
            if not allowed_users and not allowed_roles:
                accessible.append(bot)
                continue

            # Check user-specific access
            if user_id in allowed_users:
                accessible.append(bot)
                continue

            # Check role-based access
            if any(r in allowed_roles for r in user_roles):
                accessible.append(bot)

        # Return only safe public fields
        return {
            "bots": [
                {
                    "id":             b["id"],
                    "name":           b["name"],
                    "avatar_url":     b.get("avatar_url"),
                    "primary_color":  b.get("primary_color", "#1E2761"),
                    "greeting_message": b.get("greeting_message", "Hi! How can I help?"),
                }
                for b in accessible
            ]
        }

    except Exception as e:
        import traceback
        print(f"BOTS-FOR-USER ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
