import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import voyageai
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Bot config shape ───────────────────────────────────────────────────────────
class BotConfig(BaseModel):
    id:                str
    name:              str
    system_prompt:     Optional[str] = ""
    greeting_message:  Optional[str] = "Hi! How can I help you today?"
    primary_color:     Optional[str] = "#1E2761"
    avatar_url:        Optional[str] = None
    linked_folder_ids: Optional[list[str]] = []
    public_token:      Optional[str] = None
    allowed_domains:   Optional[list[str]] = []


# ── Request shapes ─────────────────────────────────────────────────────────────
class WidgetQueryRequest(BaseModel):
    question:        str
    session_id:      str
    bot_config:      BotConfig
    conversation_id: Optional[str] = None
    token:           str


class InternalQueryRequest(BaseModel):
    question:        str
    bot_config:      BotConfig
    user_id:         str
    conversation_id: Optional[str] = None


# ── Core RAG + conversational query ───────────────────────────────────────────
def run_rag_query(question: str, bot: BotConfig) -> tuple[str, list[str]]:
    """
    Runs RAG search. If relevant chunks found, answers from them.
    If no relevant chunks, the bot still responds naturally and in character
    — it never gives a cold mechanical fallback.
    """

    bot_name      = bot.name or "Assistant"
    custom_prompt = (bot.system_prompt or "").strip()

    # ── Build the system prompt ────────────────────────────────────────────────
    # The custom prompt from the admin IS the bot's personality and instructions.
    # We always append grounding rules, but the bot's character comes first.
    if custom_prompt:
        base_personality = custom_prompt
    else:
        base_personality = (
            f"You are {bot_name}, a friendly and knowledgeable AI assistant. "
            f"You help users by answering their questions clearly and warmly. "
            f"You are professional yet approachable. "
            f"You keep responses concise and useful."
        )

    # ── Try to find relevant document chunks ──────────────────────────────────
    chunks = []
    context_block = ""

    try:
        embed_result = voyage.embed([question], model="voyage-3", input_type="query")
        question_embedding = embed_result.embeddings[0]

        search_result = supabase.rpc("match_chunks", {
            "query_embedding":  question_embedding,
            "match_count":      8,
            "filter_asset_id":  None
        }).execute()

        all_chunks = search_result.data or []

        # Filter to linked folder documents if specified
        if bot.linked_folder_ids and all_chunks:
            chunks = [
                c for c in all_chunks
                if c.get("document_id") in bot.linked_folder_ids
                or c.get("asset_id") in bot.linked_folder_ids
            ]
        else:
            chunks = all_chunks

        if chunks:
            context_parts = []
            for chunk in chunks:
                file_name = chunk.get("metadata", {}).get("file_name", "Company document")
                context_parts.append(f"[{file_name}]\n{chunk['content']}")
            context_block = "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"[chatbot] RAG search error (continuing without context): {str(e)}")

    # ── Build final system content ─────────────────────────────────────────────
    if context_block:
        # Documents found — answer from them, stay in character
        system_content = f"""{base_personality}

You have access to the following company knowledge base documents to help answer questions.
Prioritise information from these documents when relevant.
If the answer is clearly in the documents, use it.
If it's a general/conversational question not needing the documents, just answer naturally.
Never say you "cannot find information" if the question is conversational — just answer it.

Documents:
{context_block}"""
    else:
        # No documents found — stay in character, be helpful, never be mechanical
        system_content = f"""{base_personality}

No specific documents are available for this query right now.
Respond naturally and helpfully based on your role and instructions above.
For greetings, small talk, and general questions — respond warmly and in character.
For specific company/product questions you don't have data for — politely let the user know
you'll need to check, and suggest they contact the relevant team if urgent.
Never give a cold or robotic response."""

    # ── Generate response ──────────────────────────────────────────────────────
    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        temperature=0.5,   # slightly higher than 0.3 for more natural conversation
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": question}
        ]
    )

    answer = response.choices[0].message.content

    sources = list(set([
        c.get("metadata", {}).get("file_name", "")
        for c in chunks
        if c.get("metadata", {}).get("file_name")
    ]))

    return answer, sources


# ── Domain check ───────────────────────────────────────────────────────────────
def verify_domain(bot: BotConfig, request: Request):
    allowed = bot.allowed_domains or []
    if not allowed:
        return
    origin  = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    source  = origin or referer
    if not any(domain in source for domain in allowed):
        raise HTTPException(
            status_code=403,
            detail="Domain not allowed. Add your domain in bot settings."
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/widget-query")
async def widget_query(request: Request, body: WidgetQueryRequest):
    try:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        verify_domain(body.bot_config, request)
        answer, sources = run_rag_query(body.question, body.bot_config)
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
        answer, sources = run_rag_query(body.question, body.bot_config)
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
