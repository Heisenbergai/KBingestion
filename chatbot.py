import os
import json
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

# ── Clients (personal Supabase only — vector DB) ───────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Bot config shape (sent by Lovable in every request) ────────────────────────
class BotConfig(BaseModel):
    id:                 str
    name:               str
    system_prompt:      Optional[str] = ""
    greeting_message:   Optional[str] = "Hi! How can I help you today?"
    primary_color:      Optional[str] = "#1E2761"
    avatar_url:         Optional[str] = None
    linked_folder_ids:  Optional[list[str]] = []   # knowledge_item IDs in linked folder
    public_token:       Optional[str] = None
    allowed_domains:    Optional[list[str]] = []


# ── Request shapes ─────────────────────────────────────────────────────────────
class WidgetQueryRequest(BaseModel):
    question:        str
    session_id:      str
    bot_config:      BotConfig
    conversation_id: Optional[str] = None
    token:           str            # public_token — verified by Lovable before calling


class InternalQueryRequest(BaseModel):
    question:        str
    bot_config:      BotConfig
    user_id:         str
    conversation_id: Optional[str] = None


# ── Core RAG query ─────────────────────────────────────────────────────────────
def run_rag_query(question: str, bot: BotConfig) -> tuple[str, list[str]]:
    """
    1. Embed the question (Voyage AI)
    2. Search personal Supabase pgvector — filtered by linked folder's document IDs
    3. Build context from top chunks
    4. Generate answer via Groq using bot's custom system prompt
    Returns: (answer, sources)
    """

    # Step 1: Embed
    embed_result = voyage.embed([question], model="voyage-3", input_type="query")
    question_embedding = embed_result.embeddings[0]

    # Step 2: Vector search
    # linked_folder_ids contains the knowledge_item IDs that belong to the bot's
    # linked folder — Lovable passes these in the request, computed from its own DB
    search_result = supabase.rpc("match_chunks", {
        "query_embedding":  question_embedding,
        "match_count":      8,
        "filter_asset_id":  None
    }).execute()

    chunks = search_result.data or []

    # Filter to linked folder documents if folder IDs provided
    if bot.linked_folder_ids and chunks:
        chunks = [
            c for c in chunks
            if c.get("document_id") in bot.linked_folder_ids
            or c.get("asset_id") in bot.linked_folder_ids
        ]

    if not chunks:
        return (
            "I couldn't find relevant information in my knowledge base. "
            "Please try rephrasing your question.",
            []
        )

    # Step 3: Build context
    context_parts = []
    for chunk in chunks:
        file_name = chunk.get("metadata", {}).get("file_name", "Company document")
        context_parts.append(f"[{file_name}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate answer with bot's persona
    bot_name = bot.name or "Assistant"
    custom_prompt = (bot.system_prompt or "").strip()

    if custom_prompt:
        system_content = custom_prompt
    else:
        system_content = (
            f"You are {bot_name}, a helpful AI assistant. "
            f"Answer questions clearly and concisely."
        )

    # Always ground the answer in provided documents
    system_content += (
        "\n\nIMPORTANT: Answer ONLY from the document context provided below. "
        "If the answer is not in the documents, say: "
        "'I don't have that information in my current knowledge base.' "
        "Never make up information."
    )

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        temperature=0.3,
        messages=[
            {"role": "system",  "content": system_content},
            {"role": "user",    "content": f"Document context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content

    sources = list(set([
        c.get("metadata", {}).get("file_name", "")
        for c in chunks
        if c.get("metadata", {}).get("file_name")
    ]))

    return answer, sources


# ── Domain verification (for external widget) ──────────────────────────────────
def verify_domain(bot: BotConfig, request: Request):
    """Only enforced when allowed_domains is non-empty."""
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
    """
    Called by the external embeddable widget.
    Lovable verifies the public_token before calling this endpoint.
    Railway runs RAG and returns the answer.
    Lovable saves the conversation to its own DB after receiving the response.
    """
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
    """
    Called by the in-app chat bubble (authenticated users).
    Lovable passes the bot_config from its own DB.
    Railway runs RAG and returns the answer.
    Lovable saves the conversation to its own DB.
    """
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
    """Serves the injectable widget JavaScript file."""
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
