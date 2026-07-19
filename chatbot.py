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
    # If this list contains raw folder UUIDs instead, the filter below will
    # never match anything and the bot will silently lose access to its
    # knowledge base — see the fallback + warning log in run_rag_query().
    linked_folder_ids: Optional[list[str]] = []
    public_token:      Optional[str] = None
    allowed_domains:   Optional[list[str]] = []


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


def run_rag_query(question: str, bot: BotConfig) -> tuple[str, list[str]]:
    """
    Searches ONLY document chunks belonging to the bot's workspace.
    workspace_id in bot_config is the single source of truth for isolation.
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

    # Search ONLY this workspace's chunks
    chunks = []
    context_block = ""

    try:
        embed_result = voyage.embed([question], model="voyage-3", input_type="query")
        question_embedding = embed_result.embeddings[0]

        # Use workspace-scoped RPC function
        search_result = supabase.rpc("match_chunks_workspace", {
            "query_embedding":     question_embedding,
            "match_count":         8,
            "filter_asset_id":     None,
            "filter_workspace_id": bot.workspace_id,  # ← workspace isolation
        }).execute()

        all_chunks = search_result.data or []

        # Additionally filter by linked folder if specified.
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
            else:
                # Defensive fallback: the filter matched nothing, which almost
                # always means linked_folder_ids contains folder UUIDs rather
                # than resolved document/asset IDs. Fail SAFE (use all
                # workspace chunks) rather than silently blinding the bot —
                # but log loudly so this is easy to spot in Railway logs.
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

Documents:
{context_block}"""
    else:
        system_content = f"""{base_personality}

No specific documents match this query right now.
Respond naturally and helpfully based on your role.
For greetings and general questions respond warmly and in character.
For specific company questions you don't have data for, politely let the user know
and suggest they contact the relevant team if urgent.
Never give a cold or robotic response."""

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        temperature=0.5,
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
