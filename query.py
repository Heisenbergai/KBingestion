import os
import ai
from fastapi import APIRouter, HTTPException
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


class QueryRequest(BaseModel):
    question:     str
    workspace_id: str           # ← REQUIRED — only search this workspace's chunks
    asset_id:     Optional[str] = None
    match_count:  Optional[int] = 5


@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Searches ONLY document chunks belonging to the caller's workspace.
    No cross-workspace data is ever returned.
    """
    try:
        if not request.workspace_id:
            raise HTTPException(
                status_code=400,
                detail="workspace_id is required for all queries."
            )

        # Embed the question (Bedrock Titan v2 via ai.py)
        question_embedding = ai.embed_texts([request.question])[0]

        # Search ONLY this workspace's chunks
        search_result = supabase.rpc("match_chunks_workspace", {
            "query_embedding":   question_embedding,
            "match_count":       request.match_count,
            "filter_asset_id":   request.asset_id,
            "filter_workspace_id": request.workspace_id,  # ← workspace isolation
        }).execute()

        chunks = search_result.data or []

        if not chunks:
            return {
                "answer":  "I couldn't find any relevant information in your knowledge base.",
                "sources": [],
                "chunks":  []
            }

        context_parts = []
        for chunk in chunks:
            file_name = chunk["metadata"].get("file_name", "Unknown document")
            context_parts.append(f"[Source: {file_name}]\n{chunk['content']}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a helpful assistant for employees at this company.
Answer questions based ONLY on the document excerpts provided below.

Content rules:
1. Answer ONLY from the provided document excerpts. Never use outside knowledge.
2. Never guess or make up information.
3. If the answer is not in the documents, respond with exactly: "I couldn't find this information in the available documents."
4. If relevant, mention which document the information comes from.

Formatting rules:
1. Match format to content — steps use numbered lists, options use bullets, comparisons use tables.
2. Use **bold** for key terms and numbers.
3. Start longer answers with a one-sentence summary.
4. Write in markdown. Keep paragraphs short."""

        answer = ai.chat(
            messages=[{
                "role": "user",
                "content": f"Document excerpts:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
            }],
            system=system_prompt,
            max_tokens=1000,
            temperature=0.2,
        )
        sources = list(set([
            chunk["metadata"].get("file_name", "Unknown")
            for chunk in chunks
        ]))
        raw_chunks = [chunk["content"] for chunk in chunks]

        return {
            "answer":       answer,
            "sources":      sources,
            "chunks":       raw_chunks,
            "workspace_id": request.workspace_id,
        }

    except Exception as e:
        import traceback
        print(f"QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
