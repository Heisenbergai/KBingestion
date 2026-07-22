import os
import re
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

GAP_MARKER = "⚠️ Not in your knowledge base:"


class QueryRequest(BaseModel):
    question:     str
    workspace_id: str           # ← REQUIRED — only search this workspace's chunks
    asset_id:     Optional[str] = None
    match_count:  Optional[int] = 8


def hybrid_search(question: str, workspace_id: str,
                  match_count: int = 8, asset_id: str = None) -> list[dict]:
    """
    Company-brain retrieval: vector + keyword fused with Reciprocal Rank
    Fusion, then boosted by source tier (official docs > curated notes >
    chat) and freshness. Falls back to pure-vector match_chunks_workspace
    if the hybrid RPC is unavailable (safety net during rollout).
    Always workspace-isolated.
    """
    embedding = ai.embed_texts([question])[0]
    try:
        result = supabase.rpc("match_chunks_hybrid", {
            "query_text":          question,
            "query_embedding":     embedding,
            "match_count":         match_count,
            "filter_workspace_id": workspace_id,
            "filter_asset_id":     asset_id,
        }).execute()
        return result.data or []
    except Exception as e:
        print(f"[query] hybrid search unavailable, falling back to vector-only: {e}")
        result = supabase.rpc("match_chunks_workspace", {
            "query_embedding":     embedding,
            "match_count":         match_count,
            "filter_asset_id":     asset_id,
            "filter_workspace_id": workspace_id,
        }).execute()
        return result.data or []


def build_context_and_citations(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Numbers each chunk as a citable source [n] and returns the LLM
    context block + a structured citations list for the frontend."""
    context_parts, citations = [], []
    for i, ch in enumerate(chunks, 1):
        meta = ch.get("metadata") or {}
        file_name = meta.get("file_name", "Unknown document")
        stype = ch.get("source_type") or meta.get("source_type") or "document"
        label = {"document": "company document", "meeting": "meeting note",
                 "slack": "team chat", "note": "curated note"}.get(stype, stype)
        context_parts.append(f"[{i}] {file_name} ({label}):\n{ch['content']}")
        citations.append({
            "index":       i,
            "file_name":   file_name,
            "snippet":     ch["content"][:200],
            "source_type": stype,
            "source_tier": ch.get("source_tier", 1),
        })
    return "\n\n---\n\n".join(context_parts), citations


def split_answer_and_gaps(text: str) -> tuple[str, Optional[str]]:
    """Separates the gap note (what the brain doesn't know) from the answer
    so the frontend can style it distinctly."""
    if GAP_MARKER in text:
        answer, gap = text.split(GAP_MARKER, 1)
        gap = gap.strip()
        return answer.strip(), (gap or None)
    return text.strip(), None


@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Company-brain search over ONLY the caller's workspace. Returns a
    synthesized answer with inline [n] citations, a structured citations
    list, and an explicit "gaps" note describing what the knowledge base
    does not yet cover. No cross-workspace data is ever returned.
    """
    try:
        if not request.workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required for all queries.")

        chunks = hybrid_search(
            request.question, request.workspace_id,
            match_count=request.match_count or 8, asset_id=request.asset_id,
        )

        if not chunks:
            return {
                "answer":  "I couldn't find anything about this in your knowledge base.",
                "citations": [],
                "sources": [],
                "chunks":  [],
                "gaps":    "Your knowledge base has no documents covering this topic yet. "
                           "Upload relevant documents or connect a data source, then try again.",
                "confidence":   "none",
                "workspace_id": request.workspace_id,
            }

        context, citations = build_context_and_citations(chunks)

        system_prompt = f"""You are the company's knowledge assistant. Answer the employee's \
question using ONLY the numbered sources below.

Citation rules:
- After each fact or claim, cite the source number(s) it came from, e.g. [1] or [2][3].
- Use ONLY information present in the sources. Never use outside knowledge or guess.
- Prefer official company documents over informal chat when sources disagree, and say so.

Honesty about gaps (important):
- If the sources only PARTIALLY answer the question, answer what you can with citations,
  then on a new final line write "{GAP_MARKER}" followed by a short description of what is missing.
- If the sources do NOT answer the question at all, write a one-line note under "{GAP_MARKER}"
  explaining what's missing, and nothing else.

Formatting:
- Markdown. Start longer answers with a one-sentence summary.
- Match structure to content (numbered steps, bullets for options, tables for comparisons).
- **Bold** key terms and numbers. Keep paragraphs short."""

        raw = ai.chat(
            messages=[{
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {request.question}\n\nAnswer with citations:"
            }],
            system=system_prompt,
            max_tokens=1000,
            temperature=0.2,
        )
        answer, gaps = split_answer_and_gaps(raw)

        # Confidence from the best chunk's semantic similarity
        top_sim = max((c.get("similarity") or 0) for c in chunks)
        confidence = "high" if top_sim >= 0.45 else "medium" if top_sim >= 0.3 else "low"

        # Only surface citations the model actually referenced (keeps the UI clean)
        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        cited = [c for c in citations if c["index"] in used] or citations

        return {
            "answer":       answer,
            "citations":    cited,
            "sources":      list(dict.fromkeys(c["file_name"] for c in cited)),  # unique, ordered
            "chunks":       [c["content"] for c in chunks],  # backward compat: decks/visuals flows
            "gaps":         gaps,
            "confidence":   confidence,
            "workspace_id": request.workspace_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
