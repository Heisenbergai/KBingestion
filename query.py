import os
import voyageai
from groq import Groq
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Clients ────────────────────────────────────────────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Test endpoint ──────────────────────────────────────────────────────────────
@router.get("/test")
async def test_components():
    results = {}

    # Test 1: Voyage AI
    try:
        voyage.embed(["test"], model="voyage-3", input_type="query")
        results["voyage"] = "OK"
    except Exception as e:
        results["voyage"] = f"FAILED: {str(e)}"

    # Test 2: Supabase
    try:
        supabase.table("document_chunks").select("id").limit(1).execute()
        results["supabase"] = "OK"
    except Exception as e:
        results["supabase"] = f"FAILED: {str(e)}"

    # Test 3: Groq
    try:
        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10
        )
        results["groq"] = f"OK: {response.choices[0].message.content}"
    except Exception as e:
        results["groq"] = f"FAILED: {str(e)}"

    return results


# ── Request shape ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:    str
    asset_id:    Optional[str] = None
    match_count: Optional[int] = 5


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/query")
async def query_documents(request: QueryRequest):
    try:
        # Step 1: Embed the question
        result = voyage.embed(
            [request.question],
            model="voyage-3",
            input_type="query"
        )
        question_embedding = result.embeddings[0]

        # Step 2: Search Supabase for most relevant chunks
        search_result = supabase.rpc("match_chunks", {
            "query_embedding":  question_embedding,
            "match_count":      request.match_count,
            "filter_asset_id":  request.asset_id
        }).execute()

        chunks = search_result.data

        if not chunks:
            return {
                "answer":  "I couldn't find any relevant information in the available documents.",
                "sources": []
            }

        # Step 3: Build context from retrieved chunks
        context_parts = []
        for chunk in chunks:
            file_name = chunk["metadata"].get("file_name", "Unknown document")
            context_parts.append(f"[Source: {file_name}]\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 4: Ask Groq
        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant for employees at this company.
Answer questions based ONLY on the document excerpts provided below.

Content rules:
1. Answer ONLY from the provided document excerpts. Never use outside knowledge.
2. Never guess or make up information.
3. If the answer is not in the documents, respond with exactly: "I couldn't find this information in the available documents."
4. If relevant, mention which document the information comes from.

Formatting rules - make answers easy to scan and digest:
1. Match the format to the content. A simple factual question gets a short, direct answer - don't pad it with unnecessary structure.
2. For processes, steps, or sequences: use a numbered list.
3. For multiple related items, options, or features: use bullet points.
4. For comparisons (e.g. X vs Y, pros/cons, before/after): use a markdown table.
5. Use **bold** to highlight key terms, numbers, or names that the reader is likely scanning for.
6. For longer answers, start with a one-sentence summary, then provide supporting detail below it.
7. Write in markdown. Keep paragraphs short - 2-3 sentences max.
8. Never sacrifice accuracy for formatting. Structure should clarify the real content, not decorate it."""
                },
                {
                    "role": "user",
                    "content": f"Document excerpts:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
                }
            ]
        )

        answer = response.choices[0].message.content

        # Step 5: Return answer + sources
        sources = list(set([
            chunk["metadata"].get("file_name", "Unknown")
            for chunk in chunks
        ]))

        return {
    "answer":  answer,
    "sources": sources,
    "chunks":  [chunk["content"] for chunk in chunks]  # raw text for presentation generation
}

    except Exception as e:
        import traceback
        print(f"QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
