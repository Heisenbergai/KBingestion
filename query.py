import os
import voyageai
import google.generativeai as genai
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

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── Request shape ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:    str            # the user's question
    asset_id:    Optional[str] = None  # if provided, search only within this asset
    match_count: Optional[int] = 5    # how many chunks to retrieve


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/query")
async def query_documents(request: QueryRequest):
    """
    Called by Lovable when a user asks a question.
    Flow: embed question → search vectors → build context → ask Claude → return answer.
    """
    try:
        # ── Step 1: Embed the question ─────────────────────────────────────────
        # input_type="query" is different from "document" — Voyage optimises
        # the embedding specifically for search queries, not storage.
        result = voyage.embed(
            [request.question],
            model="voyage-3",
            input_type="query"
        )
        question_embedding = result.embeddings[0]

        # ── Step 2: Search Supabase for the most relevant chunks ───────────────
        # This calls the match_chunks function we created in setup.sql.
        # It compares the question vector against all stored chunk vectors
        # and returns the top N most similar ones.
        search_result = supabase.rpc("match_chunks", {
            "query_embedding":  question_embedding,
            "match_count":      request.match_count,
            "filter_asset_id":  request.asset_id  # None = search all documents
        }).execute()

        chunks = search_result.data

        # If no relevant chunks found, return gracefully
        if not chunks:
            return {
                "answer":  "I couldn't find any relevant information in the available documents.",
                "sources": []
            }

        # ── Step 3: Build context from retrieved chunks ────────────────────────
        # We combine all retrieved chunks into one block of text.
        # Each chunk is labelled with its source file so Claude can reference it.
        context_parts = []
        for chunk in chunks:
            file_name = chunk["metadata"].get("file_name", "Unknown document")
            context_parts.append(f"[Source: {file_name}]\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # ── Step 4: Ask Gemini ─────────────────────────────────────────────────
        # The system instruction is the most important part of the entire system.
        # It tells Gemini:
        # - Answer ONLY from the provided documents (no hallucination)
        # - Be direct and precise
        # - If the answer isn't there, say so honestly
        system_instruction = """You are a helpful assistant for employees at this company.
Your job is to answer questions based ONLY on the document excerpts provided below.

Rules you must follow:
1. Answer ONLY from the provided document excerpts. Never use outside knowledge.
2. Be direct, precise, and clear. No fluff.
3. If the answer is not in the documents, respond with exactly: "I couldn't find this information in the available documents."
4. Never guess or make up information.
5. If relevant, mention which document the answer comes from."""

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_instruction
        )

        prompt = f"""Document excerpts:
{context}

Question: {request.question}

Answer:"""

        response = model.generate_content(prompt)
        answer = response.text

        # ── Step 5: Return answer + sources ────────────────────────────────────
        # Deduplicate source file names so user sees clean references
        sources = list(set([
            chunk["metadata"].get("file_name", "Unknown")
            for chunk in chunks
        ]))

        return {
            "answer":  answer,
            "sources": sources  # e.g. ["Employee Handbook.pdf", "Leave Policy.docx"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
