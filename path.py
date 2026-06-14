import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Clients ────────────────────────────────────────────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Request shape ──────────────────────────────────────────────────────────────
class GeneratePathRequest(BaseModel):
    document_ids: list[str]          # which processed documents to build the course from
    path_title: Optional[str] = None # optional hint for the course title


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-path")
async def generate_path(request: GeneratePathRequest):
    """
    Takes a list of already-ingested document_ids, reads ALL their content
    (not just top-5 similar chunks — the full text), and asks the AI to
    design a structured learning path: modules, objectives, sequence, duration.
    """
    try:
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids cannot be empty")

        # ── Step 1: Fetch ALL chunks for these documents, in order ─────────────
        # Unlike /query (which does similarity search for 5 chunks),
        # this pulls every chunk so the AI sees the complete document.
        result = supabase.table("document_chunks") \
            .select("document_id, content, chunk_index, metadata") \
            .in_("document_id", request.document_ids) \
            .order("document_id") \
            .order("chunk_index") \
            .execute()

        chunks = result.data

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No processed content found for the given document_ids. "
                       "Make sure these documents have completed AI Processing."
            )

        # ── Step 2: Reassemble full text per document ──────────────────────────
        docs = {}
        for chunk in chunks:
            doc_id = chunk["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "file_name":   chunk["metadata"].get("file_name", "Unknown document"),
                    "text_parts":  []
                }
            docs[doc_id]["text_parts"].append(chunk["content"])

        document_sections = []
        for doc_id, doc in docs.items():
            full_text = " ".join(doc["text_parts"])
            document_sections.append(f"=== Document: {doc['file_name']} ===\n{full_text}")

        combined_content = "\n\n".join(document_sections)

        # ── Step 3: Safety check — Groq free-tier rate limit, not context window ──
        # llama-3.1-8b-instant supports 128K context, but the FREE TIER rate limit
        # is 6,000 tokens-per-minute (input + output combined). This is the real
        # ceiling we hit in practice — context window is not the binding constraint.
        # Rough estimate: ~4 characters per token.
        OUTPUT_BUDGET = 2000   # max_tokens reserved for the AI's response
        TPM_LIMIT     = 6000   # Groq free tier limit for this model
        SAFETY_MARGIN = 500    # buffer for system prompt + formatting overhead

        approx_input_tokens = len(combined_content) // 4
        approx_total = approx_input_tokens + OUTPUT_BUDGET + SAFETY_MARGIN

        if approx_total > TPM_LIMIT:
            max_safe_input_tokens = TPM_LIMIT - OUTPUT_BUDGET - SAFETY_MARGIN
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This document set (~{approx_input_tokens} tokens) exceeds the current "
                    f"free-tier limit of ~{max_safe_input_tokens} tokens per request. "
                    f"Select fewer or shorter documents, or split this into multiple "
                    f"learning paths. (Groq free tier: 6,000 tokens/minute)"
                )
            )

        # ── Step 4: Ask the AI to design the course ─────────────────────────────
        system_prompt = """You are an instructional designer creating structured onboarding \
and training courses from company documents.

Given raw document content, design a learning path that:
1. Identifies the key topics and learning objectives.
2. Sequences them logically — foundational concepts before advanced ones.
3. Breaks the content into digestible modules (typically 3-8 modules).
4. Estimates a realistic duration for each module in minutes.
5. Notes which source document each module draws from.

Respond ONLY with valid JSON in this exact structure, nothing else, no markdown fences:
{
  "title": "string - overall course title",
  "description": "string - 1-2 sentence overview",
  "modules": [
    {
      "title": "string - module title",
      "objectives": ["string", "string"],
      "summary": "string - what this module covers, 2-3 sentences",
      "duration_minutes": number,
      "source_documents": ["string - file names this module is based on"]
    }
  ],
  "total_duration_minutes": number
}"""

        title_hint = f"\nThe course should be titled around: {request.path_title}" if request.path_title else ""

        user_prompt = f"""Design a learning path from the following document content:

{combined_content}
{title_hint}

Respond only with the JSON object described in the system prompt."""

        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.3
        )

        raw_output = response.choices[0].message.content
        path_data = json.loads(raw_output)

        return path_data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        print(f"GENERATE-PATH ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Path generation failed: {str(e)}")
