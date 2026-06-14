import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

from shared import fetch_combined_content, check_tpm_budget

load_dotenv()

router = APIRouter()

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

    This is the lightweight "outline only" version. For a full course with
    reading material, quizzes, and exercises per module, use /generate-course.
    """
    try:
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids cannot be empty")

        # Fetch and combine all document content (shared helper)
        combined_content = fetch_combined_content(request.document_ids)

        # Safety check against Groq's free-tier rate limit
        OUTPUT_BUDGET = 2000
        check_tpm_budget(combined_content, OUTPUT_BUDGET)

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
            max_tokens=OUTPUT_BUDGET,
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
