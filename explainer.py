import os
import json
import ai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateExplainerRequest(BaseModel):
    module_title:    str
    objectives:      list[str]
    reading_content: str


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-explainer")
async def generate_explainer(request: GenerateExplainerRequest):
    """
    Takes a single module's already-generated content (title, objectives,
    reading) and turns it into a short slide deck: 3-5 slides, each with a
    title, bullet points for display, and a narration script written to be
    SPOKEN aloud via the browser's text-to-speech (not read silently).

    This is per-module and operates on content already in the frontend —
    no document fetching, no TPM concerns (input is small by construction).
    """
    try:
        system_prompt = """You are creating a short narrated slide deck to explain \
a training module's content.

Create 3-5 slides. For each slide, provide:
1. "title" - short slide heading (3-6 words)
2. "bullets" - 2-4 short bullet points for on-screen display (visual, terse)
3. "narration" - 1-3 sentences written to be SPOKEN ALOUD by text-to-speech. \
Write conversationally, as if explaining to someone out loud — not as written \
bullet points. Spell things out naturally (avoid abbreviations, symbols, or \
notation that sounds awkward when read aloud).

The first slide should introduce the topic. The last slide should summarize \
the key takeaway. Together, the narrations across all slides should walk \
through the module's content in a natural spoken explanation.

Respond ONLY with valid JSON, no markdown fences:
{
  "slides": [
    { "title": "...", "bullets": ["...", "..."], "narration": "..." }
  ]
}"""

        objectives_text = "\n".join(f"- {obj}" for obj in request.objectives)

        user_prompt = f"""Module: {request.module_title}

Objectives:
{objectives_text}

Content:
{request.reading_content}

Create the slide deck as described."""

        slide_data = ai.chat_json(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            max_tokens=1200,
            temperature=0.4,
        )

        return slide_data

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        print(f"GENERATE-EXPLAINER ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Explainer generation failed: {str(e)}")
