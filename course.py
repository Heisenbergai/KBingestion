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


# ── Template definitions ─────────────────────────────────────────────────────────
# Templates differ in WHAT content to prioritize and HOW quizzes/exercises are
# framed — not in the underlying slot structure. Every course gets the same
# shape: modules with reading, quiz, exercise, plus placeholder video slots
# for Phase 2/3.
TEMPLATES = {
    "sales_training": {
        "name": "Sales Training",
        "focus": "sales techniques, pitch structure, objection handling, customer psychology, and closing strategies",
        "quiz_style": "scenario-based questions that test how the learner would respond in real sales situations (e.g. 'A customer says X — what's the best response?')",
        "exercise_style": "a roleplay or pitch-practice prompt the learner can act out"
    },
    "employee_onboarding": {
        "name": "Employee Onboarding",
        "focus": "company policies, culture, values, processes, and tools new employees need to know",
        "quiz_style": "comprehension questions that check recall of key policies and processes",
        "exercise_style": "a reflection prompt asking the learner to plan concrete actions for their first week based on what they learned"
    },
    "manager_training": {
        "name": "Manager Training",
        "focus": "leadership principles, performance management, decision-making frameworks, and conflict resolution",
        "quiz_style": "situational judgment questions presenting a management scenario with multiple response options",
        "exercise_style": "a case study reflection asking the learner to outline how they'd handle a specific situation"
    },
    "customer_support_training": {
        "name": "Customer Support Training",
        "focus": "support protocols, troubleshooting steps, escalation procedures, and communication tone standards",
        "quiz_style": "scenario-based questions about handling specific customer situations correctly",
        "exercise_style": "a prompt asking the learner to draft a response to a sample customer query"
    },
    "product_training": {
        "name": "Product Training",
        "focus": "product features, specifications, use cases, and answers to common customer questions",
        "quiz_style": "factual questions testing recall of specific features and specifications",
        "exercise_style": "a prompt asking the learner to explain a specific feature in their own words as if to a customer"
    },
}


# ── Endpoint: list available templates ───────────────────────────────────────────
@router.get("/templates")
async def list_templates():
    """
    Returns all available course templates so the frontend can show
    a selection dropdown. Returns id + display name + focus description only.
    """
    return {
        "templates": [
            {
                "id": key,
                "name": val["name"],
                "focus": val["focus"]
            }
            for key, val in TEMPLATES.items()
        ]
    }


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateCourseRequest(BaseModel):
    template_id:        str             # one of the keys in TEMPLATES
    document_ids:       list[str]       # which processed documents to build the course from
    course_title:       Optional[str] = None  # optional hint for the course title
    additional_context: Optional[str] = None  # free-text guidance from the course creator
                                                # (focus areas, tone, specific topics to emphasize)


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-course")
async def generate_course(request: GenerateCourseRequest):
    """
    Takes a template + a list of already-ingested document_ids, reads ALL
    their content, and asks the AI to assemble a complete course following
    that template: modules with reading material, a quiz, and a practical
    exercise — framed according to the template's focus area.

    video_clip and explainer_video are always included as null placeholders.
    These are Phase 2/3 features (meeting clips, AI explainer videos) — the
    schema is stable now so the frontend can be built against it today.
    """
    try:
        if request.template_id not in TEMPLATES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown template_id '{request.template_id}'. "
                       f"Valid options: {list(TEMPLATES.keys())}"
            )

        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids cannot be empty")

        template = TEMPLATES[request.template_id]

        # Fetch and combine all document content (shared helper)
        combined_content = fetch_combined_content(request.document_ids)

        # Safety check — full course generation produces more output than
        # /generate-path (reading + quiz + exercise per module), so we
        # reserve a larger output budget, which means a smaller safe input size.
        OUTPUT_BUDGET = 3000
        check_tpm_budget(combined_content, OUTPUT_BUDGET)

        # ── Build the template-aware system prompt ──────────────────────────────
        system_prompt = f"""You are an instructional designer creating a "{template['name']}" \
course from company documents.

This template focuses on: {template['focus']}.

Design a complete course with 3-6 modules. For EACH module, generate:

1. "title" - module title
2. "objectives" - array of 2-3 learning objectives
3. "reading" - object with:
   - "summary": one sentence overview
   - "content": a concise reading passage (100-200 words) covering this module's topic, written clearly for learners
4. "quiz" - array of exactly 3 multiple-choice questions, each with:
   - "question": the question text
   - "options": array of exactly 4 answer choices
   - "correct_answer": the correct option (must match one of the options exactly)
   - "explanation": one sentence explaining why this is correct
   Quiz style: {template['quiz_style']}
5. "exercise" - object with:
   - "type": short label for the exercise type
   - "prompt": the exercise instructions. Style: {template['exercise_style']}
6. "duration_minutes" - realistic estimate for this module
7. "source_documents" - array of file names this module draws from

Respond ONLY with valid JSON in this exact structure, nothing else, no markdown fences:
{{
  "title": "string - overall course title",
  "description": "string - 1-2 sentence overview",
  "modules": [
    {{
      "title": "...",
      "objectives": ["...", "..."],
      "reading": {{ "summary": "...", "content": "..." }},
      "quiz": [
        {{ "question": "...", "options": ["...","...","...","..."], "correct_answer": "...", "explanation": "..." }}
      ],
      "exercise": {{ "type": "...", "prompt": "..." }},
      "duration_minutes": number,
      "source_documents": ["..."]
    }}
  ],
  "total_duration_minutes": number
}}"""

        title_hint = f"\nThe course should be titled around: {request.course_title}" if request.course_title else ""
        context_hint = f"\nAdditional context from the course creator (use this to guide focus, tone, and emphasis): {request.additional_context}" if request.additional_context else ""

        user_prompt = f"""Source content:

{combined_content}
{title_hint}{context_hint}

Design a {template['name']} course following the structure described in the system prompt. \
Respond only with the JSON object."""

        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=OUTPUT_BUDGET,
            temperature=0.4
        )

        raw_output = response.choices[0].message.content
        course_data = json.loads(raw_output)

        # ── Add template id + placeholder video slots ───────────────────────────
        # Always present, always null for now — Phase 2/3 will populate these
        # without requiring any schema change on the frontend.
        course_data["template"] = request.template_id

        for module in course_data.get("modules", []):
            module["video_clip"] = None       # Phase 2 — meeting/training clips
            module["explainer_video"] = None  # Phase 3 — AI-generated explainer

        return course_data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        print(f"GENERATE-COURSE ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Course generation failed: {str(e)}")
