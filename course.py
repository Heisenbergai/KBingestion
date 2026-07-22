import os
import json
import ai
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from shared import fetch_combined_content, check_tpm_budget

load_dotenv()

router = APIRouter()

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")


# ── Template definitions ─────────────────────────────────────────────────────────
# Templates differ in WHAT content to prioritize and HOW quizzes/exercises are
# framed — not in the underlying block structure. Every course gets the same
# interactive blocks: intro, expandable sections, key takeaways, flashcards,
# scenario quiz, exercise.
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


# ── Unsplash (per-module hero image) ────────────────────────────────────────────

def fetch_unsplash_image_url(query: str) -> Optional[dict]:
    """
    Returns {url, thumb, credit, credit_link} for one landscape photo, or
    None on any failure. Unlike presentation.py (which embeds bytes into
    PPTX), trainings render in the browser — so we return URLs and the
    attribution Unsplash requires.
    """
    if not UNSPLASH_KEY:
        return None
    try:
        res = httpx.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
            timeout=10,
        )
        results = res.json().get("results") or []
        if not results:
            return None
        photo = results[0]

        # Unsplash API guidelines: register a download event when a photo
        # is used. Best-effort — never fail course generation over this.
        download_location = photo.get("links", {}).get("download_location")
        if download_location:
            try:
                httpx.get(download_location,
                          headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                          timeout=10)
            except Exception:
                pass

        return {
            "url":         photo["urls"]["regular"],
            "thumb":       photo["urls"]["small"],
            "credit":      photo.get("user", {}).get("name", "Unsplash"),
            "credit_link": photo.get("user", {}).get("links", {}).get("html", "https://unsplash.com"),
        }
    except Exception as e:
        print(f"[course] Unsplash fetch failed for '{query}': {e}")
        return None


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateCourseRequest(BaseModel):
    template_id:        str             # one of the keys in TEMPLATES
    document_ids:       list[str]       # which processed documents to build the course from
    course_title:       Optional[str] = None  # optional hint for the course title
    additional_context: Optional[str] = None  # free-text guidance from the course creator
    enable_images:      Optional[bool] = True # fetch an Unsplash hero image per module


# ── Generation phase 1: course outline ─────────────────────────────────────────

def generate_outline(template: dict, combined_content: str,
                     course_title: Optional[str], additional_context: Optional[str]) -> dict:
    system_prompt = f"""You are an instructional designer planning a "{template['name']}" \
course from company documents. This template focuses on: {template['focus']}.

Plan a course with 3-5 modules that build on each other — foundational
concepts first, applied/advanced material later, so the sequence forms a
real learning path. Respond ONLY with valid JSON, no markdown fences:
{{
  "title": "overall course title",
  "description": "1-2 sentence overview",
  "outcomes": ["3-5 concrete 'what you'll be able to do' statements, each starting with a verb"],
  "audience": "one short phrase: who this course is for",
  "level": "Beginner | Intermediate | Advanced",
  "image_query": "2-4 word Unsplash search phrase for the course cover image",
  "modules": [
    {{
      "title": "module title",
      "summary": "one sentence: what the learner gets from this module (shown in the syllabus)",
      "objectives": ["2-3 learning objectives"],
      "focus": "one sentence: exactly what this module should cover from the source documents",
      "image_query": "2-4 word Unsplash stock-photo search phrase that visually fits this module (e.g. 'sales handshake meeting')",
      "source_documents": ["file names this module draws from"]
    }}
  ]
}}"""
    title_hint   = f"\nThe course should be titled around: {course_title}" if course_title else ""
    context_hint = f"\nAdditional guidance from the course creator: {additional_context}" if additional_context else ""

    user_prompt = f"""Source content:

{combined_content}
{title_hint}{context_hint}

Plan the {template['name']} course. Respond only with the JSON object."""

    return ai.chat_json(
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
        max_tokens=1500,
        temperature=0.4,
    )


# ── Generation phase 2: one module's interactive content ───────────────────────

def generate_module_content(template: dict, combined_content: str, module_plan: dict) -> dict:
    system_prompt = f"""You are writing one module of a "{template['name']}" course.
Write clear, engaging learning content based ONLY on the source documents.

Respond ONLY with valid JSON in this exact structure, no markdown fences:
{{
  "intro": "1-2 sentence hook introducing this module",
  "sections": [
    {{
      "title": "short section heading",
      "content": "150-250 words of clear teaching content for this section, written directly to the learner. Use short paragraphs. You may use simple markdown (**bold**, bullet lists starting with '- ')."
    }}
  ],
  "key_takeaways": ["3-5 one-sentence takeaways"],
  "flashcards": [
    {{ "front": "term, concept, or question", "back": "concise answer or definition (max 30 words)" }}
  ],
  "quiz": [
    {{
      "question": "the question text",
      "options": ["exactly 4 answer choices"],
      "correct_answer": "the correct option, must match one option exactly",
      "explanation": "one sentence explaining why this is correct"
    }}
  ],
  "exercise": {{ "type": "short label for the exercise type", "prompt": "the exercise instructions" }},
  "duration_minutes": 8
}}

Rules:
- 2-4 sections, 4-6 flashcards, exactly 3 quiz questions.
- Quiz style: {template['quiz_style']}
- Exercise style: {template['exercise_style']}
- duration_minutes: realistic estimate to complete this module."""

    user_prompt = f"""Source documents:

{combined_content}

Write the module "{module_plan.get('title', '')}".
Module focus: {module_plan.get('focus', '')}
Learning objectives: {json.dumps(module_plan.get('objectives', []))}

Respond only with the JSON object."""

    return ai.chat_json(
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
        max_tokens=2500,
        temperature=0.4,
    )


def _normalize_quiz(quiz: list) -> list:
    """
    Guarantees every question has exactly 4 options and a correct_answer
    that matches one of them — the frontend compares strings directly,
    so a mismatch would make a question unanswerable.
    """
    normalized = []
    for q in quiz or []:
        options = [str(o) for o in (q.get("options") or [])][:4]
        if len(options) < 2:
            continue
        correct = str(q.get("correct_answer", ""))
        if correct not in options:
            match = next((o for o in options if o.strip().lower() == correct.strip().lower()), None)
            if match:
                correct = match
            else:
                print(f"[course] quiz correct_answer mismatch, defaulting to first option: {correct!r}")
                correct = options[0]
        normalized.append({
            "question":       str(q.get("question", "")),
            "options":        options,
            "correct_answer": correct,
            "explanation":    str(q.get("explanation", "")),
        })
    return normalized


def _module_to_blocks(content: dict) -> list[dict]:
    """
    Assembles the module's content into an ordered list of typed blocks.
    Lovable renders blocks generically by type, so new block types can be
    added later without breaking saved courses:
      intro | expand (checkable) | key_takeaways | flashcards | quiz | exercise
    """
    blocks = []
    if content.get("intro"):
        blocks.append({"type": "intro", "text": str(content["intro"])})

    for section in content.get("sections") or []:
        blocks.append({
            "type":      "expand",
            "title":     str(section.get("title", "Section")),
            "content":   str(section.get("content", "")),
            "checkable": True,   # learner ticks it off after reading → progress
        })

    takeaways = [str(t) for t in (content.get("key_takeaways") or [])]
    if takeaways:
        blocks.append({"type": "key_takeaways", "points": takeaways})

    cards = [
        {"front": str(c.get("front", "")), "back": str(c.get("back", ""))}
        for c in (content.get("flashcards") or [])
        if c.get("front") and c.get("back")
    ]
    if cards:
        blocks.append({"type": "flashcards", "cards": cards})

    quiz = _normalize_quiz(content.get("quiz"))
    if quiz:
        blocks.append({"type": "quiz", "questions": quiz})

    exercise = content.get("exercise") or {}
    if exercise.get("prompt"):
        blocks.append({
            "type":          "exercise",
            "exercise_type": str(exercise.get("type", "Practice")),
            "prompt":        str(exercise.get("prompt", "")),
        })

    return blocks


# ── Main route ─────────────────────────────────────────────────────────────────
# NOTE: plain `def`, not `async def` — this endpoint makes several sequential
# blocking LLM calls (30-60s total for a 5-module course). FastAPI runs sync
# endpoints in a threadpool, so the event loop stays free for other requests.
@router.post("/generate-course")
def generate_course(request: GenerateCourseRequest):
    """
    Interactive course generation (format: interactive_v2).

    Two-phase generation for reliability: one outline call plans the
    modules, then each module's content is generated in its own call —
    small JSON responses parse far more reliably than one giant one, and
    a single module's retry doesn't throw away the whole course.

    Each module = ordered interactive blocks (expand sections with check
    marks, key takeaways, flashcards, scenario quiz, exercise) plus an
    Unsplash hero image. No audio/video — that was the old V1 format.
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

        combined_content = fetch_combined_content(request.document_ids)

        OUTPUT_BUDGET = 2500  # largest single-call output (one module)
        check_tpm_budget(combined_content, OUTPUT_BUDGET)

        # Phase 1 — outline
        outline = generate_outline(
            template, combined_content, request.course_title, request.additional_context
        )
        module_plans = (outline.get("modules") or [])[:5]
        if not module_plans:
            raise HTTPException(status_code=500, detail="AI returned no modules in the course outline.")

        # Phase 2 — module content + image, one module at a time
        modules = []
        total_minutes = 0
        for plan in module_plans:
            content = generate_module_content(template, combined_content, plan)
            blocks  = _module_to_blocks(content)

            duration = content.get("duration_minutes") or 8
            try:
                duration = max(1, int(duration))
            except (TypeError, ValueError):
                duration = 8
            total_minutes += duration

            image = None
            if request.enable_images:
                image = fetch_unsplash_image_url(plan.get("image_query") or plan.get("title", ""))

            modules.append({
                "title":            str(plan.get("title", "Module")),
                "summary":          str(plan.get("summary", "")),
                "objectives":       [str(o) for o in (plan.get("objectives") or [])],
                "image":            image,   # {url, thumb, credit, credit_link} or null
                "duration_minutes": duration,
                "source_documents": [str(s) for s in (plan.get("source_documents") or [])],
                "blocks":           blocks,
                # how many blocks count toward the module's progress bar
                "progress_total":   sum(1 for b in blocks if b.get("checkable")) + 1,  # +1 for the quiz
            })

        # Course-level cover image (falls back to first module's image query)
        cover = None
        if request.enable_images:
            cover = fetch_unsplash_image_url(
                outline.get("image_query")
                or module_plans[0].get("image_query")
                or str(outline.get("title", "business training"))
            )

        level = str(outline.get("level", "Beginner"))
        if level not in ("Beginner", "Intermediate", "Advanced"):
            level = "Beginner"

        return {
            "format":                 "interactive_v2",
            "template":               request.template_id,
            "title":                  str(outline.get("title", request.course_title or "Training Course")),
            "description":            str(outline.get("description", "")),
            "outcomes":               [str(o) for o in (outline.get("outcomes") or [])],
            "audience":               str(outline.get("audience", "")),
            "level":                  level,
            "cover_image":            cover,   # {url, thumb, credit, credit_link} or null
            "modules":                modules,
            "total_duration_minutes": total_minutes,
        }

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        print(f"GENERATE-COURSE ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Course generation failed: {str(e)}")
