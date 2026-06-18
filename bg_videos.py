import os
import asyncio
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateCourseVideosRequest(BaseModel):
    training_id: str


# ── Status helpers ─────────────────────────────────────────────────────────────
def get_video_status(training_id: str) -> dict:
    result = supabase.table("generated_trainings") \
        .select("video_generation_status") \
        .eq("id", training_id) \
        .single() \
        .execute()
    return result.data.get("video_generation_status") or {}


def set_module_status(training_id: str, module_index: int,
                      status: str, video_url: str = None):
    """Updates one module's status in the video_generation_status JSON column."""
    current = get_video_status(training_id)
    current[str(module_index)] = {
        "status": status,  # pending | generating | done | failed
        "video_url": video_url
    }
    supabase.table("generated_trainings") \
        .update({"video_generation_status": current}) \
        .eq("id", training_id) \
        .execute()


def set_module_video_url(training_id: str, module_index: int, video_url: str):
    """
    Writes the video URL into the module inside course_data.
    This is what the learner view reads to show the video player.
    """
    result = supabase.table("generated_trainings") \
        .select("course_data") \
        .eq("id", training_id) \
        .single() \
        .execute()

    course_data = result.data.get("course_data", {})
    modules = course_data.get("modules", [])

    if module_index < len(modules):
        modules[module_index]["explainer_video"] = video_url
        course_data["modules"] = modules
        supabase.table("generated_trainings") \
            .update({"course_data": course_data}) \
            .eq("id", training_id) \
            .execute()


# ── Background worker ──────────────────────────────────────────────────────────
async def generate_all_videos_background(training_id: str):
    """
    Runs in the background after the HTTP response is returned.
    Generates a video for each module sequentially, updating status after each.
    """
    print(f"[bg-videos] Starting for training {training_id}")

    # Fetch training data
    result = supabase.table("generated_trainings") \
        .select("course_data, course_title") \
        .eq("id", training_id) \
        .single() \
        .execute()

    if not result.data:
        print(f"[bg-videos] Training {training_id} not found")
        return

    course_data  = result.data.get("course_data", {})
    course_title = result.data.get("title", "Course")
    modules      = course_data.get("modules", [])

    print(f"[bg-videos] {len(modules)} modules to process")

    for i, module in enumerate(modules):
        # Skip intro (index 0) and final thank-you module (last index)
        # — these are handled differently (no video, or static video)
        module_title = module.get("title", f"Module {i+1}")

        try:
            print(f"[bg-videos] Module {i+1}/{len(modules)}: {module_title}")
            set_module_status(training_id, i, "generating")

            # Call /generate-explainer to get slide content
            import httpx
            base_url = os.getenv("RAILWAY_PUBLIC_DOMAIN", "https://kbingestion-production.up.railway.app")
            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"

            async with httpx.AsyncClient(timeout=120) as client:
                # Step 1: Generate slide JSON
                explainer_res = await client.post(
                    f"{base_url}/generate-explainer",
                    json={
                        "module_title":    module_title,
                        "objectives":      module.get("objectives", []),
                        "reading_content": module.get("reading", {}).get("content", ""),
                    }
                )
                if explainer_res.status_code != 200:
                    raise Exception(f"Explainer failed: {explainer_res.text}")

                explainer_data = explainer_res.json()

                # Step 2: Generate video
                video_res = await client.post(
                    f"{base_url}/generate-video",
                    json={
                        "module_title":  module_title,
                        "module_label":  f"Module {i+1} · {course_title}",
                        "slides":        explainer_data.get("slides", []),
                        "voice":         "austin",
                    },
                    timeout=180  # video generation can take up to 3 minutes
                )
                if video_res.status_code != 200:
                    raise Exception(f"Video failed: {video_res.text}")

                video_data = video_res.json()
                video_url  = video_data.get("video_url")

                if not video_url:
                    raise Exception("No video_url in response")

            # Save URL into course_data modules
            set_module_video_url(training_id, i, video_url)
            set_module_status(training_id, i, "done", video_url)
            print(f"[bg-videos] Module {i+1} done: {video_url}")

            # Small pause between modules to avoid hitting TTS rate limits
            if i < len(modules) - 1:
                await asyncio.sleep(3)

        except Exception as e:
            print(f"[bg-videos] Module {i+1} FAILED: {str(e)}")
            set_module_status(training_id, i, "failed")

    print(f"[bg-videos] All modules processed for training {training_id}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/generate-course-videos")
async def generate_course_videos(
    request: GenerateCourseVideosRequest,
    background_tasks: BackgroundTasks
):
    """
    Fire-and-forget endpoint. Returns immediately after:
    1. Initialising all module statuses to 'pending'
    2. Scheduling the background worker

    The frontend polls /video-status/{training_id} to track progress.
    """
    # Verify training exists
    result = supabase.table("generated_trainings") \
        .select("id, course_data") \
        .eq("id", request.training_id) \
        .single() \
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training not found")

    modules = result.data.get("course_data", {}).get("modules", [])

    # Initialise all modules to 'pending'
    initial_status = {
        str(i): {"status": "pending", "video_url": None}
        for i in range(len(modules))
    }
    supabase.table("generated_trainings") \
        .update({"video_generation_status": initial_status}) \
        .eq("id", request.training_id) \
        .execute()

    # Schedule background task — returns to caller immediately
    background_tasks.add_task(
        generate_all_videos_background,
        request.training_id
    )

    return {
        "success":      True,
        "training_id":  request.training_id,
        "module_count": len(modules),
        "message":      f"Video generation started for {len(modules)} modules. Poll /video-status/{request.training_id} for progress."
    }


@router.get("/video-status/{training_id}")
async def get_video_generation_status(training_id: str):
    """
    Polled by the frontend every 10 seconds.
    Returns per-module status so the UI can update each module independently.
    """
    result = supabase.table("generated_trainings") \
        .select("video_generation_status, course_data") \
        .eq("id", training_id) \
        .single() \
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training not found")

    status = result.data.get("video_generation_status") or {}
    modules = result.data.get("course_data", {}).get("modules", [])

    # Count totals for frontend progress indicator
    statuses    = [v.get("status") for v in status.values()]
    total       = len(modules)
    done        = statuses.count("done")
    failed      = statuses.count("failed")
    generating  = statuses.count("generating")
    pending     = statuses.count("pending")
    all_done    = (done + failed) == total and total > 0

    return {
        "training_id": training_id,
        "modules":     status,       # { "0": {"status": "done", "video_url": "..."}, ... }
        "summary": {
            "total":      total,
            "done":       done,
            "failed":     failed,
            "generating": generating,
            "pending":    pending,
            "all_done":   all_done,
            "progress_pct": int((done + failed) / total * 100) if total > 0 else 0
        }
    }
