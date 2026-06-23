import os
import uuid
import asyncio
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── In-memory job store ────────────────────────────────────────────────────────
# Railway is stateless — no DB access. Jobs live in memory only.
# On Railway restart, jobs are cleared — Lovable handles persistence.
JOBS: dict[str, dict] = {}

BASE_URL = os.getenv(
    "RAILWAY_PUBLIC_DOMAIN",
    "https://kbingestion-production.up.railway.app"
)
if not BASE_URL.startswith("http"):
    BASE_URL = f"https://{BASE_URL}"


# ── Request shapes ─────────────────────────────────────────────────────────────
class ModuleInput(BaseModel):
    index:         int
    title:         str
    objectives:    list[str]
    reading:       str          # reading.content from the module
    course_title:  str


class GenerateCourseVideosRequest(BaseModel):
    modules:      list[ModuleInput]
    voice:        Optional[str] = "austin"


# ── Background worker ──────────────────────────────────────────────────────────
async def process_modules(job_id: str, modules: list[ModuleInput], voice: str):
    """
    Processes each module sequentially:
    1. Generate slide JSON (/generate-explainer)
    2. Generate video (/generate-video → R2)
    3. Update in-memory job status

    Railway never touches Lovable's DB.
    Lovable polls /job-status/{job_id} and saves video_url itself.
    """
    print(f"[job:{job_id}] Starting — {len(modules)} modules")

    for module in modules:
        idx = module.index
        JOBS[job_id]["modules"][idx]["status"] = "generating"
        print(f"[job:{job_id}] Module {idx+1}/{len(modules)}: {module.title}")

        try:
            async with httpx.AsyncClient(timeout=180) as client:

                # Step 1 — Generate slide content
                explainer_res = await client.post(
                    f"{BASE_URL}/generate-explainer",
                    json={
                        "module_title":    module.title,
                        "objectives":      module.objectives,
                        "reading_content": module.reading,
                    }
                )
                if explainer_res.status_code != 200:
                    raise Exception(f"Explainer failed ({explainer_res.status_code}): {explainer_res.text[:200]}")

                slides = explainer_res.json().get("slides", [])
                if not slides:
                    raise Exception("Explainer returned no slides")

                # Step 2 — Generate video
                video_res = await client.post(
                    f"{BASE_URL}/generate-video",
                    json={
                        "module_title":  module.title,
                        "module_label":  f"Module {idx+1} · {module.course_title}",
                        "slides":        slides,
                        "voice":         voice,
                    }
                )
                if video_res.status_code != 200:
                    raise Exception(f"Video failed ({video_res.status_code}): {video_res.text[:200]}")

                video_data = video_res.json()
                video_url  = video_data.get("video_url")

                if not video_url:
                    raise Exception("No video_url in response")

            # Update in-memory status — Lovable polls this
            JOBS[job_id]["modules"][idx]["status"]    = "done"
            JOBS[job_id]["modules"][idx]["video_url"] = video_url
            print(f"[job:{job_id}] Module {idx+1} done: {video_url}")

        except Exception as e:
            print(f"[job:{job_id}] Module {idx+1} FAILED: {str(e)}")
            JOBS[job_id]["modules"][idx]["status"] = "failed"
            JOBS[job_id]["modules"][idx]["error"]  = str(e)

        # Small pause between modules — avoids TTS rate limit bursts
        if idx < len(modules) - 1:
            await asyncio.sleep(3)

    # Mark overall job complete
    statuses = [m["status"] for m in JOBS[job_id]["modules"].values()]
    JOBS[job_id]["status"] = "failed" if all(s == "failed" for s in statuses) else "done"
    print(f"[job:{job_id}] All modules processed. Job status: {JOBS[job_id]['status']}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/generate-course-videos")
async def generate_course_videos(
    request: GenerateCourseVideosRequest,
    background_tasks: BackgroundTasks
):
    """
    Fire-and-forget endpoint.
    Lovable sends module data → Railway returns job_id immediately →
    Lovable polls /job-status/{job_id} → saves video_urls to its own DB.
    """
    if not request.modules:
        raise HTTPException(status_code=400, detail="modules list cannot be empty")

    job_id = str(uuid.uuid4())

    # Initialise job in memory
    JOBS[job_id] = {
        "status":  "processing",
        "modules": {
            str(m.index): {
                "index":     m.index,
                "title":     m.title,
                "status":    "pending",
                "video_url": None,
                "error":     None,
            }
            for m in request.modules
        }
    }

    # Schedule background processing — returns immediately
    background_tasks.add_task(process_modules, job_id, request.modules, request.voice)

    return {
        "job_id":       job_id,
        "status":       "processing",
        "module_count": len(request.modules),
        "message":      f"Processing started. Poll /job-status/{job_id} every 10 seconds."
    }


@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Polled by Lovable every 10 seconds.
    Returns per-module status + video_url as each module completes.
    Lovable saves video_url to its own DB when status is 'done'.
    """
    if job_id not in JOBS:
        raise HTTPException(
            status_code=404,
            detail="Job not found. Railway may have restarted — please regenerate."
        )

    job = JOBS[job_id]
    modules = list(job["modules"].values())

    total     = len(modules)
    done      = sum(1 for m in modules if m["status"] == "done")
    failed    = sum(1 for m in modules if m["status"] == "failed")
    all_done  = (done + failed) == total

    return {
        "job_id":   job_id,
        "status":   job["status"],
        "all_done": all_done,
        "summary": {
            "total":        total,
            "done":         done,
            "failed":       failed,
            "pending":      sum(1 for m in modules if m["status"] == "pending"),
            "generating":   sum(1 for m in modules if m["status"] == "generating"),
            "progress_pct": int((done + failed) / total * 100) if total > 0 else 0
        },
        "modules": modules   # includes video_url for completed modules
    }


@router.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Optional: Lovable can call this after saving all URLs to free memory."""
    if job_id in JOBS:
        del JOBS[job_id]
    return {"deleted": job_id}
