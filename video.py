import os
import io
import tempfile
import asyncio
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

# Import the slide renderer and TTS function
from slides import render_slide
from voiceover import synthesize_speech

load_dotenv()

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


# ── Request shape ──────────────────────────────────────────────────────────────
class Slide(BaseModel):
    title:     str
    bullets:   list[str]
    narration: str


class GenerateVideoRequest(BaseModel):
    module_title:   str
    module_label:   str         # e.g. "Module 1 · Sales Mastery"
    slides:         list[Slide]
    voice:          Optional[str] = "austin"
    document_id:    Optional[str] = None  # for storing the video URL back in Supabase


# ── Helpers ────────────────────────────────────────────────────────────────────
# TTS is now handled by synthesize_speech() imported from voiceover.py
# (Google Cloud Neural2 — 1M chars/month free, no daily token cap)


def save_to_supabase(video_bytes: bytes, document_id: str, module_label: str) -> str:
    """Uploads the video to Supabase Storage and returns a signed URL."""
    safe_label = module_label.replace(" ", "_").replace("·", "").replace("/", "_")[:60]
    path = f"explainer_videos/{document_id}/{safe_label}.mp4"

    supabase.storage.from_("knowledge-files").upload(
        path,
        video_bytes,
        {"content-type": "video/mp4", "upsert": "true"}
    )

    result = supabase.storage.from_("knowledge-files").create_signed_url(path, 86400)
    return result["signedURL"]


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-video")
async def generate_video(request: GenerateVideoRequest):
    """
    Full video pipeline for ONE module:

    1. For each slide:
       a. Render slide image (Pillow) → PNG
       b. Generate narration audio (Groq Orpheus TTS) → WAV
    2. For each slide: combine image + audio into a short video clip (FFmpeg)
    3. Concatenate all clips into one MP4
    4. Upload to Supabase Storage
    5. Return signed URL

    The frontend stores this URL in the module's explainer_video field,
    replacing the "Coming soon" placeholder.

    Processing time: ~30-90 seconds per module depending on number of slides
    and Groq TTS latency. This is expected — speed optimisation comes later.
    """
    if not request.slides:
        raise HTTPException(status_code=400, detail="slides cannot be empty")

    total = len(request.slides)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []

            for i, slide in enumerate(request.slides):
                slide_num = i + 1
                print(f"Processing slide {slide_num}/{total}: {slide.title}")

                # ── Step 1a: Render slide image ────────────────────────────────
                img = render_slide(
                    title=slide.title,
                    bullets=slide.bullets,
                    module_label=request.module_label,
                    slide_number=slide_num,
                    total_slides=total,
                )
                img_path = os.path.join(tmpdir, f"slide_{i:03d}.png")
                img.save(img_path, format="PNG")

                # ── Step 1b: Generate narration audio ──────────────────────────
                audio_bytes = synthesize_speech(slide.narration, request.voice)
                audio_path = os.path.join(tmpdir, f"audio_{i:03d}.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

                # ── Step 2: Combine slide image + audio into a video clip ───────
                # FFmpeg: use the image as a still frame for the duration of the audio.
                # -loop 1 = loop the image
                # -i audio = use audio to determine clip duration
                # -shortest = end when audio ends
                # -vf scale = ensure dimensions are divisible by 2 (required for H.264)
                # -pix_fmt yuv420p = standard MP4 compatibility
                clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
                clip_paths.append(clip_path)

                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", img_path,
                    "-i", audio_path,
                    "-shortest",
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
                    "-pix_fmt", "yuv420p",
                    "-r", "24",
                    clip_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise Exception(f"FFmpeg clip {slide_num} failed: {stderr.decode()}")

            # ── Step 3: Concatenate all clips into one MP4 ─────────────────────
            # FFmpeg concat demuxer: list all clips in a text file, then concat
            concat_list_path = os.path.join(tmpdir, "concat.txt")
            with open(concat_list_path, "w") as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path}'\n")

            final_path = os.path.join(tmpdir, "final.mp4")

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                final_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise Exception(f"FFmpeg concat failed: {stderr.decode()}")

            # ── Step 4: Upload to Supabase Storage ─────────────────────────────
            with open(final_path, "rb") as f:
                video_bytes = f.read()

            video_size_mb = len(video_bytes) / (1024 * 1024)
            print(f"Video generated: {video_size_mb:.1f} MB")

            # Always upload to Supabase and return JSON with signed URL.
            # Use document_id if provided, otherwise use a random UUID path.
            import uuid as _uuid
            storage_doc_id = request.document_id if request.document_id else str(_uuid.uuid4())
            signed_url = save_to_supabase(
                video_bytes,
                storage_doc_id,
                request.module_label
            )
            return {
                "success":     True,
                "video_url":   signed_url,
                "slide_count": total,
                "size_mb":     round(video_size_mb, 2),
                "message":     f"Video generated from {total} slides."
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"GENERATE-VIDEO ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
