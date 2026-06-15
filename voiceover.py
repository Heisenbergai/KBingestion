import os
import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateVoiceoverRequest(BaseModel):
    text:  str
    voice: Optional[str] = "austin"  # Orpheus English voice — can try others later


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-voiceover")
async def generate_voiceover(request: GenerateVoiceoverRequest):
    """
    Converts text to spoken audio using Groq's Orpheus TTS model.
    Returns raw WAV audio bytes directly — no storage, no SDK dependency,
    just a direct HTTP call to Groq's OpenAI-compatible speech endpoint.

    This is Step 1 of the video pipeline: confirm TTS works and sounds good
    before building slide rendering (Step 2) and video assembly (Step 3).
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text cannot be empty")

        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "canopylabs/orpheus-v1-english",
                "input": request.text,
                "voice": request.voice,
                "response_format": "wav",
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq TTS error: {response.text}"
            )

        return Response(content=response.content, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"VOICEOVER ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Voiceover generation failed: {str(e)}")
