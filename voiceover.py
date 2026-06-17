import os
import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def synthesize_speech(text: str, voice: str = "austin") -> bytes:
    """
    Converts text to WAV audio using Groq Orpheus TTS.
    Free tier: 3,600 tokens/day. Good for dev/demo usage.
    """
    response = httpx.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "canopylabs/orpheus-v1-english",
            "input": text,
            "voice": voice,
            "response_format": "wav",
        },
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Groq TTS error {response.status_code}: {response.text}")
    return response.content


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateVoiceoverRequest(BaseModel):
    text:  str
    voice: Optional[str] = "austin"


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-voiceover")
async def generate_voiceover(request: GenerateVoiceoverRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text cannot be empty")
        audio_bytes = synthesize_speech(request.text, request.voice)
        return Response(content=audio_bytes, media_type="audio/wav")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        print(f"VOICEOVER ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Voiceover failed: {str(e)}")
