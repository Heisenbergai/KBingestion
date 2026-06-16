import os
import base64
import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

# Available Neural2 voices (1M chars/month free — effectively unlimited at our scale)
# en-US-Neural2-D: male, deep, professional — good for training content
# en-US-Neural2-J: male, natural
# en-US-Neural2-F: female, clear
# en-US-Neural2-A: female, warm
VOICE_MAP = {
    "austin":    {"name": "en-US-Neural2-D", "gender": "MALE"},
    "male":      {"name": "en-US-Neural2-J", "gender": "MALE"},
    "female":    {"name": "en-US-Neural2-F", "gender": "FEMALE"},
    "warm":      {"name": "en-US-Neural2-A", "gender": "FEMALE"},
    "default":   {"name": "en-US-Neural2-D", "gender": "MALE"},
}


def synthesize_speech(text: str, voice_key: str = "austin") -> bytes:
    """
    Calls Google Cloud Text-to-Speech API and returns raw WAV audio bytes.

    Uses Neural2 voices — high quality, 1M characters/month free tier.
    No daily token limits (unlike Groq Orpheus which caps at 3,600/day).

    Returns LINEAR16 (WAV) audio at 24kHz sample rate.
    """
    voice = VOICE_MAP.get(voice_key, VOICE_MAP["default"])

    response = httpx.post(
        f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": voice["name"],
                "ssmlGender": voice["gender"],
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",   # WAV — compatible with FFmpeg
                "sampleRateHertz": 24000,       # 24kHz — good quality, reasonable size
                "speakingRate": 0.95,            # slightly slower than default — clearer for learning
                "pitch": 0.0,
                "volumeGainDb": 1.0,
            },
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Google TTS error {response.status_code}: {response.text}")

    audio_b64 = response.json().get("audioContent")
    if not audio_b64:
        raise RuntimeError("Google TTS returned no audio content")

    return base64.b64decode(audio_b64)


# ── Request shape ──────────────────────────────────────────────────────────────
class GenerateVoiceoverRequest(BaseModel):
    text:  str
    voice: Optional[str] = "austin"


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/generate-voiceover")
async def generate_voiceover(request: GenerateVoiceoverRequest):
    """
    Converts text to spoken WAV audio using Google Cloud Neural2 TTS.
    Returns raw WAV bytes directly.
    """
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
