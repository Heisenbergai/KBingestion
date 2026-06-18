import os
import struct
import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")

# Polly neural voice mapping — mirrors Groq voice names for easy swap
POLLY_VOICE_MAP = {
    "austin":  "Matthew",   # male, US English neural
    "male":    "Matthew",
    "female":  "Joanna",
    "warm":    "Joanna",
    "default": "Matthew",
}


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000,
               channels: int = 1, bit_depth: int = 16) -> bytes:
    """Wraps raw PCM bytes in a WAV header so FFmpeg can decode it."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size + 36,
        b"WAVE",
        b"fmt ",
        16,
        1,                                              # PCM
        channels,
        sample_rate,
        sample_rate * channels * bit_depth // 8,       # byte rate
        channels * bit_depth // 8,                     # block align
        bit_depth,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _groq_tts(text: str, voice: str) -> bytes:
    """Calls Groq Orpheus TTS. Returns WAV bytes. Raises on 429 or other errors."""
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
    if response.status_code == 429:
        raise RuntimeError("RATE_LIMIT_429")
    if response.status_code != 200:
        raise RuntimeError(f"Groq TTS error {response.status_code}: {response.text}")
    return response.content


def _polly_tts(text: str, voice: str) -> bytes:
    """
    Calls Amazon Polly Neural TTS as fallback when Groq is rate-limited.
    Uses Polly's REST API directly (no boto3 dependency).
    Returns WAV bytes.
    """
    try:
        import boto3
        polly_voice = POLLY_VOICE_MAP.get(voice, POLLY_VOICE_MAP["default"])
        client = boto3.client(
            "polly",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        response = client.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=polly_voice,
            Engine="neural",
            SampleRate="16000",
        )
        pcm_bytes = response["AudioStream"].read()
        return pcm_to_wav(pcm_bytes, sample_rate=16000)
    except Exception as e:
        raise RuntimeError(f"Amazon Polly TTS failed: {str(e)}")


def synthesize_speech(text: str, voice: str = "austin") -> bytes:
    """
    Primary: Groq Orpheus TTS (best quality, 3,600 tokens/day free).
    Fallback: Amazon Polly Neural (when Groq hits daily rate limit).
    Always returns WAV bytes.
    """
    try:
        audio = _groq_tts(text, voice)
        print(f"TTS: Groq Orpheus ({len(audio)} bytes)")
        return audio
    except RuntimeError as e:
        if "RATE_LIMIT_429" in str(e):
            print("TTS: Groq rate limit hit — falling back to Amazon Polly Neural")
            audio = _polly_tts(text, voice)
            print(f"TTS: Amazon Polly ({len(audio)} bytes)")
            return audio
        raise


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
