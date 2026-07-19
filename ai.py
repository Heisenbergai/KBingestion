"""
Central AI layer — ALL embedding and LLM calls go through this file.

Provider: AWS Bedrock (Amazon-native models only, so free AWS credits apply)
  - Chat:       Amazon Nova Lite  (us.amazon.nova-lite-v1:0 inference profile)
  - Embeddings: Amazon Titan Text Embeddings V2 (amazon.titan-embed-text-v2:0)

Titan v2 is set to 1024 dimensions — the SAME size as the old Voyage
voyage-3 embeddings, so the pgvector column and match_chunks_workspace()
need no schema change. BUT embeddings from different models are not
comparable: every document must be re-uploaded after switching so its
chunks are re-embedded with Titan. Old Voyage vectors will simply never
match Titan query vectors well.

To change models later, set env vars on Railway (no code change):
  BEDROCK_CHAT_MODEL   (default us.amazon.nova-lite-v1:0 — us.amazon.nova-micro-v1:0 is even cheaper, text-only)
  BEDROCK_EMBED_MODEL  (default amazon.titan-embed-text-v2:0)

Uses the existing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
env vars already configured on Railway (same ones Polly uses).
The IAM user needs bedrock:InvokeModel permission and model access must be
enabled once in the Bedrock console (see deploy notes).
"""
import os
import json
import re
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
CHAT_MODEL  = os.getenv("BEDROCK_CHAT_MODEL", "us.amazon.nova-lite-v1:0")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# Must stay 1024 to match the vector(1024) column in document_chunks
EMBED_DIMENSIONS = 1024

# Titan v2 accepts up to ~8k tokens of input; stay safely under it.
MAX_EMBED_CHARS = 25000

# adaptive retry mode automatically backs off and retries on
# ThrottlingException — this replaces the manual rate-limit retry
# loops we needed with Groq/Voyage free tiers.
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts with Titan v2. Titan takes ONE text per API
    call (no batch endpoint), so this loops — callers should report
    progress per batch for large documents. Normalized vectors, 1024 dims.
    """
    embeddings = []
    for text in texts:
        body = json.dumps({
            "inputText":  text[:MAX_EMBED_CHARS],
            "dimensions": EMBED_DIMENSIONS,
            "normalize":  True,
        })
        response = bedrock.invoke_model(modelId=EMBED_MODEL, body=body)
        result = json.loads(response["body"].read())
        embeddings.append(result["embedding"])
    return embeddings


def chat(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.5,
    model: str = None,
) -> str:
    """
    Chat completion via the Bedrock Converse API (works the same across
    all Bedrock models). messages = [{"role": "user"|"assistant",
    "content": "..."}] in order; system prompt is passed separately —
    Bedrock does NOT accept a system role inside the messages list.

    Bedrock requires the conversation to START with a user turn and to
    strictly ALTERNATE user/assistant. Client-supplied history can't be
    trusted to satisfy that (e.g. a bot greeting first, or two user
    messages after a failed send), so we normalize here: drop leading
    assistant turns and merge consecutive same-role turns.
    """
    normalized = []
    for m in messages:
        if not normalized and m["role"] == "assistant":
            continue  # conversation must start with a user turn
        if normalized and normalized[-1]["role"] == m["role"]:
            normalized[-1]["content"] += "\n" + m["content"]
        else:
            normalized.append({"role": m["role"], "content": m["content"]})

    conv_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in normalized
    ]
    kwargs = {
        "modelId":         model or CHAT_MODEL,
        "messages":        conv_messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    response = bedrock.converse(**kwargs)
    return response["output"]["message"]["content"][0]["text"]


def _extract_json(text: str):
    """
    Nova has no response_format={"type": "json_object"} equivalent — it
    returns plain text that *should* be JSON but may be wrapped in
    markdown fences or prose. Strip fences, then parse from the first
    '{' or '[' to its matching end.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost JSON object/array in the text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end   = text.rfind(close_ch)
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def chat_json(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 2000,
    temperature: float = 0.3,
    model: str = None,
):
    """
    Chat completion that must return parsed JSON. Retries once with an
    explicit correction message if the first response isn't valid JSON.
    """
    raw = chat(messages, system=system, max_tokens=max_tokens,
               temperature=temperature, model=model)
    try:
        return _extract_json(raw)
    except json.JSONDecodeError:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content":
                "That was not valid JSON. Respond again with ONLY the valid "
                "JSON object — no markdown fences, no explanation, no text "
                "before or after it."},
        ]
        raw = chat(retry_messages, system=system, max_tokens=max_tokens,
                   temperature=0.1, model=model)
        return _extract_json(raw)
