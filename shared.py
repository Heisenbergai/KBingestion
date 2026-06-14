import os
from fastapi import HTTPException
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ── Shared Supabase client ──────────────────────────────────────────────────────
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# ── Groq free-tier rate limit constants ─────────────────────────────────────────
# Using gemma2-9b-it: 15,000 TPM (2.5x higher than llama-3.1-8b-instant's 6,000),
# but only an 8K context window (vs 128K for llama models). At this model's size,
# the CONTEXT WINDOW is the binding constraint, not TPM — so we check both.
TPM_LIMIT      = 15000
CONTEXT_WINDOW = 8192
SAFETY_MARGIN  = 500   # buffer for system prompt + formatting overhead


def fetch_combined_content(document_ids: list[str]) -> str:
    """
    Fetches ALL chunks for the given document_ids (not similarity search —
    every chunk, in order) and reassembles full text per document.

    Used by endpoints that need to "read" entire documents, like
    /generate-path and /generate-course — as opposed to /query, which
    only needs the top-5 most relevant chunks for a specific question.
    """
    result = supabase.table("document_chunks") \
        .select("document_id, content, chunk_index, metadata") \
        .in_("document_id", document_ids) \
        .order("document_id") \
        .order("chunk_index") \
        .execute()

    chunks = result.data

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No processed content found for the given document_ids. "
                   "Make sure these documents have completed AI Processing."
        )

    # Group chunks by document and reassemble full text
    docs = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "file_name":  chunk["metadata"].get("file_name", "Unknown document"),
                "text_parts": []
            }
        docs[doc_id]["text_parts"].append(chunk["content"])

    sections = []
    for doc_id, doc in docs.items():
        full_text = " ".join(doc["text_parts"])
        sections.append(f"=== Document: {doc['file_name']} ===\n{full_text}")

    return "\n\n".join(sections)


def check_tpm_budget(combined_content: str, output_budget: int) -> int:
    """
    Raises HTTPException if the request would exceed either:
    - Groq's free-tier TPM limit (15,000 tokens/minute for gemma2-9b-it), or
    - The model's context window (8,192 tokens for gemma2-9b-it)

    Whichever is smaller wins. Returns the approximate input token count
    if within budget.

    Rough estimate: ~4 characters per token.
    """
    approx_input_tokens = len(combined_content) // 4
    approx_total = approx_input_tokens + output_budget + SAFETY_MARGIN

    max_by_tpm     = TPM_LIMIT - output_budget - SAFETY_MARGIN
    max_by_context = CONTEXT_WINDOW - output_budget - SAFETY_MARGIN
    max_safe_input_tokens = min(max_by_tpm, max_by_context)

    if approx_total > TPM_LIMIT or approx_total > CONTEXT_WINDOW:
        binding_constraint = "context window (8K)" if max_by_context < max_by_tpm else "rate limit (15K TPM)"
        raise HTTPException(
            status_code=400,
            detail=(
                f"This document set (~{approx_input_tokens} tokens) exceeds the current "
                f"free-tier limit of ~{max_safe_input_tokens} tokens per request "
                f"(binding constraint: {binding_constraint}). "
                f"Select fewer or shorter documents, or split into multiple courses/paths."
            )
        )

    return approx_input_tokens
