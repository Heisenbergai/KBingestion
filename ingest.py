import os
import re
import io
import fitz
import docx
import pptx
import voyageai
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import httpx

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


class IngestRequest(BaseModel):
    document_id:  str
    signed_url:   str
    mime_type:    str
    file_name:    str
    asset_id:     str
    workspace_id: str   # ← REQUIRED — isolates chunks per company


def download_file(signed_url: str) -> bytes:
    response = httpx.get(signed_url, timeout=60)
    response.raise_for_status()
    return response.content


def extract_text(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
         or file_name.lower().endswith(".docx"):
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation" \
         or file_name.lower().endswith(".pptx"):
        presentation = pptx.Presentation(io.BytesIO(file_bytes))
        slides_text = []
        for i, slide in enumerate(presentation.slides, 1):
            slide_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text)
            if slide_content:
                slides_text.append(f"[Slide {i}]\n" + "\n".join(slide_content))
        return "\n\n".join(slides_text)

    elif mime_type == "text/plain" or file_name.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t', ' ', text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return [c.page_content for c in splitter.create_documents([text])]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 50
    delay = 1.0

    i = 0
    while i < len(chunks):
        batch = chunks[i:i + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = voyage.embed(batch, model="voyage-3", input_type="document")
                all_embeddings.extend(result.embeddings)
                break
            except Exception as e:
                if "RateLimitError" in str(type(e)) or "rate limit" in str(e).lower():
                    wait = (attempt + 1) * 5
                    print(f"Rate limit, retrying in {wait}s")
                    time.sleep(wait)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise
        i += batch_size
        if i < len(chunks):
            time.sleep(delay)
    return all_embeddings


@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """
    Processes a document and stores chunks in the vector DB.
    workspace_id is stored with every chunk — this is what isolates
    each company's data from all other companies.
    """
    try:
        if not request.workspace_id:
            raise HTTPException(
                status_code=400,
                detail="workspace_id is required. Every document must belong to a workspace."
            )

        file_bytes = download_file(request.signed_url)
        raw_text   = extract_text(file_bytes, request.mime_type, request.file_name)

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted.")

        cleaned    = clean_text(raw_text)
        chunks     = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short to process.")

        embeddings = embed_chunks(chunks)

        rows = [
            {
                "document_id":  request.document_id,
                "asset_id":     request.asset_id,
                "workspace_id": request.workspace_id,  # ← stored with every chunk
                "content":      chunks[i],
                "embedding":    embeddings[i],
                "chunk_index":  i,
                "metadata": {
                    "file_name":    request.file_name,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "workspace_id": request.workspace_id,
                }
            }
            for i in range(len(chunks))
        ]

        supabase.table("document_chunks").insert(rows).execute()

        return {
            "success":        True,
            "document_id":    request.document_id,
            "workspace_id":   request.workspace_id,
            "chunks_created": len(chunks),
            "message":        f"Processed '{request.file_name}' into {len(chunks)} chunks."
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"INGEST ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
