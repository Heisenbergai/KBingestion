import os
import re
import io
import csv
import fitz
import docx
import pptx
import openpyxl
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


def extract_xlsx(file_bytes: bytes) -> str:
    """
    Reads every sheet in the workbook and converts each row to a readable
    'Column: Value' line, grouped by sheet. Skips fully empty rows.
    This is intentionally verbose/readable rather than compact CSV, since
    the text goes straight into embedding + LLM context — readable text
    embeds and retrieves better than raw comma-separated values.
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
    sections = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # First non-empty row is treated as the header
        header = None
        data_rows = []
        for row in rows:
            if all(cell is None for cell in row):
                continue
            if header is None:
                header = [str(c).strip() if c is not None else f"col{i}" for i, c in enumerate(row)]
                continue
            data_rows.append(row)

        if header is None or not data_rows:
            continue

        lines = [f"[Sheet: {sheet_name}]"]
        for row in data_rows:
            pairs = []
            for col_name, value in zip(header, row):
                if value is None or str(value).strip() == "":
                    continue
                pairs.append(f"{col_name}: {value}")
            if pairs:
                lines.append(" | ".join(pairs))

        if len(lines) > 1:
            sections.append("\n".join(lines))

    wb.close()
    return "\n\n".join(sections)


def extract_csv(file_bytes: bytes) -> str:
    """Same 'Column: Value' readable format as extract_xlsx, for consistency."""
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""

    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = []
        for col_name, value in zip(header, row):
            if value is None or str(value).strip() == "":
                continue
            pairs.append(f"{col_name}: {value}")
        if pairs:
            lines.append(" | ".join(pairs))

    return "\n".join(lines)


def extract_text(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    name_lower = file_name.lower()

    if mime_type == "application/pdf" or name_lower.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
         or name_lower.endswith(".docx"):
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation" \
         or name_lower.endswith(".pptx"):
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

    elif mime_type in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ) or name_lower.endswith((".xlsx", ".xlsm")):
        return extract_xlsx(file_bytes)

    elif mime_type == "text/csv" or name_lower.endswith(".csv"):
        return extract_csv(file_bytes)

    elif mime_type == "text/plain" or name_lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(
            f"Unsupported file type: {mime_type or 'unknown'} ({file_name}). "
            f"Supported: PDF, DOCX, PPTX, XLSX, CSV, TXT."
        )


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
    """
    Embeds in batches of 50 with retry/backoff on rate limits.
    For large documents (many chunks), this can take a while — that's
    expected and handled by retries, not a bug. If a batch permanently
    fails after retries, the whole request fails loudly (500) rather than
    silently dropping chunks, so partial/corrupt documents never get stored.
    """
    all_embeddings = []
    batch_size = 50
    delay = 1.0

    i = 0
    while i < len(chunks):
        batch = chunks[i:i + batch_size]
        max_retries = 4
        last_error = None
        for attempt in range(max_retries):
            try:
                result = voyage.embed(batch, model="voyage-3", input_type="document")
                all_embeddings.extend(result.embeddings)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if "RateLimitError" in str(type(e)) or "rate limit" in str(e).lower():
                    wait = (attempt + 1) * 5
                    print(f"[ingest] Rate limit on batch {i}-{i+batch_size}, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"[ingest] Embedding error on batch {i}-{i+batch_size}: {e}")
                    raise
        if last_error is not None:
            # Exhausted retries on a rate limit — fail loudly, don't silently truncate
            raise RuntimeError(
                f"Voyage AI embedding failed after {max_retries} retries on batch "
                f"{i}-{i+batch_size}: {last_error}"
            )
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

    Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT.
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
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted. The file may be empty, image-only "
                       "(scanned PDF with no OCR layer), or a spreadsheet with no data rows."
            )

        cleaned = clean_text(raw_text)
        chunks  = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short to process.")

        print(f"[ingest] {request.file_name}: {len(chunks)} chunks to embed")
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

        # Insert in batches of 200 rows — avoids one giant request for very
        # large documents (e.g. big spreadsheets can produce thousands of chunks)
        INSERT_BATCH = 200
        for i in range(0, len(rows), INSERT_BATCH):
            supabase.table("document_chunks").insert(rows[i:i + INSERT_BATCH]).execute()

        return {
            "success":        True,
            "document_id":    request.document_id,
            "workspace_id":   request.workspace_id,
            "chunks_created": len(chunks),
            "message":        f"Processed '{request.file_name}' into {len(chunks)} chunks."
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"INGEST ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
