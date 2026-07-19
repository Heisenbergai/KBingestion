import os
import re
import io
import csv
import uuid
import fitz
import docx
import pptx
import openpyxl
import threading
import ai
from datetime import datetime, timezone
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


# ── In-memory ingestion job store ───────────────────────────────────────────────
# Same pattern as bg_videos.py: Railway is stateless, jobs live in memory only.
# Lovable polls GET /ingest-status/{job_id} and owns persistence of the final
# "processed" flag on the document row. If Railway restarts mid-job, the job
# disappears — Lovable should treat a 404 on a job it was polling as "failed,
# please re-upload".
INGEST_JOBS: dict[str, dict] = {}
MAX_JOBS_KEPT = 200


def _prune_jobs():
    """Keeps the in-memory job store bounded. Oldest finished jobs go first."""
    if len(INGEST_JOBS) <= MAX_JOBS_KEPT:
        return
    finished = [
        (jid, j) for jid, j in INGEST_JOBS.items()
        if j["status"] in ("completed", "failed")
    ]
    finished.sort(key=lambda item: item[1].get("finished_at") or "")
    for jid, _ in finished[: len(INGEST_JOBS) - MAX_JOBS_KEPT]:
        INGEST_JOBS.pop(jid, None)


class IngestRequest(BaseModel):
    document_id:  str
    signed_url:   str
    mime_type:    str
    file_name:    str
    asset_id:     str
    workspace_id: str   # ← REQUIRED — isolates chunks per company
    # wait=True runs synchronously and returns the full result in one response
    # (old behavior — fine for small files and curl testing). Default is
    # background mode: returns a job_id immediately so large documents can't
    # hit the HTTP timeout, and Lovable polls /ingest-status/{job_id}.
    wait: Optional[bool] = False


def download_file(signed_url: str) -> bytes:
    response = httpx.get(signed_url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    return response.content


# ── Extraction helpers ──────────────────────────────────────────────────────────

def _table_to_lines(rows: list[list[str]]) -> list[str]:
    """
    Renders a table as readable lines. If the table has a header row,
    each data row becomes 'Header: Value | Header: Value' — the same
    readable format as extract_xlsx, because that embeds/retrieves better
    than raw pipe-separated cells.
    """
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    if len(rows) == 1:
        return [" | ".join(c for c in rows[0] if c)]
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if pairs:
            lines.append(" | ".join(pairs))
    return lines


def extract_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    if doc.page_count > 0 and len(text.strip()) < 30:
        raise ValueError(
            "This PDF appears to be scanned/image-only (no text layer). "
            "OCR is not supported yet — please upload a text-based PDF "
            "or the original document (DOCX/PPTX)."
        )
    return text


def extract_docx(file_bytes: bytes) -> str:
    """
    Extracts paragraphs AND tables, in document order where possible.
    Plain-paragraph extraction silently drops tables, which is where
    policy docs and process docs keep their most important content.
    """
    document = docx.Document(io.BytesIO(file_bytes))
    parts = []
    try:
        # python-docx >= 1.1 yields paragraphs and tables interleaved in order
        for block in document.iter_inner_content():
            if isinstance(block, docx.table.Table):
                rows = [[cell.text for cell in row.cells] for row in block.rows]
                parts.extend(_table_to_lines(rows))
            elif block.text.strip():
                parts.append(block.text)
    except AttributeError:
        # Older python-docx — paragraphs first, then all tables
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            parts.extend(_table_to_lines(rows))
    return "\n".join(parts)


def _pptx_shape_texts(shape) -> list[str]:
    """Recursively pulls text out of a shape: text frames, tables, groups."""
    texts = []
    # Grouped shapes contain child shapes
    if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
        for child in shape.shapes:
            texts.extend(_pptx_shape_texts(child))
        return texts
    if getattr(shape, "has_table", False):
        rows = [
            [cell.text for cell in row.cells]
            for row in shape.table.rows
        ]
        texts.extend(_table_to_lines(rows))
        return texts
    if hasattr(shape, "text") and shape.text.strip():
        texts.append(shape.text)
    return texts


def extract_pptx(file_bytes: bytes) -> str:
    """
    Extracts per slide: all shape text (including inside groups), table
    contents, and speaker notes. Notes often carry the actual explanation
    of a slide, so they matter a lot for search quality.
    """
    presentation = pptx.Presentation(io.BytesIO(file_bytes))
    slides_text = []
    for i, slide in enumerate(presentation.slides, 1):
        slide_content = []
        for shape in slide.shapes:
            slide_content.extend(_pptx_shape_texts(shape))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_content.append(f"(Speaker notes: {notes})")
        if slide_content:
            slides_text.append(f"[Slide {i}]\n" + "\n".join(slide_content))
    return "\n\n".join(slides_text)


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
        return extract_pdf(file_bytes)

    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
         or name_lower.endswith(".docx"):
        return extract_docx(file_bytes)

    elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation" \
         or name_lower.endswith(".pptx"):
        return extract_pptx(file_bytes)

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


def embed_chunks(chunks: list[str], on_progress=None) -> list[list[float]]:
    """
    Embeds via AWS Bedrock Titan v2 (see ai.py). Titan takes one text per
    API call, so we process in batches of 25 purely for progress
    reporting. Throttling retries are handled by boto3's adaptive retry
    mode inside ai.embed_texts. Any hard failure raises — partial/corrupt
    documents never get stored.

    on_progress(embedded_count) is called after each batch so the job
    status endpoint can report live progress to the frontend.
    """
    all_embeddings = []
    batch_size = 25

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            all_embeddings.extend(ai.embed_texts(batch))
        except Exception as e:
            print(f"[ingest] Embedding error on batch {i}-{i+batch_size}: {e}")
            raise RuntimeError(
                f"Bedrock embedding failed on batch {i}-{i+batch_size}: {e}"
            )
        if on_progress:
            on_progress(len(all_embeddings))
    return all_embeddings


# ── Core pipeline (shared by sync and background modes) ─────────────────────────

def process_document(request: IngestRequest, job: Optional[dict] = None) -> dict:
    """
    Full pipeline: download → extract → clean → chunk → embed → store.
    Updates `job` in place (if given) so /ingest-status shows live progress.
    Raises ValueError for user-fixable problems, other exceptions for real errors.
    """
    def set_stage(stage: str):
        if job is not None:
            job["stage"] = stage

    set_stage("downloading")
    file_bytes = download_file(request.signed_url)

    set_stage("extracting")
    raw_text = extract_text(file_bytes, request.mime_type, request.file_name)

    if not raw_text.strip():
        raise ValueError(
            "No text could be extracted. The file may be empty, image-only "
            "(scanned PDF with no OCR layer), or a spreadsheet with no data rows."
        )

    cleaned = clean_text(raw_text)
    chunks  = chunk_text(cleaned)

    if not chunks:
        raise ValueError("Document too short to process.")

    if job is not None:
        job["chunks_total"] = len(chunks)

    print(f"[ingest] {request.file_name}: {len(chunks)} chunks to embed")

    set_stage("embedding")

    def on_progress(done: int):
        if job is not None:
            job["chunks_embedded"] = done

    embeddings = embed_chunks(chunks, on_progress=on_progress)

    set_stage("storing")

    # Remove any previous chunks for this document first. This makes
    # re-uploading a document safe (no duplicate chunks polluting search)
    # and is what lets the workspace-isolation re-upload step replace old
    # workspace_id=null chunks instead of stacking on top of them.
    supabase.table("document_chunks") \
        .delete() \
        .eq("document_id", request.document_id) \
        .execute()

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


def _run_ingest_job(job_id: str, request: IngestRequest):
    job = INGEST_JOBS.get(job_id)
    if job is None:
        return
    try:
        result = process_document(request, job=job)
        job.update({
            "status":         "completed",
            "stage":          "completed",
            "chunks_created": result["chunks_created"],
            "finished_at":    datetime.now(timezone.utc).isoformat(),
        })
        print(f"[ingest:job {job_id}] Completed — {result['chunks_created']} chunks")
    except Exception as e:
        import traceback
        print(f"[ingest:job {job_id}] FAILED: {e}")
        print(traceback.format_exc())
        job.update({
            "status":      "failed",
            "error":       str(e),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })


# ── Routes ──────────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """
    Processes a document and stores chunks in the vector DB.
    workspace_id is stored with every chunk — this is what isolates
    each company's data from all other companies.

    Default (background) mode returns immediately:
        { success, job_id, status: "processing" }
    then the frontend polls GET /ingest-status/{job_id} until
    status is "completed" or "failed". This is what makes large
    documents work — the old synchronous mode timed out on them.

    Pass "wait": true for the old synchronous behavior (small files, curl tests).

    Supported formats: PDF, DOCX, PPTX, XLSX, CSV, TXT.
    """
    if not request.workspace_id:
        raise HTTPException(
            status_code=400,
            detail="workspace_id is required. Every document must belong to a workspace."
        )

    if request.wait:
        try:
            return process_document(request)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            import traceback
            print(f"INGEST ERROR: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    _prune_jobs()
    job_id = str(uuid.uuid4())
    INGEST_JOBS[job_id] = {
        "job_id":          job_id,
        "document_id":     request.document_id,
        "workspace_id":    request.workspace_id,
        "file_name":       request.file_name,
        "status":          "processing",
        "stage":           "queued",
        "chunks_total":    None,
        "chunks_embedded": 0,
        "chunks_created":  None,
        "error":           None,
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "finished_at":     None,
    }

    # Plain daemon thread, not BackgroundTasks — the pipeline is blocking
    # (sync HTTP, sync Voyage client, time.sleep backoff) and can run for
    # minutes on big documents; a thread keeps the event loop free.
    threading.Thread(
        target=_run_ingest_job, args=(job_id, request), daemon=True
    ).start()

    return {
        "success":     True,
        "job_id":      job_id,
        "status":      "processing",
        "document_id": request.document_id,
        "message":     f"Processing '{request.file_name}' in the background. "
                       f"Poll /ingest-status/{job_id} for progress.",
    }


@router.get("/ingest-status/{job_id}")
async def ingest_status(job_id: str):
    """
    Live status of a background ingestion job.
    status: processing | completed | failed
    stage:  queued | downloading | extracting | embedding | storing | completed
    While embedding, chunks_embedded / chunks_total gives real progress
    for a percentage bar in the UI.
    """
    job = INGEST_JOBS.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown job_id. The server may have restarted — re-upload the document."
        )
    return job
