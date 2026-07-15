import os
import re
import io
import fitz                          # PyMuPDF — reads PDFs
import docx                          # python-docx — reads Word files
import pptx                          # python-pptx — reads PowerPoint files
import voyageai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Clients ────────────────────────────────────────────────────────────────────
# We use the SERVICE KEY here (not anon key) because:
# - We need to download files from private Supabase storage
# - We need to write to the document_chunks table
# This key must NEVER be sent to the frontend — it lives only here on the server.
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))


# ── Request shape ──────────────────────────────────────────────────────────────
# This is what Lovable sends to /ingest after a file is uploaded.
# All of these values are already available in Lovable right after the insert.
class IngestRequest(BaseModel):
    document_id: str   # id from asset_documents table
    storage_path: str  # e.g. "abc123/uuid-filename.pdf"
    mime_type: str     # e.g. "application/pdf"
    file_name: str     # e.g. "Employee Handbook.pdf"
    asset_id: str      # which asset/folder this belongs to


# ── Step 1: Download ───────────────────────────────────────────────────────────
def download_file(storage_path: str) -> bytes:
    """
    Downloads the raw file bytes from Supabase Storage.
    storage_path is the path inside the 'asset-documents' bucket.
    Returns the file as raw bytes in memory.
    """
    file_bytes = supabase.storage.from_("knowledge-files").download(storage_path)
    return file_bytes


# ── Step 2: Extract text ───────────────────────────────────────────────────────
def extract_text(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    """
    Converts raw file bytes into a plain text string.
    Uses different tools depending on the file type.
    """

    # PDF → PyMuPDF reads each page and pulls all text
    if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)

    # Word document → python-docx reads each paragraph
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
         or file_name.lower().endswith(".docx"):
        document = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    # PowerPoint → python-pptx reads slide by slide, shape by shape
    elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation" \
         or file_name.lower().endswith(".pptx"):
        presentation = pptx.Presentation(io.BytesIO(file_bytes))
        slides_text = []
        for slide_num, slide in enumerate(presentation.slides, 1):
            slide_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text)
            if slide_content:
                slides_text.append(f"[Slide {slide_num}]\n" + "\n".join(slide_content))
        return "\n\n".join(slides_text)

    # Plain text → decode directly
    elif mime_type == "text/plain" or file_name.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {mime_type}. Supported: PDF, DOCX, PPTX, TXT")


# ── Step 3: Clean text ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Removes noise from extracted text.
    - Collapses 3+ newlines into 2 (removes excessive blank lines)
    - Collapses multiple spaces into one
    - Replaces tabs with spaces
    - Strips whitespace from start and end
    """
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines in a row
    text = re.sub(r' {2,}', ' ', text)        # max 1 space in a row
    text = re.sub(r'\t', ' ', text)           # no tabs
    return text.strip()


# ── Step 4: Chunk text ─────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Splits the cleaned text into smaller meaningful pieces.

    chunk_size=1000    → each chunk is ~200 words
    chunk_overlap=100  → chunks share 20 words at boundaries
                         so context is never lost at the edge of a chunk

    Separators priority: paragraph break → line break → sentence → word → character
    This means it tries to cut at natural points first.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]


# ── Step 5: Embed chunks ───────────────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Sends each chunk to Voyage AI and gets back a vector (list of 1024 numbers).
    Voyage AI allows max 128 chunks per API call, so we batch.

    input_type="document" tells Voyage this is content being stored, not a query.
    This matters — Voyage uses different internal settings for documents vs queries.
    """
    all_embeddings = []
    batch_size = 128

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = voyage.embed(batch, model="voyage-3", input_type="document")
        all_embeddings.extend(result.embeddings)

    return all_embeddings


# ── Main route ─────────────────────────────────────────────────────────────────
@router.post("/ingest")
async def ingest_document(request: IngestRequest):
    """
    Called by Lovable after a file is uploaded.
    Runs the full pipeline: download → extract → clean → chunk → embed → store.
    """
    try:
        # 1. Download raw file from Supabase Storage
        file_bytes = download_file(request.storage_path)

        # 2. Convert file to plain text
        raw_text = extract_text(file_bytes, request.mime_type, request.file_name)

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

        # 3. Remove noise
        cleaned = clean_text(raw_text)

        # 4. Cut into chunks
        chunks = chunk_text(cleaned)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short to process.")

        # 5. Convert chunks to vectors
        embeddings = embed_chunks(chunks)

        # 6. Build rows to insert into Supabase
        rows = [
            {
                "document_id": request.document_id,
                "asset_id":    request.asset_id,
                "content":     chunks[i],
                "embedding":   embeddings[i],
                "chunk_index": i,
                "metadata": {
                    "file_name":    request.file_name,
                    "chunk_index":  i,
                    "total_chunks": len(chunks)
                }
            }
            for i in range(len(chunks))
        ]

        # 7. Store all rows in Supabase
        supabase.table("document_chunks").insert(rows).execute()

        return {
            "success":        True,
            "document_id":    request.document_id,
            "chunks_created": len(chunks),
            "message":        f"Successfully processed '{request.file_name}' into {len(chunks)} chunks."
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
