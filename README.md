# Knowledge OS — Python Backend

## What This Is
A FastAPI server that handles the AI layer for Knowledge OS.
It receives files from Lovable, processes them into vectors, and answers user questions from them.

## Files
- `setup.sql`      → Run once in Supabase SQL editor
- `requirements.txt` → Python dependencies
- `.env`           → Your secret keys (never commit this to GitHub)
- `main.py`        → FastAPI app entry point
- `ingest.py`      → /ingest endpoint (file processing pipeline)
- `query.py`       → /query endpoint (question answering)

---

## STEP 1 — Supabase Setup (do this once)
1. Go to your Supabase project → SQL Editor
2. Paste the entire contents of `setup.sql`
3. Click Run
4. You should see the `document_chunks` table appear in your Table Editor

---

## STEP 2 — Get Your Anthropic API Key
1. Go to console.anthropic.com
2. API Keys → Create Key
3. Paste it into your `.env` file as `ANTHROPIC_API_KEY`

---

## STEP 3 — Deploy to Render (free)
1. Push this folder to a GitHub repo (make sure .env is in .gitignore)
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Set these settings:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (copy from your .env file):
   - SUPABASE_URL
   - SUPABASE_SERVICE_KEY
   - VOYAGE_API_KEY
   - ANTHROPIC_API_KEY
6. Click Deploy
7. Render gives you a URL like: https://your-app.onrender.com

---

## STEP 4 — Connect Lovable to This Backend

### After a file is uploaded, add this call in Lovable:

```javascript
// Add this right after your existing upload code succeeds
// data = the object returned from your supabase insert

const BACKEND_URL = "https://your-app.onrender.com";

await fetch(`${BACKEND_URL}/ingest`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    document_id:  data.id,
    storage_path: data.storage_path,
    mime_type:    data.mime_type,
    file_name:    data.file_name,
    asset_id:     data.asset_id,
  }),
});
```

### When a user asks a question:

```javascript
const response = await fetch(`${BACKEND_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "What is my notice period?",
    asset_id: currentAssetId,  // optional — omit to search all documents
  }),
});

const result = await response.json();
console.log(result.answer);   // The clean answer for the user
console.log(result.sources);  // Which documents it came from
```

---

## Endpoints

| Method | Endpoint  | What it does |
|--------|-----------|--------------|
| GET    | /health   | Confirm server is running |
| POST   | /ingest   | Process a file into vectors |
| POST   | /query    | Answer a question from documents |

---

## Supported File Types
- PDF (.pdf)
- Word Document (.docx)
- PowerPoint (.pptx)
- Plain Text (.txt)
- Video (.mp4) — coming in next phase via Whisper
# KBingestion
