from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ingest import router as ingest_router
from query import router as query_router

app = FastAPI(title="Knowledge OS API", version="1.0.0")

# Allow Lovable frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Lovable URL after launch
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(ingest_router)
app.include_router(query_router)

# Health check — Render uses this to confirm the server is alive
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Knowledge OS API is running"}
