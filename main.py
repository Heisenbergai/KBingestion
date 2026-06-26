from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ingest import router as ingest_router
from query import router as query_router
from path import router as path_router
from course import router as course_router
from explainer import router as explainer_router
from voiceover import router as voiceover_router
from slides import router as slides_router
from video import router as video_router
from bg_videos import router as bg_videos_router
from presentation import router as presentation_router

app = FastAPI(title="Knowledge OS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(path_router)
app.include_router(course_router)
app.include_router(explainer_router)
app.include_router(voiceover_router)
app.include_router(slides_router)
app.include_router(video_router)
app.include_router(bg_videos_router)
app.include_router(presentation_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Knowledge OS API is running"}
