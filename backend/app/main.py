from fastapi import FastAPI

from .api.v1.ask import router as ask_router
from .api.v1.ask_stream import router as ask_stream_router
from .api.v1.feedback import router as feedback_router
from .api.v1.ingest import router as ingest_router
from .api.v1.search import router as search_router

app = FastAPI(title="VeriRAG")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(ask_stream_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "verirag"}
