from fastapi import FastAPI

from .api.v1.ingest import router as ingest_router
from .api.v1.search import router as search_router

app = FastAPI(title="VeriRAG")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "verirag"}
