from fastapi import FastAPI

app = FastAPI(title="VeriRAG")


@app.get("/health")
def health():
    return {"status": "ok", "service": "verirag"}
