from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VeriRAG backend settings (Pydantic v2).

    All keys are optional so the Day 1 app imports and serves /health
    with no API keys set. Real keys live only in a local .env file,
    never in the repo.

    Day 2 change: Qdrant Cloud is replaced by a local LEANN dense
    index (HNSW backend first). No QDRANT settings remain.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # LEANN dense index (local, HNSW backend).
    LEANN_INDEX_PATH: str = "data/leann_index"
    LEANN_BACKEND: str = "hnsw"
    LEANN_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Chunking (word-window with overlap; bbox is the union of the
    # chunk's word boxes, so every chunk keeps page/bbox grounding).
    CHUNK_WORDS: int = 200
    CHUNK_OVERLAP_WORDS: int = 40

    # Day 5: API-key auth + rate limits. VERIRAG_API_KEY unset means
    # auth is disabled (local dev keeps working with no header).
    VERIRAG_API_KEY: str = ""
    VERIRAG_RATE_LIMIT_PER_MIN: int = 60

    # Day 5: local runtime data (JSONL logs). Gitignored.
    DATA_DIR: str = "data"


settings = Settings()
