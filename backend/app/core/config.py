from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """VeriRAG backend settings (Pydantic v2).

    All keys are optional so the Day 1 app imports and serves /health
    with no API keys set. Real keys live only in a local .env file,
    never in the repo.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


settings = Settings()
