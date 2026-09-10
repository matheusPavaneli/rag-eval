from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAGEVAL_",
        extra="ignore",
        frozen=True,
    )

    database_url: str = "postgresql://rageval:rageval@127.0.0.1:5433/rageval"
    cache_dir: Path = Path(".cache")
    documents_dir: Path = Path("documents")
    corpus_dir: Path = Path("data/corpus")
    golden_set_path: Path = Path("evals/golden-set.jsonl")
    report_dir: Path = Path("evals/reports")
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    gemini_chat_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    groq_chat_model: str = "openai/gpt-oss-120b"
    embedding_dimension: int = 768
    provider_timeout_seconds: float = 30.0
    budget_max_calls: int = 2000
    budget_max_input_chars: int = 4_000_000

    retrieval_top_k: int = 5
    embedding_batch_size: int = 64
    embedding_max_attempts: int = 8
    embedding_backoff_seconds: float = 20.0
    embedding_backoff_ceiling_seconds: float = 120.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
