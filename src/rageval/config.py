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
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
