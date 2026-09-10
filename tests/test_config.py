from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from rageval.config import Settings, get_settings

CANARY = "canary-value-that-must-not-leak"


def test_defaults_point_at_local_compose() -> None:
    settings = Settings()

    assert settings.database_url.endswith(":5433/rageval")
    assert settings.cache_dir == Path(".cache")


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGEVAL_GEMINI_API_KEY", CANARY)

    settings = Settings()

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == CANARY


def test_secret_does_not_leak_through_repr_str_or_dump() -> None:
    settings = Settings(gemini_api_key=SecretStr(CANARY))

    assert CANARY not in repr(settings)
    assert CANARY not in str(settings)
    assert CANARY not in str(settings.model_dump())


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValidationError, match="frozen"):
        settings.database_url = "postgresql://elsewhere"  # type: ignore[misc]


def test_get_settings_returns_one_instance() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
