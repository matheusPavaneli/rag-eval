from collections.abc import Iterator

import httpx
import pytest
from pydantic import SecretStr

from rageval.config import Settings
from rageval.providers.gemini import GeminiChatProvider, GeminiEmbeddingProvider
from rageval.providers.groq import GroqChatProvider

pytestmark = pytest.mark.integration

SETTINGS = Settings()
PROMPT = "Reply with the single word: pong"


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client() as open_client:
        yield open_client


def key(value: SecretStr | None, name: str) -> SecretStr:
    if value is None:
        pytest.skip(f"{name} is not set")
    return value


def test_gemini_answers_a_prompt(client: httpx.Client) -> None:
    provider = GeminiChatProvider(
        client,
        key(SETTINGS.gemini_api_key, "RAGEVAL_GEMINI_API_KEY"),
        SETTINGS.gemini_chat_model,
        SETTINGS.provider_timeout_seconds,
    )

    result = provider.complete(PROMPT)

    assert result.text.strip() != ""
    assert result.provider == "gemini"


def test_gemini_embeds_at_the_configured_dimension(client: httpx.Client) -> None:
    provider = GeminiEmbeddingProvider(
        client,
        key(SETTINGS.gemini_api_key, "RAGEVAL_GEMINI_API_KEY"),
        SETTINGS.gemini_embedding_model,
        SETTINGS.embedding_dimension,
        SETTINGS.provider_timeout_seconds,
    )

    result = provider.embed(["a span of text", "another span"], "document")

    assert len(result.vectors) == 2
    assert all(len(vector) == SETTINGS.embedding_dimension for vector in result.vectors)


def test_groq_answers_a_prompt(client: httpx.Client) -> None:
    provider = GroqChatProvider(
        client,
        key(SETTINGS.groq_api_key, "RAGEVAL_GROQ_API_KEY"),
        SETTINGS.groq_chat_model,
        SETTINGS.provider_timeout_seconds,
    )

    result = provider.complete(PROMPT)

    assert result.text.strip() != ""
    assert result.provider == "groq"
