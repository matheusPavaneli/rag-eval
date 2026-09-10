from collections.abc import Sequence
from math import sqrt

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from rageval.providers.base import (
    ChatResult,
    EmbeddingResult,
    EmbeddingTask,
    PermanentProviderError,
    Usage,
    post_json,
)

NAME = "gemini"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
TASK_TYPES: dict[EmbeddingTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


class _Part(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str = ""


class _Content(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    parts: tuple[_Part, ...] = ()


class _Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: _Content = _Content()


class _UsageMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    input_tokens: int = Field(default=0, alias="promptTokenCount")
    output_tokens: int = Field(default=0, alias="candidatesTokenCount")


class _GenerateResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    candidates: tuple[_Candidate, ...]
    usage: _UsageMetadata = Field(default=_UsageMetadata(), alias="usageMetadata")


class _Embedding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    values: tuple[float, ...]


class _BatchEmbedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    embeddings: tuple[_Embedding, ...]


class GeminiChatProvider:
    def __init__(
        self, client: httpx.Client, api_key: SecretStr, model: str, timeout: float
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return NAME

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }
        if system is not None:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        body = post_json(
            self._client,
            f"{BASE_URL}/models/{self._model}:generateContent",
            headers=_headers(self._api_key),
            payload=payload,
            timeout=self._timeout,
            provider=NAME,
            model=self._model,
        )
        parsed = _parse(_GenerateResponse, body, self._model)
        if not parsed.candidates:
            raise PermanentProviderError(f"{NAME} {self._model}: response held no candidate")

        return ChatResult(
            provider=NAME,
            model=self._model,
            text="".join(part.text for part in parsed.candidates[0].content.parts),
            usage=Usage(
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            ),
        )


class GeminiEmbeddingProvider:
    def __init__(
        self,
        client: httpx.Client,
        api_key: SecretStr,
        model: str,
        dimension: int,
        timeout: float,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._timeout = timeout

    @property
    def name(self) -> str:
        return NAME

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                provider=NAME, model=self._model, dimension=self._dimension, vectors=()
            )

        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": TASK_TYPES[task],
                    "outputDimensionality": self._dimension,
                }
                for text in texts
            ]
        }

        body = post_json(
            self._client,
            f"{BASE_URL}/models/{self._model}:batchEmbedContents",
            headers=_headers(self._api_key),
            payload=payload,
            timeout=self._timeout,
            provider=NAME,
            model=self._model,
        )
        parsed = _parse(_BatchEmbedResponse, body, self._model)
        if len(parsed.embeddings) != len(texts):
            raise PermanentProviderError(
                f"{NAME} {self._model}: asked for {len(texts)} embeddings, "
                f"received {len(parsed.embeddings)}"
            )
        for embedding in parsed.embeddings:
            if len(embedding.values) != self._dimension:
                raise PermanentProviderError(
                    f"{NAME} {self._model}: expected {self._dimension} dimensions, "
                    f"received {len(embedding.values)}"
                )

        return EmbeddingResult(
            provider=NAME,
            model=self._model,
            dimension=self._dimension,
            vectors=tuple(_unit(embedding.values) for embedding in parsed.embeddings),
        )


def _unit(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return tuple(value / norm for value in values)


def _headers(api_key: SecretStr) -> dict[str, str]:
    return {"x-goog-api-key": api_key.get_secret_value(), "content-type": "application/json"}


def _parse[T: BaseModel](model: type[T], body: object, model_id: str) -> T:
    try:
        return model.model_validate(body)
    except ValidationError as error:
        raise PermanentProviderError(
            f"{NAME} {model_id}: response did not hold the expected fields"
        ) from error
