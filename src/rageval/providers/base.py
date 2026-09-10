from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

EmbeddingTask = Literal["document", "query"]

TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    pass


class TransientProviderError(ProviderError):
    pass


class PermanentProviderError(ProviderError):
    pass


class BudgetExceededError(ProviderError):
    pass


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0


class ChatResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    text: str
    usage: Usage = Usage()


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    dimension: int
    vectors: tuple[tuple[float, ...], ...]


class ChatProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, prompt: str, system: str | None = None) -> ChatResult: ...


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult: ...


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout: float,
    provider: str,
    model: str,
) -> Mapping[str, object]:
    where = f"{provider} {model}"
    try:
        response = client.post(url, headers=dict(headers), json=payload, timeout=timeout)
    except httpx.TimeoutException as error:
        raise TransientProviderError(f"{where}: request timed out after {timeout}s") from error
    except httpx.UnsupportedProtocol as error:
        raise PermanentProviderError(f"{where}: {url} is not an http endpoint") from error
    except httpx.HTTPError as error:
        raise TransientProviderError(f"{where}: transport failure") from error

    if response.status_code in TRANSIENT_STATUS:
        raise TransientProviderError(f"{where}: HTTP {response.status_code}")
    if response.status_code >= 400:
        raise PermanentProviderError(f"{where}: HTTP {response.status_code}")

    try:
        body = response.json()
    except ValueError as error:
        raise PermanentProviderError(f"{where}: response body was not JSON") from error

    if not isinstance(body, dict):
        raise PermanentProviderError(f"{where}: response body was not a JSON object")
    return body
