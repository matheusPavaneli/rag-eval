import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from rageval.providers.base import (
    ChatProvider,
    ChatResult,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingTask,
    PermanentProviderError,
)

CHAT_NAMESPACE = "chat"
EMBEDDING_NAMESPACE = "embedding"


class CacheEntryError(PermanentProviderError):
    pass


class CacheMissError(PermanentProviderError):
    pass


class CacheKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    task: str
    inputs: tuple[str, ...]
    parameters: tuple[tuple[str, str], ...] = ()


class CachedVector(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    dimension: int
    values: tuple[float, ...]


class DiskCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    @staticmethod
    def digest(key: CacheKey) -> str:
        return hashlib.sha256(key.model_dump_json().encode("utf-8")).hexdigest()

    def get(self, namespace: str, digest: str) -> object | None:
        path = self._path(namespace, digest)
        if not path.is_file():
            return None

        try:
            stored: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CacheEntryError(
                f"cache entry {path} is unreadable; delete it to refetch"
            ) from error
        return stored

    def set(self, namespace: str, digest: str, value: BaseModel) -> None:
        path = self._path(namespace, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f"{path.name}.{os.getpid()}.tmp"
        temporary.write_text(value.model_dump_json(), encoding="utf-8")
        os.replace(temporary, path)

    def _path(self, namespace: str, digest: str) -> Path:
        return self._root / namespace / digest[:2] / f"{digest}.json"


class CachedChatProvider:
    def __init__(self, provider: ChatProvider, cache: DiskCache) -> None:
        self._provider = provider
        self._cache = cache

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        digest = DiskCache.digest(
            CacheKey(
                provider=self._provider.name,
                model=self._provider.model,
                task="complete",
                inputs=(prompt, system or ""),
            )
        )

        stored = self._cache.get(CHAT_NAMESPACE, digest)
        if stored is not None:
            return _validate(ChatResult, stored, CHAT_NAMESPACE, digest)

        result = self._provider.complete(prompt, system)
        self._cache.set(CHAT_NAMESPACE, digest, result)
        return result


class CachedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache: DiskCache) -> None:
        self._provider = provider
        self._cache = cache

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        digests = [self._digest(text, task) for text in texts]
        vectors: dict[int, tuple[float, ...]] = {}
        missing: list[int] = []

        for position, digest in enumerate(digests):
            stored = self._cache.get(EMBEDDING_NAMESPACE, digest)
            if stored is None:
                missing.append(position)
                continue
            vectors[position] = _validate(CachedVector, stored, EMBEDDING_NAMESPACE, digest).values

        if missing:
            fetched = self._provider.embed([texts[position] for position in missing], task)
            if len(fetched.vectors) != len(missing):
                raise PermanentProviderError(
                    f"{self._provider.name} {self._provider.model}: asked for "
                    f"{len(missing)} embeddings, received {len(fetched.vectors)}"
                )
            for position, values in zip(missing, fetched.vectors, strict=True):
                vectors[position] = values
                self._cache.set(
                    EMBEDDING_NAMESPACE,
                    digests[position],
                    CachedVector(model=fetched.model, dimension=fetched.dimension, values=values),
                )

        return EmbeddingResult(
            provider=self._provider.name,
            model=self._provider.model,
            dimension=self._provider.dimension,
            vectors=tuple(vectors[position] for position in range(len(texts))),
        )

    def _digest(self, text: str, task: EmbeddingTask) -> str:
        return embedding_digest(
            self._provider.name, self._provider.model, self._provider.dimension, text, task
        )


class CacheOnlyEmbeddingProvider:
    """Stands in for a provider that has no key: every text it is asked for is a miss.

    Behind CachedEmbeddingProvider it turns the cache into the only source of
    vectors, so a missing entry fails instead of reaching the network.
    """

    def __init__(self, name: str, model: str, dimension: int) -> None:
        self._name = name
        self._model = model
        self._dimension = dimension

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        raise CacheMissError(
            f"{self._name} {self._model}: no cached embedding for {len(texts)} text(s) and "
            "RAGEVAL_GEMINI_API_KEY is not set; import the vector snapshot "
            "(python -m rageval.vectors import) or set the key"
        )


def embedding_digest(
    provider: str, model: str, dimension: int, text: str, task: EmbeddingTask
) -> str:
    return DiskCache.digest(
        CacheKey(
            provider=provider,
            model=model,
            task=task,
            inputs=(text,),
            parameters=(("dimension", str(dimension)),),
        )
    )


def _validate[T: BaseModel](model: type[T], stored: object, namespace: str, digest: str) -> T:
    try:
        return model.model_validate(stored)
    except ValidationError as error:
        raise CacheEntryError(
            f"cache entry {namespace}/{digest} does not hold a {model.__name__}; "
            "delete it to refetch"
        ) from error
