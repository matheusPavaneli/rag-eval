from typing import Protocol

from rageval.providers.base import EmbeddingProvider, PermanentProviderError
from rageval.retrieval.store import ChunkStore, ScoredChunk


class QuestionRetriever(Protocol):
    @property
    def top_k(self) -> int: ...

    @property
    def corpus_version(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]: ...


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        provider: EmbeddingProvider,
        corpus_version: str,
        top_k: int,
    ) -> None:
        self._store = store
        self._provider = provider
        self._corpus_version = corpus_version
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    @property
    def embedding_model(self) -> str:
        return self._provider.model

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        result = self._provider.embed([question], "query")
        if not result.vectors:
            raise PermanentProviderError(
                f"{result.provider} {result.model}: embedding the question returned no vector"
            )

        limit = self._top_k if top_k is None else top_k
        return self._store.search(self._corpus_version, result.vectors[0], limit)
