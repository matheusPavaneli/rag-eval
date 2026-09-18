from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rageval.providers.base import EmbeddingProvider, PermanentProviderError
from rageval.retrieval.bm25 import Bm25Index
from rageval.retrieval.store import ChunkStore, RetrievalError, ScoredChunk

type RetrievalMode = Literal["dense", "fulltext", "bm25", "hybrid"]
type LexicalMode = Literal["fulltext", "bm25"]

MODES: tuple[RetrievalMode, ...] = ("dense", "fulltext", "bm25", "hybrid")
LEXICAL_MODES: tuple[LexicalMode, ...] = ("fulltext", "bm25")


class DenseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["dense"] = "dense"


class FullTextConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["fulltext"] = "fulltext"


class Bm25Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["bm25"] = "bm25"
    k1: float
    b: float


type LexicalConfig = Annotated[FullTextConfig | Bm25Config, Field(discriminator="mode")]


class HybridConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["hybrid"] = "hybrid"
    rrf_k: int
    candidates: int
    lexical: LexicalConfig


type RetrievalConfig = Annotated[
    DenseConfig | FullTextConfig | Bm25Config | HybridConfig, Field(discriminator="mode")
]


class QuestionRetriever(Protocol):
    @property
    def top_k(self) -> int: ...

    @property
    def corpus_version(self) -> str: ...

    @property
    def embedding_model(self) -> str | None: ...

    @property
    def config(self) -> DenseConfig | FullTextConfig | Bm25Config | HybridConfig: ...

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]: ...


class LexicalRetriever(Protocol):
    @property
    def config(self) -> FullTextConfig | Bm25Config: ...

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
    def embedding_model(self) -> str | None:
        return self._provider.model

    @property
    def config(self) -> DenseConfig:
        return DenseConfig()

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        result = self._provider.embed([question], "query")
        if not result.vectors:
            raise PermanentProviderError(
                f"{result.provider} {result.model}: embedding the question returned no vector"
            )

        limit = self._top_k if top_k is None else top_k
        return self._store.search(self._corpus_version, result.vectors[0], limit)


class FullTextRetriever:
    def __init__(self, store: ChunkStore, corpus_version: str, top_k: int) -> None:
        self._store = store
        self._corpus_version = corpus_version
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    @property
    def embedding_model(self) -> str | None:
        return None

    @property
    def config(self) -> FullTextConfig:
        return FullTextConfig()

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        limit = self._top_k if top_k is None else top_k
        return self._store.lexical_search(self._corpus_version, question, limit)


class Bm25Retriever:
    def __init__(
        self, store: ChunkStore, corpus_version: str, top_k: int, config: Bm25Config
    ) -> None:
        self._store = store
        self._corpus_version = corpus_version
        self._top_k = top_k
        self._config = config
        self._index: Bm25Index | None = None

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    @property
    def embedding_model(self) -> str | None:
        return None

    @property
    def config(self) -> Bm25Config:
        return self._config

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        if self._index is None:
            self._index = Bm25Index(
                self._store.chunk_terms(self._corpus_version), self._config.k1, self._config.b
            )

        limit = self._top_k if top_k is None else top_k
        return self._index.search(self._store.query_terms(question), limit)


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        lexical: LexicalRetriever,
        top_k: int,
        rrf_k: int,
        candidates: int,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._top_k = top_k
        self._config = HybridConfig(rrf_k=rrf_k, candidates=candidates, lexical=lexical.config)

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._dense.corpus_version

    @property
    def embedding_model(self) -> str | None:
        return self._dense.embedding_model

    @property
    def config(self) -> HybridConfig:
        return self._config

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        limit = self._top_k if top_k is None else top_k
        depth = self._config.candidates
        fused = reciprocal_rank_fusion(
            (self._dense.retrieve(question, depth), self._lexical.retrieve(question, depth)),
            self._config.rrf_k,
        )
        return fused[:limit]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], rrf_k: int
) -> tuple[ScoredChunk, ...]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)
            chunks.setdefault(key, chunk)

    order = sorted(scores, key=lambda key: (-scores[key], best_rank[key], key))
    return tuple(chunks[key].model_copy(update={"score": scores[key]}) for key in order)


def build_retriever(
    mode: RetrievalMode,
    store: ChunkStore,
    provider: EmbeddingProvider | None,
    corpus_version: str,
    top_k: int,
    bm25: Bm25Config,
    fuse_with: LexicalMode = "bm25",
    rrf_k: int = 60,
    candidates: int = 20,
) -> QuestionRetriever:
    fulltext = FullTextRetriever(store, corpus_version, top_k)
    ranked = Bm25Retriever(store, corpus_version, top_k, bm25)
    if mode == "fulltext":
        return fulltext
    if mode == "bm25":
        return ranked
    if provider is None:
        raise RetrievalError(f"{mode} retrieval needs an embedding provider")

    dense = Retriever(store, provider, corpus_version, top_k)
    if mode == "dense":
        return dense
    lexical: LexicalRetriever = ranked if fuse_with == "bm25" else fulltext
    return HybridRetriever(dense, lexical, top_k, rrf_k, candidates)
