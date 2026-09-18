from collections.abc import Mapping, Sequence

from rageval.ingest.chunking import Chunk
from rageval.providers.base import EmbeddingResult, EmbeddingTask
from rageval.retrieval.search import (
    Bm25Config,
    DenseConfig,
    FullTextConfig,
    HybridConfig,
    RerankConfig,
)
from rageval.retrieval.store import ChunkTerms, ScoredChunk


class FakeEmbeddingProvider:
    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension
        self.batches: list[list[str]] = []
        self.tasks: list[EmbeddingTask] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        self.batches.append(list(texts))
        self.tasks.append(task)
        return EmbeddingResult(
            provider=self.name,
            model=self.model,
            dimension=self._dimension,
            vectors=tuple(_vector(text, self._dimension) for text in texts),
        )


class FakeStore:
    def __init__(
        self,
        dimension: int = 3,
        results: Sequence[ScoredChunk] = (),
        lexical_results: Sequence[ScoredChunk] = (),
        terms: Sequence[ChunkTerms] = (),
        query_terms: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._dimension = dimension
        self._results = tuple(results)
        self._lexical_results = tuple(lexical_results)
        self.schema_calls = 0
        self.upserts: list[list[Chunk]] = []
        self.text_inserts: list[tuple[str, list[Chunk], dict[str, str]]] = []
        self.searches: list[tuple[str, tuple[float, ...], int]] = []
        self.lexical_searches: list[tuple[str, str, int]] = []
        self._terms = tuple(terms)
        self._query_terms = query_terms or {}
        self.term_reads: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def ensure_schema(self) -> None:
        self.schema_calls += 1

    def upsert(
        self,
        corpus_version: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        source_paths: dict[str, str],
    ) -> int:
        assert len(chunks) == len(vectors)
        for chunk in chunks:
            assert chunk.document_id in source_paths
        self.upserts.append(list(chunks))
        return len(chunks)

    def insert_text(
        self, corpus_version: str, chunks: Sequence[Chunk], source_paths: dict[str, str]
    ) -> int:
        self.text_inserts.append((corpus_version, list(chunks), dict(source_paths)))
        return len(chunks)

    def search(
        self, corpus_version: str, vector: Sequence[float], limit: int
    ) -> tuple[ScoredChunk, ...]:
        self.searches.append((corpus_version, tuple(vector), limit))
        return self._results[:limit]

    def lexical_search(
        self, corpus_version: str, query: str, limit: int
    ) -> tuple[ScoredChunk, ...]:
        self.lexical_searches.append((corpus_version, query, limit))
        return self._lexical_results[:limit]

    def chunk_terms(self, corpus_version: str) -> tuple[ChunkTerms, ...]:
        self.term_reads.append(corpus_version)
        return self._terms

    def query_terms(self, question: str) -> tuple[str, ...]:
        return tuple(self._query_terms.get(question, question.lower().split()))

    def count(self, corpus_version: str, embedded: bool = False) -> int:
        texts = 0 if embedded else sum(len(chunks) for _, chunks, _ in self.text_inserts)
        return sum(len(batch) for batch in self.upserts) + texts


class StubRetriever:
    def __init__(
        self,
        answers: Mapping[str, Sequence[ScoredChunk]],
        top_k: int = 5,
        corpus_version: str = "cafef00d",
        config: DenseConfig
        | FullTextConfig
        | Bm25Config
        | HybridConfig
        | RerankConfig
        | None = None,
    ) -> None:
        self._answers = answers
        self._top_k = top_k
        self._corpus_version = corpus_version
        self._config = config or DenseConfig()
        self.requests: list[tuple[str, int | None]] = []

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    @property
    def embedding_model(self) -> str | None:
        return None if isinstance(self._config, FullTextConfig | Bm25Config) else "fake-embed"

    @property
    def config(self) -> DenseConfig | FullTextConfig | Bm25Config | HybridConfig | RerankConfig:
        return self._config

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        self.requests.append((question, top_k))
        return tuple(self._answers.get(question, ()))[: top_k or self._top_k]


class FakeCrossEncoder:
    def __init__(self, scores: Mapping[str, float], extra: int = 0) -> None:
        self._scores = scores
        self._extra = extra
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def model(self) -> str:
        return "fake/cross-encoder"

    @property
    def revision(self) -> str:
        return "0123456789abcdef"

    def score(self, question: str, texts: Sequence[str]) -> tuple[float, ...]:
        self.calls.append((question, tuple(texts)))
        return tuple(self._scores.get(text, 0.0) for text in texts) + (0.0,) * self._extra


def scored(
    source_path: str, start_char: int, end_char: int, score: float = 0.5, ordinal: int = 0
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{source_path}:{start_char}:{end_char}",
        document_id=f"doc-{source_path}",
        source_path=source_path,
        ordinal=ordinal,
        text="x",
        start_char=start_char,
        end_char=end_char,
        score=score,
    )


def _vector(text: str, dimension: int) -> tuple[float, ...]:
    seed = sum(ord(character) for character in text) or 1
    return tuple(float((seed >> position) % 7 + 1) for position in range(dimension))


def terms(chunk_id: str, **counts: int) -> ChunkTerms:
    return ChunkTerms(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_path=f"{chunk_id}.md",
        ordinal=0,
        text="x",
        start_char=0,
        end_char=10,
        terms=dict(counts),
    )
