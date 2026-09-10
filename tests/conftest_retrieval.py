from collections.abc import Mapping, Sequence

from rageval.ingest.chunking import Chunk
from rageval.providers.base import EmbeddingResult, EmbeddingTask
from rageval.retrieval.store import ScoredChunk


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
    def __init__(self, dimension: int = 3, results: Sequence[ScoredChunk] = ()) -> None:
        self._dimension = dimension
        self._results = tuple(results)
        self.schema_calls = 0
        self.upserts: list[list[Chunk]] = []
        self.searches: list[tuple[str, tuple[float, ...], int]] = []

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

    def search(
        self, corpus_version: str, vector: Sequence[float], limit: int
    ) -> tuple[ScoredChunk, ...]:
        self.searches.append((corpus_version, tuple(vector), limit))
        return self._results[:limit]

    def count(self, corpus_version: str) -> int:
        return sum(len(batch) for batch in self.upserts)


class StubRetriever:
    def __init__(
        self,
        answers: Mapping[str, Sequence[ScoredChunk]],
        top_k: int = 5,
        corpus_version: str = "cafef00d",
    ) -> None:
        self._answers = answers
        self._top_k = top_k
        self._corpus_version = corpus_version

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    @property
    def embedding_model(self) -> str:
        return "fake-embed"

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
        return tuple(self._answers.get(question, ()))[: top_k or self._top_k]


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
