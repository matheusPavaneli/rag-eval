import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rageval.ingest.chunking import Chunk
from rageval.ingest.corpus import CorpusManifest
from rageval.ingest.documents import Document
from rageval.providers.base import EmbeddingProvider, TransientProviderError
from rageval.retrieval.store import ChunkStore, RetrievalError


class CorpusNotFoundError(RetrievalError):
    pass


class IndexReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunks_indexed: int
    batches: int
    embedding_model: str
    dimension: int


def index_corpus(
    corpus_dir: Path,
    corpus_version: str,
    store: ChunkStore,
    provider: EmbeddingProvider,
    batch_size: int,
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
    backoff_ceiling_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> IndexReport:
    if batch_size < 1:
        raise RetrievalError(f"batch size must be at least 1, got {batch_size}")
    if max_attempts < 1:
        raise RetrievalError(f"max attempts must be at least 1, got {max_attempts}")

    chunks, source_paths = _load(corpus_dir / corpus_version)

    store.ensure_schema()

    batches = 0
    for batch in _batched(chunks, batch_size):
        vectors = _embed(
            provider,
            batch,
            batches,
            max_attempts,
            backoff_seconds,
            backoff_ceiling_seconds,
            sleep,
        )
        store.upsert(corpus_version, batch, vectors, source_paths)
        batches += 1

    return IndexReport(
        corpus_version=corpus_version,
        chunks_indexed=len(chunks),
        batches=batches,
        embedding_model=provider.model,
        dimension=provider.dimension,
    )


def index_text(corpus_dir: Path, corpus_version: str, store: ChunkStore) -> int:
    chunks, source_paths = _load(corpus_dir / corpus_version)
    store.ensure_schema()
    return store.insert_text(corpus_version, chunks, source_paths)


def _load(root: Path) -> tuple[list[Chunk], dict[str, str]]:
    chunks = [Chunk.model_validate(record) for record in _read(root / "chunks.jsonl")]
    documents = [Document.model_validate(record) for record in _read(root / "documents.jsonl")]
    return chunks, {document.id: document.source_path for document in documents}


def _embed(
    provider: EmbeddingProvider,
    batch: Sequence[Chunk],
    number: int,
    max_attempts: int,
    backoff_seconds: float,
    backoff_ceiling_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[tuple[float, ...], ...]:
    texts = [chunk.text for chunk in batch]

    for attempt in range(1, max_attempts + 1):
        try:
            return provider.embed(texts, "document").vectors
        except TransientProviderError as error:
            if attempt == max_attempts:
                raise RetrievalError(
                    f"batch {number} of {len(batch)} chunks still failing after "
                    f"{max_attempts} attempts: {error}"
                ) from error
            sleep(min(backoff_seconds * 2 ** (attempt - 1), backoff_ceiling_seconds))

    raise RetrievalError(f"batch {number} was never attempted")


def load_manifest(corpus_dir: Path, corpus_version: str) -> CorpusManifest:
    path = corpus_dir / corpus_version / "manifest.json"
    if not path.is_file():
        raise CorpusNotFoundError(f"no corpus manifest at {path}; run python -m rageval.ingest")

    try:
        return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CorpusNotFoundError(f"corpus manifest at {path} is unreadable") from error


def latest_corpus_version(corpus_dir: Path) -> str:
    if not corpus_dir.is_dir():
        raise CorpusNotFoundError(
            f"no corpus under {corpus_dir}; run python -m rageval.ingest documents/"
        )

    versions = [path for path in sorted(corpus_dir.iterdir()) if (path / "manifest.json").is_file()]
    if not versions:
        raise CorpusNotFoundError(
            f"no corpus under {corpus_dir}; run python -m rageval.ingest documents/"
        )

    return max(versions, key=lambda path: (path / "manifest.json").stat().st_mtime).name


def _read(path: Path) -> Iterator[object]:
    if not path.is_file():
        raise CorpusNotFoundError(f"no corpus file at {path}; run python -m rageval.ingest")

    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    yield _parse(line, path, number)
    except OSError as error:
        raise CorpusNotFoundError(f"cannot read {path}") from error


def _parse(line: str, path: Path, number: int) -> object:
    try:
        parsed: object = json.loads(line)
    except ValueError as error:
        raise CorpusNotFoundError(f"{path} line {number} is not JSON") from error
    return parsed


def _batched(chunks: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]
