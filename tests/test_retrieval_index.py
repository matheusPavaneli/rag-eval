import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from conftest_retrieval import FakeEmbeddingProvider, FakeStore
from rageval.ingest.chunking import Chunk
from rageval.providers.base import (
    EmbeddingResult,
    EmbeddingTask,
    PermanentProviderError,
    TransientProviderError,
)
from rageval.retrieval.index import (
    CorpusNotFoundError,
    index_corpus,
    index_text,
    latest_corpus_version,
    load_manifest,
)
from rageval.retrieval.store import RetrievalError


def _corpus(root: Path, version: str, chunk_count: int) -> Path:
    target = root / version
    target.mkdir(parents=True)

    document = {
        "id": "doc-1",
        "source_path": "a.md",
        "title": "A",
        "text": "x" * chunk_count,
        "byte_size": chunk_count,
    }
    chunks = [
        Chunk(
            id=f"chunk-{ordinal}",
            document_id="doc-1",
            ordinal=ordinal,
            text=f"chunk {ordinal}",
            start_char=ordinal,
            end_char=ordinal + 1,
            heading_path=(),
        )
        for ordinal in range(chunk_count)
    ]

    (target / "documents.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
    (target / "chunks.jsonl").write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_version": version,
                "chunk_config": {"chunk_size": 1000, "overlap": 150},
                "document_count": 1,
                "chunk_count": chunk_count,
                "created_at": "2026-09-10T00:00:00Z",
                "rageval_version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )
    return target


def test_every_chunk_is_indexed_in_batches_of_the_configured_size(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=10)
    provider = FakeEmbeddingProvider()
    store = FakeStore()

    report = index_corpus(tmp_path, "v1", store, provider, batch_size=4)

    assert report.chunks_indexed == 10
    assert report.batches == 3
    assert [len(batch) for batch in provider.batches] == [4, 4, 2]
    assert store.schema_calls == 1

    indexed = [chunk.id for batch in store.upserts for chunk in batch]
    assert indexed == [f"chunk-{ordinal}" for ordinal in range(10)]


def test_chunks_are_embedded_as_documents_not_as_queries(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=2)
    provider = FakeEmbeddingProvider()

    index_corpus(tmp_path, "v1", FakeStore(), provider, batch_size=64)

    assert provider.tasks == ["document"]


def test_a_batch_size_below_one_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=2)

    with pytest.raises(RetrievalError, match="at least 1"):
        index_corpus(tmp_path, "v1", FakeStore(), FakeEmbeddingProvider(), batch_size=0)


def test_a_corpus_version_with_no_chunks_file_names_the_path(tmp_path: Path) -> None:
    (tmp_path / "v1").mkdir(parents=True)

    with pytest.raises(CorpusNotFoundError, match=r"chunks\.jsonl"):
        index_corpus(tmp_path, "v1", FakeStore(), FakeEmbeddingProvider(), batch_size=64)


def test_the_latest_corpus_is_the_most_recently_written_manifest(tmp_path: Path) -> None:
    older = _corpus(tmp_path, "old", chunk_count=1)
    newer = _corpus(tmp_path, "new", chunk_count=1)
    _touch(older / "manifest.json", 1_000_000)
    _touch(newer / "manifest.json", 2_000_000)

    assert latest_corpus_version(tmp_path) == "new"


def test_an_empty_corpus_directory_points_at_the_ingest_command(tmp_path: Path) -> None:
    with pytest.raises(CorpusNotFoundError, match=r"rageval\.ingest"):
        latest_corpus_version(tmp_path / "missing")


def test_the_manifest_carries_the_chunk_configuration_the_corpus_was_built_with(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path, "v1", chunk_count=3)

    manifest = load_manifest(tmp_path, "v1")

    assert manifest.chunk_count == 3
    assert manifest.chunk_config.chunk_size == 1000
    assert manifest.chunk_config.overlap == 150


def _touch(path: Path, when: int) -> None:
    import os

    os.utime(path, (when, when))


class FlakyProvider(FakeEmbeddingProvider):
    def __init__(self, failures: int, error: Exception) -> None:
        super().__init__()
        self._failures = failures
        self._error = error
        self.attempts = 0

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise self._error
        return super().embed(texts, task)


def test_a_rate_limited_batch_is_retried_and_succeeds(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=2)
    provider = FlakyProvider(2, TransientProviderError("HTTP 429"))
    slept: list[float] = []

    report = index_corpus(
        tmp_path,
        "v1",
        FakeStore(),
        provider,
        64,
        max_attempts=4,
        backoff_seconds=1.0,
        sleep=slept.append,
    )

    assert report.chunks_indexed == 2
    assert provider.attempts == 3
    assert slept == [1.0, 2.0]


def test_a_batch_that_keeps_failing_gives_up_naming_the_attempts(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=2)
    provider = FlakyProvider(99, TransientProviderError("HTTP 429"))

    with pytest.raises(RetrievalError, match="still failing after 3 attempts"):
        index_corpus(
            tmp_path,
            "v1",
            FakeStore(),
            provider,
            64,
            max_attempts=3,
            backoff_seconds=0.0,
            sleep=lambda _: None,
        )

    assert provider.attempts == 3


def test_a_permanent_failure_is_not_retried(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=2)
    provider = FlakyProvider(99, PermanentProviderError("HTTP 401"))

    with pytest.raises(PermanentProviderError):
        index_corpus(
            tmp_path,
            "v1",
            FakeStore(),
            provider,
            64,
            max_attempts=5,
            backoff_seconds=0.0,
            sleep=lambda _: None,
        )

    assert provider.attempts == 1


def test_zero_attempts_is_refused(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=1)

    with pytest.raises(RetrievalError, match="at least 1"):
        index_corpus(tmp_path, "v1", FakeStore(), FakeEmbeddingProvider(), 64, max_attempts=0)


def test_the_backoff_delay_is_capped(tmp_path: Path) -> None:
    _corpus(tmp_path, "v1", chunk_count=1)
    provider = FlakyProvider(4, TransientProviderError("HTTP 429"))
    slept: list[float] = []

    index_corpus(
        tmp_path,
        "v1",
        FakeStore(),
        provider,
        64,
        max_attempts=6,
        backoff_seconds=10.0,
        backoff_ceiling_seconds=25.0,
        sleep=slept.append,
    )

    assert slept == [10.0, 20.0, 25.0, 25.0]


def test_indexing_text_stores_every_chunk_with_its_source_path_and_embeds_nothing(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path, "v1", chunk_count=10)
    store = FakeStore()

    stored = index_text(tmp_path, "v1", store)

    assert stored == 10
    assert store.schema_calls == 1
    assert store.upserts == []
    [(version, chunks, source_paths)] = store.text_inserts
    assert version == "v1"
    assert [chunk.id for chunk in chunks] == [f"chunk-{ordinal}" for ordinal in range(10)]
    assert source_paths == {"doc-1": "a.md"}
    assert store.count("v1") == 10
    assert store.count("v1", embedded=True) == 0
