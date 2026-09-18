import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rageval.ingest.chunking import Chunk
from rageval.providers import (
    CachedEmbeddingProvider,
    CacheOnlyEmbeddingProvider,
    DiskCache,
    EmbeddingResult,
    EmbeddingTask,
)
from rageval.vectors import SnapshotError, export_snapshot, import_snapshot

VERSION = "v1"
MODEL = "embed-a"
DIMENSION = 3
CHUNKS = ("first chunk", "second chunk", "third chunk")
QUESTIONS = ("what is first?", "what is second?")


class CountingEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return MODEL

    @property
    def dimension(self) -> int:
        return DIMENSION

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        self.calls += 1
        offset = 0.5 if task == "query" else 0.0
        return EmbeddingResult(
            provider=self.name,
            model=MODEL,
            dimension=DIMENSION,
            vectors=tuple(
                tuple(len(text) / 7 + offset + index for index in range(DIMENSION))
                for text in texts
            ),
        )


def _corpus(root: Path) -> Path:
    target = root / VERSION
    target.mkdir(parents=True)
    document = {
        "id": "doc-1",
        "source_path": "a.md",
        "title": "A",
        "text": "x",
        "byte_size": 1,
    }
    chunks = [
        Chunk(
            id=f"chunk-{ordinal}",
            document_id="doc-1",
            ordinal=ordinal,
            text=text,
            start_char=ordinal,
            end_char=ordinal + 1,
            heading_path=(),
        )
        for ordinal, text in enumerate(CHUNKS)
    ]
    (target / "documents.jsonl").write_text(json.dumps(document) + "\n", encoding="utf-8")
    (target / "chunks.jsonl").write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    return root


def _warm(cache: DiskCache, chunks: Sequence[str] = CHUNKS) -> None:
    provider = CachedEmbeddingProvider(CountingEmbedding(), cache)
    provider.embed(chunks, "document")
    provider.embed(QUESTIONS, "query")


def _export(cache: DiskCache, corpus: Path, out: Path) -> tuple[int, str]:
    return export_snapshot(cache, corpus, VERSION, QUESTIONS, "gemini", MODEL, DIMENSION, out)


def _write_snapshot(path: Path, lines: Sequence[str]) -> str:
    data = gzip.compress("".join(line + "\n" for line in lines).encode("utf-8"), mtime=0)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _entries(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _corpus(tmp_path / "corpus")


def test_an_imported_snapshot_serves_every_chunk_and_question_with_no_provider(
    tmp_path: Path, corpus: Path
) -> None:
    source = DiskCache(tmp_path / "source")
    _warm(source)
    snapshot = tmp_path / "snapshot.jsonl.gz"
    count, digest = _export(source, corpus, snapshot)

    target = DiskCache(tmp_path / "target")
    imported = import_snapshot(target, snapshot, digest, MODEL, DIMENSION)

    offline = CachedEmbeddingProvider(
        CacheOnlyEmbeddingProvider("gemini", MODEL, DIMENSION), target
    )
    reference = CachedEmbeddingProvider(CountingEmbedding(), source)
    assert count == imported == len(CHUNKS) + len(QUESTIONS)
    assert offline.embed(CHUNKS, "document") == reference.embed(CHUNKS, "document")
    assert offline.embed(QUESTIONS, "query") == reference.embed(QUESTIONS, "query")


def test_two_exports_of_one_cache_are_byte_identical_sorted_and_unique(
    tmp_path: Path, corpus: Path
) -> None:
    cache = DiskCache(tmp_path / "cache")
    _warm(cache)

    first = _export(cache, corpus, tmp_path / "a.jsonl.gz")
    second = _export(cache, corpus, tmp_path / "b.jsonl.gz")

    assert first == second
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    lines = gzip.decompress((tmp_path / "a.jsonl.gz").read_bytes()).decode().splitlines()
    digests = [json.loads(line)["digest"] for line in lines]
    assert digests == sorted(set(digests))


def test_export_refuses_when_a_chunk_is_not_cached_and_writes_no_file(
    tmp_path: Path, corpus: Path
) -> None:
    cache = DiskCache(tmp_path / "cache")
    _warm(cache, CHUNKS[:-1])
    out = tmp_path / "snapshot.jsonl.gz"

    with pytest.raises(SnapshotError, match="1 of 5 vectors"):
        _export(cache, corpus, out)

    assert not out.exists()


def test_import_refuses_a_file_that_is_not_the_pinned_one_before_writing(
    tmp_path: Path, corpus: Path
) -> None:
    source = DiskCache(tmp_path / "source")
    _warm(source)
    snapshot = tmp_path / "snapshot.jsonl.gz"
    _, digest = _export(source, corpus, snapshot)
    wrong = "0" * 64

    with pytest.raises(SnapshotError, match=f"sha256 {digest}, expected {wrong}"):
        import_snapshot(DiskCache(tmp_path / "target"), snapshot, wrong, MODEL, DIMENSION)

    assert _entries(tmp_path / "target") == []


def _line(
    digest: str = "a" * 64, model: str = MODEL, dimension: int = DIMENSION, size: int = 3
) -> str:
    return json.dumps(
        {
            "digest": digest,
            "vector": {"model": model, "dimension": dimension, "values": [0.1] * size},
        }
    )


@pytest.mark.parametrize(
    "bad",
    [
        _line(model="another-model"),
        _line(dimension=4, size=4),
        _line(size=2),
        _line(digest="not-a-digest"),
    ],
    ids=["model", "dimension", "values-length", "digest"],
)
def test_import_refuses_an_entry_that_does_not_fit_and_writes_nothing(
    tmp_path: Path, bad: str
) -> None:
    snapshot = tmp_path / "snapshot.jsonl.gz"
    digest = _write_snapshot(snapshot, [_line(digest="b" * 64), bad])

    with pytest.raises(SnapshotError, match="line 2"):
        import_snapshot(DiskCache(tmp_path / "target"), snapshot, digest, MODEL, DIMENSION)

    assert _entries(tmp_path / "target") == []
