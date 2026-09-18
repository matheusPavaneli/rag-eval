"""Move the embedding cache entries one corpus version needs between machines.

A snapshot holds, for every chunk of a corpus version and every golden question,
the cache entry the embedding provider would read: the entry's digest and its
vector, exactly as cached. Imported into an empty cache it lets indexing and the
dense eval run with no key and no network call.
"""

import gzip
import hashlib
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rageval.providers.cache import (
    EMBEDDING_NAMESPACE,
    CachedVector,
    DiskCache,
    embedding_digest,
)
from rageval.retrieval.index import load_corpus


class SnapshotError(Exception):
    pass


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector: CachedVector


def export_snapshot(
    cache: DiskCache,
    corpus_dir: Path,
    corpus_version: str,
    questions: Sequence[str],
    provider: str,
    model: str,
    dimension: int,
    out: Path,
) -> tuple[int, str]:
    chunks, _ = load_corpus(corpus_dir / corpus_version)
    wanted = {
        embedding_digest(provider, model, dimension, chunk.text, "document") for chunk in chunks
    } | {embedding_digest(provider, model, dimension, question, "query") for question in questions}

    entries: list[SnapshotEntry] = []
    missing = 0
    for digest in sorted(wanted):
        stored = cache.get(EMBEDDING_NAMESPACE, digest)
        if stored is None:
            missing += 1
            continue
        try:
            entries.append(SnapshotEntry(digest=digest, vector=CachedVector.model_validate(stored)))
        except ValidationError as error:
            raise SnapshotError(
                f"cache entry {EMBEDDING_NAMESPACE}/{digest} does not hold a vector"
            ) from error

    if missing:
        raise SnapshotError(
            f"{missing} of {len(wanted)} vectors for corpus {corpus_version} are not cached; "
            "index it and run the dense eval with RAGEVAL_GEMINI_API_KEY set first"
        )

    payload = "".join(entry.model_dump_json() + "\n" for entry in entries).encode("utf-8")
    data = gzip.compress(payload, mtime=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return len(entries), hashlib.sha256(data).hexdigest()


def import_snapshot(
    cache: DiskCache, path: Path, expected_sha256: str, model: str, dimension: int
) -> int:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SnapshotError(f"cannot read snapshot {path}") from error

    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256.strip().lower():
        raise SnapshotError(
            f"snapshot {path} has sha256 {actual}, expected {expected_sha256}; "
            "it is not the pinned file"
        )

    try:
        text = gzip.decompress(data).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise SnapshotError(f"snapshot {path} is not gzip-compressed UTF-8") from error

    entries = [
        _entry(line, number, model, dimension)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    for entry in entries:
        cache.set(EMBEDDING_NAMESPACE, entry.digest, entry.vector)
    return len(entries)


def _entry(line: str, number: int, model: str, dimension: int) -> SnapshotEntry:
    try:
        entry = SnapshotEntry.model_validate_json(line)
    except ValidationError as error:
        raise SnapshotError(f"snapshot line {number} is not a cache entry") from error

    vector = entry.vector
    if vector.model != model or vector.dimension != dimension or len(vector.values) != dimension:
        raise SnapshotError(
            f"snapshot line {number} holds {vector.model} with {len(vector.values)} values "
            f"(declared {vector.dimension}); expected {model} with {dimension}"
        )
    return entry
