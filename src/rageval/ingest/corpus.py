import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rageval import __version__
from rageval.ingest.chunking import Chunk, ChunkConfig, chunk_document
from rageval.ingest.documents import Document, load_documents

CORPUS_VERSION_LENGTH = 16


class EmptyCorpusError(Exception):
    pass


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    chunk_config: ChunkConfig
    document_count: int
    chunk_count: int
    created_at: datetime
    rageval_version: str


def corpus_version(documents: Sequence[Document], config: ChunkConfig) -> str:
    digest = hashlib.sha256()
    for identifier in sorted(document.id for document in documents):
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    digest.update(config.model_dump_json().encode("utf-8"))
    return digest.hexdigest()[:CORPUS_VERSION_LENGTH]


def ingest_corpus(
    documents_dir: Path, corpus_dir: Path, config: ChunkConfig | None = None
) -> CorpusManifest:
    settings = config or ChunkConfig()

    documents = load_documents(documents_dir)
    if not documents:
        raise EmptyCorpusError(f"no .md or .txt document found under {documents_dir}")

    chunks = [chunk for document in documents for chunk in chunk_document(document, settings)]
    version = corpus_version(documents, settings)
    manifest = CorpusManifest(
        corpus_version=version,
        chunk_config=settings,
        document_count=len(documents),
        chunk_count=len(chunks),
        created_at=datetime.now(UTC),
        rageval_version=__version__,
    )

    _write(corpus_dir / version, manifest, documents, chunks)
    return manifest


def _write(
    target: Path,
    manifest: CorpusManifest,
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "documents.jsonl", documents)
    _write_jsonl(target / "chunks.jsonl", chunks)
    (target / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
