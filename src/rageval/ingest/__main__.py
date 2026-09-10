import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rageval.config import get_settings
from rageval.ingest.chunking import ChunkConfig
from rageval.ingest.corpus import EmptyCorpusError, ingest_corpus
from rageval.ingest.documents import DocumentReadError


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(prog="rageval.ingest", description="Build a versioned corpus")
    parser.add_argument("documents_dir", nargs="?", type=Path, default=settings.documents_dir)
    parser.add_argument("--chunk-size", type=int, default=ChunkConfig().chunk_size)
    parser.add_argument("--overlap", type=int, default=ChunkConfig().overlap)
    arguments = parser.parse_args(argv)

    try:
        config = ChunkConfig(chunk_size=arguments.chunk_size, overlap=arguments.overlap)
        manifest = ingest_corpus(arguments.documents_dir, settings.corpus_dir, config)
    except (DocumentReadError, EmptyCorpusError, ValueError) as error:
        print(f"ingest failed: {error}", file=sys.stderr)
        return 1

    print(
        f"corpus {manifest.corpus_version}: "
        f"{manifest.document_count} documents, {manifest.chunk_count} chunks "
        f"in {settings.corpus_dir / manifest.corpus_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
