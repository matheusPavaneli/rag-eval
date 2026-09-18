import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rageval.config import get_settings
from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.ingest.documents import DocumentReadError, load_documents
from rageval.providers.base import ProviderError
from rageval.providers.cache import DiskCache
from rageval.providers.failover import CACHE_NAMESPACE
from rageval.providers.gemini import NAME as GEMINI
from rageval.retrieval.store import RetrievalError
from rageval.vectors import SnapshotError, export_snapshot, import_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.vectors",
        description="Export or import the cached embeddings one corpus version needs",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="write every cached chunk and golden-question vector to a snapshot"
    )
    export.add_argument("--corpus-version", required=True)
    export.add_argument("--out", type=Path, required=True)

    load = commands.add_parser("import", help="load a snapshot into the cache")
    load.add_argument("path", type=Path)
    load.add_argument("--sha256", required=True, help="digest the snapshot must have")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    arguments = build_parser().parse_args(argv)
    cache = DiskCache(settings.cache_dir / CACHE_NAMESPACE)

    try:
        if arguments.command == "export":
            documents = load_documents(settings.documents_dir)
            questions = load_golden_set(settings.golden_set_path, documents)
            count, digest = export_snapshot(
                cache,
                settings.corpus_dir,
                arguments.corpus_version,
                [question.question for question in questions],
                GEMINI,
                settings.gemini_embedding_model,
                settings.embedding_dimension,
                arguments.out,
            )
            print(f"wrote {count} vectors to {arguments.out}\nsha256 {digest}")
            return 0

        count = import_snapshot(
            cache,
            arguments.path,
            arguments.sha256,
            settings.gemini_embedding_model,
            settings.embedding_dimension,
        )
        print(f"imported {count} vectors into {settings.cache_dir / CACHE_NAMESPACE}")
        return 0
    except (
        SnapshotError,
        RetrievalError,
        GoldenSetError,
        DocumentReadError,
        ProviderError,
        OSError,
    ) as error:
        print(f"vectors failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
