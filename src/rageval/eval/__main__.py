import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import psycopg

from rageval.config import get_settings
from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.eval.runner import run_eval
from rageval.ingest.documents import DocumentReadError, load_documents
from rageval.providers import build_budget, build_embedding_provider
from rageval.providers.base import ProviderError
from rageval.retrieval.index import latest_corpus_version, load_manifest
from rageval.retrieval.search import Retriever
from rageval.retrieval.store import RetrievalError, VectorStore


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"k must be at least 1, got {number}")
    return number


def build_parser(golden_set_path: Path, top_k: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.eval", description="Measure retrieval against the golden set"
    )
    parser.add_argument("--corpus-version", default=None)
    parser.add_argument("--golden-set", type=Path, default=golden_set_path)
    parser.add_argument("-k", type=_positive, default=top_k)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = build_parser(settings.golden_set_path, settings.retrieval_top_k)
    arguments = parser.parse_args(argv)

    try:
        version = arguments.corpus_version or latest_corpus_version(settings.corpus_dir)
        manifest = load_manifest(settings.corpus_dir, version)
        documents = load_documents(settings.documents_dir)
        questions = load_golden_set(arguments.golden_set, documents)
        budget = build_budget(settings)

        with (
            httpx.Client() as client,
            psycopg.connect(settings.database_url, connect_timeout=10) as connection,
        ):
            provider = build_embedding_provider(settings, client, budget)
            store = VectorStore(connection, settings.embedding_dimension)
            store.ensure_schema()

            indexed = store.count(version)
            if indexed != manifest.chunk_count:
                print(
                    f"corpus {version} has {manifest.chunk_count} chunks but "
                    f"{indexed} are indexed; run python -m rageval.retrieval first",
                    file=sys.stderr,
                )
                return 1

            report = run_eval(
                questions,
                Retriever(store, provider, version, arguments.k),
                manifest.chunk_config,
                settings.embedding_dimension,
                lambda: budget.state.calls,
            )
    except (GoldenSetError, RetrievalError, ProviderError, DocumentReadError) as error:
        print(f"eval failed: {error}", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        print(f"database unavailable: {error}", file=sys.stderr)
        return 1

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.ran_at.strftime("%Y%m%dT%H%M%SZ")
    path = settings.report_dir / f"{stamp}-{report.corpus_version}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"{report.question_count} questions, {report.network_calls} call(s) reached the network")
    print(f"context recall @{report.k}  {report.context_recall_at_k:.3f}")
    print(f"MRR @{report.k}            {report.mrr_at_k:.3f}")
    print(f"\nreport {path}\n\nREADME row:\n{report.table_row()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
