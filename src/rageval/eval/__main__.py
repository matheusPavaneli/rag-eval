import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import psycopg
from pydantic import ValidationError

from rageval.config import get_settings
from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.eval.runner import EvalReport, ReportMismatchError, compare, run_eval
from rageval.ingest.documents import DocumentReadError, load_documents
from rageval.providers import build_budget, build_embedding_provider
from rageval.providers.base import ProviderError
from rageval.retrieval.index import latest_corpus_version, load_manifest
from rageval.retrieval.rerank import OnnxCrossEncoder
from rageval.retrieval.search import (
    LEXICAL_MODES,
    MODES,
    Bm25Config,
    HybridConfig,
    RerankConfig,
    RerankRetriever,
    RetrievalConfig,
    build_retriever,
)
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
    parser.add_argument("--mode", choices=MODES, default="dense")
    parser.add_argument(
        "--fuse-with", choices=LEXICAL_MODES, default="bm25", help="lexical ranker for hybrid"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="rerank the first stage's candidates locally"
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="report to list flipped questions against"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = build_parser(settings.golden_set_path, settings.retrieval_top_k)
    arguments = parser.parse_args(argv)

    baseline = None
    if arguments.baseline is not None:
        try:
            baseline = EvalReport.model_validate_json(
                arguments.baseline.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            print(f"baseline {arguments.baseline} unreadable: {error}", file=sys.stderr)
            return 1

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
            provider = (
                None
                if arguments.mode in LEXICAL_MODES
                else build_embedding_provider(settings, client, budget)
            )
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

            retriever = build_retriever(
                arguments.mode,
                store,
                provider,
                version,
                arguments.k,
                Bm25Config(k1=settings.bm25_k1, b=settings.bm25_b),
                arguments.fuse_with,
                settings.rrf_k,
                settings.retrieval_candidates,
            )
            if arguments.rerank:
                retriever = RerankRetriever(
                    retriever,
                    OnnxCrossEncoder(
                        settings.reranker_model, settings.reranker_revision, settings.cache_dir
                    ),
                    settings.retrieval_candidates,
                )

            report = run_eval(
                questions,
                retriever,
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
    path = settings.report_dir / f"{stamp}-{report.corpus_version}-{_mode_slug(report)}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print(f"{report.question_count} questions, {report.network_calls} call(s) reached the network")
    print(f"context recall @{report.k}  {report.context_recall_at_k:.3f}")
    print(f"MRR @{report.k}            {report.mrr_at_k:.3f}")
    print(f"\nreport {path}\n\nREADME row:\n{report.table_row()}")

    if baseline is not None:
        try:
            flips = compare(baseline, report)
        except ReportMismatchError as error:
            print(f"\nnot comparable with {arguments.baseline}: {error}", file=sys.stderr)
            return 1
        print(f"\ngained vs baseline  {', '.join(flips.gained) or 'none'}")
        print(f"lost vs baseline    {', '.join(flips.lost) or 'none'}")
    return 0


def _mode_slug(report: EvalReport) -> str:
    return _config_slug(report.retrieval)


def _config_slug(config: RetrievalConfig) -> str:
    match config:
        case RerankConfig(first_stage=first_stage):
            return f"rerank-{_config_slug(first_stage)}"
        case HybridConfig(lexical=lexical):
            return f"hybrid-{lexical.mode}"
        case _:
            return config.mode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
