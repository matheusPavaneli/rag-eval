import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
import psycopg
from pydantic import ValidationError

from rageval.answer.fixture import (
    FixtureError,
    RecordingChatProvider,
    load_fixture,
    write_fixture,
)
from rageval.config import get_settings
from rageval.eval.answers import AnswerDrift, AnswerReport, answer_drift, run_answer_eval
from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.eval.runner import Drift, EvalReport, ReportMismatchError, compare, drift, run_eval
from rageval.ingest.documents import DocumentReadError, load_documents
from rageval.providers import (
    DiskCache,
    build_budget,
    build_chat_provider,
    build_embedding_provider,
    chat_chain,
)
from rageval.providers.base import ChatProvider, ProviderError
from rageval.providers.failover import CACHE_NAMESPACE
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
        "--answer",
        action="store_true",
        help="generate an answer with quoted citations per question and score the citations",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="report to list flipped questions against"
    )
    parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="exit 1 if any question's result, or an aggregate, differs from --baseline",
    )
    parser.add_argument(
        "--record-answers",
        type=Path,
        default=None,
        help="write every answer this run received to a fixture file (with --answer)",
    )
    parser.add_argument(
        "--replay-answers",
        type=Path,
        default=None,
        help="load a recorded answer fixture into the chat cache first (with --answer)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = build_parser(settings.golden_set_path, settings.retrieval_top_k)
    arguments = parser.parse_args(argv)
    if arguments.answer and arguments.rerank:
        parser.error("--answer runs over a first-stage retriever; drop --rerank")
    if not arguments.answer and (arguments.record_answers or arguments.replay_answers):
        parser.error("--record-answers and --replay-answers need --answer")
    if arguments.fail_on_change and arguments.baseline is None:
        parser.error("--fail-on-change needs --baseline")

    baseline: EvalReport | AnswerReport | None = None
    if arguments.baseline is not None:
        report_type = AnswerReport if arguments.answer else EvalReport
        try:
            baseline = report_type.model_validate_json(
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

            indexed = store.count(version, embedded=arguments.mode not in LEXICAL_MODES)
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

            report: EvalReport | AnswerReport
            if arguments.answer:
                if arguments.replay_answers is not None:
                    loaded = load_fixture(
                        DiskCache(settings.cache_dir / CACHE_NAMESPACE), arguments.replay_answers
                    )
                    print(f"loaded {loaded} recorded answers from {arguments.replay_answers}")
                chat: ChatProvider = build_chat_provider(settings, client, budget)
                recorder = RecordingChatProvider(chat)
                report = run_answer_eval(
                    questions, retriever, recorder, chat_chain(settings), lambda: budget.state.calls
                )
                if arguments.record_answers is not None:
                    written = write_fixture(
                        arguments.record_answers, chat_chain(settings), recorder.calls
                    )
                    print(f"recorded {written} answers to {arguments.record_answers}")
            else:
                report = run_eval(
                    questions,
                    retriever,
                    manifest.chunk_config,
                    settings.embedding_dimension,
                    lambda: budget.state.calls,
                )
    except (
        GoldenSetError,
        RetrievalError,
        ProviderError,
        DocumentReadError,
        FixtureError,
    ) as error:
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
    if isinstance(report, AnswerReport):
        _print_answers(report)
        print(f"\nreport {path}\n\nREADME row:\n{report.table_row()}")
        if not isinstance(baseline, AnswerReport):
            return 0
        try:
            answers = answer_drift(baseline, report)
        except ReportMismatchError as error:
            print(f"\nnot comparable with {arguments.baseline}: {error}", file=sys.stderr)
            return 1
        status = _report_answer_drift(answers, arguments.baseline)
        return status if arguments.fail_on_change else 0

    print(f"context recall @{report.k}  {report.context_recall_at_k:.3f}")
    print(f"MRR @{report.k}            {report.mrr_at_k:.3f}")
    print(f"\nreport {path}\n\nREADME row:\n{report.table_row()}")

    if isinstance(baseline, EvalReport):
        try:
            flips = compare(baseline, report)
            gate = drift(baseline, report) if arguments.fail_on_change else None
        except ReportMismatchError as error:
            print(f"\nnot comparable with {arguments.baseline}: {error}", file=sys.stderr)
            return 1
        print(f"\ngained vs baseline  {', '.join(flips.gained) or 'none'}")
        print(f"lost vs baseline    {', '.join(flips.lost) or 'none'}")
        if gate is not None:
            return _report_drift(gate, arguments.baseline)
    return 0


def _report_drift(gate: Drift, baseline: Path) -> int:
    if not gate.any:
        print(f"no drift from {baseline}")
        return 0

    print(f"\ndrift from {baseline}:", file=sys.stderr)
    for question in gate.changed:
        recall, rank = question.context_recall, question.reciprocal_rank
        print(
            f"  {question.id}  recall {recall[0]:.3f} -> {recall[1]:.3f}, "
            f"rr {rank[0]:.3f} -> {rank[1]:.3f}",
            file=sys.stderr,
        )
    recall, mrr = gate.context_recall_at_k, gate.mrr_at_k
    print(f"  context recall {recall[0]!r} -> {recall[1]!r}", file=sys.stderr)
    print(f"  MRR            {mrr[0]!r} -> {mrr[1]!r}", file=sys.stderr)
    return 1


def _report_answer_drift(gate: AnswerDrift, baseline: Path) -> int:
    if not gate.any:
        print(f"no drift from {baseline}")
        return 0

    print(f"\ndrift from {baseline}:", file=sys.stderr)
    for question in gate.changed:
        print(f"  {question.id}  {', '.join(question.fields)}", file=sys.stderr)
    for aggregate in gate.aggregates:
        print(f"  {aggregate.field} {aggregate.before!r} -> {aggregate.after!r}", file=sys.stderr)
    return 1


def _print_answers(report: AnswerReport) -> None:
    print(f"citation hit rate @{report.k}            {report.citation_hit_rate:.3f}")
    print(
        f"  over retrieved gold spans     {report.citation_hit_rate_retrieved:.3f} "
        f"({report.retrieved_count} questions)"
    )
    print(f"citations resolved             {report.resolution_rate:.3f} of {report.citation_count}")
    print(
        f"mean citation / cited chunk    {report.mean_citation_chars:.0f} / "
        f"{report.mean_cited_chunk_chars:.0f} chars"
    )
    print(f"unparseable responses          {report.parse_failures}")
    print(f"answered by                    {report.providers}")


def _mode_slug(report: EvalReport | AnswerReport) -> str:
    slug = _config_slug(report.retrieval)
    return f"answer-{slug}" if isinstance(report, AnswerReport) else slug


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
