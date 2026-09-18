import argparse
import sys
from collections.abc import Sequence

import httpx
import psycopg

from rageval.answer.citations import ResolvedCitation
from rageval.answer.generate import answer_question
from rageval.config import get_settings
from rageval.providers import build_budget, build_chat_provider, build_embedding_provider
from rageval.providers.base import ProviderError
from rageval.retrieval.index import index_corpus, latest_corpus_version
from rageval.retrieval.search import LEXICAL_MODES, MODES, Bm25Config, build_retriever
from rageval.retrieval.store import RetrievalError, VectorStore


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"k must be at least 1, got {number}")
    return number


def build_parser(top_k: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval.retrieval", description="Index a corpus into pgvector, or query it"
    )
    parser.add_argument("--corpus-version", default=None)
    parser.add_argument("--query", default=None, help="run one question instead of indexing")
    parser.add_argument("-k", type=_positive, default=top_k)
    parser.add_argument("--mode", choices=MODES, default="dense", help="retrieval mode for --query")
    parser.add_argument(
        "--fuse-with", choices=LEXICAL_MODES, default="bm25", help="lexical ranker for hybrid"
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="with --query, answer from the retrieved chunks and show each cited span",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = build_parser(settings.retrieval_top_k)
    arguments = parser.parse_args(argv)
    if arguments.answer and arguments.query is None:
        parser.error("--answer needs --query")

    try:
        version = arguments.corpus_version or latest_corpus_version(settings.corpus_dir)
        budget = build_budget(settings)

        with (
            httpx.Client() as client,
            psycopg.connect(settings.database_url, connect_timeout=10) as connection,
        ):
            store = VectorStore(connection, settings.embedding_dimension)

            if arguments.query is None:
                provider = build_embedding_provider(settings, client, budget)
                report = index_corpus(
                    settings.corpus_dir,
                    version,
                    store,
                    provider,
                    settings.embedding_batch_size,
                    settings.embedding_max_attempts,
                    settings.embedding_backoff_seconds,
                    settings.embedding_backoff_ceiling_seconds,
                )
                print(
                    f"indexed {report.chunks_indexed} chunks of corpus {report.corpus_version} "
                    f"in {report.batches} batches "
                    f"({report.embedding_model}, {report.dimension}d); "
                    f"{budget.state.calls} call(s) reached the network"
                )
                return 0

            store.ensure_schema()
            retriever = build_retriever(
                arguments.mode,
                store,
                None
                if arguments.mode in LEXICAL_MODES
                else build_embedding_provider(settings, client, budget),
                version,
                arguments.k,
                Bm25Config(k1=settings.bm25_k1, b=settings.bm25_b),
                arguments.fuse_with,
                settings.rrf_k,
                settings.retrieval_candidates,
            )
            chunks = retriever.retrieve(arguments.query)
            for rank, chunk in enumerate(chunks, start=1):
                head = " ".join(chunk.text.split())[:120]
                print(
                    f"{rank}. {chunk.score:.3f}  {chunk.source_path}"
                    f"[{chunk.start_char}:{chunk.end_char}]\n   {head}"
                )

            if arguments.answer:
                answer = answer_question(
                    arguments.query, chunks, build_chat_provider(settings, client, budget)
                )
                print(f"\nanswer ({answer.provider} {answer.model}):")
                if answer.parse_error is not None:
                    print(f"   unparseable response: {answer.parse_error}")
                    return 1
                print(f"   {answer.text}")
                for citation in answer.citations:
                    if isinstance(citation, ResolvedCitation):
                        where = f"{citation.source_path}[{citation.start_char}:{citation.end_char}]"
                    else:
                        where = f"unresolved ({citation.reason})"
                    print(f"   [{citation.chunk}] {where}\n       {citation.quote!r}")
    except (RetrievalError, ProviderError) as error:
        print(f"retrieval failed: {error}", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        print(f"database unavailable: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
