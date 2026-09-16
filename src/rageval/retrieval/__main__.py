import argparse
import sys
from collections.abc import Sequence

import httpx
import psycopg

from rageval.config import get_settings
from rageval.providers import build_budget, build_embedding_provider
from rageval.providers.base import ProviderError
from rageval.retrieval.index import index_corpus, latest_corpus_version
from rageval.retrieval.search import Retriever
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()

    parser = build_parser(settings.retrieval_top_k)
    arguments = parser.parse_args(argv)

    try:
        version = arguments.corpus_version or latest_corpus_version(settings.corpus_dir)
        budget = build_budget(settings)

        with (
            httpx.Client() as client,
            psycopg.connect(settings.database_url, connect_timeout=10) as connection,
        ):
            provider = build_embedding_provider(settings, client, budget)
            store = VectorStore(connection, settings.embedding_dimension)

            if arguments.query is None:
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
            for rank, chunk in enumerate(
                Retriever(store, provider, version, arguments.k).retrieve(arguments.query), start=1
            ):
                head = " ".join(chunk.text.split())[:120]
                print(
                    f"{rank}. {chunk.score:.3f}  {chunk.source_path}"
                    f"[{chunk.start_char}:{chunk.end_char}]\n   {head}"
                )
    except (RetrievalError, ProviderError) as error:
        print(f"retrieval failed: {error}", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        print(f"database unavailable: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
