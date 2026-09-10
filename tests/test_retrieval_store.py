from collections.abc import Iterator, Sequence

import psycopg
import pytest
from psycopg import sql

from rageval.ingest.chunking import Chunk
from rageval.retrieval.store import (
    TABLE,
    DimensionMismatchError,
    RetrievalError,
    VectorStore,
)

pytestmark = pytest.mark.integration

DIMENSION = 3
CORPUS = "test-corpus"
SOURCE_PATHS = {"doc-1": "a.md"}


@pytest.fixture
def store(
    database: psycopg.Connection[tuple[object, ...]],
) -> Iterator[VectorStore]:
    _drop(database)
    built = VectorStore(database, DIMENSION)
    built.ensure_schema()
    yield built
    _drop(database)


def _drop(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(TABLE)))
    connection.commit()


def _chunk(ordinal: int, text: str = "a chunk") -> Chunk:
    return Chunk(
        id=f"chunk-{ordinal}",
        document_id="doc-1",
        ordinal=ordinal,
        text=text,
        start_char=ordinal * 10,
        end_char=ordinal * 10 + 5,
        heading_path=("A",),
    )


def test_the_schema_is_created_on_a_clean_database(store: VectorStore) -> None:
    assert store.count(CORPUS) == 0


def test_ensure_schema_is_safe_to_run_twice(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0)], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    store.ensure_schema()

    assert store.count(CORPUS) == 1


def test_upserting_the_same_chunk_twice_leaves_one_row(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "first")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    store.upsert(CORPUS, [_chunk(0, "second")], [[0.0, 1.0, 0.0]], SOURCE_PATHS)

    assert store.count(CORPUS) == 1
    assert store.search(CORPUS, [0.0, 1.0, 0.0], limit=1)[0].text == "second"


def test_search_returns_the_nearest_vector_first(store: VectorStore) -> None:
    store.upsert(
        CORPUS,
        [_chunk(0, "east"), _chunk(1, "north"), _chunk(2, "west")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        SOURCE_PATHS,
    )

    result = store.search(CORPUS, [0.9, 0.1, 0.0], limit=3)

    assert [chunk.text for chunk in result] == ["east", "north", "west"]
    assert result[0].score > result[1].score > result[2].score


def test_a_result_carries_the_span_back_into_its_source_document(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(4)], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    found = store.search(CORPUS, [1.0, 0.0, 0.0], limit=1)[0]

    assert found.source_path == "a.md"
    assert (found.start_char, found.end_char) == (40, 45)
    assert found.ordinal == 4


def test_another_corpus_version_is_not_searched(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0)], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    store.upsert("other", [_chunk(1)], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    assert store.count(CORPUS) == 1
    assert len(store.search(CORPUS, [1.0, 0.0, 0.0], limit=10)) == 1


def test_a_table_built_at_another_width_is_refused_by_name(
    database: psycopg.Connection[tuple[object, ...]], store: VectorStore
) -> None:
    wider = VectorStore(database, DIMENSION + 1)

    with pytest.raises(DimensionMismatchError, match=f"stores {DIMENSION}-dimension"):
        wider.ensure_schema()


def test_a_vector_of_the_wrong_width_never_reaches_the_database(store: VectorStore) -> None:
    with pytest.raises(DimensionMismatchError, match="has 2 dimensions"):
        store.upsert(CORPUS, [_chunk(0)], [[1.0, 0.0]], SOURCE_PATHS)

    assert store.count(CORPUS) == 0


def test_a_chunk_without_its_vector_is_refused(store: VectorStore) -> None:
    chunks: Sequence[Chunk] = [_chunk(0), _chunk(1)]

    with pytest.raises(RetrievalError, match="2 chunks were given 1 vectors"):
        store.upsert(CORPUS, chunks, [[1.0, 0.0, 0.0]], SOURCE_PATHS)
