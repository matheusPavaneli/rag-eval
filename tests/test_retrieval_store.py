from collections.abc import Iterator, Sequence

import psycopg
import pytest
from psycopg import sql

from rageval.ingest.chunking import Chunk
from rageval.retrieval.store import (
    DimensionMismatchError,
    RetrievalError,
    VectorStore,
)

pytestmark = pytest.mark.integration

DIMENSION = 3
CORPUS = "test-corpus"
# Never the production table: a dropped "chunks" would take the indexed corpus
# behind the published baseline with it.
TABLE = "chunks_under_test"
SOURCE_PATHS = {"doc-1": "a.md"}


@pytest.fixture
def store(
    database: psycopg.Connection[tuple[object, ...]],
) -> Iterator[VectorStore]:
    _drop(database)
    built = VectorStore(database, DIMENSION, table=TABLE)
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


def test_a_renamed_document_updates_the_stored_path_on_re_index(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "same text")], [[1.0, 0.0, 0.0]], {"doc-1": "a.md"})
    store.upsert(CORPUS, [_chunk(0, "same text")], [[1.0, 0.0, 0.0]], {"doc-1": "renamed.md"})

    found = store.search(CORPUS, [1.0, 0.0, 0.0], limit=1)[0]

    assert found.source_path == "renamed.md"


def test_a_moved_span_updates_the_stored_offsets_on_re_index(store: VectorStore) -> None:
    moved = _chunk(0).model_copy(update={"start_char": 900, "end_char": 950})
    store.upsert(CORPUS, [_chunk(0)], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    store.upsert(CORPUS, [moved], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    found = store.search(CORPUS, [1.0, 0.0, 0.0], limit=1)[0]

    assert (found.start_char, found.end_char) == (900, 950)


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
    wider = VectorStore(database, DIMENSION + 1, table=TABLE)

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


def test_ensure_schema_adds_the_lexical_column_to_a_table_built_before_it(
    database: psycopg.Connection[tuple[object, ...]], store: VectorStore
) -> None:
    store.upsert(CORPUS, [_chunk(0, "structural subtyping")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    with database.cursor() as cursor:
        cursor.execute(sql.SQL("ALTER TABLE {} DROP COLUMN lexical").format(sql.Identifier(TABLE)))
    database.commit()

    store.ensure_schema()
    store.ensure_schema()

    assert [chunk.text for chunk in store.lexical_search(CORPUS, "subtyping", limit=5)] == [
        "structural subtyping"
    ]


def test_lexical_search_ranks_the_chunk_with_the_query_terms_first(store: VectorStore) -> None:
    store.upsert(
        CORPUS,
        [
            _chunk(0, "a protocol class defines structural subtyping for a protocol"),
            _chunk(1, "an enumeration is a set of named constants"),
            _chunk(2, "the protocol is described elsewhere"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        SOURCE_PATHS,
    )

    result = store.lexical_search(CORPUS, "What is a protocol class?", limit=5)

    assert [chunk.ordinal for chunk in result] == [0, 2]
    assert result[0].score > result[1].score


def test_lexical_search_matches_when_only_some_of_the_terms_appear(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "context variables")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    result = store.lexical_search(CORPUS, "what problem do context variables solve", limit=5)

    assert [chunk.text for chunk in result] == ["context variables"]


def test_lexical_search_finds_nothing_for_a_query_with_no_matching_term(
    store: VectorStore,
) -> None:
    store.upsert(CORPUS, [_chunk(0, "an enumeration")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    assert store.lexical_search(CORPUS, "asynchronous generators", limit=5) == ()
    assert store.lexical_search(CORPUS, "what is the", limit=5) == ()


def test_lexical_search_does_not_search_another_corpus_version(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "an enumeration")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    assert store.lexical_search("another-corpus", "enumeration", limit=5) == ()


def test_chunk_terms_are_stemmed_counted_and_free_of_stopwords(store: VectorStore) -> None:
    store.upsert(
        CORPUS,
        [_chunk(0, "The classes define a class of protocols"), _chunk(1, "the of")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        SOURCE_PATHS,
    )

    found = {chunk.chunk_id: chunk for chunk in store.chunk_terms(CORPUS)}

    assert found["chunk-0"].terms == {"class": 2, "defin": 1, "protocol": 1}
    assert found["chunk-0"].source_path == "a.md"
    assert found["chunk-1"].terms == {}


def test_chunk_terms_read_only_the_requested_corpus_version(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "an enumeration")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    assert store.chunk_terms("another-corpus") == ()


def test_a_question_is_normalised_to_the_lexemes_its_chunk_stores(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "Protocol classes")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    stored = set(store.chunk_terms(CORPUS)[0].terms)

    assert set(store.query_terms("What is a protocol class?")) == stored == {"class", "protocol"}


def test_a_text_only_chunk_is_lexical_but_invisible_to_dense(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "an embedded enumeration")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    store.insert_text(CORPUS, [_chunk(1, "a bare enumeration")], SOURCE_PATHS)

    assert store.count(CORPUS) == 2
    assert store.count(CORPUS, embedded=True) == 1
    assert [chunk.chunk_id for chunk in store.search(CORPUS, [1.0, 0.0, 0.0], limit=10)] == [
        "chunk-0"
    ]
    assert {chunk.chunk_id for chunk in store.lexical_search(CORPUS, "enumeration", 10)} == {
        "chunk-0",
        "chunk-1",
    }
    assert store.chunk_terms(CORPUS)[1].terms == {"bare": 1, "enumer": 1}


def test_storing_text_over_an_embedded_chunk_leaves_it_untouched(store: VectorStore) -> None:
    store.upsert(CORPUS, [_chunk(0, "original")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)

    store.insert_text(CORPUS, [_chunk(0, "replacement")], SOURCE_PATHS)

    assert store.count(CORPUS, embedded=True) == 1
    assert store.search(CORPUS, [1.0, 0.0, 0.0], limit=1)[0].text == "original"


def test_embedding_a_text_only_chunk_gives_it_the_vector(store: VectorStore) -> None:
    store.insert_text(CORPUS, [_chunk(0, "later")], SOURCE_PATHS)

    store.upsert(CORPUS, [_chunk(0, "later")], [[0.0, 1.0, 0.0]], SOURCE_PATHS)

    assert store.count(CORPUS) == 1
    assert store.count(CORPUS, embedded=True) == 1
    assert store.search(CORPUS, [0.0, 1.0, 0.0], limit=1)[0].text == "later"


def test_ensure_schema_relaxes_a_table_built_with_a_required_vector(
    database: psycopg.Connection[tuple[object, ...]], store: VectorStore
) -> None:
    store.upsert(CORPUS, [_chunk(0, "kept")], [[1.0, 0.0, 0.0]], SOURCE_PATHS)
    with database.cursor() as cursor:
        cursor.execute(
            sql.SQL("ALTER TABLE {} ALTER COLUMN embedding SET NOT NULL").format(
                sql.Identifier(TABLE)
            )
        )
    database.commit()

    store.ensure_schema()
    store.insert_text(CORPUS, [_chunk(1, "bare")], SOURCE_PATHS)

    assert store.count(CORPUS) == 2
    assert store.count(CORPUS, embedded=True) == 1
