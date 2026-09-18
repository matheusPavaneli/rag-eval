from collections.abc import Sequence
from typing import Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict

from rageval.ingest.chunking import Chunk

TABLE = "chunks"


class RetrievalError(Exception):
    pass


class DimensionMismatchError(RetrievalError):
    pass


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_path: str
    ordinal: int
    text: str
    start_char: int
    end_char: int
    score: float


class ChunkTerms(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source_path: str
    ordinal: int
    text: str
    start_char: int
    end_char: int
    terms: dict[str, int]

    def scored(self, score: float) -> ScoredChunk:
        return ScoredChunk(**self.model_dump(exclude={"terms"}), score=score)


class ChunkStore(Protocol):
    @property
    def dimension(self) -> int: ...

    def ensure_schema(self) -> None: ...

    def upsert(
        self,
        corpus_version: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        source_paths: dict[str, str],
    ) -> int: ...

    def insert_text(
        self, corpus_version: str, chunks: Sequence[Chunk], source_paths: dict[str, str]
    ) -> int: ...

    def search(
        self, corpus_version: str, vector: Sequence[float], limit: int
    ) -> tuple[ScoredChunk, ...]: ...

    def lexical_search(
        self, corpus_version: str, query: str, limit: int
    ) -> tuple[ScoredChunk, ...]: ...

    def chunk_terms(self, corpus_version: str) -> tuple[ChunkTerms, ...]: ...

    def query_terms(self, question: str) -> tuple[str, ...]: ...

    def count(self, corpus_version: str, embedded: bool = False) -> int: ...


class VectorStore:
    def __init__(
        self,
        connection: psycopg.Connection[tuple[object, ...]],
        dimension: int,
        table: str = TABLE,
    ) -> None:
        self._connection = connection
        self._dimension = dimension
        self._table_name = table
        self._table = sql.Identifier(table)

    @property
    def dimension(self) -> int:
        return self._dimension

    def ensure_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        corpus_version text    NOT NULL,
                        chunk_id       text    NOT NULL,
                        document_id    text    NOT NULL,
                        source_path    text    NOT NULL,
                        ordinal        integer NOT NULL,
                        text           text    NOT NULL,
                        start_char     integer NOT NULL,
                        end_char       integer NOT NULL,
                        embedding      vector({dimension}),
                        PRIMARY KEY (corpus_version, chunk_id)
                    )
                    """
                ).format(table=self._table, dimension=sql.Literal(self._dimension))
            )
            cursor.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN embedding DROP NOT NULL").format(
                    table=self._table
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    ALTER TABLE {table} ADD COLUMN IF NOT EXISTS lexical tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                    """
                ).format(table=self._table)
            )
            cursor.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {table} USING gin (lexical)").format(
                    index=sql.Identifier(f"{self._table_name}_lexical_idx"), table=self._table
                )
            )
        self._connection.commit()
        self._assert_dimension()

    def upsert(
        self,
        corpus_version: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        source_paths: dict[str, str],
    ) -> int:
        if len(chunks) != len(vectors):
            raise RetrievalError(
                f"{len(chunks)} chunks were given {len(vectors)} vectors; "
                "every chunk must carry exactly one embedding"
            )

        rows = [
            (
                corpus_version,
                chunk.id,
                chunk.document_id,
                source_paths[chunk.document_id],
                chunk.ordinal,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                _literal(vector, self._dimension),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        with self._connection.cursor() as cursor:
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (corpus_version, chunk_id, document_id, source_path,
                                         ordinal, text, start_char, end_char, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (corpus_version, chunk_id) DO UPDATE
                        SET document_id = EXCLUDED.document_id,
                            source_path = EXCLUDED.source_path,
                            ordinal = EXCLUDED.ordinal,
                            text = EXCLUDED.text,
                            start_char = EXCLUDED.start_char,
                            end_char = EXCLUDED.end_char,
                            embedding = EXCLUDED.embedding
                    """
                ).format(table=self._table),
                rows,
            )
        self._connection.commit()
        return len(rows)

    def insert_text(
        self, corpus_version: str, chunks: Sequence[Chunk], source_paths: dict[str, str]
    ) -> int:
        rows = [
            (
                corpus_version,
                chunk.id,
                chunk.document_id,
                source_paths[chunk.document_id],
                chunk.ordinal,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
            )
            for chunk in chunks
        ]

        with self._connection.cursor() as cursor:
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {table} (corpus_version, chunk_id, document_id, source_path,
                                         ordinal, text, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (corpus_version, chunk_id) DO NOTHING
                    """
                ).format(table=self._table),
                rows,
            )
        self._connection.commit()
        return len(rows)

    def search(
        self, corpus_version: str, vector: Sequence[float], limit: int
    ) -> tuple[ScoredChunk, ...]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT chunk_id, document_id, source_path, ordinal, text,
                           start_char, end_char, 1 - (embedding <=> %s::vector) AS score
                    FROM {table}
                    WHERE corpus_version = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ).format(table=self._table),
                (
                    _literal(vector, self._dimension),
                    corpus_version,
                    _literal(vector, self._dimension),
                    limit,
                ),
            )
            rows = cursor.fetchall()

        return tuple(_scored(row) for row in rows)

    def lexical_search(
        self, corpus_version: str, query: str, limit: int
    ) -> tuple[ScoredChunk, ...]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT chunk_id, document_id, source_path, ordinal, text,
                           start_char, end_char, ts_rank_cd(lexical, query) AS score
                    FROM {table},
                         CAST(replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')
                              AS tsquery) AS query
                    WHERE corpus_version = %s AND lexical @@ query
                    ORDER BY score DESC, chunk_id
                    LIMIT %s
                    """
                ).format(table=self._table),
                (query, corpus_version, limit),
            )
            rows = cursor.fetchall()

        return tuple(_scored(row) for row in rows)

    def chunk_terms(self, corpus_version: str) -> tuple[ChunkTerms, ...]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    SELECT chunk.chunk_id, chunk.document_id, chunk.source_path, chunk.ordinal,
                           chunk.text, chunk.start_char, chunk.end_char, term.lexeme,
                           coalesce(array_length(term.positions, 1), 1)
                    FROM {table} AS chunk
                    LEFT JOIN LATERAL unnest(chunk.lexical) AS term(lexeme, positions, weights)
                        ON true
                    WHERE chunk.corpus_version = %s
                    ORDER BY chunk.chunk_id, term.lexeme
                    """
                ).format(table=self._table),
                (corpus_version,),
            )
            rows = cursor.fetchall()

        grouped: dict[str, tuple[tuple[object, ...], dict[str, int]]] = {}
        for row in rows:
            _, terms = grouped.setdefault(str(row[0]), (row, {}))
            if row[7] is not None:
                terms[str(row[7])] = int(row[8])  # type: ignore[call-overload]

        return tuple(
            ChunkTerms(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                source_path=str(row[2]),
                ordinal=int(row[3]),  # type: ignore[call-overload]
                text=str(row[4]),
                start_char=int(row[5]),  # type: ignore[call-overload]
                end_char=int(row[6]),  # type: ignore[call-overload]
                terms=terms,
            )
            for row, terms in grouped.values()
        )

    def query_terms(self, question: str) -> tuple[str, ...]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT lexeme FROM unnest(to_tsvector('english', %s)) ORDER BY lexeme",
                (question,),
            )
            rows = cursor.fetchall()
        return tuple(str(row[0]) for row in rows)

    def count(self, corpus_version: str, embedded: bool = False) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT count(*) FROM {table} "
                    "WHERE corpus_version = %s AND (%s OR embedding IS NOT NULL)"
                ).format(table=self._table),
                (corpus_version, not embedded),
            )
            row = cursor.fetchone()
        return 0 if row is None else int(row[0])  # type: ignore[call-overload]

    def _assert_dimension(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = %s::regclass AND attname = 'embedding'
                """,
                (self._table_name,),
            )
            row = cursor.fetchone()

        if row is None:
            raise RetrievalError(f"table {self._table_name} has no embedding column")

        stored = int(row[0])  # type: ignore[call-overload]
        if stored != self._dimension:
            raise DimensionMismatchError(
                f"table {self._table_name} stores {stored}-dimension vectors but this run is "
                f"for {self._dimension}: drop the table or set RAGEVAL_EMBEDDING_DIMENSION "
                f"to {stored}"
            )


def _scored(row: tuple[object, ...]) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=str(row[0]),
        document_id=str(row[1]),
        source_path=str(row[2]),
        ordinal=int(row[3]),  # type: ignore[call-overload]
        text=str(row[4]),
        start_char=int(row[5]),  # type: ignore[call-overload]
        end_char=int(row[6]),  # type: ignore[call-overload]
        score=float(row[7]),  # type: ignore[arg-type]
    )


def _literal(vector: Sequence[float], dimension: int) -> str:
    if len(vector) != dimension:
        raise DimensionMismatchError(
            f"vector has {len(vector)} dimensions, the store is built for {dimension}"
        )
    return f"[{','.join(repr(float(value)) for value in vector)}]"
