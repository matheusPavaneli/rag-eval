from collections.abc import Sequence
from typing import Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict

from rageval.ingest.chunking import Chunk

TABLE = "chunks"

_TABLE = sql.Identifier(TABLE)


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

    def search(
        self, corpus_version: str, vector: Sequence[float], limit: int
    ) -> tuple[ScoredChunk, ...]: ...

    def count(self, corpus_version: str) -> int: ...


class VectorStore:
    def __init__(self, connection: psycopg.Connection[tuple[object, ...]], dimension: int) -> None:
        self._connection = connection
        self._dimension = dimension

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
                        embedding      vector({dimension}) NOT NULL,
                        PRIMARY KEY (corpus_version, chunk_id)
                    )
                    """
                ).format(table=_TABLE, dimension=sql.Literal(self._dimension))
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
                        SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """
                ).format(table=_TABLE),
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
                    WHERE corpus_version = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """
                ).format(table=_TABLE),
                (
                    _literal(vector, self._dimension),
                    corpus_version,
                    _literal(vector, self._dimension),
                    limit,
                ),
            )
            rows = cursor.fetchall()

        return tuple(
            ScoredChunk(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                source_path=str(row[2]),
                ordinal=int(row[3]),  # type: ignore[call-overload]
                text=str(row[4]),
                start_char=int(row[5]),  # type: ignore[call-overload]
                end_char=int(row[6]),  # type: ignore[call-overload]
                score=float(row[7]),  # type: ignore[arg-type]
            )
            for row in rows
        )

    def count(self, corpus_version: str) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {table} WHERE corpus_version = %s").format(
                    table=_TABLE
                ),
                (corpus_version,),
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
                (TABLE,),
            )
            row = cursor.fetchone()

        if row is None:
            raise RetrievalError(f"table {TABLE} has no embedding column")

        stored = int(row[0])  # type: ignore[call-overload]
        if stored != self._dimension:
            raise DimensionMismatchError(
                f"table {TABLE} stores {stored}-dimension vectors but this run is configured "
                f"for {self._dimension}: drop the table or set RAGEVAL_EMBEDDING_DIMENSION "
                f"to {stored}"
            )


def _literal(vector: Sequence[float], dimension: int) -> str:
    if len(vector) != dimension:
        raise DimensionMismatchError(
            f"vector has {len(vector)} dimensions, the store is built for {dimension}"
        )
    return f"[{','.join(repr(float(value)) for value in vector)}]"
