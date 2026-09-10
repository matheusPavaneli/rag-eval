import hashlib
import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rageval.ingest.documents import Document

SEPARATORS: tuple[str, ...] = ("\n#", "\n\n", "\n", " ")

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class ChunkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _overlap_fits_inside_a_chunk(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    ordinal: int
    text: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]


def chunk_id(document: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{document}:{start}:{end}".encode()).hexdigest()


def chunk_document(document: Document, config: ChunkConfig | None = None) -> list[Chunk]:
    settings = config or ChunkConfig()
    offsets, paths = _heading_index(document.text)

    chunks: list[Chunk] = []
    for span_start, span_end, anchor in _spans(document.text, settings):
        start, end = _trim(document.text, span_start, span_end)
        if start == end:
            continue
        heading_at, _ = _trim(document.text, max(start, anchor), end)
        chunks.append(
            Chunk(
                id=chunk_id(document.id, start, end),
                document_id=document.id,
                ordinal=len(chunks),
                text=document.text[start:end],
                start_char=start,
                end_char=end,
                heading_path=paths[bisect_right(offsets, heading_at) - 1],
            )
        )
    return chunks


def _spans(text: str, config: ChunkConfig) -> list[tuple[int, int, int]]:
    pieces = list(_segments(text, 0, SEPARATORS, config.chunk_size))

    spans: list[tuple[int, int, int]] = []
    index = 0
    while index < len(pieces):
        offset, piece = pieces[index]
        start = offset if not spans else max(spans[-1][0] + 1, offset - config.overlap)
        end = offset + len(piece)
        index += 1

        while index < len(pieces):
            next_offset, next_piece = pieces[index]
            if next_offset + len(next_piece) - start > config.chunk_size:
                break
            end = next_offset + len(next_piece)
            index += 1

        if end - start > config.chunk_size:
            start = offset
        spans.append((start, end, offset))
    return spans


def _segments(
    text: str, offset: int, separators: Sequence[str], limit: int
) -> Iterator[tuple[int, str]]:
    if len(text) <= limit or not separators:
        yield offset, text
        return

    pieces = _partition(text, separators[0], offset)
    if len(pieces) == 1:
        yield from _segments(text, offset, separators[1:], limit)
        return

    for piece_offset, piece in pieces:
        yield from _segments(piece, piece_offset, separators[1:], limit)


def _partition(text: str, separator: str, offset: int) -> list[tuple[int, str]]:
    pieces: list[tuple[int, str]] = []
    cursor = 0
    while True:
        index = text.find(separator, cursor + 1)
        if index == -1:
            pieces.append((offset + cursor, text[cursor:]))
            return pieces
        pieces.append((offset + cursor, text[cursor:index]))
        cursor = index


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _heading_index(text: str) -> tuple[list[int], list[tuple[str, ...]]]:
    offsets = [0]
    paths: list[tuple[str, ...]] = [()]

    stack: list[str] = []
    position = 0
    fenced = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            position += len(line)
            continue

        match = None if fenced else _HEADING.match(line.rstrip("\n"))
        if match is not None:
            level = len(match.group(1))
            del stack[level - 1 :]
            stack.append(match.group(2))
            offsets.append(position)
            paths.append(tuple(stack))
        position += len(line)
    return offsets, paths
