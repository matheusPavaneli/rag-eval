from itertools import pairwise

import pytest
from pydantic import ValidationError

from rageval.ingest.chunking import ChunkConfig, chunk_document
from rageval.ingest.documents import Document, document_id

CONFIG = ChunkConfig(chunk_size=200, overlap=40)

PROSE = "\n\n".join(f"Paragraph {index} " + "word " * 30 for index in range(12))

MARKDOWN = """# Handbook

Intro line.

## Retrieval

Retrieval body that is long enough to matter.

### Reranking

Reranking body.

## Evaluation

Evaluation body.
"""


def document(text: str, name: str = "doc.md") -> Document:
    return Document(
        id=document_id(text),
        source_path=name,
        title=name,
        text=text,
        byte_size=len(text.encode("utf-8")),
    )


@pytest.mark.parametrize("text", [PROSE, MARKDOWN, "one paragraph only", "a" * 700])
def test_every_chunk_span_reproduces_its_text(text: str) -> None:
    source = document(text)

    for chunk in chunk_document(source, CONFIG):
        assert source.text[chunk.start_char : chunk.end_char] == chunk.text


def test_no_chunk_exceeds_chunk_size_and_overlap_is_bounded() -> None:
    chunks = chunk_document(document(PROSE), CONFIG)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.end_char - chunk.start_char <= CONFIG.chunk_size
    for previous, following in pairwise(chunks):
        assert following.start_char > previous.start_char
        assert previous.end_char - following.start_char <= CONFIG.overlap


def test_ordinals_are_dense_and_ordered() -> None:
    chunks = chunk_document(document(PROSE), CONFIG)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_a_short_document_is_one_chunk_spanning_its_whole_text() -> None:
    source = document("Short body, nothing to split.")

    chunks = chunk_document(source, CONFIG)

    assert len(chunks) == 1
    assert (chunks[0].start_char, chunks[0].end_char) == (0, len(source.text))


def test_an_empty_document_yields_no_chunks() -> None:
    assert chunk_document(document("   \n\n  ")) == []


def test_a_word_longer_than_chunk_size_becomes_its_own_chunk() -> None:
    word = "x" * (CONFIG.chunk_size * 2)
    source = document(f"lead in\n\n{word}\n\ntail")

    chunks = chunk_document(source, CONFIG)

    assert any(chunk.text == word for chunk in chunks)


def test_a_chunk_carries_the_heading_path_in_force_at_its_start() -> None:
    chunks = chunk_document(document(MARKDOWN), ChunkConfig(chunk_size=60, overlap=10))

    by_text = {chunk.text: chunk.heading_path for chunk in chunks}
    reranking = next(path for text, path in by_text.items() if "Reranking body" in text)

    assert reranking == ("Handbook", "Retrieval", "Reranking")


def test_a_comment_inside_a_code_fence_is_not_a_heading() -> None:
    text = (
        "# Guide\n\n## Setup\n\n```python\n# install the client\nrun()\n```\n\n"
        "After the block, prose that still belongs to Setup.\n"
    )

    chunks = chunk_document(document(text), ChunkConfig(chunk_size=40, overlap=5))

    assert all(chunk.heading_path[0] == "Guide" for chunk in chunks)
    prose = next(chunk for chunk in chunks if "still belongs" in chunk.text)
    assert prose.heading_path == ("Guide", "Setup")


def test_chunk_ids_are_stable_and_distinct() -> None:
    first = chunk_document(document(PROSE), CONFIG)
    second = chunk_document(document(PROSE), CONFIG)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="overlap must be smaller"):
        ChunkConfig(chunk_size=100, overlap=100)


def test_chunking_is_independent_of_the_source_path() -> None:
    assert chunk_document(document(PROSE, "a.md"), CONFIG) == chunk_document(
        document(PROSE, "b.md"), CONFIG
    )
