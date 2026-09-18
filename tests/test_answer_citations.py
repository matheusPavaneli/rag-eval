import pytest

from rageval.answer.citations import (
    Citation,
    ModelAnswer,
    ParseFailure,
    ResolvedCitation,
    UnresolvedCitation,
    parse_response,
    resolve,
)
from rageval.retrieval.store import ScoredChunk

DOCUMENT = (
    "# PEP 8\n\n"
    "Limit all lines to a maximum of 79 characters.\n\n"
    "For flowing long blocks of text with fewer structural restrictions\n"
    "(docstrings or comments), the line length should be limited to 72\n"
    "characters.\n"
)


def chunk_of(text: str, start: int, end: int, source_path: str = "pep-0008.md") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{source_path}:{start}:{end}",
        document_id=f"doc-{source_path}",
        source_path=source_path,
        ordinal=0,
        text=text[start:end],
        start_char=start,
        end_char=end,
        score=0.5,
    )


# The document is split into two chunks at an offset that is not zero, so a resolver
# that forgot to add the chunk's start_char would produce the wrong document span.
FIRST = chunk_of(DOCUMENT, 0, 60)
SECOND = chunk_of(DOCUMENT, 58, len(DOCUMENT))
CHUNKS = (FIRST, SECOND)


def test_a_well_formed_response_parses_into_an_answer_and_its_citations() -> None:
    parsed = parse_response('{"answer": "79.", "citations": [{"chunk": 1, "quote": "79"}]}')

    assert parsed == ModelAnswer(answer="79.", citations=(Citation(chunk=1, quote="79"),))


def test_a_response_inside_a_json_fence_parses_the_same_as_bare_json() -> None:
    bare = '{"answer": "79.", "citations": [{"chunk": 1, "quote": "79"}]}'

    assert parse_response(f"```json\n{bare}\n```") == parse_response(bare)
    assert parse_response(f"```\n{bare}\n```\n") == parse_response(bare)


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 79 characters.",
        '{"citations": []}',
        '{"answer": "79.", "citations": [{"chunk": 0, "quote": "79"}]}',
        '{"answer": "79.", "citations": [{"chunk": 1, "quote": ""}]}',
    ],
)
def test_a_malformed_response_is_a_failure_that_keeps_the_raw_text(text: str) -> None:
    parsed = parse_response(text)

    assert isinstance(parsed, ParseFailure)
    assert parsed.raw == text
    assert parsed.error


def test_a_verbatim_quote_resolves_to_its_span_in_the_source_document() -> None:
    quote = "maximum of 79 characters"
    resolved = resolve(Citation(chunk=1, quote=quote), CHUNKS)

    assert isinstance(resolved, ResolvedCitation)
    assert resolved.start_char == FIRST.start_char + FIRST.text.index(quote)
    assert DOCUMENT[resolved.start_char : resolved.end_char] == quote
    assert resolved.source_path == "pep-0008.md"


def test_the_span_is_offset_by_where_the_cited_chunk_starts_in_the_document() -> None:
    quote = "limited to 72"
    resolved = resolve(Citation(chunk=2, quote=quote), CHUNKS)

    assert isinstance(resolved, ResolvedCitation)
    assert resolved.start_char == DOCUMENT.index(quote)
    assert DOCUMENT[resolved.start_char : resolved.end_char] == quote


def test_a_quote_differing_only_in_whitespace_resolves_to_the_original_characters() -> None:
    # The source breaks the line after "restrictions"; the model wrote a space and
    # doubled another one. The span must still start and end on the quoted words.
    quote = "  fewer structural  restrictions (docstrings or comments), "
    resolved = resolve(Citation(chunk=2, quote=quote), CHUNKS)

    assert isinstance(resolved, ResolvedCitation)
    span = DOCUMENT[resolved.start_char : resolved.end_char]
    assert span == "fewer structural restrictions\n(docstrings or comments),"
    assert resolved.quote == quote


@pytest.mark.parametrize(
    ("citation", "reason"),
    [
        (Citation(chunk=1, quote="lines must not exceed 79 characters"), "not_in_chunk"),
        (Citation(chunk=1, quote="limit all lines to a maximum"), "case_mismatch"),
        (Citation(chunk=3, quote="Limit all lines"), "no_such_chunk"),
        (Citation(chunk=1, quote=" \n "), "empty_quote"),
    ],
)
def test_a_quote_that_is_not_in_its_chunk_is_unresolved_with_the_reason(
    citation: Citation, reason: str
) -> None:
    resolved = resolve(citation, CHUNKS)

    assert isinstance(resolved, UnresolvedCitation)
    assert resolved.reason == reason


def test_a_quote_is_only_searched_for_in_the_chunk_it_cites() -> None:
    resolved = resolve(Citation(chunk=1, quote="limited to 72"), CHUNKS)

    assert isinstance(resolved, UnresolvedCitation)
    assert resolved.reason == "not_in_chunk"
