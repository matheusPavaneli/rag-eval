import json
import re
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rageval.retrieval.store import ScoredChunk

FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)

type UnresolvedReason = Literal["no_such_chunk", "empty_quote", "case_mismatch", "not_in_chunk"]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: int = Field(ge=1)
    quote: str = Field(min_length=1)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[Citation, ...] = ()


class ParseFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str
    raw: str


class ResolvedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["resolved"] = "resolved"
    chunk: int
    quote: str
    source_path: str
    start_char: int
    end_char: int


class UnresolvedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["unresolved"] = "unresolved"
    chunk: int
    quote: str
    reason: UnresolvedReason


type ResolutionResult = Annotated[
    ResolvedCitation | UnresolvedCitation, Field(discriminator="status")
]


def parse_response(text: str) -> ModelAnswer | ParseFailure:
    fenced = FENCE.match(text)
    body = fenced.group(1) if fenced else text

    try:
        return ModelAnswer.model_validate(json.loads(body))
    except ValueError as error:
        # ValidationError is a ValueError, so this covers bad JSON and a bad shape alike.
        kind = "invalid shape" if isinstance(error, ValidationError) else "not JSON"
        return ParseFailure(error=f"{kind}: {error}".splitlines()[0], raw=text)


def resolve(
    citation: Citation, chunks: Sequence[ScoredChunk]
) -> ResolvedCitation | UnresolvedCitation:
    if citation.chunk > len(chunks):
        return _unresolved(citation, "no_such_chunk")
    chunk = chunks[citation.chunk - 1]

    quote = " ".join(citation.quote.split())
    if not quote:
        return _unresolved(citation, "empty_quote")

    collapsed, origin = _collapse(chunk.text)
    found = collapsed.find(quote)
    if found < 0:
        reason: UnresolvedReason = (
            "case_mismatch" if quote.casefold() in collapsed.casefold() else "not_in_chunk"
        )
        return _unresolved(citation, reason)

    # The quote is stripped, so both ends of the match are non-space characters,
    # and origin maps each of them to its exact index in the chunk text.
    start = origin[found]
    end = origin[found + len(quote) - 1] + 1
    return ResolvedCitation(
        chunk=citation.chunk,
        quote=citation.quote,
        source_path=chunk.source_path,
        start_char=chunk.start_char + start,
        end_char=chunk.start_char + end,
    )


def _collapse(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space, as str.split does, keeping each output
    character's index in the original text."""
    characters: list[str] = []
    origin: list[int] = []
    pending: int | None = None

    for index, character in enumerate(text):
        if character.isspace():
            if characters and pending is None:
                pending = index
            continue
        if pending is not None:
            characters.append(" ")
            origin.append(pending)
            pending = None
        characters.append(character)
        origin.append(index)

    return "".join(characters), origin


def _unresolved(citation: Citation, reason: UnresolvedReason) -> UnresolvedCitation:
    return UnresolvedCitation(chunk=citation.chunk, quote=citation.quote, reason=reason)
