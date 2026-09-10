import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rageval.ingest.documents import Document


class GoldenSetError(Exception):
    pass


class Support(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def _span_is_not_empty(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError(f"end_char must be greater than start_char, got {self!r}")
        return self


class GoldenQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answer: str
    supports: tuple[Support, ...] = Field(min_length=1)


def load_golden_set(path: Path, documents: Sequence[Document]) -> tuple[GoldenQuestion, ...]:
    by_path: Mapping[str, Document] = {document.source_path: document for document in documents}

    questions: list[GoldenQuestion] = []
    seen: set[str] = set()

    for number, line in _lines(path):
        question = _question(line, path, number)

        if question.id in seen:
            raise GoldenSetError(f"{path} line {number}: question id {question.id!r} is used twice")
        seen.add(question.id)

        for support in question.supports:
            _resolve(support, by_path, question, path, number)

        questions.append(question)

    if not questions:
        raise GoldenSetError(f"{path} holds no question")
    return tuple(questions)


def _resolve(
    support: Support,
    by_path: Mapping[str, Document],
    question: GoldenQuestion,
    path: Path,
    number: int,
) -> None:
    document = by_path.get(support.source_path)
    if document is None:
        raise GoldenSetError(
            f"{path} line {number}: question {question.id!r} supports "
            f"{support.source_path!r}, which is not in the corpus"
        )

    if support.end_char > len(document.text):
        raise GoldenSetError(
            f"{path} line {number}: question {question.id!r} supports characters "
            f"{support.start_char}-{support.end_char} of {support.source_path!r}, "
            f"which holds {len(document.text)}"
        )


def _lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GoldenSetError(f"cannot read the golden set at {path}") from error

    return [
        (number, line)
        for number, line in enumerate(text.split("\n"), start=1)
        if line.strip() and not line.lstrip().startswith("//")
    ]


def _question(line: str, path: Path, number: int) -> GoldenQuestion:
    try:
        record: object = json.loads(line)
    except ValueError as error:
        raise GoldenSetError(f"{path} line {number} is not JSON") from error

    try:
        return GoldenQuestion.model_validate(record)
    except ValidationError as error:
        raise GoldenSetError(f"{path} line {number} is not a golden question: {error}") from error
