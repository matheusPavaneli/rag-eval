import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rageval.answer.citations import ModelAnswer, ResolutionResult, parse_response, resolve
from rageval.providers.base import ChatProvider
from rageval.retrieval.store import ScoredChunk

SYSTEM_PROMPT = """\
You answer questions about Python Enhancement Proposals using only the numbered \
passages you are given.

Reply with one JSON object and nothing else, in this shape:
{"answer": "<the answer, one to three sentences>", \
"citations": [{"chunk": <passage number>, "quote": "<text copied from that passage>"}]}

Rules:
- Every quote must be copied character for character from the passage it cites. \
Do not paraphrase, correct, reformat, or join text from different places.
- Keep each quote as short as it can be while still supporting the answer; \
one sentence is usually enough.
- If the passages do not answer the question, say so in "answer" and return an \
empty "citations" list."""

PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str | None
    citations: tuple[ResolutionResult, ...]
    parse_error: str | None
    raw_response: str | None
    provider: str
    model: str


def build_prompt(question: str, chunks: Sequence[ScoredChunk]) -> str:
    passages = "\n\n".join(
        f"[{number}] {chunk.source_path}\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def answer_question(question: str, chunks: Sequence[ScoredChunk], chat: ChatProvider) -> Answer:
    result = chat.complete(build_prompt(question, chunks), SYSTEM_PROMPT)
    parsed = parse_response(result.text)

    if not isinstance(parsed, ModelAnswer):
        return Answer(
            text=None,
            citations=(),
            parse_error=parsed.error,
            raw_response=parsed.raw,
            provider=result.provider,
            model=result.model,
        )

    return Answer(
        text=parsed.answer,
        citations=tuple(resolve(citation, chunks) for citation in parsed.citations),
        parse_error=None,
        raw_response=None,
        provider=result.provider,
        model=result.model,
    )
