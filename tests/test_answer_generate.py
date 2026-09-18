from conftest_retrieval import scored
from rageval.answer.citations import ResolvedCitation, UnresolvedCitation
from rageval.answer.generate import SYSTEM_PROMPT, answer_question, build_prompt
from rageval.providers.base import ChatResult
from rageval.retrieval.store import ScoredChunk


class ScriptedChat:
    """A chat provider standing in front of another one, as failover does: its own
    name and model differ from the ones on the result it returns."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[str, str | None]] = []

    @property
    def name(self) -> str:
        return "failover"

    @property
    def model(self) -> str:
        return "primary-model"

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        self.calls.append((prompt, system))
        return ChatResult(provider="groq", model="fallback-model", text=self._reply)


def passage(source_path: str, start: int, text: str) -> ScoredChunk:
    return scored(source_path, start, start + len(text)).model_copy(update={"text": text})


CHUNKS = (
    passage("pep-0008.md", 100, "Limit all lines to a maximum of 79 characters."),
    passage("pep-0257.md", 0, "A docstring is a string literal."),
)


def test_the_prompt_numbers_the_passages_in_retrieval_order() -> None:
    prompt = build_prompt("How long may a line be?", CHUNKS)

    first, second = prompt.index("[1] pep-0008.md"), prompt.index("[2] pep-0257.md")
    assert first < second
    assert CHUNKS[0].text in prompt[first:second]
    assert CHUNKS[1].text in prompt[second:]
    assert prompt.endswith("Question: How long may a line be?")


def test_the_same_inputs_build_the_same_prompt_so_the_cache_key_is_stable() -> None:
    assert build_prompt("q", CHUNKS) == build_prompt("q", CHUNKS)


def test_an_answer_resolves_its_citations_against_the_retrieved_chunks() -> None:
    chat = ScriptedChat(
        '{"answer": "79 characters.", "citations": ['
        '{"chunk": 1, "quote": "maximum of 79 characters"},'
        '{"chunk": 2, "quote": "not there"}]}'
    )

    answer = answer_question("How long may a line be?", CHUNKS, chat)

    assert chat.calls == [(build_prompt("How long may a line be?", CHUNKS), SYSTEM_PROMPT)]
    assert answer.text == "79 characters."
    resolved, unresolved = answer.citations
    assert isinstance(resolved, ResolvedCitation)
    assert (resolved.start_char, resolved.end_char) == (100 + 21, 100 + 45)
    assert isinstance(unresolved, UnresolvedCitation)


def test_the_answer_records_the_provider_that_produced_it_not_the_wrapper() -> None:
    answer = answer_question("q", CHUNKS, ScriptedChat('{"answer": "a", "citations": []}'))

    assert (answer.provider, answer.model) == ("groq", "fallback-model")


def test_an_unparseable_reply_is_kept_as_a_failed_answer_instead_of_raising() -> None:
    answer = answer_question("q", CHUNKS, ScriptedChat("Lines may be 79 characters."))

    assert answer.text is None
    assert answer.citations == ()
    assert answer.parse_error is not None
    assert answer.raw_response == "Lines may be 79 characters."
