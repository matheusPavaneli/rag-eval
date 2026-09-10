import pytest

from rageval.providers import (
    ChatResult,
    FailoverChatProvider,
    PermanentProviderError,
    ProviderUnavailableError,
    TransientProviderError,
    Usage,
)


class StubChat:
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.calls = 0
        self._name = name
        self._error = error

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return f"{self._name}-model"

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ChatResult(
            provider=self._name, model=self.model, text="answer", usage=Usage(input_tokens=1)
        )


def test_a_transient_failure_is_answered_by_the_next_provider() -> None:
    first = StubChat("gemini", TransientProviderError("gemini model: HTTP 429"))
    second = StubChat("groq")

    answer = FailoverChatProvider([first, second]).complete("question")

    assert answer.provider == "groq"
    assert (first.calls, second.calls) == (1, 1)


def test_a_permanent_failure_stops_the_chain_where_it_happened() -> None:
    first = StubChat("gemini", PermanentProviderError("gemini model: HTTP 401"))
    second = StubChat("groq")

    with pytest.raises(PermanentProviderError, match="401"):
        FailoverChatProvider([first, second]).complete("question")
    assert second.calls == 0


def test_when_every_provider_fails_the_last_error_is_kept_as_the_cause() -> None:
    last = TransientProviderError("groq model: HTTP 503")
    first = StubChat("gemini", TransientProviderError("gemini model: HTTP 429"))
    second = StubChat("groq", last)

    with pytest.raises(ProviderUnavailableError) as raised:
        FailoverChatProvider([first, second]).complete("question")

    assert raised.value.__cause__ is last
    assert "gemini" in str(raised.value)
    assert "groq" in str(raised.value)


def test_a_chain_with_no_provider_names_the_variables_that_would_fill_it() -> None:
    with pytest.raises(PermanentProviderError, match="RAGEVAL_GEMINI_API_KEY"):
        FailoverChatProvider([])


def test_the_chain_reports_the_model_of_its_first_choice() -> None:
    chain = FailoverChatProvider([StubChat("gemini"), StubChat("groq")])

    assert chain.model == "gemini-model"
    assert chain.name == "failover"
