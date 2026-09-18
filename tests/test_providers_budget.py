from pathlib import Path

import pytest

from rageval.providers import (
    Budget,
    BudgetedChatProvider,
    BudgetExceededError,
    CachedChatProvider,
    ChatResult,
    DiskCache,
    Usage,
)


class RecordingChat:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return "model-a"

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        self.calls.append(prompt)
        return ChatResult(
            provider=self.name, model=self.model, text="answer", usage=Usage(input_tokens=1)
        )


def test_a_call_past_the_call_limit_is_refused_before_it_is_made() -> None:
    provider = RecordingChat()
    budgeted = BudgetedChatProvider(provider, Budget(max_calls=1, max_input_chars=1000))

    budgeted.complete("first")

    with pytest.raises(BudgetExceededError, match="1 calls"):
        budgeted.complete("second")
    assert provider.calls == ["first"]


def test_a_call_past_the_character_limit_is_refused_before_it_is_made() -> None:
    provider = RecordingChat()
    budgeted = BudgetedChatProvider(provider, Budget(max_calls=10, max_input_chars=8))

    with pytest.raises(BudgetExceededError, match="9 characters"):
        budgeted.complete("123456789")
    assert provider.calls == []


def test_the_system_message_counts_towards_the_character_budget() -> None:
    budget = Budget(max_calls=10, max_input_chars=100)
    budgeted = BudgetedChatProvider(RecordingChat(), budget)

    budgeted.complete("1234", "5678")

    assert budget.state.input_chars == 8
    assert budget.state.calls == 1


def test_an_error_message_points_at_the_setting_that_raises_the_limit() -> None:
    budgeted = BudgetedChatProvider(RecordingChat(), Budget(max_calls=0, max_input_chars=100))

    with pytest.raises(BudgetExceededError, match="RAGEVAL_BUDGET_MAX_CALLS"):
        budgeted.complete("anything")


def test_a_cached_answer_spends_no_budget(tmp_path: Path) -> None:
    provider = RecordingChat()
    budget = Budget(max_calls=1, max_input_chars=1000)
    cached = CachedChatProvider(
        BudgetedChatProvider(provider, budget),
        DiskCache(tmp_path / "providers"),
        ("recording/model-a",),
    )

    cached.complete("repeated")
    cached.complete("repeated")

    assert budget.state.calls == 1
    assert provider.calls == ["repeated"]
