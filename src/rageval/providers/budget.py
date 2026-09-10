from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rageval.providers.base import (
    BudgetExceededError,
    ChatProvider,
    ChatResult,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingTask,
)


class BudgetState(BaseModel):
    model_config = ConfigDict(frozen=True)

    calls: int
    input_chars: int
    max_calls: int
    max_input_chars: int


class Budget:
    def __init__(self, max_calls: int, max_input_chars: int) -> None:
        self._max_calls = max_calls
        self._max_input_chars = max_input_chars
        self._calls = 0
        self._input_chars = 0

    @property
    def state(self) -> BudgetState:
        return BudgetState(
            calls=self._calls,
            input_chars=self._input_chars,
            max_calls=self._max_calls,
            max_input_chars=self._max_input_chars,
        )

    def spend(self, chars: int, who: str) -> None:
        if self._calls + 1 > self._max_calls:
            raise BudgetExceededError(
                f"{who}: budget of {self._max_calls} calls is spent; "
                "raise RAGEVAL_BUDGET_MAX_CALLS or run against the cache"
            )
        if self._input_chars + chars > self._max_input_chars:
            raise BudgetExceededError(
                f"{who}: this call sends {chars} characters and the budget of "
                f"{self._max_input_chars} has {self._max_input_chars - self._input_chars} left"
            )
        self._calls += 1
        self._input_chars += chars


class BudgetedChatProvider:
    def __init__(self, provider: ChatProvider, budget: Budget) -> None:
        self._provider = provider
        self._budget = budget

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        self._budget.spend(len(prompt) + len(system or ""), f"{self.name} {self.model}")
        return self._provider.complete(prompt, system)


class BudgetedEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, budget: Budget) -> None:
        self._provider = provider
        self._budget = budget

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        self._budget.spend(sum(len(text) for text in texts), f"{self.name} {self.model}")
        return self._provider.embed(texts, task)
