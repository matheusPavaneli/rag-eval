from collections.abc import Sequence

import httpx

from rageval.config import Settings
from rageval.providers.base import (
    ChatProvider,
    ChatResult,
    EmbeddingProvider,
    PermanentProviderError,
    ProviderError,
    TransientProviderError,
)
from rageval.providers.budget import Budget, BudgetedChatProvider, BudgetedEmbeddingProvider
from rageval.providers.cache import (
    CachedChatProvider,
    CachedEmbeddingProvider,
    CacheOnlyEmbeddingProvider,
    DiskCache,
)
from rageval.providers.gemini import NAME as GEMINI
from rageval.providers.gemini import GeminiChatProvider, GeminiEmbeddingProvider
from rageval.providers.groq import GroqChatProvider

CACHE_NAMESPACE = "providers"


class ProviderUnavailableError(ProviderError):
    pass


class FailoverChatProvider:
    def __init__(self, providers: Sequence[ChatProvider]) -> None:
        if not providers:
            raise PermanentProviderError(
                "no chat provider is configured: set RAGEVAL_GEMINI_API_KEY or RAGEVAL_GROQ_API_KEY"
            )
        self._providers = tuple(providers)

    @property
    def name(self) -> str:
        return "failover"

    @property
    def model(self) -> str:
        return self._providers[0].model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        failures: list[str] = []
        last: TransientProviderError | None = None

        for provider in self._providers:
            try:
                return provider.complete(prompt, system)
            except TransientProviderError as error:
                failures.append(f"{provider.name} ({error})")
                last = error

        raise ProviderUnavailableError(
            f"every chat provider failed: {', '.join(failures)}"
        ) from last


def build_budget(settings: Settings) -> Budget:
    return Budget(settings.budget_max_calls, settings.budget_max_input_chars)


def build_chat_provider(
    settings: Settings, client: httpx.Client, budget: Budget | None = None
) -> ChatProvider:
    spend = budget if budget is not None else build_budget(settings)
    cache = DiskCache(settings.cache_dir / CACHE_NAMESPACE)
    providers: list[ChatProvider] = []

    if settings.gemini_api_key is not None:
        providers.append(
            CachedChatProvider(
                BudgetedChatProvider(
                    GeminiChatProvider(
                        client,
                        settings.gemini_api_key,
                        settings.gemini_chat_model,
                        settings.provider_timeout_seconds,
                    ),
                    spend,
                ),
                cache,
            )
        )

    if settings.groq_api_key is not None:
        providers.append(
            CachedChatProvider(
                BudgetedChatProvider(
                    GroqChatProvider(
                        client,
                        settings.groq_api_key,
                        settings.groq_chat_model,
                        settings.provider_timeout_seconds,
                    ),
                    spend,
                ),
                cache,
            )
        )

    return FailoverChatProvider(providers)


def build_embedding_provider(
    settings: Settings, client: httpx.Client, budget: Budget | None = None
) -> EmbeddingProvider:
    cache = DiskCache(settings.cache_dir / CACHE_NAMESPACE)
    if settings.gemini_api_key is None:
        # Groq exposes no embedding API, so without a Gemini key the cache is the
        # only source of vectors: a miss fails instead of reaching the network.
        return CachedEmbeddingProvider(
            CacheOnlyEmbeddingProvider(
                GEMINI, settings.gemini_embedding_model, settings.embedding_dimension
            ),
            cache,
        )

    spend = budget if budget is not None else build_budget(settings)
    return CachedEmbeddingProvider(
        BudgetedEmbeddingProvider(
            GeminiEmbeddingProvider(
                client,
                settings.gemini_api_key,
                settings.gemini_embedding_model,
                settings.embedding_dimension,
                settings.provider_timeout_seconds,
            ),
            spend,
        ),
        cache,
    )
