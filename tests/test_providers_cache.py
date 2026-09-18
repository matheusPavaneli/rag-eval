from collections.abc import Sequence
from pathlib import Path

import pytest

from rageval.providers import (
    CachedChatProvider,
    CachedEmbeddingProvider,
    CacheEntryError,
    CacheMissError,
    CacheOnlyEmbeddingProvider,
    ChatResult,
    DiskCache,
    EmbeddingResult,
    EmbeddingTask,
    PermanentProviderError,
    Usage,
    embedding_digest,
)


class RecordingChat:
    def __init__(self, model: str = "model-a", text: str = "answer") -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._model = model
        self._text = text

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        self.calls.append((prompt, system))
        return ChatResult(
            provider=self.name, model=self._model, text=self._text, usage=Usage(input_tokens=7)
        )


class RecordingEmbedding:
    def __init__(self, dimension: int = 3) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return "embed-a"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        self.calls.append((tuple(texts), task))
        return EmbeddingResult(
            provider=self.name,
            model=self.model,
            dimension=self._dimension,
            vectors=tuple(
                tuple(float(len(text) + offset) for offset in range(self._dimension))
                for text in texts
            ),
        )


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(tmp_path / "providers")


def test_second_identical_call_is_answered_from_disk(cache: DiskCache) -> None:
    provider = RecordingChat()
    cached = CachedChatProvider(provider, cache)

    first = cached.complete("what is a span?", "be terse")
    second = cached.complete("what is a span?", "be terse")

    assert first == second
    assert provider.calls == [("what is a span?", "be terse")]


def test_a_changed_prompt_or_system_message_is_a_different_entry(cache: DiskCache) -> None:
    provider = RecordingChat()
    cached = CachedChatProvider(provider, cache)

    cached.complete("one", "be terse")
    cached.complete("two", "be terse")
    cached.complete("one", None)

    assert len(provider.calls) == 3


def test_another_model_never_reads_the_first_models_answer(cache: DiskCache) -> None:
    first = CachedChatProvider(RecordingChat(model="model-a", text="from a"), cache)
    second_provider = RecordingChat(model="model-b", text="from b")
    second = CachedChatProvider(second_provider, cache)

    first.complete("same prompt")
    answer = second.complete("same prompt")

    assert answer.text == "from b"
    assert second_provider.calls == [("same prompt", None)]


def test_an_entry_survives_into_a_fresh_cache_over_the_same_directory(tmp_path: Path) -> None:
    root = tmp_path / "providers"
    provider = RecordingChat()

    CachedChatProvider(provider, DiskCache(root)).complete("persisted")
    answer = CachedChatProvider(provider, DiskCache(root)).complete("persisted")

    assert answer.text == "answer"
    assert len(provider.calls) == 1


def test_writing_an_entry_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    root = tmp_path / "providers"

    CachedChatProvider(RecordingChat(), DiskCache(root)).complete("written")

    assert list(root.rglob("*.tmp")) == []
    assert len(list(root.rglob("*.json"))) == 1


def test_an_unreadable_entry_names_its_path_instead_of_refetching(tmp_path: Path) -> None:
    root = tmp_path / "providers"
    provider = RecordingChat()
    CachedChatProvider(provider, DiskCache(root)).complete("corrupted")
    entry = next(iter(root.rglob("*.json")))
    entry.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(CacheEntryError, match=entry.name):
        CachedChatProvider(provider, DiskCache(root)).complete("corrupted")


def test_an_entry_holding_the_wrong_shape_is_reported_not_returned(tmp_path: Path) -> None:
    root = tmp_path / "providers"
    provider = RecordingChat()
    CachedChatProvider(provider, DiskCache(root)).complete("wrong shape")
    entry = next(iter(root.rglob("*.json")))
    entry.write_text('{"unexpected": true}', encoding="utf-8")

    with pytest.raises(CacheEntryError, match="ChatResult"):
        CachedChatProvider(provider, DiskCache(root)).complete("wrong shape")


def test_only_the_texts_that_are_new_reach_the_provider(cache: DiskCache) -> None:
    provider = RecordingEmbedding()
    cached = CachedEmbeddingProvider(provider, cache)

    first = cached.embed(["alpha", "beta"])
    second = cached.embed(["alpha", "gamma"])

    assert provider.calls == [(("alpha", "beta"), "document"), (("gamma",), "document")]
    assert second.vectors[0] == first.vectors[0]


def test_cached_vectors_come_back_in_the_order_they_were_asked_for(cache: DiskCache) -> None:
    provider = RecordingEmbedding()
    cached = CachedEmbeddingProvider(provider, cache)
    cached.embed(["alpha"])

    result = cached.embed(["beta", "alpha", "gamma"])

    assert result.vectors == (
        (4.0, 5.0, 6.0),
        (5.0, 6.0, 7.0),
        (5.0, 6.0, 7.0),
    )


def test_the_same_text_as_a_document_and_as_a_query_are_two_entries(cache: DiskCache) -> None:
    provider = RecordingEmbedding()
    cached = CachedEmbeddingProvider(provider, cache)

    cached.embed(["shared text"], "document")
    cached.embed(["shared text"], "query")

    assert provider.calls == [(("shared text",), "document"), (("shared text",), "query")]


class ShortEmbedding(RecordingEmbedding):
    def embed(self, texts: Sequence[str], task: EmbeddingTask = "document") -> EmbeddingResult:
        full = super().embed(texts, task)
        return full.model_copy(update={"vectors": full.vectors[:-1]})


def test_a_provider_that_returns_too_few_vectors_is_reported_not_indexed(
    cache: DiskCache,
) -> None:
    cached = CachedEmbeddingProvider(ShortEmbedding(), cache)

    with pytest.raises(PermanentProviderError, match="asked for 2 embeddings"):
        cached.embed(["alpha", "beta"])


def _warm(cache: DiskCache, texts: Sequence[str]) -> RecordingEmbedding:
    provider = RecordingEmbedding()
    CachedEmbeddingProvider(provider, cache).embed(texts)
    return provider


def _cache_only(cache: DiskCache) -> CachedEmbeddingProvider:
    return CachedEmbeddingProvider(CacheOnlyEmbeddingProvider("recording", "embed-a", 3), cache)


def test_a_cache_only_provider_serves_every_cached_text(cache: DiskCache) -> None:
    warmed = _warm(cache, ["alpha", "beta"]).embed(["alpha", "beta"])

    result = _cache_only(cache).embed(["beta", "alpha"])

    assert result.vectors == (warmed.vectors[1], warmed.vectors[0])


def test_a_cache_only_miss_names_the_count_and_the_key_and_writes_nothing(
    cache: DiskCache, tmp_path: Path
) -> None:
    _warm(cache, ["alpha"])
    before = sorted((tmp_path / "providers").rglob("*.json"))

    with pytest.raises(CacheMissError, match=r"1 text\(s\).*RAGEVAL_GEMINI_API_KEY"):
        _cache_only(cache).embed(["alpha", "unseen"])

    assert sorted((tmp_path / "providers").rglob("*.json")) == before


def test_a_cache_only_miss_is_permanent_so_indexing_never_retries_it(cache: DiskCache) -> None:
    with pytest.raises(PermanentProviderError):
        _cache_only(cache).embed(["unseen"])


def test_embedding_digest_is_the_key_the_cached_provider_writes(
    cache: DiskCache, tmp_path: Path
) -> None:
    _warm(cache, ["alpha"])

    document = embedding_digest("recording", "embed-a", 3, "alpha", "document")
    query = embedding_digest("recording", "embed-a", 3, "alpha", "query")

    assert cache.get("embedding", document) is not None
    assert cache.get("embedding", query) is None
    assert document != query
