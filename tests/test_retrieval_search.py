import pytest

from conftest_retrieval import FakeEmbeddingProvider, FakeStore, scored
from rageval.providers.base import EmbeddingResult, EmbeddingTask, PermanentProviderError
from rageval.retrieval.search import Retriever


def test_the_question_is_embedded_as_a_query_not_as_a_document() -> None:
    provider = FakeEmbeddingProvider()
    retriever = Retriever(FakeStore(), provider, "v1", top_k=5)

    retriever.retrieve("what does PEP 8 say about line length?")

    assert provider.tasks == ["query"]
    assert provider.batches == [["what does PEP 8 say about line length?"]]


def test_the_store_ranking_is_returned_unchanged() -> None:
    ranked = [scored("a.md", 0, 10, score=0.9), scored("b.md", 5, 15, score=0.4)]
    retriever = Retriever(FakeStore(results=ranked), FakeEmbeddingProvider(), "v1", top_k=5)

    result = retriever.retrieve("a question")

    assert [chunk.source_path for chunk in result] == ["a.md", "b.md"]
    assert [chunk.score for chunk in result] == [0.9, 0.4]


def test_top_k_is_passed_to_the_store_and_can_be_overridden_per_call() -> None:
    store = FakeStore(results=[scored("a.md", 0, 10)])
    retriever = Retriever(store, FakeEmbeddingProvider(), "corpus-1", top_k=5)

    retriever.retrieve("a question")
    retriever.retrieve("another question", top_k=2)

    assert [call[2] for call in store.searches] == [5, 2]
    assert {call[0] for call in store.searches} == {"corpus-1"}


def test_a_provider_that_returns_no_vector_is_an_error_not_an_empty_result() -> None:
    class EmptyProvider(FakeEmbeddingProvider):
        def embed(self, texts: object = (), task: EmbeddingTask = "document") -> EmbeddingResult:
            return EmbeddingResult(provider="fake", model="fake-embed", dimension=3, vectors=())

    retriever = Retriever(FakeStore(), EmptyProvider(), "v1", top_k=5)

    with pytest.raises(PermanentProviderError, match="no vector"):
        retriever.retrieve("a question")
