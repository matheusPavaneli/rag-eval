from collections.abc import Sequence

import pytest

from conftest_retrieval import FakeEmbeddingProvider, FakeStore, scored, terms
from rageval.providers.base import EmbeddingResult, EmbeddingTask, PermanentProviderError
from rageval.retrieval.search import (
    Bm25Config,
    Bm25Retriever,
    DenseConfig,
    FullTextConfig,
    FullTextRetriever,
    HybridConfig,
    HybridRetriever,
    LexicalMode,
    RetrievalMode,
    Retriever,
    build_retriever,
    reciprocal_rank_fusion,
)
from rageval.retrieval.store import RetrievalError, ScoredChunk

BM25 = Bm25Config(k1=1.2, b=0.75)


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


def test_a_top_k_of_zero_asks_for_nothing_rather_than_the_default() -> None:
    store = FakeStore(results=[scored("a.md", 0, 10)])
    retriever = Retriever(store, FakeEmbeddingProvider(), "corpus-1", top_k=5)

    retriever.retrieve("a question", top_k=0)

    assert [call[2] for call in store.searches] == [0]


def test_a_provider_that_returns_no_vector_is_an_error_not_an_empty_result() -> None:
    class EmptyProvider(FakeEmbeddingProvider):
        def embed(self, texts: object = (), task: EmbeddingTask = "document") -> EmbeddingResult:
            return EmbeddingResult(provider="fake", model="fake-embed", dimension=3, vectors=())

    retriever = Retriever(FakeStore(), EmptyProvider(), "v1", top_k=5)

    with pytest.raises(PermanentProviderError, match="no vector"):
        retriever.retrieve("a question")


def _ids(chunks: Sequence[ScoredChunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]


def _hybrid(store: FakeStore, lexical: FullTextRetriever | Bm25Retriever) -> HybridRetriever:
    return HybridRetriever(
        Retriever(store, FakeEmbeddingProvider(), "v1", top_k=5),
        lexical,
        top_k=5,
        rrf_k=60,
        candidates=20,
    )


def test_a_chunk_ranked_high_in_both_lists_outranks_one_first_in_only_one() -> None:
    shared = scored("shared.md", 0, 10)
    dense_only = scored("dense.md", 0, 10)
    lexical_only = scored("lexical.md", 0, 10)

    fused = reciprocal_rank_fusion([[dense_only, shared], [lexical_only, shared]], rrf_k=60)

    assert _ids(fused)[0] == shared.chunk_id


def test_a_chunk_present_in_both_lists_appears_once() -> None:
    shared = scored("shared.md", 0, 10)

    fused = reciprocal_rank_fusion([[shared], [shared]], rrf_k=60)

    assert _ids(fused) == [shared.chunk_id]
    assert fused[0].score == pytest.approx(2 / 61)


def test_fusing_a_single_list_preserves_its_order() -> None:
    ranked = [scored("a.md", 0, 10), scored("b.md", 0, 10), scored("c.md", 0, 10)]

    fused = reciprocal_rank_fusion([ranked, []], rrf_k=60)

    assert _ids(fused) == _ids(ranked)


def test_fused_ties_are_broken_the_same_way_every_run() -> None:
    first = scored("z.md", 0, 10)
    second = scored("a.md", 0, 10)

    one = reciprocal_rank_fusion([[first], [second]], rrf_k=60)
    other = reciprocal_rank_fusion([[second], [first]], rrf_k=60)

    assert _ids(one) == _ids(other) == [second.chunk_id, first.chunk_id]


def test_the_hybrid_asks_each_retriever_for_the_candidate_depth_and_returns_top_k() -> None:
    dense = [scored(f"d{index}.md", 0, 10) for index in range(30)]
    lexical = [scored(f"l{index}.md", 0, 10) for index in range(30)]
    store = FakeStore(results=dense, lexical_results=lexical)

    result = _hybrid(store, FullTextRetriever(store, "v1", top_k=5)).retrieve("a question")

    assert [call[2] for call in store.searches] == [20]
    assert [call[2] for call in store.lexical_searches] == [20]
    assert len(result) == 5


def test_a_hybrid_with_no_lexical_match_ranks_exactly_like_dense() -> None:
    dense = [scored("a.md", 0, 10), scored("b.md", 0, 10), scored("c.md", 0, 10)]
    store = FakeStore(results=dense)

    hybrid = _hybrid(store, FullTextRetriever(store, "v1", top_k=5))

    assert _ids(hybrid.retrieve("a question")) == _ids(dense)


def test_the_hybrid_fuses_whichever_lexical_retriever_it_is_given() -> None:
    store = FakeStore(terms=[terms("rare", protocol=1)])
    bm25 = Bm25Retriever(store, "v1", top_k=5, config=BM25)

    hybrid = _hybrid(store, bm25)

    assert _ids(hybrid.retrieve("protocol")) == ["rare"]
    assert store.lexical_searches == []
    assert hybrid.config == HybridConfig(rrf_k=60, candidates=20, lexical=BM25)
    assert _hybrid(store, FullTextRetriever(store, "v1", 5)).config.lexical == FullTextConfig()


def test_the_fulltext_retriever_never_embeds_and_names_no_embedding_model() -> None:
    store = FakeStore(lexical_results=[scored("a.md", 0, 10)])
    fulltext = FullTextRetriever(store, "corpus-1", top_k=5)

    result = fulltext.retrieve("what is a protocol class?", top_k=3)

    assert store.searches == []
    assert store.lexical_searches == [("corpus-1", "what is a protocol class?", 3)]
    assert _ids(result) == [scored("a.md", 0, 10).chunk_id]
    assert fulltext.embedding_model is None


def test_bm25_reads_the_corpus_terms_once_and_never_embeds() -> None:
    store = FakeStore(terms=[terms("a", protocol=2), terms("b", enumeration=1)])
    bm25 = Bm25Retriever(store, "corpus-1", top_k=5, config=BM25)

    first = bm25.retrieve("protocol")
    second = bm25.retrieve("enumeration")

    assert (_ids(first), _ids(second)) == (["a"], ["b"])
    assert store.term_reads == ["corpus-1"]
    assert store.searches == []
    assert bm25.embedding_model is None


def test_bm25_normalises_the_question_through_the_store() -> None:
    store = FakeStore(
        terms=[terms("a", protocol=1)], query_terms={"What is a Protocol?": ["protocol"]}
    )

    result = Bm25Retriever(store, "v1", top_k=5, config=BM25).retrieve("What is a Protocol?")

    assert _ids(result) == ["a"]


def test_each_retriever_reports_its_own_configuration() -> None:
    store = FakeStore()
    dense = Retriever(store, FakeEmbeddingProvider(), "v1", top_k=5)

    assert dense.config == DenseConfig()
    assert FullTextRetriever(store, "v1", 5).config == FullTextConfig()
    assert Bm25Retriever(store, "v1", 5, BM25).config == BM25


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_a_mode_that_embeds_refuses_to_be_built_without_a_provider(
    mode: RetrievalMode,
) -> None:
    with pytest.raises(RetrievalError, match=f"{mode} retrieval needs an embedding provider"):
        build_retriever(mode, FakeStore(), None, "v1", 5, BM25)


@pytest.mark.parametrize("mode", ["fulltext", "bm25"])
def test_lexical_modes_are_built_without_a_provider(mode: RetrievalMode) -> None:
    assert build_retriever(mode, FakeStore(), None, "v1", 5, BM25).config.mode == mode


@pytest.mark.parametrize("fuse_with", ["fulltext", "bm25"])
def test_the_hybrid_is_built_over_the_requested_lexical_ranker(fuse_with: LexicalMode) -> None:
    retriever = build_retriever(
        "hybrid", FakeStore(), FakeEmbeddingProvider(), "v1", 5, BM25, fuse_with
    )

    assert isinstance(retriever.config, HybridConfig)
    assert retriever.config.lexical.mode == fuse_with
