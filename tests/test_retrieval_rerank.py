import pytest

from conftest_retrieval import FakeCrossEncoder, StubRetriever, scored
from rageval.config import get_settings
from rageval.retrieval.rerank import OnnxCrossEncoder
from rageval.retrieval.search import (
    Bm25Config,
    HybridConfig,
    RerankConfig,
    RerankRetriever,
)
from rageval.retrieval.store import RetrievalError, ScoredChunk

QUESTION = "what does yield from do?"


def _chunk(text: str, score: float = 0.5) -> ScoredChunk:
    return scored(f"{text}.md", 0, 10, score).model_copy(update={"text": text})


def _first_stage(*texts: str, top_k: int = 5) -> StubRetriever:
    return StubRetriever({QUESTION: [_chunk(text) for text in texts]}, top_k=top_k)


def test_candidates_are_ordered_by_the_encoder_and_carry_its_score() -> None:
    encoder = FakeCrossEncoder({"a": 0.1, "b": 2.5, "c": 1.0})
    retriever = RerankRetriever(_first_stage("a", "b", "c"), encoder, candidates=20)

    ranked = retriever.retrieve(QUESTION)

    assert [chunk.text for chunk in ranked] == ["b", "c", "a"]
    assert [chunk.score for chunk in ranked] == [2.5, 1.0, 0.1]


def test_the_first_stage_is_asked_for_the_candidate_depth_and_k_are_returned() -> None:
    first_stage = _first_stage(*"abcdefgh", top_k=3)
    encoder = FakeCrossEncoder({text: float(ord(text)) for text in "abcdefgh"})
    retriever = RerankRetriever(first_stage, encoder, candidates=8)

    assert [chunk.text for chunk in retriever.retrieve(QUESTION)] == ["h", "g", "f"]
    assert [chunk.text for chunk in retriever.retrieve(QUESTION, top_k=1)] == ["h"]
    assert first_stage.requests == [(QUESTION, 8), (QUESTION, 8)]
    assert retriever.top_k == 3


def test_equal_scores_keep_the_first_stage_order() -> None:
    retriever = RerankRetriever(_first_stage("a", "b", "c"), FakeCrossEncoder({}), candidates=20)

    assert [chunk.text for chunk in retriever.retrieve(QUESTION)] == ["a", "b", "c"]


def test_the_encoder_scores_every_candidate_once_and_is_skipped_when_there_are_none() -> None:
    encoder = FakeCrossEncoder({})
    retriever = RerankRetriever(_first_stage("a", "b"), encoder, candidates=20)

    retriever.retrieve(QUESTION)
    assert retriever.retrieve("a question nothing matches") == ()
    assert encoder.calls == [(QUESTION, ("a", "b"))]


def test_a_score_count_that_does_not_match_the_candidates_is_refused() -> None:
    retriever = RerankRetriever(
        _first_stage("a", "b"), FakeCrossEncoder({}, extra=1), candidates=20
    )

    with pytest.raises(RetrievalError, match="3 scores for 2 candidates"):
        retriever.retrieve(QUESTION)


def test_the_config_nests_the_first_stage_and_names_the_model_revision() -> None:
    hybrid = HybridConfig(rrf_k=60, candidates=20, lexical=Bm25Config(k1=1.2, b=0.75))
    first_stage = StubRetriever({}, config=hybrid)

    config = RerankRetriever(first_stage, FakeCrossEncoder({}), candidates=20).config

    assert config == RerankConfig(
        model="fake/cross-encoder", revision="0123456789abcdef", candidates=20, first_stage=hybrid
    )


def test_a_reranker_refuses_to_rerank_another_reranker() -> None:
    inner = RerankRetriever(_first_stage("a"), FakeCrossEncoder({}), candidates=20)

    with pytest.raises(RetrievalError, match="cannot rerank another reranker"):
        RerankRetriever(inner, FakeCrossEncoder({}), candidates=20)


@pytest.mark.integration
def test_the_pinned_model_scores_an_answering_passage_above_an_unrelated_one() -> None:
    settings = get_settings()
    encoder = OnnxCrossEncoder(
        settings.reranker_model, settings.reranker_revision, settings.cache_dir
    )

    answer, unrelated = encoder.score(
        QUESTION,
        [
            "The yield from expression delegates part of a generator's operations "
            "to a subgenerator, passing values and exceptions through to it.",
            "Enumerations are created using the class syntax and have members with names.",
        ],
    )

    assert answer > unrelated
