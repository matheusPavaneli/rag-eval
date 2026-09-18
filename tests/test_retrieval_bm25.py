import math

import pytest

from conftest_retrieval import terms
from rageval.retrieval.bm25 import Bm25Index
from rageval.retrieval.store import RetrievalError


def _score(index: Bm25Index, query: list[str], chunk_id: str) -> float:
    return next(chunk.score for chunk in index.search(query, 10) if chunk.chunk_id == chunk_id)


def test_the_score_is_the_bm25_formula_computed_by_hand() -> None:
    corpus = [
        terms("a", protocol=2, class_=1),
        terms("b", protocol=1, enumer=3),
        terms("c", enumer=1),
    ]
    index = Bm25Index(corpus, k1=1.2, b=0.75)

    average_length = (3 + 4 + 1) / 3
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    norm = 1 - 0.75 + 0.75 * 3 / average_length
    expected = idf * 2 * (1.2 + 1) / (2 + 1.2 * norm)

    assert _score(index, ["protocol"], "a") == pytest.approx(expected)


def test_a_common_term_repeated_cannot_outrank_a_rare_term_matched_once() -> None:
    corpus = [
        terms("common", pep=2),
        terms("rare", length=1, other=1),
        *(terms(f"filler-{index}", pep=1, other=1) for index in range(20)),
    ]
    index = Bm25Index(corpus, k1=1.2, b=0.75)

    ranked = index.search(["pep", "length"], limit=3)

    assert ranked[0].chunk_id == "rare"
    assert index.idf("pep") < index.idf("length")


def test_term_frequency_saturates_below_the_k1_ceiling() -> None:
    corpus = [terms("once", term=1, pad=9), terms("tenfold", term=10), terms("other", pad=10)]
    index = Bm25Index(corpus, k1=1.2, b=0.0)

    once = _score(index, ["term"], "once")
    tenfold = _score(index, ["term"], "tenfold")

    assert once < tenfold < 2 * once
    assert tenfold < index.idf("term") * (1.2 + 1)


def test_the_shorter_chunk_wins_on_equal_counts_only_when_length_is_normalised() -> None:
    corpus = [terms("short", term=1), terms("long", term=1, pad=20), terms("other", pad=5)]

    normalised = Bm25Index(corpus, k1=1.2, b=0.75)
    flat = Bm25Index(corpus, k1=1.2, b=0.0)

    assert _score(normalised, ["term"], "short") > _score(normalised, ["term"], "long")
    assert _score(flat, ["term"], "short") == pytest.approx(_score(flat, ["term"], "long"))


def test_a_query_with_no_indexed_term_returns_nothing() -> None:
    index = Bm25Index([terms("a", protocol=1)], k1=1.2, b=0.75)

    assert index.search(["enumer"], limit=5) == ()
    assert index.search([], limit=5) == ()


def test_ties_break_by_chunk_id_and_at_most_limit_chunks_come_back() -> None:
    corpus = [terms(name, term=1) for name in ("c", "a", "b")] + [terms("d", pad=1)]
    index = Bm25Index(corpus, k1=1.2, b=0.75)

    assert [chunk.chunk_id for chunk in index.search(["term"], limit=2)] == ["a", "b"]


def test_a_repeated_query_term_counts_once() -> None:
    index = Bm25Index([terms("a", term=1), terms("b", pad=1)], k1=1.2, b=0.75)

    assert _score(index, ["term", "term"], "a") == pytest.approx(_score(index, ["term"], "a"))


def test_an_empty_corpus_searches_to_nothing() -> None:
    assert Bm25Index([], k1=1.2, b=0.75).search(["term"], limit=5) == ()


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.2, -0.1), (1.2, 1.1)])
def test_parameters_outside_their_range_are_refused(k1: float, b: float) -> None:
    with pytest.raises(RetrievalError, match="BM25"):
        Bm25Index([], k1=k1, b=b)
