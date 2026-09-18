from pathlib import Path

import pytest

from conftest_retrieval import StubRetriever, scored
from rageval.eval.golden import GoldenQuestion, Support
from rageval.eval.runner import EvalReport, ReportMismatchError, compare, run_eval
from rageval.ingest.chunking import ChunkConfig
from rageval.retrieval.search import Bm25Config, DenseConfig, FullTextConfig, HybridConfig
from rageval.retrieval.store import ScoredChunk

CONFIG = ChunkConfig(chunk_size=800, overlap=120)


def _question(identifier: str, start: int = 100, end: int = 200) -> GoldenQuestion:
    return GoldenQuestion(
        id=identifier,
        question=f"question {identifier}",
        answer="an answer",
        supports=(Support(source_path="a.md", start_char=start, end_char=end),),
    )


def _answers(**pairs: list[ScoredChunk]) -> dict[str, list[ScoredChunk]]:
    return {f"question {identifier}": chunks for identifier, chunks in pairs.items()}


def test_the_report_names_the_configuration_that_produced_it() -> None:
    retriever = StubRetriever(_answers(q1=[scored("a.md", 100, 200)]), top_k=5, corpus_version="ab")

    report = run_eval([_question("q1")], retriever, CONFIG, dimension=768, network_calls=lambda: 3)

    assert report.corpus_version == "ab"
    assert report.embedding_model == "fake-embed"
    assert report.dimension == 768
    assert report.chunk_size == 800
    assert report.overlap == 120
    assert report.k == 5
    assert report.question_count == 1
    assert report.network_calls == 3


def test_the_network_count_is_read_after_the_run_not_before_it() -> None:
    calls = 0

    class CountingRetriever(StubRetriever):
        def retrieve(self, question: str, top_k: int | None = None) -> tuple[ScoredChunk, ...]:
            nonlocal calls
            calls += 1
            return super().retrieve(question, top_k)

    retriever = CountingRetriever(
        _answers(q1=[scored("a.md", 100, 200)], q2=[scored("a.md", 100, 200)]), top_k=5
    )

    report = run_eval(
        [_question("q1"), _question("q2")],
        retriever,
        CONFIG,
        dimension=768,
        network_calls=lambda: calls,
    )

    assert report.network_calls == 2


def test_both_metrics_are_the_mean_over_the_questions() -> None:
    answers = _answers(
        q1=[scored("a.md", 100, 200)],
        q2=[scored("b.md", 0, 10), scored("a.md", 150, 250)],
        q3=[scored("b.md", 0, 10)],
    )
    questions = [_question("q1"), _question("q2"), _question("q3")]

    report = run_eval(questions, StubRetriever(answers, top_k=5), CONFIG, dimension=3)

    assert report.context_recall_at_k == pytest.approx(2 / 3)
    assert report.mrr_at_k == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_each_question_keeps_its_own_result_and_what_was_retrieved() -> None:
    answers = _answers(q1=[scored("b.md", 0, 10), scored("a.md", 100, 200)])

    report = run_eval([_question("q1")], StubRetriever(answers, top_k=5), CONFIG, dimension=3)

    assert report.results[0].id == "q1"
    assert report.results[0].context_recall == 1.0
    assert report.results[0].reciprocal_rank == 0.5
    assert report.results[0].retrieved_paths == ("b.md", "a.md")


def test_a_question_nothing_relevant_was_found_for_scores_zero() -> None:
    report = run_eval(
        [_question("q1")],
        StubRetriever(_answers(q1=[scored("b.md", 0, 10)]), top_k=5),
        CONFIG,
        dimension=3,
    )

    assert report.context_recall_at_k == 0.0
    assert report.mrr_at_k == 0.0


def test_the_table_row_carries_the_date_the_configuration_and_both_numbers() -> None:
    report = run_eval(
        [_question("q1")],
        StubRetriever(_answers(q1=[scored("a.md", 100, 200)]), top_k=5, corpus_version="cafe"),
        CONFIG,
        dimension=768,
    )

    row = report.table_row()

    assert row.startswith(f"| {report.ran_at.date().isoformat()} |")
    assert "fake-embed, 768d, chunk 800/120, k=5" in row
    assert "corpus `cafe`" in row
    assert row.endswith("| 1.000 | 1.000 |")


BASELINE_REPORT = Path(__file__).parent.parent / Path(
    "evals/reports/20260916T154348Z-26b03ce9a1c2c1d4.json"
)


def _report(hits: dict[str, bool], corpus_version: str = "cafe", top_k: int = 5) -> EvalReport:
    answers = {
        f"question {identifier}": [scored("a.md", 100, 200)] if hit else []
        for identifier, hit in hits.items()
    }
    return run_eval(
        [_question(identifier) for identifier in hits],
        StubRetriever(answers, top_k=top_k, corpus_version=corpus_version),
        CONFIG,
        dimension=768,
    )


def _row(config: FullTextConfig | Bm25Config | HybridConfig) -> str:
    report = run_eval([_question("q1")], StubRetriever({}, config=config), CONFIG, dimension=768)
    assert report.retrieval == config
    return report.table_row()


def test_the_report_records_the_retrieval_config_and_the_row_names_it() -> None:
    bm25 = Bm25Config(k1=1.2, b=0.75)

    assert "| fulltext: Postgres full-text (ts_rank_cd, english), chunk 800/120" in _row(
        FullTextConfig()
    )
    assert "| bm25: BM25 (k1=1.2, b=0.75) over Postgres english lexemes, chunk" in _row(bm25)
    assert "fake-embed" not in _row(bm25)
    assert (
        "| hybrid (RRF k=60, 20 candidates each): fake-embed, 768d + "
        "BM25 (k1=1.2, b=0.75) over Postgres english lexemes, chunk"
    ) in _row(HybridConfig(rrf_k=60, candidates=20, lexical=bm25))
    assert "768d + Postgres full-text (ts_rank_cd, english)" in _row(
        HybridConfig(rrf_k=60, candidates=20, lexical=FullTextConfig())
    )


def test_a_hybrid_report_round_trips_with_the_ranker_it_fused() -> None:
    config = HybridConfig(rrf_k=60, candidates=20, lexical=Bm25Config(k1=1.2, b=0.75))
    report = run_eval([_question("q1")], StubRetriever({}, config=config), CONFIG, dimension=768)

    assert EvalReport.model_validate_json(report.model_dump_json()).retrieval == config


def test_the_committed_f3_report_still_loads_and_reads_as_dense() -> None:
    report = EvalReport.model_validate_json(BASELINE_REPORT.read_text(encoding="utf-8"))

    assert report.retrieval == DenseConfig()
    assert report.question_count == 30


def test_compare_lists_misses_that_became_hits_and_hits_that_became_misses() -> None:
    baseline = _report({"q1": True, "q2": False, "q3": True, "q4": False})
    candidate = _report({"q1": True, "q2": True, "q3": False, "q4": False})

    flips = compare(baseline, candidate)

    assert flips.gained == ("q2",)
    assert flips.lost == ("q3",)


def test_compare_ignores_a_hit_that_only_moved_rank() -> None:
    baseline = _report({"q1": True})
    moved = run_eval(
        [_question("q1")],
        StubRetriever(
            _answers(q1=[scored("b.md", 0, 10), scored("a.md", 100, 200)]), corpus_version="cafe"
        ),
        CONFIG,
        dimension=768,
    )

    flips = compare(baseline, moved)

    assert (flips.gained, flips.lost) == ((), ())


@pytest.mark.parametrize(
    ("candidate", "named"),
    [
        ({"corpus_version": "other"}, "corpus_version differs"),
        ({"top_k": 3}, "k differs"),
        ({"extra": True}, "question set differs"),
    ],
)
def test_compare_refuses_reports_that_do_not_measure_the_same_thing(
    candidate: dict[str, object], named: str
) -> None:
    hits = {"q1": True, "q9": True} if "extra" in candidate else {"q1": True}
    other = _report(
        hits,
        corpus_version=str(candidate.get("corpus_version", "cafe")),
        top_k=int(str(candidate.get("top_k", 5))),
    )

    with pytest.raises(ReportMismatchError, match=named):
        compare(_report({"q1": True}), other)
