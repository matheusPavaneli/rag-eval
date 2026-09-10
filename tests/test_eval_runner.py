import pytest

from conftest_retrieval import StubRetriever, scored
from rageval.eval.golden import GoldenQuestion, Support
from rageval.eval.runner import run_eval
from rageval.ingest.chunking import ChunkConfig
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

    report = run_eval([_question("q1")], retriever, CONFIG, dimension=768, network_calls=3)

    assert report.corpus_version == "ab"
    assert report.embedding_model == "fake-embed"
    assert report.dimension == 768
    assert report.chunk_size == 800
    assert report.overlap == 120
    assert report.k == 5
    assert report.question_count == 1
    assert report.network_calls == 3


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
