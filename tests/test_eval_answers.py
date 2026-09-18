from collections.abc import Mapping
from pathlib import Path

import pytest

from conftest_retrieval import StubRetriever, scored
from rageval.answer.generate import PROMPT_VERSION
from rageval.eval.answers import AnswerReport, answer_drift, run_answer_eval
from rageval.eval.golden import GoldenQuestion, Support, golden_digest
from rageval.eval.runner import EvalReport, ReportMismatchError
from rageval.providers.base import ChatResult
from rageval.retrieval.store import ScoredChunk

REPORTS = Path(__file__).resolve().parent.parent / "evals" / "reports"
CHAIN = ("stub/stub-model", "backup/backup-model")


class ChatByQuestion:
    def __init__(self, replies: Mapping[str, str]) -> None:
        self._replies = replies

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return "stub-model"

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        question = prompt.rsplit("Question: ", 1)[1]
        return ChatResult(provider="stub", model="stub-model", text=self._replies[question])


def passage(source_path: str, start: int, text: str) -> ScoredChunk:
    return scored(source_path, start, start + len(text)).model_copy(update={"text": text})


def golden(identifier: str, source_path: str, start: int, end: int) -> GoldenQuestion:
    return GoldenQuestion(
        id=identifier,
        question=f"question {identifier}",
        answer="a",
        supports=(Support(source_path=source_path, start_char=start, end_char=end),),
    )


# q1: gold span retrieved, cited correctly           -> hit
# q2: gold span retrieved, cites another sentence    -> miss; one citation unresolved
# q3: gold span not retrieved, reply is not JSON     -> miss; a parse failure
LINE = passage("a.md", 1000, "Lines are 79 characters. Tabs are not used.")
QUESTIONS = (
    golden("q1", "a.md", 1000, 1024),
    golden("q2", "a.md", 1000, 1024),
    golden("q3", "b.md", 0, 10),
)
RETRIEVER = StubRetriever(
    {"question q1": [LINE], "question q2": [LINE], "question q3": [LINE]}, top_k=5
)
CHAT = ChatByQuestion(
    {
        "question q1": '{"answer": "79", "citations": [{"chunk": 1, "quote": "79 characters"}]}',
        "question q2": (
            '{"answer": "no tabs", "citations": ['
            '{"chunk": 1, "quote": "Tabs are not used."}, {"chunk": 2, "quote": "x"}]}'
        ),
        "question q3": "I cannot tell.",
    }
)


@pytest.fixture
def report() -> AnswerReport:
    return run_answer_eval(QUESTIONS, RETRIEVER, CHAT, CHAIN, network_calls=lambda: 7)


def test_the_hit_rate_counts_questions_whose_citation_lands_on_the_gold_span(
    report: AnswerReport,
) -> None:
    assert [result.citation_hit for result in report.results] == [True, False, False]
    assert report.citation_hit_rate == pytest.approx(1 / 3)


def test_the_retrieved_hit_rate_only_counts_questions_whose_gold_span_was_retrieved(
    report: AnswerReport,
) -> None:
    assert report.retrieved_count == 2
    assert report.citation_hit_rate_retrieved == 0.5


def test_resolution_and_parse_failures_are_counted_over_every_citation(
    report: AnswerReport,
) -> None:
    assert report.citation_count == 3
    assert report.resolution_rate == pytest.approx(2 / 3)
    assert report.parse_failures == 1
    assert report.results[2].raw_response == "I cannot tell."


def test_citation_length_is_compared_with_the_length_of_the_chunk_it_came_from(
    report: AnswerReport,
) -> None:
    assert report.mean_citation_chars == (len("79 characters") + len("Tabs are not used.")) / 2
    assert report.mean_cited_chunk_chars == len(LINE.text)


def test_the_report_names_what_produced_the_number(report: AnswerReport) -> None:
    assert report.prompt_version == PROMPT_VERSION
    assert report.chat_chain == CHAIN
    assert report.golden_set_digest == golden_digest(QUESTIONS)
    assert report.providers == {"stub/stub-model": 3}
    assert (report.k, report.corpus_version, report.network_calls) == (5, "cafef00d", 7)
    assert "`" + PROMPT_VERSION + "`" in report.table_row()


def test_the_table_row_names_the_models_that_answered_not_the_configured_primary(
    report: AnswerReport,
) -> None:
    mixed = report.model_copy(
        update={
            "chat_chain": ("gemini/primary-model", "groq/fallback-model"),
            "providers": {"gemini/primary-model": 1, "groq/fallback-model": 29},
        }
    )

    row = mixed.table_row()

    assert "groq/fallback-model (29), gemini/primary-model (1)" in row
    assert "| primary-model" not in row


def test_an_answer_report_round_trips_through_json(report: AnswerReport) -> None:
    assert AnswerReport.model_validate_json(report.model_dump_json()) == report


def test_existing_retrieval_reports_still_load_as_retrieval_reports() -> None:
    retrieval_reports = [path for path in REPORTS.glob("*.json") if "-answer-" not in path.name]

    assert retrieval_reports
    for path in retrieval_reports:
        EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_a_report_has_no_answer_drift_from_itself(report: AnswerReport) -> None:
    assert not answer_drift(report, report).any


def test_answer_drift_names_the_question_and_the_field_that_moved(report: AnswerReport) -> None:
    first = report.results[0]
    moved_citation = first.citations[0].model_copy(update={"end_char": 1024})
    moved = report.model_copy(
        update={
            "results": (
                first.model_copy(update={"citations": (moved_citation,), "provider": "other"}),
                *report.results[1:],
            ),
            "citation_hit_rate": 0.0,
        }
    )

    drift = answer_drift(report, moved)

    assert [(question.id, question.fields) for question in drift.changed] == [
        ("q1", ("citations", "provider"))
    ]
    assert [
        (aggregate.field, aggregate.before, aggregate.after) for aggregate in drift.aggregates
    ] == [("citation_hit_rate", report.citation_hit_rate, 0.0)]
    assert drift.any


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_version", "another"),
        ("chat_chain", ("stub/another-model",)),
        ("golden_set_digest", "0" * 64),
        ("golden_set_digest", None),
        ("corpus_version", "another"),
        ("k", 3),
    ],
)
def test_answer_drift_refuses_a_report_that_is_not_comparable(
    report: AnswerReport, field: str, value: object
) -> None:
    with pytest.raises(ReportMismatchError):
        answer_drift(report, report.model_copy(update={field: value}))


def test_answer_drift_refuses_a_different_question_set(report: AnswerReport) -> None:
    with pytest.raises(ReportMismatchError, match="question set differs"):
        answer_drift(report, report.model_copy(update={"results": report.results[:2]}))
