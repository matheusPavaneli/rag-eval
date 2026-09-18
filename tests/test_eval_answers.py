from collections.abc import Mapping
from pathlib import Path

import pytest

from conftest_retrieval import StubRetriever, scored
from rageval.answer.generate import PROMPT_VERSION
from rageval.eval.answers import AnswerReport, run_answer_eval
from rageval.eval.golden import GoldenQuestion, Support
from rageval.eval.runner import EvalReport
from rageval.providers.base import ChatResult
from rageval.retrieval.store import ScoredChunk

REPORTS = Path(__file__).resolve().parent.parent / "evals" / "reports"


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
    return run_answer_eval(QUESTIONS, RETRIEVER, CHAT, network_calls=lambda: 7)


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
    assert report.chat_model == "stub-model"
    assert report.providers == {"stub/stub-model": 3}
    assert (report.k, report.corpus_version, report.network_calls) == (5, "cafef00d", 7)
    assert "`" + PROMPT_VERSION + "`" in report.table_row()


def test_the_table_row_names_the_models_that_answered_not_the_configured_primary(
    report: AnswerReport,
) -> None:
    mixed = report.model_copy(
        update={
            "chat_model": "primary-model",
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
