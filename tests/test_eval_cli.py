from datetime import UTC, datetime
from pathlib import Path

import pytest

from rageval.eval.__main__ import _config_slug, _mode_slug, _report_answer_drift, _report_drift
from rageval.eval.__main__ import build_parser as eval_parser
from rageval.eval.__main__ import main as eval_main
from rageval.eval.answers import AggregateDrift, AnswerDrift, AnswerQuestionDrift, AnswerReport
from rageval.eval.runner import Drift, QuestionDrift
from rageval.retrieval.__main__ import build_parser as retrieval_parser
from rageval.retrieval.__main__ import main as retrieval_main
from rageval.retrieval.search import Bm25Config, DenseConfig, HybridConfig, RerankConfig

DEFAULT = Path("evals/golden-set.jsonl")


def test_a_golden_set_given_on_the_command_line_arrives_as_a_path() -> None:
    arguments = eval_parser(DEFAULT, top_k=5).parse_args(["--golden-set", "evals/other.jsonl"])

    assert arguments.golden_set == Path("evals/other.jsonl")


def test_the_default_golden_set_is_the_configured_one() -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args([]).golden_set == DEFAULT


@pytest.mark.parametrize("value", ["0", "-1"])
def test_the_eval_refuses_a_k_below_one_instead_of_asking_postgres(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        eval_parser(DEFAULT, top_k=5).parse_args(["-k", value])

    assert error.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1"])
def test_the_retriever_refuses_a_k_below_one_instead_of_asking_postgres(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        retrieval_parser(top_k=5).parse_args(["-k", value])

    assert error.value.code == 2


def test_the_eval_mode_defaults_to_dense_so_the_baseline_command_is_unchanged() -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args([]).mode == "dense"


@pytest.mark.parametrize("mode", ["dense", "fulltext", "bm25", "hybrid"])
def test_every_retrieval_mode_is_accepted(mode: str) -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args(["--mode", mode]).mode == mode
    assert retrieval_parser(top_k=5).parse_args(["--mode", mode]).mode == mode


def test_the_old_lexical_name_is_refused_rather_than_silently_meaning_fulltext() -> None:
    with pytest.raises(SystemExit) as error:
        eval_parser(DEFAULT, top_k=5).parse_args(["--mode", "lexical"])

    assert error.value.code == 2


def test_the_hybrid_fuses_with_bm25_unless_told_otherwise() -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args([]).fuse_with == "bm25"
    assert retrieval_parser(top_k=5).parse_args(["--fuse-with", "fulltext"]).fuse_with == (
        "fulltext"
    )


def test_a_baseline_given_on_the_command_line_arrives_as_a_path() -> None:
    arguments = eval_parser(DEFAULT, top_k=5).parse_args(["--baseline", "evals/reports/x.json"])

    assert arguments.baseline == Path("evals/reports/x.json")


def test_reranking_is_off_unless_asked_for_and_combines_with_any_mode() -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args([]).rerank is False
    arguments = eval_parser(DEFAULT, top_k=5).parse_args(["--mode", "hybrid", "--rerank"])

    assert (arguments.mode, arguments.rerank) == ("hybrid", True)


def test_a_rerank_report_file_is_named_after_its_first_stage() -> None:
    hybrid = HybridConfig(rrf_k=60, candidates=20, lexical=Bm25Config(k1=1.2, b=0.75))

    def rerank(first_stage: DenseConfig | HybridConfig) -> RerankConfig:
        return RerankConfig(model="m", revision="r", candidates=20, first_stage=first_stage)

    assert _config_slug(rerank(DenseConfig())) == "rerank-dense"
    assert _config_slug(rerank(hybrid)) == "rerank-hybrid-bm25"
    assert _config_slug(hybrid) == "hybrid-bm25"
    assert _config_slug(Bm25Config(k1=1.2, b=0.75)) == "bm25"


def test_answering_is_off_unless_asked_for() -> None:
    assert eval_parser(DEFAULT, top_k=5).parse_args([]).answer is False
    assert retrieval_parser(top_k=5).parse_args([]).answer is False


def test_an_answer_report_file_is_named_after_its_retriever() -> None:
    report = AnswerReport(
        ran_at=datetime(2026, 9, 18, tzinfo=UTC),
        rageval_version="0",
        corpus_version="c",
        embedding_model="e",
        chat_chain=("gemini/m", "groq/n"),
        prompt_version="p",
        k=5,
        question_count=0,
        retrieved_count=0,
        citation_hit_rate=0.0,
        citation_hit_rate_retrieved=0.0,
        citation_count=0,
        resolution_rate=0.0,
        mean_citation_chars=0.0,
        mean_cited_chunk_chars=0.0,
        parse_failures=0,
        providers={},
        network_calls=0,
        results=(),
    )

    assert _mode_slug(report) == "answer-dense"


def test_answering_refuses_a_reranker_before_touching_anything() -> None:
    with pytest.raises(SystemExit) as refused:
        eval_main(["--answer", "--rerank"])

    assert refused.value.code == 2


def test_an_answer_run_can_be_gated_against_a_baseline() -> None:
    arguments = eval_parser(DEFAULT, top_k=5).parse_args(
        ["--answer", "--baseline", "evals/reports/x.json", "--fail-on-change"]
    )

    assert (arguments.answer, arguments.fail_on_change) == (True, True)


@pytest.mark.parametrize("flag", ["--record-answers", "--replay-answers"])
def test_recording_or_replaying_answers_needs_an_answer_run(flag: str) -> None:
    with pytest.raises(SystemExit) as refused:
        eval_main([flag, "answers.jsonl"])

    assert refused.value.code == 2


def test_answering_one_question_needs_the_question() -> None:
    with pytest.raises(SystemExit) as refused:
        retrieval_main(["--answer"])

    assert refused.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [["--fail-on-change"], ["--fail-on-change", "--answer"]],
)
def test_the_gate_needs_a_baseline_before_touching_anything(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as refused:
        eval_main(arguments)

    assert refused.value.code == 2


def test_indexing_lexically_refuses_a_query() -> None:
    with pytest.raises(SystemExit) as refused:
        retrieval_main(["--lexical-only", "--query", "what is an enumeration?"])

    assert refused.value.code == 2


def test_no_drift_passes_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    gate = Drift(changed=(), context_recall_at_k=(0.6, 0.6), mrr_at_k=(0.465, 0.465))

    assert _report_drift(gate, Path("base.json")) == 0
    assert "no drift from base.json" in capsys.readouterr().out


def test_drift_fails_the_gate_naming_each_question(capsys: pytest.CaptureFixture[str]) -> None:
    gate = Drift(
        changed=(QuestionDrift(id="q07", context_recall=(1.0, 0.0), reciprocal_rank=(0.5, 0.0)),),
        context_recall_at_k=(0.6, 0.567),
        mrr_at_k=(0.465, 0.448),
    )

    assert _report_drift(gate, Path("base.json")) == 1
    err = capsys.readouterr().err
    assert "q07  recall 1.000 -> 0.000, rr 0.500 -> 0.000" in err
    assert "0.6 -> 0.567" in err


def test_no_answer_drift_passes_the_gate(capsys: pytest.CaptureFixture[str]) -> None:
    assert _report_answer_drift(AnswerDrift(changed=(), aggregates=()), Path("base.json")) == 0
    assert "no drift from base.json" in capsys.readouterr().out


def test_answer_drift_fails_the_gate_naming_each_question_and_aggregate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = AnswerDrift(
        changed=(AnswerQuestionDrift(id="q04", fields=("citations", "citation_hit")),),
        aggregates=(AggregateDrift(field="citation_hit_rate", before=0.533, after=0.5),),
    )

    assert _report_answer_drift(gate, Path("base.json")) == 1
    err = capsys.readouterr().err
    assert "q04  citations, citation_hit" in err
    assert "citation_hit_rate 0.533 -> 0.5" in err
