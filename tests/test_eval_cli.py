from pathlib import Path

import pytest

from rageval.eval.__main__ import build_parser as eval_parser
from rageval.retrieval.__main__ import build_parser as retrieval_parser

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
