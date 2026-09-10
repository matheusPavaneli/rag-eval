import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rageval.eval.golden import GoldenSetError, load_golden_set
from rageval.ingest.documents import Document

DOCUMENTS = [
    Document(id="doc-a", source_path="a.md", title="A", text="x" * 500, byte_size=500),
    Document(id="doc-b", source_path="b.md", title="B", text="y" * 80, byte_size=80),
]


def _write(path: Path, records: Sequence[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )
    return path


def _question(
    identifier: str = "q1", source_path: str = "a.md", start: int = 0, end: int = 100
) -> dict[str, object]:
    return {
        "id": identifier,
        "question": "what?",
        "answer": "that",
        "supports": [{"source_path": source_path, "start_char": start, "end_char": end}],
    }


def test_a_well_formed_set_loads_with_its_spans(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [_question("q1"), _question("q2", "b.md", 0, 80)])

    questions = load_golden_set(path, DOCUMENTS)

    assert [question.id for question in questions] == ["q1", "q2"]
    assert questions[1].supports[0].source_path == "b.md"
    assert questions[1].supports[0].end_char == 80


def test_a_duplicate_question_id_names_the_id(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [_question("q1"), _question("q1")])

    with pytest.raises(GoldenSetError, match="'q1' is used twice"):
        load_golden_set(path, DOCUMENTS)


def test_a_span_past_the_end_of_its_document_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [_question("q1", "b.md", 0, 81)])

    with pytest.raises(GoldenSetError, match="which holds 80"):
        load_golden_set(path, DOCUMENTS)


def test_a_source_path_outside_the_corpus_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [_question("q1", "nowhere.md")])

    with pytest.raises(GoldenSetError, match="not in the corpus"):
        load_golden_set(path, DOCUMENTS)


def test_an_empty_span_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [_question("q1", "a.md", 100, 100)])

    with pytest.raises(GoldenSetError, match="not a golden question"):
        load_golden_set(path, DOCUMENTS)


def test_a_question_with_no_support_is_refused(tmp_path: Path) -> None:
    record = _question("q1")
    record["supports"] = []
    path = _write(tmp_path / "g.jsonl", [record])

    with pytest.raises(GoldenSetError, match="not a golden question"):
        load_golden_set(path, DOCUMENTS)


def test_a_line_that_is_not_json_names_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(_question()) + "\nnot json\n", encoding="utf-8", newline="\n")

    with pytest.raises(GoldenSetError, match="line 2 is not JSON"):
        load_golden_set(path, DOCUMENTS)


def test_an_empty_file_is_refused_rather_than_measuring_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path / "g.jsonl", [])

    with pytest.raises(GoldenSetError, match="holds no question"):
        load_golden_set(path, DOCUMENTS)


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="cannot read the golden set"):
        load_golden_set(tmp_path / "absent.jsonl", DOCUMENTS)
