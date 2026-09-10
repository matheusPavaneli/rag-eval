import json
from pathlib import Path

import pytest

from rageval.ingest.__main__ import main
from rageval.ingest.chunking import ChunkConfig
from rageval.ingest.corpus import CorpusManifest, EmptyCorpusError, ingest_corpus

CONFIG = ChunkConfig(chunk_size=200, overlap=40)


def corpus(root: Path, **documents: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in documents.items():
        (root / f"{name}.md").write_text(text, encoding="utf-8", newline="")
    return root


def body(marker: str) -> str:
    return f"# {marker}\n\n" + "\n\n".join(
        f"{marker} paragraph {index} " + "w " * 40 for index in range(6)
    )


def test_the_same_directory_ingests_to_the_same_version(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"), beta=body("Beta"))
    out = tmp_path / "data"

    first = ingest_corpus(documents, out, CONFIG)
    second = ingest_corpus(documents, out, CONFIG)

    assert first.corpus_version == second.corpus_version
    assert (first.document_count, first.chunk_count) == (second.document_count, second.chunk_count)


def test_touching_a_file_does_not_change_the_version(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))
    out = tmp_path / "data"
    before = ingest_corpus(documents, out, CONFIG)

    (documents / "alpha.md").touch()

    assert ingest_corpus(documents, out, CONFIG).corpus_version == before.corpus_version


def test_adding_a_document_changes_the_version(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))
    out = tmp_path / "data"
    before = ingest_corpus(documents, out, CONFIG)

    corpus(documents, beta=body("Beta"))

    assert ingest_corpus(documents, out, CONFIG).corpus_version != before.corpus_version


def test_changing_only_the_chunk_config_changes_the_version(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))
    out = tmp_path / "data"

    before = ingest_corpus(documents, out, CONFIG)
    after = ingest_corpus(documents, out, ChunkConfig(chunk_size=400, overlap=40))

    assert before.corpus_version != after.corpus_version


def test_the_written_corpus_matches_the_manifest(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"), beta=body("Beta"))
    out = tmp_path / "data"

    manifest = ingest_corpus(documents, out, CONFIG)

    target = out / manifest.corpus_version
    chunk_lines = (target / "chunks.jsonl").read_text("utf-8").splitlines()
    document_lines = (target / "documents.jsonl").read_text("utf-8").splitlines()
    chunks = [json.loads(line) for line in chunk_lines]
    records = [json.loads(line) for line in document_lines]
    written = CorpusManifest.model_validate_json((target / "manifest.json").read_text("utf-8"))

    assert len(chunks) == manifest.chunk_count
    assert len(records) == manifest.document_count
    assert written.corpus_version == manifest.corpus_version
    assert written.chunk_config == CONFIG
    assert {chunk["document_id"] for chunk in chunks} == {record["id"] for record in records}


def test_re_ingesting_rewrites_rather_than_appends(tmp_path: Path) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))
    out = tmp_path / "data"

    first = ingest_corpus(documents, out, CONFIG)
    ingest_corpus(documents, out, CONFIG)

    lines = (out / first.corpus_version / "chunks.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == first.chunk_count


def test_a_directory_without_supported_documents_fails(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.pdf").write_bytes(b"%PDF-1.4")

    with pytest.raises(EmptyCorpusError, match="documents"):
        ingest_corpus(documents, tmp_path / "data", CONFIG)


def test_cli_reports_the_corpus_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))

    assert main([str(documents)]) == 0
    assert "1 documents" in capsys.readouterr().out


def test_cli_exits_non_zero_naming_the_missing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "absent")]) == 1
    assert "absent" in capsys.readouterr().err


def test_cli_rejects_an_overlap_that_does_not_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = corpus(tmp_path / "documents", alpha=body("Alpha"))

    assert main([str(documents), "--chunk-size", "100", "--overlap", "100"]) == 1
    assert "overlap" in capsys.readouterr().err
