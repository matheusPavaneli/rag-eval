from pathlib import Path

import pytest

from rageval.ingest.documents import DocumentReadError, load_document, load_documents

BOM = "﻿"


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_identical_content_under_two_names_shares_one_id(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "# Title\n\nSame body.\n")
    write(tmp_path, "nested/b.md", "# Title\n\nSame body.\n")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source_path == "a.md"


def test_one_changed_character_changes_the_id(tmp_path: Path) -> None:
    first = load_document(write(tmp_path, "a.md", "body\n"), tmp_path)
    second = load_document(write(tmp_path, "b.md", "bodY\n"), tmp_path)

    assert first.id != second.id


def test_line_endings_do_not_change_the_id(tmp_path: Path) -> None:
    lf = load_document(write(tmp_path, "lf.md", "one\ntwo\n"), tmp_path)
    crlf = load_document(write(tmp_path, "crlf.md", "one\r\ntwo\r\n"), tmp_path)

    assert lf.id == crlf.id
    assert "\r" not in crlf.text


def test_byte_order_mark_does_not_enter_the_text(tmp_path: Path) -> None:
    path = write(tmp_path, "bom.md", BOM + "# Title\n\nbody\n")

    document = load_document(path, tmp_path)

    assert not document.text.startswith(BOM)
    assert document.text.startswith("# Title")
    assert document.title == "Title"


def test_title_falls_back_to_the_filename_stem(tmp_path: Path) -> None:
    document = load_document(write(tmp_path, "notes.txt", "no heading here\n"), tmp_path)

    assert document.title == "notes"


def test_unsupported_suffixes_are_not_loaded(tmp_path: Path) -> None:
    write(tmp_path, "keep.md", "kept\n")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")

    documents = load_documents(tmp_path)

    assert [document.source_path for document in documents] == ["keep.md"]


def test_a_file_that_is_not_utf8_fails_naming_its_path(tmp_path: Path) -> None:
    path = tmp_path / "latin1.md"
    path.write_bytes(b"caf\xe9\n")

    with pytest.raises(DocumentReadError, match=r"latin1\.md"):
        load_documents(tmp_path)


def test_a_missing_directory_fails_naming_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "absent"

    with pytest.raises(DocumentReadError, match="absent"):
        load_documents(missing)
