import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})

_BYTE_ORDER_MARK = "﻿"
_HEADING = re.compile(r"^#\s+(.+?)\s*$")


class DocumentReadError(Exception):
    pass


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source_path: str
    title: str
    text: str
    byte_size: int


def normalise(text: str) -> str:
    return text.removeprefix(_BYTE_ORDER_MARK).replace("\r\n", "\n").replace("\r", "\n")


def document_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_document(path: Path, root: Path) -> Document:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DocumentReadError(f"cannot read {path}") from error

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DocumentReadError(f"{path} is not valid UTF-8") from error

    text = normalise(decoded)
    return Document(
        id=document_id(text),
        source_path=path.relative_to(root).as_posix(),
        title=_title(text, path),
        text=text,
        byte_size=len(raw),
    )


def load_documents(root: Path) -> list[Document]:
    if not root.is_dir():
        raise DocumentReadError(f"{root} is not a directory")

    documents: dict[str, Document] = {}
    for path in _supported_paths(root):
        document = load_document(path, root)
        documents.setdefault(document.id, document)
    return list(documents.values())


def _supported_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is not None:
            return match.group(1)
    return path.stem
