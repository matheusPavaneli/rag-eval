"""Build the measured corpus: fetch a pinned set of PEPs and convert them to markdown.

Run once. The output under documents/ is committed, because data/ is gitignored
and a published number is only reproducible if its input is in the repository.
"""

import hashlib
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx

SOURCE_URL = "https://raw.githubusercontent.com/python/peps/main/peps/pep-{number:04d}.rst"

PEPS: tuple[int, ...] = (
    8, 20, 202, 234, 238, 249, 255, 257, 263, 273,
    282, 289, 302, 308, 309, 318, 328, 333, 342, 343,
    372, 380, 393, 405, 420, 428, 435, 440, 443, 448,
    465, 484, 492, 498, 503, 508, 515, 517, 518, 525,
    544, 557, 561, 563, 567, 572, 585, 604, 634, 695,
)  # fmt: skip

HEADER_FIELDS = ("Title", "Author", "Status", "Type", "Created", "Python-Version")

_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+_])\1{2,}\s*$")
_FIELD = re.compile(r"^([A-Za-z][A-Za-z-]*):\s*(.*)$")
_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([a-z-]+)::\s*(.*)$")
_COMMENT = re.compile(r"^\s*\.\.(\s|$)")
_ROLE = re.compile(r":([a-z:-]+):`([^`]+)`", re.S)
# The trailing lookahead matters: without it, `a` and `__b__` lets the reference
# regex close on the SECOND backtick and swallow the leading __ of the next name.
# The backtick in the class covers the same trap around a literal underscore.
_LINK = re.compile(r"`([^`<]+?)\s*<[^`>]+>`_+(?![\w`])", re.S)
_ANONYMOUS = re.compile(r"`([^`]+)`_+(?![\w`])", re.S)
_LITERAL = re.compile(r"``([^`]+)``", re.S)
_FOOTNOTE = re.compile(r"\[(\d+|#[\w-]*)\]_")
_LITERAL_BLOCK = re.compile(r"::$", re.M)
_TARGET = re.compile(r"^(.*?)\s*<([^<>]+)>$")


def main(argv: Sequence[str] | None = None) -> int:
    target = Path(argv[0]) if argv else Path("documents")
    target.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for number in PEPS:
            url = SOURCE_URL.format(number=number)
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                print(f"failed to fetch PEP {number}: {error}", file=sys.stderr)
                return 1

            raw = response.content
            title, body = convert(raw.decode("utf-8"), number)
            (target / f"pep-{number:04d}.md").write_text(body, encoding="utf-8", newline="\n")
            rows.append(
                f"| {number} | {title} | [{url.rsplit('/', 1)[-1]}]({url}) "
                f"| `{hashlib.sha256(raw).hexdigest()[:16]}` |"
            )
            print(f"pep-{number:04d}.md  {title}")

    sources = Path("docs/corpus-sources.md")
    sources.parent.mkdir(parents=True, exist_ok=True)
    sources.write_text(_sources(rows), encoding="utf-8", newline="\n")

    print(f"\n{len(PEPS)} documents in {target}, provenance in {sources}")
    return 0


def convert(text: str, number: int) -> tuple[str, str]:
    """Return the PEP title and its body as markdown."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    metadata, start = _header(lines)
    title = metadata.get("Title", f"PEP {number}")

    title = _inline(title)
    out = [f"# PEP {number} -- {title}", ""]
    out.extend(
        f"- **{field}:** {_inline(metadata[field])}" for field in HEADER_FIELDS if field in metadata
    )
    out.append("")
    out.extend(_body(lines[start:]))

    return title, _polish(_squeeze(out))


def _header(lines: Sequence[str]) -> tuple[dict[str, str], int]:
    """Read the RFC 2822 preamble every PEP opens with."""
    metadata: dict[str, str] = {}
    field = ""
    for index, line in enumerate(lines):
        if not line.strip():
            if metadata:
                return metadata, index + 1
            continue

        match = _FIELD.match(line)
        if match is not None:
            field = match.group(1)
            metadata[field] = match.group(2).strip()
        elif field and line.startswith((" ", "\t")):
            metadata[field] = f"{metadata[field]} {line.strip()}".strip()

    return metadata, len(lines)


def _body(lines: Sequence[str]) -> Iterator[str]:
    levels: dict[str, int] = {}
    skip_until: int | None = None
    index = 0

    while index < len(lines):
        line = lines[index]

        if skip_until is not None:
            if line.strip() and _indent(line) < skip_until:
                skip_until = None
            else:
                index += 1
                continue

        block = _DIRECTIVE.match(line)
        if block is not None:
            yield from _block(lines, index, block)
            skip_until = len(block.group(1)) + 1
            index += 1
            continue

        if _COMMENT.match(line):
            skip_until = _indent(line) + 1
            index += 1
            continue

        heading = _heading(lines, index, levels)
        if heading is not None:
            yield heading
            yield ""
            index += 2
            continue

        yield _inline(line)
        index += 1


def _block(lines: Sequence[str], index: int, match: re.Match[str]) -> Iterator[str]:
    """Render a directive as a fenced block or a blockquote, keeping its content."""
    name, argument = match.group(2), match.group(3).strip()
    body = list(_indented(lines, index + 1, len(match.group(1))))

    if name in {"code-block", "code", "sourcecode"}:
        yield ""
        yield f"```{argument or 'text'}"
        yield from body
        yield "```"
        yield ""
        return

    if name in {"note", "warning", "important", "caution"}:
        yield ""
        yield f"> **{name.capitalize()}**"
        yield from (f"> {_inline(entry)}" if entry.strip() else ">" for entry in body)
        yield ""
        return

    yield from (_inline(entry) for entry in body)


def _indented(lines: Sequence[str], start: int, outer: int) -> Iterator[str]:
    block: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            block.append("")
            continue
        if _indent(line) <= outer:
            break
        block.append(line)

    inner = min((_indent(line) for line in block if line), default=0)
    for line in block:
        yield line[inner:] if line else ""


def _heading(lines: Sequence[str], index: int, levels: dict[str, int]) -> str | None:
    """An RST section is a line of text underlined by punctuation at least as long."""
    if index + 1 >= len(lines):
        return None

    text = lines[index].strip()
    underline = lines[index + 1]
    if not text or _indent(lines[index]) or not _UNDERLINE.match(underline):
        return None
    if len(underline.strip()) < len(text):
        return None

    character = underline.strip()[0]
    level = levels.setdefault(character, len(levels) + 2)
    return f"{'#' * min(level, 6)} {_inline(text)}"


def _inline(line: str) -> str:
    line = _LITERAL.sub(r"`\1`", line)
    line = _LINK.sub(r"\1", line)
    line = _ANONYMOUS.sub(r"\1", line)
    line = _ROLE.sub(_role, line)
    return line.rstrip()


def _role(match: re.Match[str]) -> str:
    name, value = match.group(1), " ".join(match.group(2).split())

    target = _TARGET.match(value)
    if target is not None:
        text, value = target.group(1), target.group(2)
        if name not in {"pep", "rfc"}:
            return text

    if name == "pep":
        return f"PEP {value}"
    if name == "rfc":
        return f"RFC {value}"
    return f"`{value}`" if name in {"class", "func", "meth", "mod", "attr", "data"} else value


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _polish(text: str) -> str:
    """A second pass for markup a line-at-a-time pass cannot see."""
    text = _LITERAL.sub(r"`\1`", text)
    text = _LINK.sub(lambda match: " ".join(match.group(1).split()), text)
    text = _ROLE.sub(_role, text)
    text = _FOOTNOTE.sub(r"[\1]", text)
    return _LITERAL_BLOCK.sub(":", text)


def _squeeze(lines: Sequence[str]) -> str:
    out: list[str] = []
    for line in lines:
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def _sources(rows: Sequence[str]) -> str:
    return "\n".join(
        [
            "# Corpus sources",
            "",
            "This file lives outside `documents/` on purpose: ingest treats every",
            "markdown file under that directory as corpus content, and a table of",
            "hashes is not something the retrieval baseline should be searching.",
            "",
            f"{len(rows)} Python Enhancement Proposals, fetched from the `python/peps`",
            "repository and converted from reStructuredText to markdown by",
            "`scripts/fetch_peps.py`. The conversion touches structure only: headings,",
            "code blocks, admonitions and inline roles. No prose is rewritten.",
            "",
            "PEPs are placed in the public domain or under CC0-1.0 (PEP 1), so the",
            "corpus can be redistributed here without further condition.",
            "",
            "The digest is the sha256 of the source `.rst` as fetched, truncated to 16",
            "characters. It pins what was converted: upstream may edit a PEP, and this",
            "is how a later re-run is known to have produced a different corpus.",
            "",
            "| PEP | Title | Source | sha256 |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
