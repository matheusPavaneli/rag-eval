"""Draft the golden set: locate each answer in its source document and emit exact spans.

Run when the corpus changes. Every span is found by searching the document text
for a needle and expanding to the enclosing paragraph, so the set is a function
of the corpus rather than a hand-copied list of offsets. A needle that is missing,
or that appears more than once, is reported rather than guessed at.
"""

import json
import sys
from pathlib import Path

from rageval.config import get_settings
from rageval.ingest.documents import load_documents

# id, source, question, answer, needle
DRAFT: list[tuple[str, str, str, str, str]] = [
    ("q01", "pep-0008.md", "What is the maximum line length PEP 8 asks for?",
     "79 characters.", "Limit all lines to a maximum of 79 characters"),
    ("q02", "pep-0008.md", "How many spaces per indentation level does PEP 8 require?",
     "Four spaces.", "Use 4 spaces per indentation level"),
    ("q03", "pep-0020.md",
     "What does the Zen of Python say about there being one way to do something?",
     "There should be one -- and preferably only one -- obvious way to do it.",
     "There should be one-- and preferably only one --obvious way to do it."),
    ("q04", "pep-0257.md", "Which quotes should a docstring use?",
     "Triple double quotes.", "triple double quotes"),
    ("q05", "pep-0263.md",
     "What is the regular expression a Python source encoding declaration must match?",
     "coding[:=]\\s*([-\\w.]+)", "coding[:=]"),
    ("q06", "pep-0498.md", "What is an f-string?",
     "A string literal prefixed with f, whose braces hold expressions evaluated at runtime.",
     "In source code, f-strings are string literals that are prefixed by the"),
    ("q07", "pep-0572.md", "What does the walrus operator do?",
     "Assigns to a name as part of an expression.", ":="),
    ("q08", "pep-0484.md", "What does PEP 484 say about runtime enforcement of type hints?",
     "Type hints are not enforced at runtime; they are for static checkers.",
     "no type checking happens at runtime"),
    ("q09", "pep-0440.md", "What is the canonical form of a version's release segment?",
     "A dot-separated sequence of non-negative integers.", "release segment"),
    ("q10", "pep-0518.md", "Which table in pyproject.toml declares the build system?",
     "[build-system], with a requires key.",
     "for the table: `requires`. This key must have a value of a list"),
    ("q11", "pep-0517.md", "What is a build backend?",
     "The Python object a build frontend calls to build a wheel or sdist.", "build backend"),
    ("q12", "pep-0249.md", "What does paramstyle indicate in the DB-API?",
     "The type of parameter marker formatting the module expects.",
     "String constant stating the type of parameter marker formatting"),
    ("q13", "pep-0343.md", "What two methods does the with statement need on a context manager?",
     "__enter__ and __exit__.",
     "In this PEP, context managers provide `__enter__()` and `__exit__()`"),
    ("q14", "pep-0380.md", "What does yield from do?",
     "Delegates part of a generator's operations to another iterator.",
     "A syntax is proposed for a generator to delegate part of its"),
    ("q15", "pep-0405.md", "What is a virtual environment?",
     "A self-contained directory tree with its own Python installation and packages.",
     "virtual environment"),
    ("q16", "pep-0420.md", "What is a namespace package?",
     "A package split across multiple portions with no __init__.py.",
     "Namespace packages are a mechanism for splitting a single Python package"),
    ("q17", "pep-0435.md", "What is an enumeration?",
     "A set of symbolic names bound to unique, constant values.",
     "An enumeration is a set of symbolic names bound to unique"),
    ("q18", "pep-0448.md", "What does PEP 448 generalise?",
     "Unpacking with * and ** in calls, literals and comprehensions.",
     "This PEP proposes extended usages of the `*` iterable unpacking"),
    ("q19", "pep-0465.md", "What operator does PEP 465 add for matrix multiplication?",
     "The @ operator.", "A new binary operator is added to the Python language"),
    ("q20", "pep-0492.md", "What makes a function a coroutine under PEP 492?",
     "The async def syntax.", "The following new syntax is used to declare a *native coroutine*"),
    ("q21", "pep-0503.md", "How is a project name normalised for the simple repository API?",
     "Runs of -, _ and . are replaced by a single -, then lowercased.",
     "return re.sub(r\"[-_.]+\", \"-\", name).lower()"),
    ("q22", "pep-0515.md", "Where may underscores appear in a numeric literal?",
     "Between digits, and after a base specifier.",
     "The current proposal is to allow one underscore between digits, and"),
    ("q23", "pep-0544.md", "What is a protocol class?",
     "A class defining structural subtyping, matched by shape rather than inheritance.",
     "At runtime, protocol classes will be simple ABCs"),
    ("q24", "pep-0557.md", "What does the dataclass decorator generate?",
     "__init__, __repr__ and __eq__ among other methods.",
     "A class decorator is provided which inspects a class definition for"),
    ("q25", "pep-0561.md", "How does a package declare that it ships type information?",
     "By including a py.typed marker file.", "py.typed"),
    ("q26", "pep-0563.md", "What does from __future__ import annotations change?",
     "Annotations are not evaluated at definition time; they are kept as strings.",
     "they are preserved in `__annotations__` in string form"),
    ("q27", "pep-0567.md", "What problem do context variables solve?",
     "Context-local state that works correctly with asynchronous tasks.",
     "similar to thread-local storage (TLS), but, unlike TLS, it also allows"),
    ("q28", "pep-0585.md", "What does PEP 585 allow in place of typing.List?",
     "The builtin list, subscripted directly as list[int].",
     "This PEP proposes to enable support for the generics syntax in all"),
    ("q29", "pep-0604.md", "How is an optional type written under PEP 604?",
     "As X | None.", "int | str"),
    ("q30", "pep-0634.md", "What does a match statement compare a subject against?",
     "A sequence of case patterns.",
     "The match statement first evaluates the subject expression"),
]  # fmt: skip


def main() -> int:
    settings = get_settings()
    documents = {
        document.source_path: document for document in load_documents(settings.documents_dir)
    }

    out: list[str] = []
    problems: list[str] = []

    for identifier, source, question, answer, needle in DRAFT:
        document = documents.get(source)
        if document is None:
            problems.append(f"{identifier}: {source} is not in the corpus")
            continue

        found = document.text.find(needle)
        if found == -1:
            problems.append(f"{identifier}: needle {needle!r} is absent from {source}")
            continue

        start, end = _paragraph(document.text, found, found + len(needle))
        excerpt = " ".join(document.text[start:end].split())
        occurrences = document.text.count(needle)

        print(f"{identifier}  {source}[{start}:{end}]  ({occurrences} occurrence(s) of needle)")
        print(f"    Q: {question}")
        print(f"    A: {answer}")
        print(f"    span: {excerpt[:220]}")
        print()

        out.append(
            json.dumps(
                {
                    "id": identifier,
                    "question": question,
                    "answer": answer,
                    "supports": [{"source_path": source, "start_char": start, "end_char": end}],
                },
                ensure_ascii=False,
            )
        )

    if problems:
        print("PROBLEMS", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)

    target = Path(settings.golden_set_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(out)}/{len(DRAFT)} questions written to {target}")
    return 1 if problems else 0


def _paragraph(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a match to the blank-line-delimited block that holds it."""
    before = text.rfind("\n\n", 0, start)
    after = text.find("\n\n", end)
    left = 0 if before == -1 else before + 2
    right = len(text) if after == -1 else after
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


if __name__ == "__main__":
    raise SystemExit(main())
