from datetime import UTC, datetime
from pathlib import Path

import pytest

from rageval.answer.citations import ResolvedCitation, UnresolvedCitation
from rageval.eval.answers import AnswerReport, AnswerResult
from rageval.eval.golden import GoldenQuestion, Support, golden_digest, load_golden_set
from rageval.eval.runner import EvalReport, QuestionResult
from rageval.ingest.chunking import ChunkConfig
from rageval.ingest.corpus import corpus_version
from rageval.ingest.documents import Document, document_id, load_documents
from rageval.retrieval.search import DenseConfig, RerankConfig
from rageval.site import (
    Line,
    Mark,
    SiteExportError,
    build_site,
    configuration_label,
    load_report,
    segment,
)
from rageval.site.__main__ import ANSWER_BASELINE, GATED, RECORDED

ROOT = Path(__file__).resolve().parent.parent

TEXT = (
    "# Title\n\nThe \U0001d518nicode line has a span.\nIt crosses\ninto here.\n``text\ncode\n``\n"
)


def mark(identifier: str, start: int, end: int, kind: str = "cited") -> Mark:
    return Mark.model_validate(
        {
            "id": identifier,
            "kind": kind,
            "source_path": "doc.md",
            "start_char": start,
            "end_char": end,
        }
    )


def marked_positions(lines: tuple[Line, ...], identifier: str) -> list[int]:
    positions: list[int] = []
    offset = 0
    for line in lines:
        for piece in line.segments:
            if identifier in piece.marks:
                positions.extend(range(offset, offset + len(piece.text)))
            offset += len(piece.text)
        offset += 1
    return positions


def expected_positions(text: str, start: int, end: int) -> list[int]:
    return [index for index in range(start, end) if text[index] != "\n"]


def test_the_lines_reproduce_the_document_exactly() -> None:
    lines = segment(TEXT, [mark("c1", 12, 20), mark("e1", 15, 40)])

    rebuilt = "\n".join("".join(piece.text for piece in line.segments) for line in lines)

    assert rebuilt == TEXT


def test_a_span_after_an_astral_character_marks_exactly_its_code_points() -> None:
    start = TEXT.index("has a span")
    end = start + len("has a span")

    lines = segment(TEXT, [mark("c1", start, end)])

    assert marked_positions(lines, "c1") == expected_positions(TEXT, start, end)
    marked = "".join(piece.text for line in lines for piece in line.segments if "c1" in piece.marks)
    assert marked == "has a span"


def test_overlapping_marks_split_into_segments_that_name_both_where_they_overlap() -> None:
    lines = segment(TEXT, [mark("c1", 9, 16), mark("e1", 13, 22, "expected")])

    marks = [(piece.text, piece.marks) for piece in lines[2].segments]

    assert marks == [
        ("The ", ("c1",)),
        ("\U0001d518ni", ("c1", "e1")),
        ("code l", ("e1",)),
        ("ine has a span.", ()),
    ]


def test_a_mark_crossing_a_line_break_is_split_per_line_and_anchored_once() -> None:
    start = TEXT.index("span.")
    end = TEXT.index("into") + len("into")

    lines = segment(TEXT, [mark("c1", start, end)])

    assert marked_positions(lines, "c1") == expected_positions(TEXT, start, end)
    anchors = [piece.anchors for line in lines for piece in line.segments if piece.anchors]
    assert anchors == [("c1",)]
    assert lines[2].segments[-1].anchors == ("c1",)


def test_a_mark_starting_on_a_line_break_anchors_on_the_next_line() -> None:
    start = TEXT.index("\nIt crosses")

    lines = segment(TEXT, [mark("c1", start, start + 3)])

    assert lines[3].segments[0].anchors == ("c1",)
    assert lines[3].segments[0].text == "It"


def test_lines_are_classified_from_their_raw_text() -> None:
    kinds = [line.kind for line in segment(TEXT, [])]

    assert kinds == ["heading", "blank", "text", "text", "text", "fence", "code", "fence", "blank"]


def test_a_mark_outside_the_document_is_refused() -> None:
    with pytest.raises(SiteExportError, match=r"c1 \[5, 999\) is outside doc.md"):
        segment(TEXT, [mark("c1", 5, 999)])


def test_configurations_are_labelled_by_what_they_are() -> None:
    rerank = RerankConfig(
        model="m", revision="r", candidates=20, first_stage=DenseConfig()
    ).model_dump(mode="json")

    assert configuration_label(DenseConfig().model_dump(mode="json")) == "Dense"
    assert configuration_label(rerank) == "Rerank over Dense"


def document(text: str = TEXT) -> Document:
    return Document(
        id=document_id(text), source_path="doc.md", title="Title", text=text, byte_size=len(text)
    )


def golden() -> tuple[GoldenQuestion, ...]:
    return (
        GoldenQuestion(
            id="q01",
            question="Where is the span?",
            answer="Here.",
            supports=(Support(source_path="doc.md", start_char=13, end_char=31),),
        ),
    )


def answer_report(
    version: str, digest: str | None, citations: tuple[ResolvedCitation | UnresolvedCitation, ...]
) -> AnswerReport:
    return AnswerReport(
        ran_at=datetime(2026, 9, 18, tzinfo=UTC),
        rageval_version="0.1.0",
        corpus_version=version,
        golden_set_digest=digest,
        embedding_model="e",
        chat_chain=("groq/m",),
        prompt_version="p",
        k=5,
        question_count=1,
        retrieved_count=1,
        citation_hit_rate=1.0,
        citation_hit_rate_retrieved=1.0,
        citation_count=len(citations),
        resolution_rate=0.5,
        mean_citation_chars=10.0,
        mean_cited_chunk_chars=100.0,
        parse_failures=0,
        providers={"groq/m": 1},
        network_calls=0,
        results=(
            AnswerResult(
                id="q01",
                question="Where is the span?",
                context_recall=1.0,
                answer="Here.",
                citations=citations,
                cited_chunk_chars=(100,),
                parse_error=None,
                raw_response=None,
                provider="groq",
                model="m",
                citation_hit=True,
            ),
        ),
    )


def retrieval_report(version: str, digest: str | None, ids: tuple[str, ...]) -> EvalReport:
    return EvalReport(
        ran_at=datetime(2026, 9, 18, tzinfo=UTC),
        rageval_version="0.1.0",
        corpus_version=version,
        golden_set_digest=digest,
        embedding_model="e",
        dimension=768,
        chunk_size=1000,
        overlap=150,
        k=5,
        question_count=len(ids),
        context_recall_at_k=1.0,
        mrr_at_k=1.0,
        network_calls=0,
        results=tuple(
            QuestionResult(
                id=identifier,
                question="Where is the span?",
                context_recall=1.0,
                reciprocal_rank=1.0,
                retrieved_paths=("doc.md",),
            )
            for identifier in ids
        ),
    )


RESOLVED = ResolvedCitation(
    chunk=1, quote="has a span", source_path="doc.md", start_char=27, end_char=37
)
UNRESOLVED = UnresolvedCitation(chunk=2, quote="not there", reason="not_in_chunk")


def test_an_answer_report_of_another_corpus_is_refused_naming_both_versions() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())

    with pytest.raises(SiteExportError, match=rf"is corpus 0000000000000000, .* corpus {version}"):
        build_site(
            docs,
            golden(),
            ("a.json", answer_report("0000000000000000", golden_digest(golden()), (RESOLVED,))),
            [],
        )


def test_an_answer_report_scored_against_another_golden_set_is_refused() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())

    with pytest.raises(SiteExportError, match=r"a\.json: golden set differs"):
        build_site(docs, golden(), ("a.json", answer_report(version, "f" * 64, (RESOLVED,))), [])


def test_an_answer_report_pinned_to_no_golden_set_is_refused() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())

    with pytest.raises(SiteExportError, match="records no golden-set digest"):
        build_site(docs, golden(), ("a.json", answer_report(version, None, (RESOLVED,))), [])


def test_an_undigested_report_of_other_questions_is_refused() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())
    answers = answer_report(version, golden_digest(golden()), (RESOLVED,))

    with pytest.raises(SiteExportError, match=r"r.json answers questions \['q02'\]"):
        build_site(
            docs,
            golden(),
            ("a.json", answers),
            [("r.json", retrieval_report(version, None, ("q02",)), False)],
        )


def test_a_gated_report_must_record_its_golden_set() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())
    answers = answer_report(version, golden_digest(golden()), (RESOLVED,))

    with pytest.raises(SiteExportError, match=r"marked gated but records no golden-set digest"):
        build_site(
            docs,
            golden(),
            ("a.json", answers),
            [("r.json", retrieval_report(version, None, ("q01",)), True)],
        )


def test_an_undigested_report_of_the_same_questions_is_exported_as_not_gated() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())
    answers = answer_report(version, golden_digest(golden()), (RESOLVED,))

    site = build_site(
        docs,
        golden(),
        ("a.json", answers),
        [("r.json", retrieval_report(version, None, ("q01",)), False)],
    )

    assert [(row.label, row.gated) for row in site.payload.configurations] == [("Dense", False)]


def test_an_unresolved_citation_keeps_its_reason_and_marks_nothing() -> None:
    docs = [document()]
    version = corpus_version(docs, ChunkConfig())
    answers = answer_report(version, golden_digest(golden()), (UNRESOLVED, RESOLVED))

    page = build_site(docs, golden(), ("a.json", answers), []).pages[0]

    assert page.citations[0] == UNRESOLVED
    assert [(item.id, item.kind) for item in page.documents[0].marks] == [
        ("c2", "cited"),
        ("e1", "expected"),
    ]


def test_the_committed_baseline_marks_every_resolved_citation_exactly() -> None:
    documents = load_documents(ROOT / "documents")
    questions = load_golden_set(ROOT / "evals" / "golden-set.jsonl", documents)
    by_path = {item.source_path: item.text for item in documents}
    retrieval = [
        (path.name, load_report(ROOT / path, EvalReport), gated)
        for paths, gated in ((GATED, True), (RECORDED, False))
        for path in paths
    ]

    site = build_site(
        documents,
        questions,
        (ANSWER_BASELINE.name, load_report(ROOT / ANSWER_BASELINE, AnswerReport)),
        retrieval,
    )

    exact = 0
    unresolved: list[tuple[str, str]] = []
    for page in site.pages:
        for number, citation in enumerate(page.citations, start=1):
            if isinstance(citation, UnresolvedCitation):
                unresolved.append((page.id, citation.reason))
                continue
            view = next(v for v in page.documents if v.source_path == citation.source_path)
            text = by_path[citation.source_path]
            assert marked_positions(view.lines, f"c{number}") == expected_positions(
                text, citation.start_char, citation.end_char
            )
            exact += 1

    assert len(site.pages) == 30
    assert exact == 51
    assert unresolved == [("q15", "not_in_chunk")]
    assert len(site.payload.configurations) == 7
    assert [row.gated for row in site.payload.configurations].count(True) == 3
