import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from rageval.answer.citations import ResolutionResult, ResolvedCitation
from rageval.eval.answers import AnswerReport, AnswerResult
from rageval.eval.golden import GoldenQuestion, Support, golden_digest
from rageval.eval.runner import EvalReport, ReportMismatchError, check_golden_set
from rageval.ingest.chunking import ChunkConfig
from rageval.ingest.corpus import corpus_version
from rageval.ingest.documents import Document

type LineKind = Literal["heading", "fence", "code", "blank", "text"]
type MarkKind = Literal["cited", "expected"]

_FENCE = re.compile(r"^`{2,}[^`]*$")


class SiteExportError(Exception):
    pass


class Mark(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: MarkKind
    source_path: str
    start_char: int
    end_char: int


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    marks: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()


class Line(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: LineKind
    segments: tuple[Segment, ...]


class DocumentView(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    title: str
    length: int
    marks: tuple[Mark, ...]
    lines: tuple[Line, ...]


class QuestionPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    expected_answer: str
    answer: str | None
    parse_error: str | None
    provider: str
    model: str
    citation_hit: bool
    context_recall: float
    citations: tuple[ResolutionResult, ...]
    supports: tuple[Support, ...]
    documents: tuple[DocumentView, ...]


class QuestionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    citation_hit: bool
    retrieved: bool


class QuestionScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    context_recall: float
    reciprocal_rank: float


class ConfigurationRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    report: str
    gated: bool
    context_recall_at_k: float
    mrr_at_k: float
    results: tuple[QuestionScore, ...]


class SitePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_version: str
    golden_set_digest: str
    answer_report: str
    prompt_version: str
    chat_chain: tuple[str, ...]
    providers: Mapping[str, int]
    embedding_model: str | None
    k: int
    question_count: int
    document_count: int
    retrieved_count: int
    citation_hit_rate: float
    citation_hit_rate_retrieved: float
    citation_count: int
    resolved_count: int
    resolution_rate: float
    questions: tuple[QuestionSummary, ...]
    configurations: tuple[ConfigurationRow, ...]


class Site(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: SitePayload
    pages: tuple[QuestionPage, ...]


def segment(text: str, marks: Sequence[Mark]) -> tuple[Line, ...]:
    for mark in marks:
        if not 0 <= mark.start_char < mark.end_char <= len(text):
            raise SiteExportError(
                f"mark {mark.id} [{mark.start_char}, {mark.end_char}) is outside "
                f"{mark.source_path} ({len(text)} characters)"
            )

    anchored: set[str] = set()
    lines: list[Line] = []
    in_fence = False
    line_start = 0
    for raw in text.split("\n"):
        line_end = line_start + len(raw)
        kind, in_fence = _kind(raw, in_fence)
        cuts = {line_start, line_end}
        for mark in marks:
            cuts.update(
                point for point in (mark.start_char, mark.end_char) if line_start < point < line_end
            )
        ordered = sorted(cuts)
        segments: list[Segment] = []
        for start, end in pairwise(ordered):
            covering = tuple(
                mark.id for mark in marks if mark.start_char <= start and end <= mark.end_char
            )
            anchors = tuple(identifier for identifier in covering if identifier not in anchored)
            anchored.update(anchors)
            segments.append(Segment(text=text[start:end], marks=covering, anchors=anchors))
        lines.append(Line(kind=kind, segments=tuple(segments)))
        line_start = line_end + 1
    return tuple(lines)


def _kind(raw: str, in_fence: bool) -> tuple[LineKind, bool]:
    if _FENCE.match(raw):
        return "fence", not in_fence
    if in_fence:
        return "code", True
    if raw.strip() == "":
        return "blank", False
    if raw.startswith("#"):
        return "heading", False
    return "text", False


def configuration_label(retrieval: Mapping[str, Any]) -> str:
    mode = retrieval.get("mode")
    if mode == "dense":
        return "Dense"
    if mode == "fulltext":
        return "Full-text"
    if mode == "bm25":
        return "BM25"
    if mode == "hybrid":
        return f"Hybrid · dense + {configuration_label(retrieval['lexical'])}"
    if mode == "rerank":
        return f"Rerank over {configuration_label(retrieval['first_stage'])}"
    raise SiteExportError(f"no label for retrieval mode {mode!r}")


def build_site(
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    answer_report: tuple[str, AnswerReport],
    retrieval_reports: Sequence[tuple[str, EvalReport, bool]],
    chunk_config: ChunkConfig | None = None,
) -> Site:
    answer_name, answers = answer_report
    expected_version = corpus_version(documents, chunk_config or ChunkConfig())
    digest = golden_digest(questions)

    _check_corpus(answer_name, answers.corpus_version, expected_version)
    if answers.golden_set_digest is None:
        raise SiteExportError(
            f"{answer_name} records no golden-set digest; the page is built only from a "
            "report pinned to the golden set it was scored against"
        )
    _check_digest(answer_name, answers.golden_set_digest, digest)
    _check_questions(answer_name, [result.id for result in answers.results], questions)

    rows: list[ConfigurationRow] = []
    for name, report, gated in retrieval_reports:
        _check_corpus(name, report.corpus_version, expected_version)
        if report.k != answers.k:
            raise SiteExportError(f"{name} has k={report.k}, the answer report k={answers.k}")
        if report.golden_set_digest is not None:
            _check_digest(name, report.golden_set_digest, digest)
        elif gated:
            raise SiteExportError(f"{name} is marked gated but records no golden-set digest")
        _check_questions(name, [result.id for result in report.results], questions)
        rows.append(_row(name, report, gated))

    by_path = {document.source_path: document for document in documents}
    golden = {question.id: question for question in questions}
    pages = tuple(_page(result, golden[result.id], by_path) for result in answers.results)
    resolved = sum(
        isinstance(citation, ResolvedCitation)
        for result in answers.results
        for citation in result.citations
    )

    payload = SitePayload(
        corpus_version=expected_version,
        golden_set_digest=digest,
        answer_report=answer_name,
        prompt_version=answers.prompt_version,
        chat_chain=answers.chat_chain,
        providers=answers.providers,
        embedding_model=answers.embedding_model,
        k=answers.k,
        question_count=answers.question_count,
        document_count=len(documents),
        retrieved_count=answers.retrieved_count,
        citation_hit_rate=answers.citation_hit_rate,
        citation_hit_rate_retrieved=answers.citation_hit_rate_retrieved,
        citation_count=answers.citation_count,
        resolved_count=resolved,
        resolution_rate=answers.resolution_rate,
        questions=tuple(
            QuestionSummary(
                id=result.id,
                question=result.question,
                citation_hit=result.citation_hit,
                retrieved=result.context_recall > 0,
            )
            for result in answers.results
        ),
        configurations=tuple(rows),
    )
    return Site(payload=payload, pages=pages)


def _check_corpus(name: str, recorded: str, expected: str) -> None:
    if recorded != expected:
        raise SiteExportError(
            f"{name} is corpus {recorded}, but documents/ is corpus {expected}; "
            "re-run the eval or check out the documents it was scored on"
        )


def _check_digest(name: str, recorded: str, expected: str) -> None:
    try:
        check_golden_set(recorded, expected)
    except ReportMismatchError as error:
        raise SiteExportError(f"{name}: {error}") from error


def _check_questions(name: str, ids: Sequence[str], questions: Sequence[GoldenQuestion]) -> None:
    expected = [question.id for question in questions]
    if list(ids) != expected:
        raise SiteExportError(f"{name} answers questions {list(ids)}, the golden set {expected}")


def _row(name: str, report: EvalReport, gated: bool) -> ConfigurationRow:
    return ConfigurationRow(
        label=configuration_label(report.retrieval.model_dump(mode="json")),
        report=name,
        gated=gated,
        context_recall_at_k=report.context_recall_at_k,
        mrr_at_k=report.mrr_at_k,
        results=tuple(
            QuestionScore(
                id=result.id,
                context_recall=result.context_recall,
                reciprocal_rank=result.reciprocal_rank,
            )
            for result in report.results
        ),
    )


def _page(
    result: AnswerResult, question: GoldenQuestion, by_path: Mapping[str, Document]
) -> QuestionPage:
    if result.question != question.question:
        raise SiteExportError(f"{result.id} asks {result.question!r}, the golden set another")

    marks: list[Mark] = [
        Mark(
            id=f"c{number}",
            kind="cited",
            source_path=citation.source_path,
            start_char=citation.start_char,
            end_char=citation.end_char,
        )
        for number, citation in enumerate(result.citations, start=1)
        if isinstance(citation, ResolvedCitation)
    ]
    marks += [
        Mark(
            id=f"e{number}",
            kind="expected",
            source_path=support.source_path,
            start_char=support.start_char,
            end_char=support.end_char,
        )
        for number, support in enumerate(question.supports, start=1)
    ]

    paths = list(dict.fromkeys(mark.source_path for mark in marks))
    views: list[DocumentView] = []
    for path in paths:
        document = by_path.get(path)
        if document is None:
            raise SiteExportError(f"{result.id} marks {path}, which is not in documents/")
        own = tuple(mark for mark in marks if mark.source_path == path)
        views.append(
            DocumentView(
                source_path=path,
                title=document.title,
                length=len(document.text),
                marks=own,
                lines=segment(document.text, own),
            )
        )

    return QuestionPage(
        id=result.id,
        question=result.question,
        expected_answer=question.answer,
        answer=result.answer,
        parse_error=result.parse_error,
        provider=result.provider,
        model=result.model,
        citation_hit=result.citation_hit,
        context_recall=result.context_recall,
        citations=result.citations,
        supports=question.supports,
        documents=tuple(views),
    )


def load_report[T: BaseModel](path: Path, model: type[T]) -> T:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise SiteExportError(f"cannot read report {path}") from error


def write_site(site: Site, out: Path) -> None:
    pages_dir = out / "questions"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for stale in pages_dir.glob("*.json"):
        stale.unlink()
    for page in site.pages:
        _write(pages_dir / f"{page.id}.json", page)
    _write(out / "site.json", site.payload)


def _write(path: Path, model: BaseModel) -> None:
    partial = path.with_suffix(".json.partial")
    partial.write_text(model.model_dump_json(), encoding="utf-8")
    partial.replace(path)
