from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from rageval import __version__
from rageval.answer.citations import ResolutionResult, ResolvedCitation
from rageval.answer.generate import PROMPT_VERSION, answer_question
from rageval.eval.golden import GoldenQuestion, golden_digest
from rageval.eval.metrics import citation_hit, context_recall, mean
from rageval.eval.runner import ReportMismatchError, check_golden_set
from rageval.providers.base import ChatProvider
from rageval.retrieval.search import DenseConfig, QuestionRetriever, RetrievalConfig


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    context_recall: float
    answer: str | None
    citations: tuple[ResolutionResult, ...]
    cited_chunk_chars: tuple[int, ...]
    parse_error: str | None
    raw_response: str | None
    provider: str
    model: str
    citation_hit: bool


class AnswerReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ran_at: datetime
    rageval_version: str
    corpus_version: str
    golden_set_digest: str | None = None
    retrieval: RetrievalConfig = DenseConfig()
    embedding_model: str | None
    chat_chain: tuple[str, ...] = ()
    prompt_version: str
    k: int
    question_count: int
    retrieved_count: int
    citation_hit_rate: float
    citation_hit_rate_retrieved: float
    citation_count: int
    resolution_rate: float
    mean_citation_chars: float
    mean_cited_chunk_chars: float
    parse_failures: int
    providers: dict[str, int]
    network_calls: int
    results: tuple[AnswerResult, ...]

    def table_row(self) -> str:
        # Failover can answer questions with a model other than the configured primary,
        # so the row names the models that actually answered, most frequent first.
        answered_by = ", ".join(
            f"{model} ({count})"
            for model, count in sorted(self.providers.items(), key=lambda item: -item[1])
        )
        return (
            f"| {self.ran_at.date().isoformat()} "
            f"| {answered_by}, prompt `{self.prompt_version}`, over {self.retrieval.mode} "
            f"retrieval ({self.embedding_model}), k={self.k}, corpus `{self.corpus_version}` "
            f"| {self.citation_hit_rate:.3f} "
            f"| {self.citation_hit_rate_retrieved:.3f} ({self.retrieved_count}) "
            f"| {self.resolution_rate:.3f} |"
        )


def run_answer_eval(
    questions: Sequence[GoldenQuestion],
    retriever: QuestionRetriever,
    chat: ChatProvider,
    chat_chain: Sequence[str],
    network_calls: Callable[[], int] = lambda: 0,
) -> AnswerReport:
    k = retriever.top_k
    results: list[AnswerResult] = []

    for question in questions:
        retrieved = retriever.retrieve(question.question)[:k]
        answer = answer_question(question.question, retrieved, chat)
        cited = [
            retrieved[citation.chunk - 1]
            for citation in answer.citations
            if isinstance(citation, ResolvedCitation)
        ]
        results.append(
            AnswerResult(
                id=question.id,
                question=question.question,
                context_recall=context_recall(retrieved, question.supports, k),
                answer=answer.text,
                citations=answer.citations,
                cited_chunk_chars=tuple(chunk.end_char - chunk.start_char for chunk in cited),
                parse_error=answer.parse_error,
                raw_response=answer.raw_response,
                provider=answer.provider,
                model=answer.model,
                citation_hit=citation_hit(answer.citations, question.supports),
            )
        )

    citations = [citation for result in results for citation in result.citations]
    resolved = [citation for citation in citations if isinstance(citation, ResolvedCitation)]
    retrieved_hits = [result for result in results if result.context_recall > 0]

    return AnswerReport(
        ran_at=datetime.now(UTC),
        rageval_version=__version__,
        corpus_version=retriever.corpus_version,
        golden_set_digest=golden_digest(questions),
        retrieval=retriever.config,
        embedding_model=retriever.embedding_model,
        chat_chain=tuple(chat_chain),
        prompt_version=PROMPT_VERSION,
        k=k,
        question_count=len(results),
        retrieved_count=len(retrieved_hits),
        citation_hit_rate=_rate(results),
        citation_hit_rate_retrieved=_rate(retrieved_hits),
        citation_count=len(citations),
        resolution_rate=len(resolved) / len(citations) if citations else 0.0,
        mean_citation_chars=mean([cited.end_char - cited.start_char for cited in resolved]),
        mean_cited_chunk_chars=mean(
            [chars for result in results for chars in result.cited_chunk_chars]
        ),
        parse_failures=sum(1 for result in results if result.parse_error is not None),
        providers=dict(Counter(f"{result.provider}/{result.model}" for result in results)),
        network_calls=network_calls(),
        results=tuple(results),
    )


# What the pipeline derives from one recorded answer. A change in any of them is drift.
ANSWER_FIELDS = (
    "answer",
    "parse_error",
    "raw_response",
    "citations",
    "cited_chunk_chars",
    "citation_hit",
    "context_recall",
    "provider",
    "model",
)
AGGREGATE_FIELDS = (
    "retrieved_count",
    "citation_hit_rate",
    "citation_hit_rate_retrieved",
    "citation_count",
    "resolution_rate",
    "mean_citation_chars",
    "mean_cited_chunk_chars",
    "parse_failures",
    "providers",
)


class AnswerQuestionDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    fields: tuple[str, ...]


class AggregateDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    before: object
    after: object


class AnswerDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    changed: tuple[AnswerQuestionDrift, ...]
    aggregates: tuple[AggregateDrift, ...]

    @property
    def any(self) -> bool:
        return bool(self.changed or self.aggregates)


def answer_drift(baseline: AnswerReport, candidate: AnswerReport) -> AnswerDrift:
    for field in ("corpus_version", "k", "retrieval", "prompt_version", "chat_chain"):
        before, after = getattr(baseline, field), getattr(candidate, field)
        if before != after:
            raise ReportMismatchError(f"{field} differs: baseline {before}, candidate {after}")

    before = {result.id: result for result in baseline.results}
    after = {result.id: result for result in candidate.results}
    if before.keys() != after.keys():
        missing = sorted(before.keys() - after.keys())
        extra = sorted(after.keys() - before.keys())
        raise ReportMismatchError(
            f"question set differs: missing {missing or 'none'}, extra {extra or 'none'}"
        )
    check_golden_set(baseline.golden_set_digest, candidate.golden_set_digest)

    changed = tuple(
        AnswerQuestionDrift(id=question, fields=fields)
        for question in sorted(after)
        if (
            fields := tuple(
                field
                for field in ANSWER_FIELDS
                if getattr(before[question], field) != getattr(after[question], field)
            )
        )
    )
    aggregates = tuple(
        AggregateDrift(
            field=field, before=getattr(baseline, field), after=getattr(candidate, field)
        )
        for field in AGGREGATE_FIELDS
        if getattr(baseline, field) != getattr(candidate, field)
    )
    return AnswerDrift(changed=changed, aggregates=aggregates)


def _rate(results: Sequence[AnswerResult]) -> float:
    return mean([1.0 if result.citation_hit else 0.0 for result in results])
