from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from rageval import __version__
from rageval.answer.citations import ResolutionResult, ResolvedCitation
from rageval.answer.generate import PROMPT_VERSION, answer_question
from rageval.eval.golden import GoldenQuestion
from rageval.eval.metrics import citation_hit, context_recall, mean
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
    retrieval: RetrievalConfig = DenseConfig()
    embedding_model: str | None
    chat_model: str
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
        retrieval=retriever.config,
        embedding_model=retriever.embedding_model,
        chat_model=chat.model,
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


def _rate(results: Sequence[AnswerResult]) -> float:
    return mean([1.0 if result.citation_hit else 0.0 for result in results])
