from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from rageval import __version__
from rageval.eval.golden import GoldenQuestion
from rageval.eval.metrics import context_recall, mean, reciprocal_rank
from rageval.ingest.chunking import ChunkConfig
from rageval.retrieval.search import QuestionRetriever


class QuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    context_recall: float
    reciprocal_rank: float
    retrieved_paths: tuple[str, ...]


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ran_at: datetime
    rageval_version: str
    corpus_version: str
    embedding_model: str
    dimension: int
    chunk_size: int
    overlap: int
    k: int
    question_count: int
    context_recall_at_k: float
    mrr_at_k: float
    network_calls: int
    results: tuple[QuestionResult, ...]

    def table_row(self) -> str:
        return (
            f"| {self.ran_at.date().isoformat()} "
            f"| {self.embedding_model}, {self.dimension}d, chunk {self.chunk_size}/{self.overlap}, "
            f"k={self.k}, corpus `{self.corpus_version}` "
            f"| {self.context_recall_at_k:.3f} | {self.mrr_at_k:.3f} |"
        )


def run_eval(
    questions: Sequence[GoldenQuestion],
    retriever: QuestionRetriever,
    chunk_config: ChunkConfig,
    dimension: int,
    network_calls: Callable[[], int] = lambda: 0,
) -> EvalReport:
    k = retriever.top_k
    results: list[QuestionResult] = []

    for question in questions:
        retrieved = retriever.retrieve(question.question)
        results.append(
            QuestionResult(
                id=question.id,
                question=question.question,
                context_recall=context_recall(retrieved, question.supports, k),
                reciprocal_rank=reciprocal_rank(retrieved, question.supports, k),
                retrieved_paths=tuple(chunk.source_path for chunk in retrieved),
            )
        )

    return EvalReport(
        ran_at=datetime.now(UTC),
        rageval_version=__version__,
        corpus_version=retriever.corpus_version,
        embedding_model=retriever.embedding_model,
        dimension=dimension,
        chunk_size=chunk_config.chunk_size,
        overlap=chunk_config.overlap,
        k=k,
        question_count=len(results),
        context_recall_at_k=mean([result.context_recall for result in results]),
        mrr_at_k=mean([result.reciprocal_rank for result in results]),
        network_calls=network_calls(),
        results=tuple(results),
    )
