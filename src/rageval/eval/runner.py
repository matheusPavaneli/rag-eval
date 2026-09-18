from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from rageval import __version__
from rageval.eval.golden import GoldenQuestion, golden_digest
from rageval.eval.metrics import context_recall, mean, reciprocal_rank
from rageval.ingest.chunking import ChunkConfig
from rageval.retrieval.search import (
    Bm25Config,
    DenseConfig,
    FullTextConfig,
    HybridConfig,
    QuestionRetriever,
    RerankConfig,
    RetrievalConfig,
)


class ReportMismatchError(Exception):
    pass


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
    golden_set_digest: str | None = None
    retrieval: RetrievalConfig = DenseConfig()
    embedding_model: str | None
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
            f"| {self._retrieval_label()}, chunk {self.chunk_size}/{self.overlap}, "
            f"k={self.k}, corpus `{self.corpus_version}` "
            f"| {self.context_recall_at_k:.3f} | {self.mrr_at_k:.3f} |"
        )

    def _retrieval_label(
        self,
        config: DenseConfig
        | FullTextConfig
        | Bm25Config
        | HybridConfig
        | RerankConfig
        | None = None,
    ) -> str:
        dense = f"{self.embedding_model}, {self.dimension}d"
        match self.retrieval if config is None else config:
            case DenseConfig():
                return f"dense: {dense}"
            case FullTextConfig() | Bm25Config() as lexical_ranker:
                return f"{lexical_ranker.mode}: {_lexical_label(lexical_ranker)}"
            case HybridConfig(rrf_k=rrf_k, candidates=candidates, lexical=lexical):
                return (
                    f"hybrid (RRF k={rrf_k}, {candidates} candidates each): "
                    f"{dense} + {_lexical_label(lexical)}"
                )
            case RerankConfig(
                model=model, revision=revision, candidates=candidates, first_stage=first_stage
            ):
                return (
                    f"rerank ({model}@{revision[:7]}, top {candidates}) over "
                    f"{self._retrieval_label(first_stage)}"
                )


def _lexical_label(config: FullTextConfig | Bm25Config) -> str:
    match config:
        case FullTextConfig():
            return "Postgres full-text (ts_rank_cd, english)"
        case Bm25Config(k1=k1, b=b):
            return f"BM25 (k1={k1}, b={b}) over Postgres english lexemes"


class Flips(BaseModel):
    model_config = ConfigDict(frozen=True)

    gained: tuple[str, ...]
    lost: tuple[str, ...]


def compare(baseline: EvalReport, candidate: EvalReport) -> Flips:
    for field in ("corpus_version", "k"):
        before, after = getattr(baseline, field), getattr(candidate, field)
        if before != after:
            raise ReportMismatchError(f"{field} differs: baseline {before}, candidate {after}")

    before_ids = {result.id for result in baseline.results}
    after_ids = {result.id for result in candidate.results}
    if before_ids != after_ids:
        missing = sorted(before_ids - after_ids)
        extra = sorted(after_ids - before_ids)
        raise ReportMismatchError(
            f"question set differs: missing {missing or 'none'}, extra {extra or 'none'}"
        )
    check_golden_set(baseline.golden_set_digest, candidate.golden_set_digest)

    hit_before = {result.id for result in baseline.results if result.context_recall > 0}
    hit_after = {result.id for result in candidate.results if result.context_recall > 0}
    return Flips(
        gained=tuple(sorted(hit_after - hit_before)), lost=tuple(sorted(hit_before - hit_after))
    )


def check_golden_set(baseline: str | None, candidate: str | None) -> None:
    if baseline != candidate:
        raise ReportMismatchError(
            f"golden set differs: baseline {baseline or 'records none'}, candidate "
            f"{candidate or 'records none'}; re-cut the baseline if the golden set changed "
            "on purpose"
        )


class QuestionDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    context_recall: tuple[float, float]
    reciprocal_rank: tuple[float, float]


class Drift(BaseModel):
    model_config = ConfigDict(frozen=True)

    changed: tuple[QuestionDrift, ...]
    context_recall_at_k: tuple[float, float]
    mrr_at_k: tuple[float, float]

    @property
    def any(self) -> bool:
        return (
            bool(self.changed)
            or self.context_recall_at_k[0] != self.context_recall_at_k[1]
            or self.mrr_at_k[0] != self.mrr_at_k[1]
        )


def drift(baseline: EvalReport, candidate: EvalReport) -> Drift:
    compare(baseline, candidate)
    if baseline.retrieval != candidate.retrieval:
        raise ReportMismatchError(
            f"retrieval config differs: baseline {baseline.retrieval.model_dump()}, "
            f"candidate {candidate.retrieval.model_dump()}"
        )

    before = {result.id: result for result in baseline.results}
    changed = tuple(
        QuestionDrift(
            id=after.id,
            context_recall=(before[after.id].context_recall, after.context_recall),
            reciprocal_rank=(before[after.id].reciprocal_rank, after.reciprocal_rank),
        )
        for after in sorted(candidate.results, key=lambda result: result.id)
        if (before[after.id].context_recall, before[after.id].reciprocal_rank)
        != (after.context_recall, after.reciprocal_rank)
    )
    return Drift(
        changed=changed,
        context_recall_at_k=(baseline.context_recall_at_k, candidate.context_recall_at_k),
        mrr_at_k=(baseline.mrr_at_k, candidate.mrr_at_k),
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
        golden_set_digest=golden_digest(questions),
        retrieval=retriever.config,
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
