from collections.abc import Sequence

from rageval.answer.citations import ResolvedCitation, UnresolvedCitation
from rageval.eval.golden import Support
from rageval.retrieval.store import ScoredChunk


def covers(chunk: ScoredChunk, support: Support) -> bool:
    if chunk.source_path != support.source_path:
        return False
    return chunk.start_char < support.end_char and support.start_char < chunk.end_char


def context_recall(retrieved: Sequence[ScoredChunk], supports: Sequence[Support], k: int) -> float:
    if not supports:
        return 0.0

    top = retrieved[:k]
    found = sum(1 for support in supports if any(covers(chunk, support) for chunk in top))
    return found / len(supports)


def reciprocal_rank(retrieved: Sequence[ScoredChunk], supports: Sequence[Support], k: int) -> float:
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if any(covers(chunk, support) for support in supports):
            return 1.0 / rank
    return 0.0


def citation_hit(
    citations: Sequence[ResolvedCitation | UnresolvedCitation], supports: Sequence[Support]
) -> bool:
    return any(
        citation.source_path == support.source_path
        and citation.start_char < support.end_char
        and support.start_char < citation.end_char
        for citation in citations
        if isinstance(citation, ResolvedCitation)
        for support in supports
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
