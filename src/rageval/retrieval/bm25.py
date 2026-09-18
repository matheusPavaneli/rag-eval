import math
from collections.abc import Sequence

from rageval.retrieval.store import ChunkTerms, RetrievalError, ScoredChunk


class Bm25Index:
    def __init__(self, documents: Sequence[ChunkTerms], k1: float, b: float) -> None:
        if k1 < 0:
            raise RetrievalError(f"BM25 k1 must be at least 0, got {k1}")
        if not 0 <= b <= 1:
            raise RetrievalError(f"BM25 b must be between 0 and 1, got {b}")

        self._documents = tuple(documents)
        self._k1 = k1
        self._b = b
        self._lengths = tuple(sum(document.terms.values()) for document in self._documents)
        total = sum(self._lengths)
        self._average_length = total / len(self._documents) if self._documents else 0.0

        self._postings: dict[str, list[tuple[int, int]]] = {}
        for position, document in enumerate(self._documents):
            for term, count in document.terms.items():
                self._postings.setdefault(term, []).append((position, count))

    def idf(self, term: str) -> float:
        frequency = len(self._postings.get(term, ()))
        count = len(self._documents)
        return math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))

    def search(self, terms: Sequence[str], limit: int) -> tuple[ScoredChunk, ...]:
        scores: dict[int, float] = {}
        for term in dict.fromkeys(terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            weight = self.idf(term)
            for position, count in postings:
                scores[position] = scores.get(position, 0.0) + weight * self._saturate(
                    count, self._lengths[position]
                )

        order = sorted(
            (position for position, score in scores.items() if score > 0),
            key=lambda position: (-scores[position], self._documents[position].chunk_id),
        )
        return tuple(
            self._documents[position].scored(scores[position]) for position in order[:limit]
        )

    def _saturate(self, count: int, length: int) -> float:
        norm = 1 - self._b + self._b * length / self._average_length
        return count * (self._k1 + 1) / (count + self._k1 * norm)
