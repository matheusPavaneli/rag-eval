from conftest_retrieval import scored
from rageval.eval.golden import Support
from rageval.eval.metrics import context_recall, covers, mean, reciprocal_rank

SUPPORT = Support(source_path="a.md", start_char=100, end_char=200)


def test_a_chunk_overlapping_by_one_character_supports_the_answer() -> None:
    assert covers(scored("a.md", 50, 101), SUPPORT)
    assert covers(scored("a.md", 199, 400), SUPPORT)


def test_a_chunk_ending_where_the_span_begins_shares_no_character() -> None:
    assert not covers(scored("a.md", 50, 100), SUPPORT)
    assert not covers(scored("a.md", 200, 400), SUPPORT)


def test_a_chunk_from_another_document_never_supports_the_answer() -> None:
    assert not covers(scored("b.md", 100, 200), SUPPORT)


def test_recall_counts_supports_covered_not_chunks_returned() -> None:
    second = Support(source_path="b.md", start_char=0, end_char=10)
    retrieved = [
        scored("a.md", 100, 150),
        scored("a.md", 150, 200),
        scored("a.md", 120, 180),
    ]

    assert context_recall(retrieved, [SUPPORT, second], k=5) == 0.5


def test_recall_only_looks_at_the_top_k() -> None:
    retrieved = [scored("b.md", 0, 10), scored("c.md", 0, 10), scored("a.md", 100, 200)]

    assert context_recall(retrieved, [SUPPORT], k=2) == 0.0
    assert context_recall(retrieved, [SUPPORT], k=3) == 1.0


def test_reciprocal_rank_is_the_position_of_the_first_supporting_chunk() -> None:
    retrieved = [scored("b.md", 0, 10), scored("a.md", 100, 200), scored("a.md", 120, 180)]

    assert reciprocal_rank(retrieved, [SUPPORT], k=5) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_in_the_top_k_supports_the_answer() -> None:
    retrieved = [scored("b.md", 0, 10), scored("c.md", 0, 10)]

    assert reciprocal_rank(retrieved, [SUPPORT], k=5) == 0.0
    assert reciprocal_rank([scored("a.md", 100, 200)], [SUPPORT], k=0) == 0.0


def test_a_question_with_no_support_scores_zero_rather_than_dividing_by_zero() -> None:
    assert context_recall([scored("a.md", 100, 200)], [], k=5) == 0.0
    assert mean([]) == 0.0


def test_mean_averages_the_per_question_scores() -> None:
    assert mean([1.0, 0.0, 0.5]) == 0.5
