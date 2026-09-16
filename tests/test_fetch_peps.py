from fetch_peps import _indented


def test_a_block_is_dedented_by_its_own_minimum_not_by_a_running_one() -> None:
    lines = ["", "        deep = 1", "    shallow = 2", "        deeper = 3", "outside"]

    assert list(_indented(lines, start=0, outer=0)) == [
        "",
        "    deep = 1",
        "shallow = 2",
        "    deeper = 3",
    ]


def test_the_block_ends_at_the_first_line_back_at_the_outer_indent() -> None:
    lines = ["    inside", "outside", "    inside again"]

    assert list(_indented(lines, start=0, outer=0)) == ["inside"]


def test_a_uniformly_indented_block_keeps_its_relative_shape() -> None:
    lines = ["    def f():", "        return 1", "", "    f()"]

    assert list(_indented(lines, start=0, outer=0)) == ["def f():", "    return 1", "", "f()"]
