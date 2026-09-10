import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in [key for key in os.environ if key.startswith("RAGEVAL_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield
