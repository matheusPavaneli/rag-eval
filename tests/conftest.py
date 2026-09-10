import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from rageval.config import Settings

DATABASE_URL = os.environ.get("RAGEVAL_DATABASE_URL") or Settings().database_url


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in [key for key in os.environ if key.startswith("RAGEVAL_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def database() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as connection:
        yield connection
