"""Recorded answers, committed so the answer eval replays with no key.

A fixture holds one chat cache entry per line: the chain digest of a prompt and
the result the chain returned for it. Loading one writes those entries into the
chat cache, where the keyless chat path serves them and refuses anything else.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rageval.providers.base import ChatProvider, ChatResult
from rageval.providers.cache import CHAT_NAMESPACE, DiskCache, chat_digest


class FixtureError(Exception):
    pass


class FixtureEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: ChatResult


class RecordedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    system: str | None
    result: ChatResult


class RecordingChatProvider:
    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider
        self.calls: list[RecordedCall] = []

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        result = self._provider.complete(prompt, system)
        self.calls.append(RecordedCall(prompt=prompt, system=system, result=result))
        return result


def write_fixture(path: Path, chain: Sequence[str], calls: Sequence[RecordedCall]) -> int:
    entries: dict[str, FixtureEntry] = {}
    for call in calls:
        digest = chat_digest(chain, call.prompt, call.system)
        entries[digest] = FixtureEntry(digest=digest, result=call.result)
    payload = "".join(entries[digest].model_dump_json() + "\n" for digest in sorted(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8"))
    return len(entries)


def load_fixture(cache: DiskCache, path: Path) -> int:
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise FixtureError(f"cannot read answer fixture {path}") from error

    entries = [
        _entry(line, number, path)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    for entry in entries:
        cache.set(CHAT_NAMESPACE, entry.digest, entry.result)
    return len(entries)


def _entry(line: str, number: int, path: Path) -> FixtureEntry:
    try:
        return FixtureEntry.model_validate_json(line)
    except ValidationError as error:
        raise FixtureError(f"{path} line {number} is not a recorded answer") from error
