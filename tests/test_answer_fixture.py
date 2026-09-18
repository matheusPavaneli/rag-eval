import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from rageval.answer.fixture import (
    FixtureError,
    RecordingChatProvider,
    load_fixture,
    write_fixture,
)
from rageval.config import Settings
from rageval.providers import DiskCache, build_chat_provider, chat_chain
from rageval.providers.failover import CACHE_NAMESPACE


def groq_body(text: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def record(settings: Settings, fixture: Path) -> list[str]:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json=groq_body(f"answer to {prompt}"))

    keyed = settings.model_copy(update={"groq_api_key": SecretStr("key")})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        recorder = RecordingChatProvider(build_chat_provider(keyed, client))
        answers = [recorder.complete(prompt, "system").text for prompt in ("two", "one", "two")]
    write_fixture(fixture, chat_chain(settings), recorder.calls)
    return answers


def test_recorded_answers_replay_from_an_empty_cache_with_no_key_and_no_request(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "answers.jsonl"
    recorded = record(Settings(cache_dir=tmp_path / "recording"), fixture)
    replaying = Settings(cache_dir=tmp_path / "replaying")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    loaded = load_fixture(DiskCache(replaying.cache_dir / CACHE_NAMESPACE), fixture)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        keyless = build_chat_provider(replaying, client)
        replayed = [keyless.complete(prompt, "system").text for prompt in ("two", "one", "two")]

    assert loaded == 2
    assert replayed == recorded
    assert requests == []


def test_a_fixture_has_one_sorted_entry_per_prompt_and_lf_line_endings(tmp_path: Path) -> None:
    fixture = tmp_path / "answers.jsonl"
    record(Settings(cache_dir=tmp_path / "recording"), fixture)

    data = fixture.read_bytes()
    digests = [json.loads(line)["digest"] for line in data.decode("utf-8").splitlines()]

    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert len(digests) == 2
    assert digests == sorted(digests)


def test_a_malformed_line_is_refused_by_number_and_nothing_is_written(tmp_path: Path) -> None:
    fixture = tmp_path / "answers.jsonl"
    record(Settings(cache_dir=tmp_path / "recording"), fixture)
    first = fixture.read_text(encoding="utf-8").splitlines()[0]
    fixture.write_text(f'{first}\n{{"digest": "not-hex"}}\n', encoding="utf-8")
    root = tmp_path / "replaying"

    with pytest.raises(FixtureError, match="line 2"):
        load_fixture(DiskCache(root), fixture)
    assert not root.exists()
