import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from rageval.config import Settings
from rageval.providers import (
    CacheMissError,
    ChatProvider,
    PermanentProviderError,
    TransientProviderError,
    build_budget,
    build_chat_provider,
    build_embedding_provider,
)
from rageval.providers.gemini import GeminiChatProvider, GeminiEmbeddingProvider
from rageval.providers.groq import GroqChatProvider

KEY = "canary-key-that-must-not-leak"

Handler = Callable[[httpx.Request], httpx.Response]


def client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def gemini_chat_body(text: str = "answer") -> dict[str, object]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
    }


def gemini_embed_body(vectors: list[list[float]]) -> dict[str, object]:
    return {"embeddings": [{"values": values} for values in vectors]}


def groq_chat_body(text: str = "answer") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def test_gemini_sends_the_key_in_a_header_and_never_in_the_url() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=gemini_chat_body())

    provider = GeminiChatProvider(client(handler), SecretStr(KEY), "gemini-test", 5.0)
    result = provider.complete("question", "be terse")

    request = seen[0]
    payload = json.loads(request.content)
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url)
    assert request.url.path.endswith("/models/gemini-test:generateContent")
    assert payload["generationConfig"]["temperature"] == 0
    assert payload["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert result.text == "answer"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (11, 3)


def test_groq_sends_the_key_as_a_bearer_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=groq_chat_body())

    provider = GroqChatProvider(client(handler), SecretStr(KEY), "llama-test", 5.0)
    result = provider.complete("question", "be terse")

    request = seen[0]
    payload = json.loads(request.content)
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url)
    assert payload["temperature"] == 0
    assert payload["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "question"},
    ]
    assert result.provider == "groq"
    assert result.text == "answer"


def test_the_two_adapters_return_the_same_shape() -> None:
    gemini = GeminiChatProvider(
        client(lambda _: httpx.Response(200, json=gemini_chat_body("same"))),
        SecretStr(KEY),
        "gemini-test",
        5.0,
    )
    groq = GroqChatProvider(
        client(lambda _: httpx.Response(200, json=groq_chat_body("same"))),
        SecretStr(KEY),
        "llama-test",
        5.0,
    )

    first = gemini.complete("question")
    second = groq.complete("question")

    assert first.text == second.text
    assert (first.provider, second.provider) == ("gemini", "groq")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_a_transient_status_is_transient(status: int) -> None:
    provider = GroqChatProvider(
        client(lambda _: httpx.Response(status, json={})), SecretStr(KEY), "llama-test", 5.0
    )

    with pytest.raises(TransientProviderError, match=str(status)):
        provider.complete("question")


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_rejected_request_is_permanent(status: int) -> None:
    provider = GroqChatProvider(
        client(lambda _: httpx.Response(status, json={})), SecretStr(KEY), "llama-test", 5.0
    )

    with pytest.raises(PermanentProviderError, match=str(status)):
        provider.complete("question")


def test_a_timeout_is_transient_and_names_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = GroqChatProvider(client(handler), SecretStr(KEY), "llama-test", 5.0)

    with pytest.raises(TransientProviderError, match=r"5\.0s"):
        provider.complete("question")


def test_a_success_missing_the_expected_field_is_permanent_not_a_key_error() -> None:
    provider = GeminiChatProvider(
        client(lambda _: httpx.Response(200, json={"nothing": "useful"})),
        SecretStr(KEY),
        "gemini-test",
        5.0,
    )

    with pytest.raises(PermanentProviderError, match="expected fields"):
        provider.complete("question")


def test_a_body_that_is_not_json_is_permanent() -> None:
    provider = GroqChatProvider(
        client(lambda _: httpx.Response(200, text="<html>gateway</html>")),
        SecretStr(KEY),
        "llama-test",
        5.0,
    )

    with pytest.raises(PermanentProviderError, match="not JSON"):
        provider.complete("question")


def test_no_provider_error_carries_the_key_even_when_the_body_echoes_it() -> None:
    rejection = {"error": {"message": f"API key {KEY} is not valid"}}
    providers: list[ChatProvider] = [
        GeminiChatProvider(
            client(lambda _: httpx.Response(401, json=rejection)),
            SecretStr(KEY),
            "gemini-test",
            5.0,
        ),
        GroqChatProvider(
            client(lambda _: httpx.Response(401, json=rejection)),
            SecretStr(KEY),
            "llama-test",
            5.0,
        ),
    ]

    for provider in providers:
        with pytest.raises(PermanentProviderError) as raised:
            provider.complete("question")
        assert KEY not in str(raised.value)
        assert KEY not in repr(raised.value)


def test_a_transport_failure_carries_no_key_either() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = GeminiChatProvider(client(handler), SecretStr(KEY), "gemini-test", 5.0)

    with pytest.raises(TransientProviderError) as raised:
        provider.complete("question")
    assert KEY not in str(raised.value)


def test_embeddings_come_back_one_per_text_in_order_at_the_asked_for_dimension() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=gemini_embed_body([[1.0, 0.0], [0.0, 1.0]]))

    provider = GeminiEmbeddingProvider(client(handler), SecretStr(KEY), "embed-test", 2, 5.0)
    result = provider.embed(["alpha", "beta"], "query")

    payload = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/models/embed-test:batchEmbedContents")
    assert [entry["content"]["parts"][0]["text"] for entry in payload["requests"]] == [
        "alpha",
        "beta",
    ]
    assert payload["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert payload["requests"][0]["outputDimensionality"] == 2
    assert result.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert result.dimension == 2


def test_a_document_and_a_query_ask_gemini_for_different_task_types() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=gemini_embed_body([[0.1, 0.2]]))

    provider = GeminiEmbeddingProvider(client(handler), SecretStr(KEY), "embed-test", 2, 5.0)
    provider.embed(["text"], "document")
    provider.embed(["text"], "query")

    task_types = [json.loads(request.content)["requests"][0]["taskType"] for request in seen]
    assert task_types == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_a_short_or_missing_vector_is_reported_rather_than_stored() -> None:
    wrong_length = GeminiEmbeddingProvider(
        client(lambda _: httpx.Response(200, json=gemini_embed_body([[0.1]]))),
        SecretStr(KEY),
        "embed-test",
        2,
        5.0,
    )
    missing_one = GeminiEmbeddingProvider(
        client(lambda _: httpx.Response(200, json=gemini_embed_body([[0.1, 0.2]]))),
        SecretStr(KEY),
        "embed-test",
        2,
        5.0,
    )

    with pytest.raises(PermanentProviderError, match="expected 2 dimensions"):
        wrong_length.embed(["alpha"])
    with pytest.raises(PermanentProviderError, match="asked for 2 embeddings"):
        missing_one.embed(["alpha", "beta"])


def test_an_empty_batch_asks_nothing_of_the_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made")

    provider = GeminiEmbeddingProvider(client(handler), SecretStr(KEY), "embed-test", 2, 5.0)

    assert provider.embed([]).vectors == ()


def test_without_a_key_the_chat_builder_replays_the_cache_and_never_the_network() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=groq_chat_body("recorded"))

    settings = Settings()
    build_chat_provider(
        settings.model_copy(update={"groq_api_key": SecretStr(KEY)}), client(handler)
    ).complete("question", "system")
    requests.clear()

    keyless = build_chat_provider(settings, client(handler))

    replayed = keyless.complete("question", "system")
    assert (replayed.provider, replayed.text) == ("groq", "recorded")
    with pytest.raises(CacheMissError, match=r"RAGEVAL_GEMINI_API_KEY.*RAGEVAL_GROQ_API_KEY"):
        keyless.complete("never answered")
    assert requests == []


def test_a_recorded_answer_is_replayed_even_after_the_first_provider_recovers() -> None:
    settings = Settings(gemini_api_key=SecretStr(KEY), groq_api_key=SecretStr(KEY))
    requests: list[httpx.Request] = []
    gemini_status = 429

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "googleapis" in request.url.host:
            return httpx.Response(gemini_status, json=gemini_chat_body("gemini answer"))
        return httpx.Response(200, json=groq_chat_body("groq answer"))

    recorded = build_chat_provider(settings, client(handler)).complete("question")
    requests.clear()
    gemini_status = 200

    replayed = build_chat_provider(settings, client(handler)).complete("question")

    assert (recorded.provider, recorded.text) == ("groq", "groq answer")
    assert replayed == recorded
    assert requests == []


def test_a_changed_chat_model_is_a_miss_not_a_stale_answer() -> None:
    settings = Settings(groq_api_key=SecretStr(KEY))
    build_chat_provider(
        settings, client(lambda _: httpx.Response(200, json=groq_chat_body()))
    ).complete("question")

    keyless = build_chat_provider(
        Settings(groq_chat_model="another/model"),
        client(lambda _: httpx.Response(200, json=groq_chat_body())),
    )

    with pytest.raises(CacheMissError):
        keyless.complete("question")


def test_a_replayed_answer_spends_no_budget() -> None:
    settings = Settings(groq_api_key=SecretStr(KEY))
    transport = client(lambda _: httpx.Response(200, json=groq_chat_body()))
    build_chat_provider(settings, transport).complete("question")
    budget = build_budget(settings)

    build_chat_provider(settings, transport, budget).complete("question")

    assert budget.state.calls == 0


def test_without_a_key_the_embedding_builder_serves_the_cache_and_never_the_network(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=gemini_embed_body([[0.5, 0.5]]))

    settings = Settings(cache_dir=tmp_path, embedding_dimension=2)
    keyed = build_embedding_provider(
        settings.model_copy(update={"gemini_api_key": SecretStr(KEY)}), client(handler)
    )
    warmed = keyed.embed(["cached"], "query")
    requests.clear()

    keyless = build_embedding_provider(settings, client(handler))

    assert (keyless.name, keyless.model, keyless.dimension) == (
        "gemini",
        settings.gemini_embedding_model,
        2,
    )
    assert keyless.embed(["cached"], "query").vectors == warmed.vectors
    with pytest.raises(CacheMissError, match="RAGEVAL_GEMINI_API_KEY"):
        keyless.embed(["never embedded"], "query")
    assert requests == []


def test_one_key_is_enough_to_build_a_chain() -> None:
    settings = Settings(groq_api_key=SecretStr(KEY))

    chain = build_chat_provider(
        settings, client(lambda _: httpx.Response(200, json=groq_chat_body()))
    )

    assert chain.complete("question").provider == "groq"


def test_the_chain_prefers_gemini_and_falls_over_to_groq() -> None:
    settings = Settings(gemini_api_key=SecretStr(KEY), groq_api_key=SecretStr(KEY))

    def handler(request: httpx.Request) -> httpx.Response:
        if "googleapis" in request.url.host:
            return httpx.Response(429, json={})
        return httpx.Response(200, json=groq_chat_body())

    chain = build_chat_provider(settings, client(handler))

    assert chain.complete("question").provider == "groq"


def test_a_vector_shorter_than_the_models_default_comes_back_normalised() -> None:
    provider = GeminiEmbeddingProvider(
        client(lambda _: httpx.Response(200, json=gemini_embed_body([[3.0, 4.0]]))),
        SecretStr(KEY),
        "embed-test",
        2,
        5.0,
    )

    assert provider.embed(["alpha"]).vectors == ((0.6, 0.8),)


def test_an_endpoint_that_is_not_http_is_permanent_rather_than_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.UnsupportedProtocol("ftp is not supported", request=request)

    provider = GroqChatProvider(client(handler), SecretStr(KEY), "llama-test", 5.0)

    with pytest.raises(PermanentProviderError, match="not an http endpoint"):
        provider.complete("question")
