# ADR 0003 — Providers: one Protocol, a cache that makes a demo repeatable

Date: 2026-09-10 · Slice: F2 · Status: accepted

## Context

F3 measures retrieval, which means embedding a whole corpus and asking a model
questions, repeatedly, on a free tier that is rate limited and can change under
the project. Every later slice calls a model through whatever is decided here,
so the contract, the cost control and the failure behaviour are settled in this
slice rather than negotiated in each of the ones after it.

## Decisions

### One Protocol, two adapters, no vendor SDK

`ChatProvider` and `EmbeddingProvider` are `Protocol`s over frozen result
models. `GeminiChatProvider`, `GeminiEmbeddingProvider` and `GroqChatProvider`
speak raw REST through an injected `httpx.Client`.

Rejected: the `google-genai` and `groq` clients. Two SDKs mean two object
models, two retry policies and two release cadences inside code whose entire
point is that the caller cannot tell which provider answered. The adapters are
about forty lines each, and the API surface used — one POST, one JSON body — is
the part of a vendor API that changes least.

The cost accepted: a rename of a request field is ours to notice. That is what
`tests/test_providers_live.py` is for, and it is the only place in the suite
that talks to a real provider.

`httpx` is the one dependency the slice adds: a timeout per request, and a
`MockTransport` that lets every adapter test assert on a real request object
without a network or a key.

### Transient and permanent are different failures

`TransientProviderError` covers 429, 5xx, timeouts and connection errors.
`PermanentProviderError` covers 400, 401, 403, 404 and any 200 whose body does
not hold the expected fields. Only a transient error may fail over.

An invalid key is therefore never answered by the second provider. Failing over
on an auth error would turn a broken configuration into a silent change of
model, and the number in the README would name a configuration that did not
produce it.

No error message carries the key, the `Authorization` header or the body of a
rejection — a provider that echoes an invalid key back is the ordinary case, and
that body would otherwise land in a log. The test asserts the canary key appears
in no message, `repr` or `str`, for a rejected key and a transport failure
alike.

### The cache sits inside the chain, one entry per provider and model

`Cached*` wraps `Budgeted*` wraps the adapter, each implementing the same
Protocol, so composition is the only wiring and no adapter knows either exists.
The key is the sha256 of a frozen `CacheKey` — provider, model, task, inputs,
dimension — so an entry can never be served for a model that did not produce it.
Entries are written to a temporary file and moved into place, so an interrupted
run leaves no half-written file that later reads as a hit.

Rejected: one cache in front of the failover chain. It would key on the request
alone, and the answer would then depend on which provider happened to be up when
the entry was first written — a cached run and a live run could disagree with
nothing in the corpus or configuration to explain it.

Embeddings are cached per text, not per batch, so a corpus that gained one
document re-embeds one document. Embedding the same text as a document and as a
query are two entries, because they are two different requests.

A cache file that does not parse raises an error naming the path instead of
being treated as a miss. Silent self-healing here would hide the one thing worth
knowing: that something wrote into `.cache/` that this code did not.

### The budget is a stop, not a meter

`Budget` caps calls and input characters per run and raises before the request
is made. It counts only what reaches the network, so a fully cached eval spends
nothing.

The point is not accounting — it is that a loop with a bug cannot burn a day's
free-tier quota in a minute. Output tokens are not counted: nothing in F2
consumes them, and F3 is what learns what a full eval run actually costs.

### Embeddings have one provider, and the code says so

Gemini embeds; Groq serves no embedding API. `build_embedding_provider` raises
naming the missing key and the reason there is no second provider, rather than
returning a one-element chain that pretends failover exists.

`embedding_dimension` defaults to 768 rather than the model's own default: F3
needs a fixed number for its pgvector column, and a stated dimension is what
makes a re-run comparable. The adapter normalises every vector to unit length
before returning it, because `gemini-embedding-001` only guarantees a normalised
vector at its full 3072 dimensions — an unnormalised vector would make cosine
similarity in F3 measure length as well as direction, and the cache would keep
it that way.

`taskType` is on the contract from the start — Gemini embeds a question and a
passage differently, and adding the distinction later would invalidate every
cached vector.

### Chunking stays in characters

ADR 0002 left open that F2 might switch chunking to the tokenizer of the chosen
embedding model. It does not.

Changing it now would change `corpus_version`, and there is no baseline yet for
the new corpus to be compared against — the switch would cost the one thing it
was supposed to improve. F4 is where a chunking change can be measured against
F3's number.

## Consequences

- Nothing in `rageval.ingest` imports `rageval.providers`; F1 keeps running
  with no key present.
- `uv run pytest` still needs no network and no key: the adapter tests drive
  `httpx.MockTransport`, and the three live tests carry the `integration` marker
  and skip when their key is absent.
- A repeated demo answers from `.cache/providers/` and makes no request, which
  is the mitigation the frame asked for against a free-tier quota running out
  mid-demo.
- F3 inherits `build_chat_provider` and `build_embedding_provider`, one shared
  `Budget` per run, and `embedding_dimension` as the number its vector column
  is built on.
