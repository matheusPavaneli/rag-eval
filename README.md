# rageval

A retrieval-augmented generation pipeline that measures its own quality and
proves every answer against a verifiable span in the source document.

Not a chatbot demo. The point of the project is the part tutorials skip: a
golden set, a number, and a CI gate that fails when retrieval quality drops.

## Status

Built in slices, each one measured before the next begins.

| Slice | Scope | State |
| --- | --- | --- |
| F0 | Reproducible skeleton: uv, ruff, mypy strict, pytest, Postgres + pgvector, CI | done |
| F1 | Ingest: loading, chunking with metadata, content hashing, versioned corpus | done |
| F2 | Provider layer: one Protocol, Gemini and Groq adapters, disk cache, budget, failover | done |
| F3 | Vector retrieval baseline, 30-question golden set, first measured number | pending |
| F4 | Hybrid retrieval and local reranking, measured against the F3 baseline | pending |
| F5 | Answers with citations resolved to a character span in the source document | pending |
| F6 | Eval harness wired into CI, failing the build on quality regression | pending |
| F7 | Next.js frontend: chat where a citation click highlights the source span | pending |

## Retrieval quality

No numbers yet. This table is filled by F3 and must never hold a figure that was
not produced by a recorded run.

| Date | Configuration | Context recall @5 | MRR @5 |
| --- | --- | --- | --- |
| — | — | — | — |

## Requirements

Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker.

## Setup

```bash
cp .env.example .env
uv sync --all-groups
docker compose up -d
```

On Windows, Docker runs inside WSL: `wsl.exe -e docker compose up -d`.

Both keys are free tier and neither asks for a credit card. Gemini is required
from F2 on, because it is the only one of the two that embeds:

- `RAGEVAL_GEMINI_API_KEY` — https://aistudio.google.com/apikey
- `RAGEVAL_GROQ_API_KEY` — https://console.groq.com/keys

## Ingest

```bash
uv run python -m rageval.ingest documents/
```

Loads every `.md` and `.txt` under the directory, chunks each one, and writes
`data/corpus/<corpus_version>/` — `documents.jsonl`, `chunks.jsonl` and a
`manifest.json`. Nothing here calls a model or the network.

Every chunk carries the character span it occupies in its source document, so a
citation resolves to exact characters rather than to a copy of the text.

`corpus_version` is the hash of the document contents and the chunk parameters,
so it is stable across re-runs and different the moment either changes. Every
number this project publishes names the corpus version that produced it: without
that, two measurements cannot be compared.

## Providers

Every model call goes through one `Protocol`, so no caller knows which provider
answered:

```python
from httpx import Client

from rageval.config import get_settings
from rageval.providers import build_chat_provider, build_embedding_provider

settings = get_settings()
with Client() as client:
    vectors = build_embedding_provider(settings, client).embed(["a chunk"])
    answer = build_chat_provider(settings, client).complete("a question")
```

Gemini answers first and Groq takes over when Gemini fails *transiently* — a
rate limit, a timeout, a 5xx. A rejected key is a permanent failure and stops
the run: failing over on a bad key would quietly change the model behind a
published number.

Every answer is cached on disk under `.cache/providers/`, keyed by the hash of
the provider, the model and the exact input. A repeated run answers from disk
and makes no request, which is what keeps a demo working when the free-tier
quota is gone. Delete the directory to force a refetch.

`RAGEVAL_BUDGET_MAX_CALLS` and `RAGEVAL_BUDGET_MAX_INPUT_CHARS` cap what one run
may send; the call that would exceed either is refused before it is made, so a
loop with a bug cannot spend a day's quota. Cached answers cost nothing against
the budget.

Groq exposes no embedding API, so embeddings are Gemini only and say so rather
than pretending a fallback exists.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run pytest -m integration
```

Pure tests run everywhere. Tests that need Postgres or the network carry the
`integration` marker and are excluded by default, so a suite that fails is a
real failure rather than a missing service.

## Design decisions

Every slice records what was chosen, what was rejected, and why, under
[`docs/adr/`](docs/adr/). Code carries no explanatory comments: the reasoning
lives in the ADRs, where it can be dated and revised.
