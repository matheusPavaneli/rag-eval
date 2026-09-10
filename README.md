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
| F1 | Ingest: loading, chunking with metadata, content hashing, versioned corpus | pending |
| F2 | Provider layer: one Protocol, Gemini and Groq adapters, disk cache, budget, failover | pending |
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

`.env` is optional until F2. Both keys are free tier and neither asks for a
credit card:

- `RAGEVAL_GEMINI_API_KEY` — https://aistudio.google.com/apikey
- `RAGEVAL_GROQ_API_KEY` — https://console.groq.com/keys

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
