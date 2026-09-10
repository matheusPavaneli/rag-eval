# ADR 0001 — Project skeleton

Date: 2026-09-09 · Slice: F0 · Status: accepted

## Context

The project has to be reproducible from an empty machine and has to make quality
measurable from the first slice. Every choice below is judged against one
question: does it make the pipeline easier to explain and to prove?

## Decisions

### Postgres with pgvector, not a dedicated vector database

Chosen because one service holds both the dense vectors and, later, the lexical
index — F4 needs both, and a single store means no consistency problem between
two systems that must be kept in sync.

Rejected: Qdrant and Chroma. Both are pleasant to start with and add a second
service to operate. Running pgvector inside Postgres is also the more common
production arrangement and the more useful thing to be able to discuss.

Rejected: SQLite with sqlite-vec. Lighter, but it pushes the lexical half of F4
back into application code, which is exactly the seam this decision avoids.

### BM25 deliberately left undecided

Postgres full-text search ranks with `ts_rank_cd`, which is not BM25. It is a
different formula with different behaviour on term saturation and document
length. Calling it BM25 in the README would be false.

The choice — `pg_search`, an in-process BM25 implementation, or accepting
`ts_rank_cd` and naming it correctly — belongs to F4, where a measured baseline
exists to judge it against. Recording the tension here so the F4 decision starts
from the constraint rather than rediscovering it.

### No RAG framework

Chosen: every stage written directly against the provider APIs.

Rejected: LangChain and LlamaIndex. They shorten the first afternoon and hide
the retrieval and prompt assembly behind abstractions. Since the object of the
project is to be able to explain and defend each stage, hiding those stages
removes the value.

Cost accepted: more code to write, and reranking, chunking and retry logic have
to be implemented rather than imported.

### Secrets typed as `SecretStr`

A plain `str` key reaches logs through three routes: `repr` in a traceback, an
f-string in a log line, and `model_dump` in structured logging. `SecretStr`
closes all three, and `test_secret_does_not_leak_through_repr_str_or_dump`
asserts each one, so the guarantee is enforced rather than intended.

### Integration tests behind a marker

Pure tests run with no services. Anything needing Postgres or the network is
marked `integration` and excluded by default.

A suite that requires Docker to be up is a suite that gets skipped locally, and
a skipped suite stops catching regressions. Splitting them keeps a red run
meaningful.

### `mypy --strict` with the pydantic plugin, without `disallow_any_explicit`

The plugin type-checks model constructors, which is where most mistakes with
settings and schemas occur. `disallow_any_explicit` was tried and removed: it
flags `Any` inside third-party base classes, producing errors no change in this
repository can fix.

### Reasoning lives in ADRs, not in comments

Code carries no explanatory comments. Comments drift from the code and carry no
date, no status and no record of what was rejected. An ADR has all three, and a
decision that changes is superseded by a new one instead of being silently
edited.

## Consequences

- `docker compose up -d` plus `uv sync --all-groups` is the whole setup.
- CI runs quality checks and integration tests as separate jobs; the integration
  job creates the `vector` extension explicitly, because the compose init script
  does not run under GitHub's service containers.
- Port 5433 is used locally to avoid colliding with an existing Postgres.
- The README quality table stays empty until F3 produces a recorded run.
