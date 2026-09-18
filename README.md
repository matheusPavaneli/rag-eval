# rageval

A retrieval-augmented generation pipeline that measures its own quality, on a
corpus you can read, with a number you can reproduce.

The interesting part of a RAG system is not that it answers. It is whether the
passage it answered from was the right one — and most RAG projects never find
out. This one starts there: a golden set of thirty questions, ground truth
recorded as character spans in the source documents, and a baseline published
before anything is optimised.

## The current numbers

| Date | Configuration | Context recall @5 | MRR @5 |
| --- | --- | --- | --- |
| 2026-09-16 | dense: gemini-embedding-001, 768d, chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.767 | 0.603 |
| 2026-09-18 | fulltext: Postgres full-text (ts_rank_cd, english), chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.400 | 0.222 |
| 2026-09-18 | bm25: BM25 (k1=1.2, b=0.75) over Postgres english lexemes, chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.600 | 0.465 |
| 2026-09-18 | hybrid (RRF k=60, 20 candidates each): gemini-embedding-001, 768d + Postgres full-text (ts_rank_cd, english), chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.767 | 0.439 |
| 2026-09-18 | hybrid (RRF k=60, 20 candidates each): gemini-embedding-001, 768d + BM25 (k1=1.2, b=0.75) over Postgres english lexemes, chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.800 | 0.570 |
| 2026-09-18 | rerank (cross-encoder/ms-marco-MiniLM-L6-v2@233902d, top 20) over dense: gemini-embedding-001, 768d, chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.833 | 0.490 |
| 2026-09-18 | rerank (cross-encoder/ms-marco-MiniLM-L6-v2@233902d, top 20) over hybrid (RRF k=60, 20 candidates each): gemini-embedding-001, 768d + BM25 (k1=1.2, b=0.75) over Postgres english lexemes, chunk 1000/150, k=5, corpus `26b03ce9a1c2c1d4` | 0.800 | 0.451 |

Measured over fifty Python Enhancement Proposals (~1.3 MB, 1829 chunks), over the
whole golden set — the harness refuses to report over a partial index or a corpus
version it cannot resolve every span against. The dense row is the F3 baseline;
it was re-run on 2026-09-18 and reproduced to the digit with no network call.
Each row's report, carrying the per-question result and every parameter that
produced it, is in [`evals/reports/`](evals/reports/).

What the rows say, read with the per-question flips rather than the aggregates:

- **Hybrid with BM25 trades rank for recall, and the gain is one question.** It
  finds two passages dense misses (q17, q27) and loses one dense found (q19): net
  +0.033 recall. MRR falls 0.603 → 0.570 — eight questions dense already answered
  are ranked lower, three higher. On thirty questions that is not a win; it is a
  different retriever with a different failure set.
- **BM25 over full-text is the clearest number here.** Both rank the *same*
  Postgres lexemes, so the gap — +0.200 recall, +0.243 MRR — is what IDF and
  term saturation are worth on this corpus. `ts_rank_cd` has no IDF, and `pep`
  occurs in 524 of 1829 chunks.
- **Fusing a weak ranker costs precision.** Hybrid over full-text keeps dense's
  recall and drops MRR to 0.439.
- **The reranker raised recall and lowered rank — the opposite of its target.**
  Over dense it finds three passages dense ranked 6th–20th (q11, q17, q27) and
  loses one (q08): recall 0.767 → 0.833, the highest in the table. But MRR falls
  to 0.490: it demotes the right chunk below another chunk *of the same PEP* in
  fourteen questions. Some of those are a real answer the single-span golden set
  does not count (q04); others are keyword attraction (q09, q28). Over the hybrid
  it does no better: 0.800 / 0.451.
- Four questions are missed by every configuration: q14, q21, q23, q29. The
  first two are in no first stage's top twenty, so no reranker could reach them;
  the reranker lifted q11, the fifth F4 miss, and none of these.

**Context recall @5** is the share of supporting passages that appear somewhere in
the top five. **MRR @5** is how far down the list the first correct passage sat.
Both are reported because either alone misleads: recall hides a retriever that
always ranks the answer fifth, MRR hides one that ranks its few hits well and
misses most questions.

Reproduce it:

```bash
uv run python -m rageval.ingest documents/
uv run python -m rageval.retrieval
uv run python -m rageval.eval
```

The second command is the slow one — the free embedding tier meters tokens per
minute, so the first index takes minutes and backs off when it is throttled.
Every vector is then cached on disk by content hash, so the run after that makes
no network call at all.

## What is measured, and what that is worth

Ground truth is a **character span in a source document**, not the id of a chunk.
A chunk id is a function of the chunk size, so a golden set keyed on one would
have to be rewritten the moment chunking changed — which is exactly the
experiment the set exists to judge. A retrieved chunk counts as relevant when it
overlaps a supporting span in the same document.

Every span is resolved against the corpus when the golden set loads. An unknown
file, a span past the end of its document, or a duplicated question id is an
error naming the question. A harness that silently skips questions it cannot
resolve reports a better number than it earned.

The set is drafted by `scripts/build_golden_set.py`, which holds each question
next to the needle that locates its answer and widens the match to the enclosing
paragraph. It is committed because the spans are offsets into documents that
have changed: re-running it is how the set follows a new corpus version.

The honest caveat, stated here rather than buried: the thirty questions were
drafted against this corpus and then verified span by span against the source.
About half the initial anchors were moved because they had landed on a heading
or a code sample instead of on a sentence that answered the question. A set
written with knowledge of the corpus flatters recall. Treat the figure as an
upper bound on this corpus, not as an estimate of production quality.

## Getting it running

Python 3.13, [uv](https://docs.astral.sh/uv/), Docker, and a free Gemini key.

```bash
cp .env.example .env      # then add RAGEVAL_GEMINI_API_KEY
uv sync --all-groups
docker compose up -d
```

On Windows, Docker runs inside WSL: `wsl.exe -e docker compose up -d`.

Two keys are read, both free tier, neither asking for a card. Gemini is required
because it is the only one of the two that embeds; Groq is the failover for chat.

- `RAGEVAL_GEMINI_API_KEY` — https://aistudio.google.com/apikey
- `RAGEVAL_GROQ_API_KEY` — https://console.groq.com/keys

## How it fits together

**Ingest** turns a directory of markdown into a versioned corpus. Every chunk
carries the character span it occupies in its source document, so a citation can
later resolve to exact characters rather than to a copy of the text that may
appear twice. The corpus version is a hash of the document contents and the
chunk parameters — stable across re-runs, different the moment either changes.
Every number this project publishes names the corpus version that produced it.

**Providers** put Gemini and Groq behind one `Protocol`, so no caller knows which
answered. Gemini goes first and Groq takes over on a *transient* failure — a rate
limit, a timeout, a 5xx. A rejected key is permanent and stops the run: failing
over on a bad key would quietly change the model behind a published number.
Responses are cached on disk keyed by the hash of provider, model and input, and
a per-run budget refuses the call that would exceed it before it is made.

**Retrieval** stores chunks and their vectors in pgvector and searches by exact
cosine distance over the whole corpus — no approximate index. At this scale the
speed is not needed, and an ANN index would fold its own recall loss into the
baseline without saying so. Adding one later becomes its own slice, with the
recall it costs measured against this number instead of assumed to be zero.

Two lexical rankers sit beside it, and both are named for what they are.
**Full-text** is Postgres `ts_rank_cd` over a generated `tsvector` column: cover
density, with no inverse document frequency — it is not BM25, and is not called
BM25. **BM25** is computed in memory from the lexemes that same column stores,
so the two differ in the ranking function and nothing else. **Hybrid** fuses
dense with either through reciprocal rank fusion, which combines rank positions
rather than scores that live on incomparable scales. **Rerank** (`--rerank`, on
any mode) scores the first stage's top twenty with a local cross-encoder —
`ms-marco-MiniLM-L6-v2`, run through ONNX Runtime at a pinned revision, no API
call — and keeps its top k.

**Eval** runs the golden set through the retriever and writes a frozen report to
`evals/reports/` holding both metrics and the entire configuration that produced
them: corpus version, retrieval configuration, model, dimension, chunk size,
overlap, k, and how many calls actually reached the network. Given
`--baseline <report>`, it also lists the questions that flipped in each
direction, and refuses to compare reports of a different corpus, `k` or
question set.

```bash
uv run python -m rageval.eval --mode hybrid --fuse-with bm25 \
  --baseline evals/reports/20260916T154348Z-26b03ce9a1c2c1d4.json
```

## The corpus

Fifty PEPs, converted from reStructuredText by `scripts/fetch_peps.py` and
committed, with `docs/corpus-sources.md` recording each source URL and the
sha256 of the bytes that were converted. They are public domain or CC0, dense
enough to be a real retrieval problem, and familiar enough that a bad result is
legible rather than abstract.

The source documents are in the repository because `data/` is not: a published
number whose input is fetched at eval time drifts silently when upstream edits a
file. The conversion rewrites no prose — headings, code blocks and inline markup
only.

## Working on it

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                 # no network, no database, no key
uv run pytest -m integration  # needs Postgres, and a key for the live provider tests
```

Every unit test runs offline. Indexing, retrieval and the eval harness are all
driven through protocols — `ChunkStore`, `EmbeddingProvider`, `QuestionRetriever` —
so the suite substitutes fakes rather than reaching for a service, and a failing
`pytest` is a real failure instead of a missing container.

Useful while poking at it:

```bash
uv run python -m rageval.retrieval --query "what is the maximum line length?" -k 5
uv run python -m rageval.retrieval --query "what is an enumeration?" --mode bm25
```

## Where this is going

Built in slices, each one measured before the next begins: **F0** a reproducible
skeleton, **F1** ingest with spans, **F2** the provider layer with caching and
failover, **F3** the pgvector baseline and its number.

**F4** measured lexical retrieval and fusion against that baseline. Its first
lexical ranker was Postgres full-text; measuring it exposed the missing IDF, so
the slice added real BM25 over the same lexemes rather than tuning around the
gap. The result is in the table: BM25 is far better than full-text, and fusing
it with dense gains one question of recall and gives back some rank. The local
reranker was then measured against that target — recovering the lost rank — with
its model, revision and depth fixed first. It missed: it found more passages and
ranked them worse, and that result is published as it came out. Dense stays the
default. **F5** resolves
citations to a character span in the source. **F6** turns the harness into a CI
gate that fails the build on a quality regression. **F7** puts a frontend on it
where clicking a citation highlights the span it came from.

The ordering is deliberate, and it is the argument the project is making:
measure the baseline before optimising anything, or you cannot prove the
optimisation helped.

## Design decisions

Every slice records what was chosen, what was rejected, and why, in
[`docs/adr/`](docs/adr/) — including the two conversion bugs found by reading the
corpus output instead of trusting it. The code carries no explanatory comments:
the reasoning lives in the ADRs, where it can be dated and revised.
