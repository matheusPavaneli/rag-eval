# rageval

A retrieval-augmented generation pipeline that measures its own quality, on a
corpus you can read, with a number you can reproduce.

The interesting part of a RAG system is not that it answers. It is whether the
passage it answered from was the right one — and most RAG projects never find
out. This one starts there: a golden set of thirty questions, ground truth
recorded as character spans in the source documents, and a baseline published
before anything is optimised.

**See it:** [matheuspavaneli.github.io/rag-eval](https://matheuspavaneli.github.io/rag-eval/)
— each of the thirty answers beside its source PEP. Select a citation and the
characters it resolved to are highlighted, next to the span the golden set
expected.

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

### Citations

Each answer cites passages by quoting them; code finds every quote in the chunk
it cites and turns it into a character span in the source document. A quote that
is not there verbatim — only whitespace is forgiven — does not count.

| Date | Configuration | Citation hit @5 | over retrieved gold spans | Citations resolved |
| --- | --- | --- | --- | --- |
| 2026-09-18 | groq/openai/gpt-oss-120b (29), gemini/gemini-3.5-flash (1), prompt `682ec1326b2f`, over dense retrieval (gemini-embedding-001), k=5, corpus `26b03ce9a1c2c1d4` | 0.533 | 0.696 (23) | 0.981 |

- **Quotes resolve.** 51 of 52 were found verbatim; the one that was not joined
  two sentences with `...`. A resolved citation averages 104 characters, against
  847 for the chunk it came from.
- **The hit rate is a lower bound.** A hit means the citation overlaps the one
  golden span recorded for the question. In at least four of the seven retrieved
  misses the model cited a different sentence of the same PEP that also answers
  it (q19, q20, q26, q30).
- **The model never abstained.** All seven questions whose gold span was not
  retrieved got a confident answer citing something else.
- **The number is mostly Groq's.** Gemini was rate-limited from the first
  question and failover answered 29 of 30; the row says so. The run replays
  field for field with 0 network calls, with or without keys: chat is cached in
  front of failover, so a recorded answer is served whichever provider is up.
  Details in [ADR 0007](docs/adr/0007-citations.md) and
  [ADR 0010](docs/adr/0010-chat-cache.md).

```bash
uv run python -m rageval.eval --answer
uv run python -m rageval.retrieval --query "What is the maximum line length PEP 8 asks for?" --answer
```

### What CI protects

Every pull request and every push to `main` rebuilds the corpus from
`documents/`, runs the BM25, dense and hybrid evals and the answer eval, and
compares each with its frozen report. Any difference fails the build: a question
whose recall or rank moved in either direction, an answer whose text, citations
or spans changed, any aggregate, or a report from another configuration or
another golden set. No key is involved and no call reaches a model provider.

Dense needs vectors, and CI has no key to make them. Without a key the embedding
provider serves only from its cache and refuses on a miss. CI fills that cache
from a snapshot of the 1859 vectors the corpus and golden set need, attached to
the release
[`vectors-26b03ce9a1c2c1d4`](https://github.com/matheusPavaneli/rag-eval/releases/tag/vectors-26b03ce9a1c2c1d4)
and pinned by sha256 in the workflow.

The answers are replayed from
[`evals/answers/`](evals/answers/), the 30 recorded model responses committed as
a fixture: the model's text is fixed, and everything the pipeline derives from
it is compared. Every report records a digest of the golden set it was scored
against, so an edited golden span is refused rather than passed.

A green check covers ingest, chunking, the lexeme config, the golden set and its
loader, the metrics, BM25, dense retrieval, fusion, and the answer pipeline:
parsing, citation resolution and spans. It does **not** cover the reranker,
whether a live model would still answer the same way, or whether an answer is
correct. Ubuntu reproduced the Windows baselines exactly. Across the planted
regressions, the lexical gate caught two that the unit suite missed. The dense
gate refused a query embedded as a document rather than calling the network,
and it caught a fusion bug that *raised* MRR. The answer gate refused a changed
prompt without a call and caught a golden-span shift that nothing caught
before. Details in [ADR 0008](docs/adr/0008-quality-gate.md),
[ADR 0009](docs/adr/0009-dense-gate.md) and
[ADR 0011](docs/adr/0011-answer-gate.md).

```bash
uv run python -m rageval.ingest documents/
uv run python -m rageval.retrieval --corpus-version 26b03ce9a1c2c1d4 --lexical-only
uv run python -m rageval.eval --mode bm25 --corpus-version 26b03ce9a1c2c1d4 \
  --baseline evals/reports/20260918T175145Z-26b03ce9a1c2c1d4-bm25.json --fail-on-change

# dense and hybrid, with no key: fill the cache from the pinned snapshot
gh release download vectors-26b03ce9a1c2c1d4 --pattern embeddings-26b03ce9a1c2c1d4.jsonl.gz
uv run python -m rageval.vectors import embeddings-26b03ce9a1c2c1d4.jsonl.gz \
  --sha256 49f22cf2bc87a061005adce96737f5d22e7803bca063cbeba1b362bbde4bcc82
uv run python -m rageval.retrieval --corpus-version 26b03ce9a1c2c1d4
uv run python -m rageval.eval --mode dense --corpus-version 26b03ce9a1c2c1d4 \
  --baseline evals/reports/20260918T175148Z-26b03ce9a1c2c1d4-dense.json --fail-on-change

# answers, with no key: replay the recorded answers
uv run python -m rageval.eval --answer --corpus-version 26b03ce9a1c2c1d4   --replay-answers evals/answers/26b03ce9a1c2c1d4-682ec1326b2f.jsonl   --baseline evals/reports/20260918T175155Z-26b03ce9a1c2c1d4-answer-dense.json --fail-on-change
```

A change to chunking or to the embedding model is a new corpus version: embed
it locally with a key, run `python -m rageval.vectors export`, publish a new
`vectors-<corpus>` release, and update the pins and baselines in `ci.yml` in the
same PR. A change to the prompt, to retrieval or to a chat model is recorded
again the same way: answer live with keys and `--record-answers`, then commit
the new fixture and answer baseline.

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
Embeddings are cached on disk keyed by the hash of provider, model and input;
chat answers are cached in front of failover, keyed on the prompt and the
configured chain, so a replay does not depend on which provider is up. A
per-run budget refuses the call that would exceed it before it is made.

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

**Answer** (`--answer`) sends the question and the top k chunks, numbered, to the
chat provider under one fixed system prompt, and asks for JSON: an answer and
citations, each a passage number and a verbatim quote. The model's word is not
taken for where the quote is. `resolve` finds it in the passage it cites, after
collapsing whitespace and nothing more, and adds the chunk's offset to get the
span in the document; anything else is recorded as unresolved with a reason. The
answer report keeps every citation, the model that answered each question, and a
hash of the prompt.

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

The page lives in `web/` (Astro, Node 24, pnpm) and is built from data the
exporter writes; it ships no JavaScript:

```bash
uv run python -m rageval.site --out web/src/data
pnpm --dir web install
pnpm --dir web dev            # or: build, then test (checks every highlighted span)
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
default.

**F5** added the first answer, with citations the model quotes and code locates:
almost every quote resolved, and about half the answers cite the exact golden
span — a lower bound, since the golden set records one span where the source
often has several. **F6** made the harness a CI gate, starting where the number
is deterministic without a key: BM25, compared per question with zero tolerance.
A follow-up gated dense and hybrid the same way, from pinned cached vectors
with no key. The chat cache then moved in front of failover so a recorded answer
replays, and the answers joined the gate from a committed fixture, with every
report pinned to its golden set. A correctness judge and abstention were left
out on purpose: each needs live calls and would replace the published number.
**F7** put a static page on it: each answer beside its source, a citation
highlighting the exact characters it resolved to, cut in Python so the offsets
stay code points, and checked in CI against the report for all 51 resolved
citations ([ADR 0012](docs/adr/0012-citation-viewer.md)). Free-form questions
against a live retriever were left out: they need a public endpoint holding a
key.

The ordering is deliberate, and it is the argument the project is making:
measure the baseline before optimising anything, or you cannot prove the
optimisation helped.

## Design decisions

Every slice records what was chosen, what was rejected, and why, in
[`docs/adr/`](docs/adr/) — including the two conversion bugs found by reading the
corpus output instead of trusting it. The code carries no explanatory comments:
the reasoning lives in the ADRs, where it can be dated and revised.
