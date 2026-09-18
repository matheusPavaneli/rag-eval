# ADR 0009 — Gate dense and hybrid retrieval from pinned cached vectors

Date: 2026-09-18 · Slice: F6 follow-up (RAG-9) · Status: accepted

## Context

ADR 0008 put BM25 behind a CI gate and said what a green check did not cover:
the dense numbers, fusion, the reranker and the answers. The dense row is the
README's headline number (recall@5 0.767, MRR@5 0.603), so the gap was the
largest one the check left.

Two things kept dense out of CI, and neither was the eval itself:

- **The embedding provider could not be built without a key.**
  `build_embedding_provider` raised before the cache was consulted, even when
  every vector it would be asked for was cached.
- **CI had no cache, and a miss went to the network.** `.cache/` is gitignored.
  `CachedEmbeddingProvider` fetches whatever it does not hold, so a cold run
  would embed 1829 chunks against the free tier.

The dense query itself needed nothing: it is an exact scan ordered by cosine
distance, with no HNSW or IVFFlat index, so there is no approximate search whose
results depend on how an index was built.

## Decisions

### No key means cache-only, not a flag

Without `RAGEVAL_GEMINI_API_KEY` the builder now returns the same cached provider
over a stand-in that raises `CacheMissError` for any text it is asked to embed.
A hit is served; a miss fails with the count and the variable that would fix it.
It is a permanent error, so indexing does not retry it. With a key nothing
changes.

Rejected: an `--offline` flag. The only situation in which the cache must be the
sole source is the one where there is no key, and a flag would allow the
contradictory "offline with a key" and "online without one". The cost of the
implicit rule: a local run that forgot its `.env` now fails at the first miss
rather than at construction, with a message naming the variable.

A blank key variable now reads as no key. `.env.example` ships them blank, so a
copied template used to send an empty key to the API.

### The vectors travel as a release asset pinned by sha256

`python -m rageval.vectors export` writes the cache entries one corpus version
needs: every chunk as a document, every golden question as a query. That is 1859
entries, sorted by digest, as JSON lines in gzip with a zero timestamp, so the
same cache always produces the same bytes. Two exports gave the same sha256. The
file is 12.6 MB and is attached to the release
[`vectors-26b03ce9a1c2c1d4`](https://github.com/matheusPavaneli/rag-eval/releases/tag/vectors-26b03ce9a1c2c1d4).

`ci.yml` pins the sha256. `import` hashes the file before it parses a line and
refuses any other file; it validates every entry's digest, model and dimension
before writing any of them.

Rejected:

- **The 33 MB JSON cache in the repository.** About fifty times the size of the
  whole git history, and history keeps every copy.
- **A committed snapshot.** 12.6 MB in history, again for every re-embedding.
- **Git LFS.** Keeps history small, but every CI checkout counts against the
  free tier's 1 GB of monthly bandwidth.
- **An Actions cache.** ADR 0008 rejected it because eviction led to cold calls
  to the free tier. With a cache-only provider eviction would be a refusal
  instead, but still a red build caused by something other than code.

A release is public and permanent, and it adds a download from GitHub to the
gate. That download involves no key and reaches no model provider; the pin makes
its content as fixed as a committed file.

### Float64, as cached

The snapshot keeps each value exactly as the cache holds it, so the literal
`VectorStore` sends to Postgres is byte-identical to the one sent on the machine
that produced the baselines. Storing float32 would halve the file. Since
pgvector stores float32 anyway, that would probably change nothing, but
"probably" would then need an argument about double rounding. The release is
outside git history, so the extra bytes cost nothing.

### Hybrid is gated with dense

Hybrid-bm25 uses the same vectors and the same lexemes, so it takes one more
step and no new input, and fusion moves from "not covered" to covered. The
reranker stays out: an 88 MB model download on every run, for a mode that
lowered MRR.

## Results

**Ubuntu reproduces the Windows baselines exactly.** The first CI run imported
the snapshot, indexed 1829 chunks and reported no drift for BM25, dense (0.767 /
0.603) or hybrid-bm25 (0.800 / 0.570), each with `0 call(s) reached the network`
and no key in the job. The gate job went from 20 s to 29 s.

Before the run, the smallest gap between adjacent dense scores in any
question's top six was 2.8 × 10⁻⁵ (q11); the median was 9.4 × 10⁻⁴. Float32
rounding sits around 10⁻⁷, so a flip caused by CPU differences needed to be
about two hundred times larger than it could be.

Four regressions were fixed in the frame before the gate existed, then planted
one at a time. Each ran the gate's commands with no key, against an empty cache
filled only from the snapshot and a fresh database. The unit suite also ran for
each.

| Planted regression | Gate | Unit suite |
| --- | --- | --- |
| none (control, before and after) | pass | pass |
| distance `<=>` (cosine) → `<->` (L2) | pass: no result changes | pass |
| question embedded as `document`, not `query` | **fail** by refusal: `no cached embedding for 1 text(s)` | **fail** |
| vector literal rounded to 4 decimals, re-indexed | pass: no result changes | pass |
| fusion computes `1 / rank`, ignoring `rrf_k` (config still says 60) | **fail**: 8 hybrid questions changed, recall 0.800 → 0.767, MRR 0.570 → 0.628; dense unchanged | **fail** |

- **The L2 plant is a no-op, and the frame was wrong about why it would not
  be.** The frame expected truncated 768-dimension vectors to have different
  lengths. The Gemini adapter normalises every vector to unit length
  (`_unit` in `gemini.py`), and for unit vectors ‖a − b‖² = 2 − 2 cos(a, b), so
  L2 and cosine rank identically. It is a real regression only for vectors that
  are not unit length. Neither the gate nor the unit suite can see it on this
  data.
- **Rounding to four decimals changes nothing either.** It does not move any
  score far enough to reorder a top five. The 2.8 × 10⁻⁵ margin that absorbs
  this plant is the same one that absorbs cross-CPU float differences.
- **The query-task plant fails the way it should.** A question embedded as a
  document hashes to a key the snapshot does not hold, so the gate refuses
  instead of reaching the network. This is the plant that matters: it shows a
  miss cannot turn into a call.
- **The fusion plant makes the number look better.** MRR rises from 0.570 to
  0.628, and the report still records `rrf_k` 60, so the README row it prints
  would publish a better number under a configuration that did not produce it.
  A gate that only failed when a number fell would have passed it. This is the
  case ADR 0008 gave for checking both directions, and it has now happened.
- **This time the unit suite caught everything the gate caught.** The two
  plants that changed results each broke a unit test as well. For these four
  plants, the gate's value is that it ran on real vectors and confirmed the
  published numbers on another operating system, not that it caught a
  regression the tests missed. ADR 0008's lexical plants were the opposite case.

## Consequences

- **A green check now covers dense retrieval and fusion**, as well as everything
  ADR 0008 listed. It still does not cover the reranker, generation or the
  answer metrics, or a golden-span edit smaller than a chunk.
- **Re-embedding is a release.** A new chunking or embedding model is a new
  corpus version: index and evaluate locally with a key, export, publish a new
  `vectors-<corpus>` release, and update `VECTORS_*`, `CORPUS` and the three
  baselines in `ci.yml` in the same PR.
- **The gate depends on a release existing.** Deleting it turns the dense steps
  red with a download error. BM25 runs first, so its result is never hidden.
- **The vectors are public.** They are Gemini outputs for public-domain and CC0
  PEPs and the project's own questions, with nothing secret or personal in them.
- Next: the chat cache in front of failover, then a gate on answers.
