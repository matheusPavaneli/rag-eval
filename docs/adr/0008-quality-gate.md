# ADR 0008 — A CI gate on lexical retrieval quality

Date: 2026-09-18 · Slice: F6 · Status: accepted

## Context

Every number in the README was produced by running the eval by hand and diffing
it against a frozen report with `--baseline`. That works when someone remembers
to do it. CI ran lint, types, the unit suite and the integration suite, and
nothing in it ingested the corpus or scored the golden set, so a change to
chunking, the lexeme config, the golden-set loader or the metric could merge
green with the published number changed.

The obvious gate — rerun the dense eval, or the answer eval, in CI — is not
available yet:

- CI has no provider key and no cache. A dense run would embed 1829 chunks cold
  against the Gemini free tier on every push.
- The embedding provider refuses to start without a key even when every vector
  is cached, so a committed cache would not help on its own.
- The chat cache sits behind failover (ADR 0007). A generated answer does not
  reproduce, so gating on one would gate on Gemini's quota that day.

BM25 needs none of that. It ranks from lexemes Postgres derives from the chunk
text, ties break by chunk id, and the eval already skips the embedding provider
for lexical modes. Checking the code showed one gap in that: *indexing* still
needed a vector for every chunk, because `chunks.embedding` was `NOT NULL` and
the index command always embedded.

## Decisions

### Gate the mode that is deterministic without a key

The gate ingests the committed `documents/`, stores the chunk text of corpus
`26b03ce9a1c2c1d4` without embedding it, runs the BM25 eval, and compares the
result with the frozen BM25 report. No key enters CI and no call leaves it: the
eval prints `0 call(s) reached the network`.

BM25, not full-text: both rank the same lexemes, so gating both would catch
nothing extra, and BM25 is the one the hybrid uses.

### A chunk may be stored without a vector

`embedding` is now nullable. `rageval.retrieval --lexical-only` stores each
chunk's text with no vector, and does nothing for a chunk that is already
stored, so it can never erase a real vector. Dense search skips chunks without a
vector, and the eval's partial-index check counts only chunks with a vector for
dense and hybrid modes: over a lexical-only index a dense eval refuses with
"1829 chunks but 0 are indexed" instead of ranking nothing.

Rejected: storing a stand-in vector of zeros. It needs no schema change, but a
lexical-only pass over the real local table would overwrite the embeddings behind
the published dense numbers, since indexing updates on conflict, and a zero vector
has no cosine distance. NULL plus "do nothing on conflict" cannot corrupt
anything.

### Zero tolerance, in both directions, per question

`--fail-on-change` exits 1 when any question's recall or reciprocal rank
differs from the baseline, or either aggregate does, or the baseline was
produced by a different retrieval config. Values are compared exactly.

- **Zero, not a margin.** A margin exists to absorb variance, and BM25 over a
  fixed corpus has none. The parent frame's "thresholds with margin" was written
  for generated answers, which this gate does not touch.
- **Both directions.** A change that raises the number is as suspect as one that
  lowers it: an off-by-one in an overlap predicate makes retrieval look better.
  An intended improvement is a PR that also commits a new baseline, where the
  change is visible in the diff.
- **Per question, not flips.** The existing `compare()` reports questions that
  went from hit to miss or back. It cannot see a hit that dropped from rank 1 to
  rank 5. In the planted `simple` regression below, flips named 7 questions;
  the gate named 13.

Rejected: dense in CI by committing ~33 MB of vectors, or by an Actions cache.
The first needs a cache-only embedding path that does not exist yet; the second
evicts after seven idle days and then calls the free tier cold, which is a red
build caused by quota. Rejected for now: an LLM judge in CI. It is a second
model with an unmeasured error rate, and it would put provider quota back into
the gate.

## Results

Four regressions were fixed in the frame before the gate existed, then planted
one at a time. Each ran the gate's exact commands against a fresh database and
a fresh corpus directory, and ran the unit suite. They ran locally, not as CI
runs: the workflow runs on pull requests and on `main`, so throwaway branches
would not have produced a run.

| Planted regression | Gate | Unit suite |
| --- | --- | --- |
| none (control) | pass | pass |
| chunk overlap 150 → 0 | **fail**: corpus version changes to `a33c34b0c987fe1d`, 1544 chunks; `26b03ce9a1c2c1d4` is not found | pass |
| text-search config `english` → `simple` | **fail**: 13 questions changed, recall 0.600 → 0.500, MRR 0.465 → 0.281 | pass |
| overlap predicate `<` → `<=` | pass: no BM25 result changes | **fail**: `test_a_chunk_ending_where_the_span_begins_shares_no_character` |
| q01's golden span shifted by 40 characters | pass | pass |

- **The gate catches what the unit suite does not.** The two regressions that
  change retrieval on real text passed every unit test. That is the case for
  building it; had the gate caught only what the tests already caught, it would
  have been ceremony.
- **The overlap regression fails by refusal, not by a list of questions.** A
  different chunking is a different corpus version, and the gate names the
  version it expected. That is the right failure: the numbers are not
  comparable, so the gate does not compare them.
- **The `<=` plant is a no-op on this data.** No retrieved chunk ends exactly
  where a golden span begins, so no metric moves. The unit test that pins the
  boundary is what protects it.
- **The span shift was caught by nothing.** Moving q01's span 40 characters
  keeps it inside the same 1000-character chunk, so every retrieval metric is
  unchanged. The gate measures at chunk granularity; a ground-truth edit below
  that resolution is invisible to it. The citation metric would see it, and it
  is not gated. Until it is, the golden set is protected by the rule that it is
  never edited after measuring, and by that file showing up in a PR's diff.
  *Closed by [ADR 0011](0011-answer-gate.md): every report records a golden-set
  digest, and the same shift now fails all four gates.*

## Consequences

- **A green check protects the lexical path only.** Ingest, chunking, the
  lexeme config, the golden-set loader, the metrics and BM25 are covered. The
  vector store's dense query, the embedding provider, fusion, the reranker and
  generation are not. The README says so next to the gate.
- **Every deliberate lexical change now needs a new baseline.** The PR commits
  the new report and points the workflow's `BASELINE` at it, so the number change
  is a line in the diff that has to be explained.
- **Existing databases lose a constraint.** `ensure_schema` drops `NOT NULL`
  from `chunks.embedding`. Every row written before this change has a vector,
  so nothing reads differently. It is reversible by hand: delete the rows with no
  vector, then set the constraint again.
- **Ingest reproduces across operating systems.** The first CI run, on an
  Ubuntu runner, rebuilt corpus `26b03ce9a1c2c1d4` with the same 1829 chunks as
  the Windows machine that produced the baseline, and the gate reported no
  drift with zero network calls. The repository forces LF line endings, the
  likeliest reason the rebuild matches.
- Next, in order: a dense gate (an embedding path that serves from cache without
  a key), the chat cache in front of failover, then a gate on answers.
