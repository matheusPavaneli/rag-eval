# ADR 0005 — Lexical retrieval, BM25 and rank fusion

Date: 2026-09-18 · Slice: F4 (first half) · Status: accepted

## Context

F3 published one number: dense retrieval at context recall @5 0.767 and MRR @5
0.603, on corpus `26b03ce9a1c2c1d4`. Seven of the thirty questions retrieved
nothing relevant, and they were mostly short and term-heavy — "What is an
enumeration?", "What does `yield from` do?". The hypothesis F4 tests is that a
lexical signal recovers what a dense embedding blurs.

What counts as a result was fixed before measuring, so it could not drift
afterwards: both metrics for every configuration, the questions that flipped in
each direction against the baseline, and a hybrid that fails to beat dense is
published the same way as one that succeeds. The golden set is the contract
ADR 0004 froze, and it is byte-identical to F3's.

## Decisions

### Postgres full-text first, named for what it is

The first lexical ranker is a generated `tsvector` column
(`to_tsvector('english', text)`) with a GIN index, ranked by `ts_rank_cd`. The
column is generated, so ingest and upsert did not change and the 1829 indexed
chunks were filled by the `ALTER TABLE` with no re-embedding. The query is the
question's lexemes joined by OR: `plainto_tsquery` joins them by AND, which for a
short question returns nothing more often than not.

ADR 0001 recorded that `ts_rank_cd` is not BM25 — it has no inverse document
frequency — and deferred the choice to F4. It is exposed as `fulltext` and never
called BM25.

Rejected: `pg_search` (ParadeDB). Real BM25 inside Postgres, but it replaces the
stock `pgvector/pgvector:pg17` image, and changing the database to test a ranking
function is the wrong way round.

### Measuring full-text exposed the missing IDF

Full-text scored 0.400 / 0.222, and its hybrid kept dense's recall while dropping
MRR to 0.439. The cause was checked rather than assumed: the lexeme `pep` occurs
in 524 of 1829 chunks, and without IDF it weighs as much as `length` in "What is
the maximum line length PEP 8 asks for?". The top full-text hit for that
question came from PEP 257.

The response was not to tune `ts_rank_cd`'s normalisation flags or the query
until the number moved — on thirty questions that is fitting the golden set. It
was to measure the ranker the README had promised, and keep the full-text row as
the evidence for why.

### BM25 over the same lexemes, so one variable changes

BM25 is computed in `rageval.retrieval.bm25` from the lexemes the `tsvector`
column already stores: `chunk_terms` unnests them with their counts, and
`query_terms` normalises the question through the same `english` configuration.
Full-text and BM25 therefore see identical tokens — same stemming, same
stopwords — and differ only in the ranking function. The gap between their rows
is what IDF and term saturation are worth on this corpus, and nothing else.

The formula is Lucene's: `idf = ln(1 + (N − df + 0.5) / (df + 0.5))`, which
cannot go negative, and `k1 = 1.2`, `b = 0.75`, the conventional defaults, fixed
before measuring and not tuned since. Document length is the number of lexemes
after stopword removal; that is consistent across every chunk and the average,
and a test computing the score by hand pins the definition. A test that disables
IDF fails, which is what keeps the `pep` failure from returning.

Rejected: `rank_bm25` or `bm25s`. Each is a dependency and a tokeniser of its
own, and a second tokeniser would have made the comparison with full-text
confound two changes. Fifty lines of scoring the author can derive at a
whiteboard are cheaper to own than a library that cannot be explained.

Rejected: stemming or stopword changes for BM25. `from` in `yield from` is an
English stopword and is gone before either ranker sees it; that limit is the same
for both rows and is recorded here rather than fixed with a tokeniser tuned after
the fact.

### Reciprocal rank fusion, with fixed parameters

Hybrid retrieval asks dense and the chosen lexical ranker for 20 candidates each
and fuses them by `Σ 1 / (60 + rank)`, deduplicating by chunk and breaking ties
by best single rank and then chunk id, so two runs give the same order.

RRF combines positions, not scores. Cosine similarity and either lexical score
live on scales that cannot be compared, and normalising them into one would be a
second model to justify. `rrf_k = 60` is the value from the original paper and
the candidate depth of 20 gives a chunk ranked eighth by one retriever room to
rise; neither was tuned.

### The report records the whole retrieval configuration

A retriever exposes one `RetrievalConfig`, a discriminated union of dense,
full-text, BM25 and hybrid — the hybrid variant carrying `rrf_k`, the candidate
depth and which lexical ranker it fused. A dense report cannot carry fusion
parameters and a hybrid report cannot omit its ranker. The F3 report, written
before the field existed, still loads and reads as dense; a test holds that.

`--baseline` lists the questions that went from a miss to a hit and back, and
refuses to compare reports whose corpus version, `k` or question set differ. On
thirty questions one question is 0.033 recall: an aggregate without its flips is
not evidence.

## Results

Corpus `26b03ce9a1c2c1d4`, k = 5, every run with no network call. Flips are
against the F3 dense report.

| Retrieval | Recall @5 | MRR @5 | Gained | Lost |
| --- | --- | --- | --- | --- |
| dense (F3, re-run) | 0.767 | 0.603 | — | — |
| full-text | 0.400 | 0.222 | q17 | q01 q03 q07 q08 q09 q12 q13 q18 q19 q24 q26 q28 |
| BM25 | 0.600 | 0.465 | q17 q27 | q08 q12 q13 q16 q19 q20 q25 |
| hybrid, dense + full-text | 0.767 | 0.439 | q17 | q19 |
| hybrid, dense + BM25 | 0.800 | 0.570 | q17 q27 | q19 |

Read plainly:

- **The hypothesis held for two questions.** "What is an enumeration?" (q17) and
  "What problem do context variables solve?" (q27) are found lexically and not
  by dense; BM25 ranks q27's passage first.
- **Fusion cost one.** "What operator does PEP 465 add for matrix
  multiplication?" (q19) is found by dense at rank two and by neither lexical
  ranker, and fusion pushes it out of the top five.
- **Recall rose by one question and rank fell.** Hybrid with BM25 ranks eight
  already-answered questions lower and three higher. That is not a win to claim
  on thirty questions; it is evidence that the two retrievers fail differently,
  which is the precondition for a reranker to be worth anything.
- **IDF is worth +0.200 recall and +0.243 MRR here**, the cleanest comparison
  in the slice, because nothing but the ranking function differs.
- **Five questions are missed by everything**: q11, q14, q21, q23, q29. They are
  the next slice's evidence, not this one's to fix.

The caveat ADR 0004 recorded applies with more force to lexical retrieval: each
question was drafted by locating a needle in the corpus with text search, so the
set shares vocabulary with its spans. Any lexical gain above may be partly that
bias. This set cannot separate the two.

## Consequences

- Dense stays the default mode: nothing in this slice beats it on both metrics.
- The reranker (F4's second half) now has a concrete target — the rank fusion
  gave back — and a candidate list that contains passages dense alone misses.
  It is gated on the same rules: both metrics, the flips, parameters fixed first.
- `ensure_schema` adds the `lexical` column to a table built before it. Dropping
  it (`ALTER TABLE chunks DROP COLUMN lexical`) leaves dense retrieval unaffected.
- BM25 is rebuilt in memory once per process from a single query. At 1829 chunks
  that is immediate; a much larger corpus would want it cached per corpus
  version, and a much larger `chunk_size` could meet Postgres' cap of 256
  positions per lexeme and undercount term frequency.
