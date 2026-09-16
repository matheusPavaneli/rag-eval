# ADR 0004 — A measured retrieval baseline

Date: 2026-09-10 · Slice: F3 · Status: accepted

## Context

Until this slice the project had no number. Every claim about retrieval was an
opinion, and F4's hybrid retrieval and reranking would have had nothing to be
measured against. This slice exists to produce one figure that a later change
can be compared to, and to make the way it was produced auditable.

Measuring first is deliberate. Building BM25 or a reranker before a baseline
existed would have made it impossible to say either helped.

## Decisions

### The corpus is 50 Python Enhancement Proposals, committed

`documents/` holds fifty PEPs converted from reStructuredText to markdown by
`scripts/fetch_peps.py`, with `docs/corpus-sources.md` recording each source URL
and the sha256 of the bytes that were converted.

Rejected: keeping the corpus out of the repository and fetching it at eval time.
`data/` is gitignored and the corpus is derived, so the *source* documents have
to be committed or the published number cannot be reproduced by anyone. A
corpus fetched at eval time also drifts silently: upstream edits a PEP, the
number moves, and nothing in the repository explains why.

Rejected: this repository's own documentation as the corpus. About ten short
files is too small for recall @5 to separate a good retriever from a lucky one,
and a project that measures itself on itself invites the obvious objection.

PEPs were chosen because they are public domain or CC0 (PEP 1), so redistributing
them here needs no further condition; because they are dense technical prose with
factual, checkable claims; and because an interviewer knows the content, which
makes a bad retrieval result legible rather than abstract.

The conversion touches structure only — headings, code blocks, admonitions,
inline roles — and rewrites no prose. Two conversion bugs were found by reading
the output rather than by trusting it: reStructuredText's anonymous-reference
syntax closes on a backtick, so `` `a` and `__b__` `` had the leading
underscores of the second name swallowed, corrupting nine dunder names; the same
pattern ate a literal underscore in PEP 503's normalisation rule. Both are fixed
with a lookahead, and the fix is the reason `corpus_version` changed twice
during the slice.

### The corpus is `documents/`, and provenance lives outside it

`docs/corpus-sources.md` is not under `documents/` because ingest treats every
markdown file there as corpus content. The first ingest of this slice produced
51 documents, not 50: a table of hashes had become a searchable document. A
metrics harness that quietly measures its own bookkeeping is worse than one that
measures nothing.

### Ground truth is a character span, not a chunk id

A golden question names one or more supporting spans as `source_path`,
`start_char`, `end_char`. A retrieved chunk counts as relevant when it comes from
that document and its own span overlaps, half-open: a chunk ending exactly where
the supporting span begins shares no character and is not a hit.

Rejected: recording the id of the chunk that should be retrieved. A chunk id is
a function of `chunk_size` and `overlap`, so the golden set would have to be
rewritten every time chunking changed — and changing chunking to see whether the
number moves is precisely what F4 is for. A set that cannot survive the
experiment it exists to judge is not ground truth.

The cost accepted: spans are meaningful only against the *normalised* document
text, which ties the golden set to the normalisation contract fixed in ADR 0002.
`load_golden_set` therefore resolves every span against the corpus at load time —
an unknown `source_path`, a span past the end of its document, or a duplicate
question id is an error naming the question, never a row silently skipped. A
harness that quietly drops questions it cannot resolve reports a better number
than it earned.

### Exact search, no ANN index

Retrieval orders by pgvector's `<=>` over every row of the corpus. There is no
HNSW or IVFFlat index.

Rejected: building an approximate index now. An ANN index trades recall for
speed, and at this scale — 1829 chunks — the speed is not needed. Worse, its
recall loss would be folded into the baseline without being named, and F4 could
then "improve" retrieval by doing nothing but raising `ef_search`. The baseline
has to be the ceiling that later slices are measured against.

The cost accepted: this does not scale. When the corpus is large enough for exact
search to hurt, adding an index becomes its own slice, with the recall it costs
measured against this number rather than assumed to be zero.

The `vector` column is declared at a fixed width, and the store asserts that
width before it writes. A table built at another dimension does not fail loudly:
pgvector rejects or truncates, and a truncated vector would degrade the number
without ever raising.

### The golden set is model-drafted and human-verified, and says so

Thirty questions were drafted against the corpus, each anchored to a passage
located by searching the document text and widened to the paragraph containing
it. Every span was then read against its source before the set landed;
roughly half the initial anchors were re-pointed because they had settled on a
heading, a code sample, or a sentence that mentioned the topic without answering
the question.

The drafting script is committed as `scripts/build_golden_set.py`, holding the
question, the answer and the needle that locates it. It is the provenance of the
set: the spans are offsets into documents that changed three times during this
slice, and a file of bare offsets nobody can regenerate is not evidence of
anything. Re-running it against the current corpus reproduces
`evals/golden-set.jsonl` byte for byte.

This is worth stating plainly rather than burying: a set written with knowledge
of the corpus flatters recall relative to questions a stranger would ask. The
number below is an upper bound on this corpus, not an estimate of production
quality.

Rejected: an unverified generated set. The verification is the only thing that
separates this from a system grading its own homework, and it is cheap at thirty
questions.

Rejected, for now: a set written entirely by hand with no drafting. Stronger,
and the right move if this number ever has to carry weight beyond a
conversation. At thirty questions the cost was not worth the marginal
credibility over verified drafting.

### The budget was right and the rate limit was not where it was expected

F2's budget caps calls and input characters per run. Neither was the binding
constraint. The first full index answered `HTTP 429` on its first batch: the
free tier meters *tokens per minute*, so a corpus of any size is refused partway
through however small the batches are.

`index_corpus` therefore retries a transient failure with exponential backoff and
gives up after a configured number of attempts, naming the batch. A permanent
failure is not retried — waiting on a rejected key only spends the wall clock.
Because F2 caches per text rather than per batch, a retried or resumed run
re-sends only what has not landed yet, which is what makes a fifteen-minute first
index a one-off rather than a recurring tax.

### Both numbers, or neither

The report carries `corpus_version`, embedding model, dimension, `chunk_size`,
`overlap`, `k`, the question count, the run date and how many calls reached the
network, alongside the two metrics. `EvalReport` is frozen and every run is
written to `evals/reports/`, which is committed: `data/` is gitignored, so a
report written there could not stand behind a number published in the README.

Context recall @k answers "did we retrieve the evidence at all". MRR @k answers
"how far down the list was it". Recall alone would hide a retriever that finds
the right passage at rank five every time; MRR alone would hide one that ranks
its few hits well and misses most questions. Publishing one without the other
invites the wrong conclusion.

Recall is counted over supporting spans, not over retrieved chunks. Retrieving
the same passage five times answers the question once.

## Consequences

- `evals/golden-set.jsonl` is now the contract F4 must not change. Editing a
  question invalidates the comparison it exists to enable.
- Every unit test still runs with no Postgres, no network and no key: indexing,
  retrieval and the harness are all driven through `ChunkStore`,
  `EmbeddingProvider` and `QuestionRetriever` protocols, with the pgvector store
  covered behind the `integration` marker.
- The number in the README names its configuration. A figure without the corpus
  version, the model and `k` is not comparable to anything, including a later run
  of this same harness.
- F4 inherits a baseline it can be measured against, and the obligation to report
  the same two numbers on the same golden set and corpus version.
- F5's citations already have what they need: a retrieved chunk carries the
  character span it occupies in its source document.
