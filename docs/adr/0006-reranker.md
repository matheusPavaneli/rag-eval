# ADR 0006 — A local cross-encoder reranker

Date: 2026-09-18 · Slice: F4 (second half) · Status: accepted

## Context

ADR 0005 left F4 with a concrete target. Hybrid retrieval (dense + BM25) raised
context recall @5 from 0.767 to 0.800 but lowered MRR @5 from 0.603 to 0.570:
it put the right passage in the top five more often and at rank one less often.
A reranker reads the question and each candidate together instead of comparing
two independent vectors, so it is the standard answer to "the right passage is
in the list but not at the top". This slice measures whether it is the answer
here.

What counts as a result was fixed before measuring, as in ADR 0005: both
metrics, the questions that flipped in each direction, and a reranker that
misses its target is published the same way as one that hits it. The model, its
revision, the candidate depth and both first stages were written into the plan
before any number existed, and none of them changed after.

## Decisions

### A cross-encoder, not a second retriever

`RerankRetriever` wraps any first stage. It asks the first stage for 20
candidates, scores each `(question, chunk text)` pair once, sorts by that score
with the first-stage rank breaking ties, and returns the top k. It reads nothing
from the store — candidates already carry their text — and it calls no
provider. Reranking a reranker is refused.

Twenty is `retrieval_candidates`, the depth the hybrid already fuses, so every
first stage hands the reranker the same number of passages and there is no
second knob to tune.

The report records a `RerankConfig` nesting the first stage's own config, so a
reranked hybrid row still says which lexical ranker was fused, and the file is
named `rerank-<first stage>`.

### `ms-marco-MiniLM-L6-v2`, through ONNX Runtime, at a pinned revision

The model is `cross-encoder/ms-marco-MiniLM-L6-v2`: 22 M parameters, a 91 MB
fp32 ONNX export published in the model's own repository, and a max input of 512
tokens; a 1000-character chunk and a short question should stay under it, and
longer pairs are truncated rather than refused (not measured). It is downloaded
from Hugging Face at commit `233902d25c440f23af6f7d6e94d2946bac0bee0a` into the
gitignored `.cache/models`. Pinned to a commit, `hf_hub_download` resolves from
the cache without a request: with the Hub endpoint pointed at a closed port, the
second load scored identically. The eval stays offline after the first run.

Four runtime dependencies, each for one reason:

- `onnxruntime` — runs the model on CPU without PyTorch.
- `tokenizers` — the model's own `tokenizer.json`, so tokenisation is the one it
  was trained with, not a reimplementation.
- `huggingface-hub` — the download, pinned to a commit SHA and cached.
- `numpy` — the tensors ONNX Runtime takes and returns; already installed by it,
  declared because the code imports it.

Rejected: `sentence-transformers`. It is the reference `CrossEncoder`, and it
pulls PyTorch — over a gigabyte of wheels to run a 91 MB model.

Rejected: `fastembed`. One direct dependency instead of four, but it brings
Pillow, loguru, mmh3, a stemmer and requests with it, and hides which revision
of a model it fetched. A reproducible number needs the revision in the report.

Rejected: `mxbai-rerank-xsmall-v1` (284 MB) and `bge-reranker-base` (1.1 GB
ONNX). Possibly stronger, but the second is outside the 90–300 MB the F4 frame
budgeted, and choosing between models after seeing their numbers on thirty
questions is fitting the golden set. One model was chosen before measuring.

## Results

Corpus `26b03ce9a1c2c1d4`, k = 5, no provider call in any run. Flips are against
the F3 dense report.

| Retrieval | Recall @5 | MRR @5 | Gained | Lost |
| --- | --- | --- | --- | --- |
| dense (F3) | 0.767 | 0.603 | — | — |
| hybrid, dense + BM25 | 0.800 | 0.570 | q17 q27 | q19 |
| rerank over dense | 0.833 | 0.490 | q11 q17 q27 | q08 |
| rerank over hybrid, dense + BM25 | 0.800 | 0.451 | q11 q17 q27 | q08 q20 |

The ceiling — what the reranker was handed — measured by running each first
stage at k = 20 (`evals/reports/20260918T141703Z-…-dense.json` and
`20260918T141705Z-…-hybrid-bm25.json`):

| First stage | Recall @20 |
| --- | --- |
| dense | 0.933 |
| hybrid, dense + BM25 | 0.900 |

Read plainly:

- **The target was missed.** MRR fell, over both first stages, to below every
  row it was meant to improve on. The reranker did not recover the rank fusion
  lost; it lost more.
- **Recall rose.** Over dense, three passages dense ranked 6th to 20th reached
  the top five — q11 "What is a build backend?", q17, q27 — and one fell out
  (q08). 0.833 is the highest recall in the project.
- **The rank was lost inside the right document.** Over dense, the reranker
  lowered the reciprocal rank of fourteen questions and raised six. In the
  demotions checked by hand, the chunk it put first came from the correct PEP
  and was not the annotated span. Sometimes it is an answer the golden set does
  not count: for q04 "Which quotes should a docstring use?" it chose PEP 257's
  "Triple quotes are used even though the string fits on one line". Sometimes it
  is keyword attraction: for q09 it chose a passage about the *pre-release*
  segment, for q28 one about standard collections that never says what replaces
  `typing.List`.
- **Fusion does not help the reranker.** Its input over the hybrid is smaller
  (0.900 vs 0.933 at 20: q23 "What is a protocol class?" is at dense rank 20 and
  fusion pushes it out), and the result is worse on both metrics.
- **Two questions are unreachable**: q14 "What does `yield from` do?" and q21
  are in no first stage's top twenty. q23 and q29 are in dense's top twenty and
  the reranker did not lift them into the top five. Of the five questions every
  F4 mode missed, the reranker recovered one, q11.

Two biases make the MRR drop partly, not wholly, an artefact. The golden set
records one span per question (ADR 0004); a second passage that also answers
scores zero. And `ms-marco` was trained on web search queries against web
passages, not on specification prose. This set cannot separate how much of the
drop is each; q09 and q28 show that some of it is the model being wrong.

Eval time: each rerank run takes about 26 seconds end to end, against 1.5 to 2.5
without — model load plus 600 pair scores on CPU. Inside "a short break". Run
twice, all four reports reproduced question by question.

## Consequences

- Dense stays the default mode. No F4 configuration beats it on both metrics.
- `--rerank` stays in the harness as a measured option: it is the best recall
  configuration, and a later golden set with multiple accepted spans per question
  is the way to find out how much of its MRR loss is real.
- The F4 answer, for the interview: a lexical signal recovers two of seven
  misses; a reranker recovers a third and ranks everything else worse; neither
  is worth shipping as the default on this evidence.
- Not tried, deliberately: a second reranker model, a different candidate depth,
  or score blending between the first stage and the cross-encoder. Each is a
  parameter chosen after seeing this number.
- The CI integration job downloads the 91 MB model on every run, because the
  test runs in a temporary directory. If that becomes slow or flaky, cache it.
- The report's `network_calls` counts provider calls. The one-time model
  download is not one, and is not counted.
