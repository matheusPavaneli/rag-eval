# ADR 0011 — Gate the answers from recorded answers, and pin the golden set

Date: 2026-09-18 · Slice: RAG-11 · Status: accepted

## Context

After [ADR 0009](0009-dense-gate.md), a green check covered retrieval (BM25,
dense and fusion) but not the answers. [ADR 0010](0010-chat-cache.md) made the
published answer run replay with no network calls, but only on the machine that
held the cache: the 30 recorded answers lived in a gitignored directory, and the
eval CLI refused `--answer` together with `--baseline`.

[ADR 0008](0008-quality-gate.md) left a second gap open. Moving q01's golden
span by 40 characters was caught by nothing, because no report recorded which
golden set it had been scored against.

This is the last backend slice before the frontend (F7). Its scope was chosen
on 2026-09-18: the answer gate and what it needs, and nothing more. There is no
correctness judge: it would cost 30 or more live calls on a rate-limited free
tier, publish a new number, and its verdicts would need the same cache and
gate. Abstention is also out: it changes the prompt, so all 30 questions would
be answered again live and the published 0.533 replaced. Both are left unbuilt
on purpose, and the README says so.

## Decisions

### The recorded answers are committed as a fixture

`evals/answers/26b03ce9a1c2c1d4-682ec1326b2f.jsonl` has 30 lines and 21 KB.
Each line holds the chain digest of one prompt and the result the chain
returned for it. The fixture is written by `rageval.eval --answer
--record-answers PATH`, which wraps the chat provider in a recorder, and never
by hand. The lines are sorted by digest and end in LF, so recording the same
answers again produces the same bytes. `--replay-answers PATH` validates every
line before it writes any of them into the chat cache. From there the keyless
chat path of ADR 0010 serves them and refuses anything else.

The file name carries the corpus version and the prompt version, because a
change to either changes every prompt.

Rejected:

- **A release asset, as for the vectors.** The vectors needed one at 12.6 MB.
  At 21 KB, the answers are better off in the diff where a reviewer can read
  them. They are Groq and Gemini outputs about public-domain and CC0 PEPs.
- **Serving replies straight from the fixture, bypassing the cache.** CI would
  then test a path production never takes. Loading into the cache puts the
  real keyless path under the gate.

### What counts as drift for an answer

The model's text is fixed by the fixture, so the gate compares everything the
pipeline derives from it, per question and with zero tolerance, in both
directions: the answer text, parse error, raw response, every citation's status
and span, the cited chunk sizes, the citation hit, context recall, and the
provider and model credited. Aggregates are compared as well: the hit rates,
resolution rate, citation counts and lengths, parse failures and the providers
counter. A report is refused as not comparable when its corpus, k, retrieval
config, prompt version, chat chain, golden set or question set differ.

A different prompt also cannot slip through as drift, because it misses the
cache. With no key, a miss is a refusal. Any change to retrieval or to the
prompt therefore stops the answer step until the answers are recorded again.

### Every report records the golden set

`golden_set_digest` is a sha256 of the parsed questions (ids, questions,
answers and spans), not of the file bytes. Git checks the file out with CRLF on
Windows and LF on Ubuntu, so hashing the file would make the two platforms
disagree. Comparing two reports now refuses a golden-set mismatch, and it
refuses a baseline that records no digest at all. Old reports still load; they
are simply no longer comparable.

### The answer report names the configured chain

`chat_model` held failover's first model, `gemini-3.5-flash`, while Groq wrote
29 of the 30 answers. It is replaced by `chat_chain`, the configured
`provider/model` order, which is also what the chat cache key is built from.
Which provider actually answered stays per question and in the `providers`
counter.

## Results

The four baselines were re-cut, since the schema changed. Each new report was
compared field by field with the report it replaces. Every number is identical;
only `ran_at`, `golden_set_digest` and, for answers, `chat_chain` differ. The
old reports stay committed as history.

| Gate | Old baseline | New baseline |
| --- | --- | --- |
| BM25 | `20260918T134848Z` | `20260918T175145Z` |
| dense | `20260918T134844Z` | `20260918T175148Z` |
| hybrid-bm25 | `20260918T134853Z` | `20260918T175152Z` |
| answers | `20260918T144440Z` | `20260918T175155Z` |

The whole gate was then run the way CI runs it: no key, an empty cache filled
only from the vector release and the answer fixture, and all four comparisons
with `--fail-on-change`. None drifted, and every eval reported
`0 call(s) reached the network`.

Five regressions were fixed in the frame before building, then planted one at a
time:

| Planted regression | Gate | Unit suite |
| --- | --- | --- |
| resolver matches the raw chunk text, without collapsing whitespace | **fail**: 24 questions; hit rate 0.533 → 0.200, resolved 0.981 → 0.288 | **fail** |
| every resolved span one character longer | **fail**: all 30 questions, spans only | **fail** |
| passage header `[1]` → `(1)` in the prompt | **fail** by refusal: `no cached answer for this prompt…` | **fail** |
| `citation_hit` overlap `<` → `<=` | pass: no question changes | **fail** |
| q01's golden span shifted by 40 characters | **fail** on all four gates: `golden set differs` | pass |

- **The span shift is no longer invisible.** This is the ADR 0008 plant, which
  nothing caught before. The unit suite still passes it; the gate is the only
  thing that fails.
- **The `<=` plant cannot fire on this data.** No citation ends exactly where a
  golden span begins, as with the overlap plant in ADR 0008. The unit test
  pinning that boundary is what protects it.
- **The prompt plant fails by refusal, not by a call.** With no key, a new
  prompt cannot reach a provider.

## Consequences

- **A green check now covers the answer pipeline**: parsing, citation
  resolution, spans, the answer metrics, and the golden set itself, as well as
  everything ADR 0009 listed. It does not cover the reranker. It also cannot say
  whether a live model would still answer the same way, or whether an answer is
  correct: that needs a judge, which was not built.
- **Changing the prompt, the retrieval or a chat model means recording again.**
  That takes a live run with keys, `--record-answers`, a new fixture and a new
  answer baseline, repointed in `ci.yml` in the same PR. The fixture's name
  changes with the prompt version.
- **Editing the golden set means re-cutting every baseline.** That is the
  intended cost of closing the gap.
- Next: F7, the frontend.
