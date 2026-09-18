# ADR 0007 — Answers whose citations resolve to character spans

Date: 2026-09-18 · Slice: F5 · Status: accepted

## Context

Up to F4 the project retrieved and stopped: nothing called a chat model, and the
claim in the README — every answer backed by a span you can check in the source
— had no code behind it. Citing a whole chunk would have been almost free, since
every chunk already carries its `start_char`/`end_char` (ADR 0002). It would also
prove little. A chunk is about 1000 characters, and the thirty golden support
spans run from 35 to 822. "The answer is somewhere in this block" is not a span
anyone can check.

So F5 adds the first generation step, and makes each citation a verbatim quote
that code — not the model — locates in the source. What counts as a result was
fixed before measuring, as in ADRs 0005 and 0006: one prompt, one match rule,
dense retrieval at k=5, and no edit to any of them after the number was seen.

## Decisions

### The model quotes; code finds the quote

The model is asked for one JSON object: an answer, and a list of citations, each
a passage number and a quote copied from that passage. `resolve` then looks for
the quote in the chunk it cites — only that chunk — and translates the position
into document offsets by adding the chunk's `start_char`. That addition is exact
because chunk text is a slice of the document text, not a copy.

A citation either resolves to `source_path[start:end]` or is recorded as
unresolved, with a reason: `no_such_chunk`, `empty_quote`, `case_mismatch` or
`not_in_chunk`. The reasons exist for diagnosis; none of them is scored as a
partial hit.

A reply that is not the JSON asked for is kept, raw, as a parse failure. It does
not stop the run and it does not score.

### Exact match after whitespace normalisation, nothing looser

Runs of whitespace are collapsed on both sides before matching, because the
source hard-wraps its prose and a model copying a sentence writes it on one line.
An index map carries each collapsed character back to its original position, so
the span in the document starts and ends on the quote's first and last
characters, line breaks and all.

Nothing else is forgiven. Rejected:

- **Fuzzy matching** (edit distance, token overlap). Any threshold is a knob,
  and it would be tuned after seeing which citations failed. A citation that
  resolves only approximately cannot be highlighted exactly, which is the whole
  point of F7.
- **Case folding.** Recorded as `case_mismatch` so it is visible, not accepted.
- **Searching every retrieved chunk** when the cited one does not contain the
  quote. A model that cites passage 2 for text in passage 4 has cited wrongly;
  finding it elsewhere would hide that.

### Citation hit rate, scored against the golden spans

A question is a hit when at least one resolved citation overlaps a golden
support span, by the same half-open rule that decides whether a retrieved chunk
covers one. It is reported over all thirty questions and over the questions
whose gold span dense retrieval actually returned, since a citation cannot land
on a passage the model never saw.

Rejected: an **LLM judge** for faithfulness. It is a second model whose error
rate on this corpus is unmeasured, and calibrating it against the golden answers
is a slice of its own. It belongs next to F6, where a CI gate needs it.

### A separate report, and a demo for one question

`--answer` on the eval writes an `AnswerReport` to
`evals/reports/<stamp>-<corpus>-answer-dense.json`. It is a new model rather than
new fields on `EvalReport`, so every retrieval report already published still
loads unchanged. Each question records its answer, every citation with its
offsets or its reason, the provider and model that answered, and the raw reply
when it could not be parsed. The report also records a hash of the system
prompt, so two runs with different prompts cannot be mistaken for one.

`--answer --query "…"` on the retrieval CLI answers one question and prints each
citation as `source_path[start:end]` with its quote — the thing to show someone.

## Results

Measured 2026-09-18 over corpus `26b03ce9a1c2c1d4`, dense retrieval, k=5, prompt
`682ec1326b2f`. Report: `evals/reports/20260918T144440Z-26b03ce9a1c2c1d4-answer-dense.json`.

| | |
| --- | --- |
| Citation hit rate @5, all questions | **0.533** (16 of 30) |
| over questions whose gold span was retrieved | **0.696** (16 of 23) |
| Citations resolved | 0.981 (51 of 52) |
| Mean resolved citation / mean cited chunk | 104 / 847 characters |
| Unparseable replies | 0 |
| Answered by | Groq `openai/gpt-oss-120b` 29, `gemini-3.5-flash` 1 |

What the numbers say:

- **Resolution is not the problem.** 51 of 52 quotes were found verbatim in the
  chunk they cited, with nothing looser than whitespace forgiven. The one that
  failed (q15) joined two sentences with `...` — exactly what the prompt forbids,
  and exactly what an exact rule should refuse. It would otherwise have hit.
- **The spans are narrow.** A resolved citation averages 104 characters, an
  eighth of the chunk it came from. That is the difference between pointing at a
  sentence and pointing at a page.
- **Most "misses" are the golden set's limit, not the model's.** Of the seven
  questions whose gold span was retrieved but not cited, at least four cite a
  different sentence of the same PEP that also answers the question: q19 cites
  "a new binary operator … called `@`", q20 cites "`async def` functions are
  always coroutines", q30 cites the sentence right after the gold span, q26 the
  sentence saying postponed evaluation solves the problem. The golden set records
  one support span per question, so the hit rate is a **lower bound** on correct
  citation, not an estimate of it. The set is not edited to fix this; ADR 0004
  makes it the contract, and widening it after seeing which citations missed
  would be tuning the ruler.
- **No abstention.** All seven questions whose gold span was not retrieved got
  a confident answer with resolved citations from somewhere else (q14, q21, q23,
  q29 among them). Some of those answers are right from the model's own
  knowledge and cite text that only loosely supports them. The prompt allows
  "the passages do not answer the question"; the model never used it.

## Consequences

- **The number is mostly Groq's.** Gemini returned HTTP 429 from the first
  question, and failover answered 29 of 30 with Groq `openai/gpt-oss-120b`. The
  report says so per question, and the README row names the models that
  answered — the first version of the row named the configured primary instead,
  which would have credited Gemini with Groq's number. That was fixed and is
  covered by a test.
- **A rerun is not reproducible yet.** The chat cache sits inside each provider,
  behind failover. While Gemini is rate-limited, a rerun calls Gemini again
  before it reaches Groq's cached answer; once Gemini recovers, a rerun would
  answer with a different model and produce a different number. The frozen
  report is the record of this run. Moving the cache in front of failover is a
  change to the provider layer, and F6 needs it before it can gate CI on a
  generated answer. *Done in [ADR 0010](0010-chat-cache.md): this report now
  replays with 0 network calls, with or without keys.*
- **Answer correctness is not measured.** A citation that lands on the gold span
  under a wrong answer counts as a hit. Named here, not hidden; the judge that
  would measure it is F6's.
- **Abstention is its own slice.** It changes the prompt and needs a metric of
  its own (answers withheld on the seven retrieval misses vs answers withheld
  wrongly), measured as a new row, not an edit to this one.
- F7 has what it needs: every resolved citation is a `source_path` and two
  offsets into a committed document.
