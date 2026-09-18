# ADR 0010 — Cache chat in front of failover, keyed on the configured chain

Date: 2026-09-18 · Slice: RAG-10 · Status: accepted · Supersedes: the cache
placement section of [ADR 0003](0003-providers.md)

## Context

ADR 0003 put one chat cache inside each provider, behind failover, keyed by the
provider and model that answered. It rejected a single cache in front of the
chain because the answer "would then depend on which provider happened to be up
when the entry was first written".

The first answer run showed the cost of that choice. Gemini returned HTTP 429
from the first question, and failover answered 29 of 30 with Groq
([ADR 0007](0007-citations.md)). The cache then held 29 Groq entries and one
Gemini entry, and none of them could be replayed:

- **With keys, a rerun asks Gemini first.** Gemini's entry misses for the 29
  Groq questions, so each one goes to the network. If Gemini answers, the run
  records a new answer from a different model. If it is still rate-limited, the
  run spends a call and then reaches Groq's cached answer. The result depends on
  Gemini's quota at the time of the run.
- **Without a key, the chain cannot be built.** `build_chat_provider` refused
  even though all 30 answers were on disk.

So the published answer number (citation hit 0.533) was reproducible by no
one, including its author, and a gate on answers had nothing to compare against.

## Decisions

### One cache in front of the chain

`build_chat_provider` now returns `Cached(Failover(Budgeted(Gemini),
Budgeted(Groq)))`. A hit never reaches failover, so it never reaches a provider
or the budget. A miss goes through failover as before, and the answer is stored
with the `provider` and `model` that produced it.

ADR 0003's objection still holds: an entry records whichever provider answered
first. What changed is that this is visible. The answer report credits each
question to the provider in the stored result, and the README row lists the
models that answered. A replay that matches the record is more useful for an
eval than a live run that matches nothing.

### The key is the configured chain, not the keys that are set

The key is the prompt, the system message and every configured provider/model,
in failover order: `gemini/gemini-3.5-flash > groq/openai/gpt-oss-120b`. It comes
from settings, not from which API keys are present. The same run therefore uses
the same key with both keys, with one, or with none.

Rejected:

- **Key on the prompt alone.** After `RAGEVAL_GROQ_CHAT_MODEL` changes, the
  cache would still serve answers from the old model under the new name.
- **Key on the providers that have keys.** A run with no key would miss every
  entry that a keyed run wrote, which is exactly the replay this slice exists
  for.
- **Keep the per-provider caches and look them up in failover order before
  calling anything.** This replays the record, but only through a second lookup
  path that keeps both layouts alive indefinitely.

The cost: changing either configured model invalidates every cached answer,
including those the other provider wrote.

### No key means cache-only

With neither `RAGEVAL_GEMINI_API_KEY` nor `RAGEVAL_GROQ_API_KEY`, the cache
wraps a stand-in that raises `CacheMissError` for every prompt, as the embedding
path does since [ADR 0009](0009-dense-gate.md). A hit is served. A miss fails
with a message naming both variables and never makes a request.

### The recorded answers were re-keyed, not re-measured

Re-answering the 30 questions live would have spent free-tier calls and
produced a new number from whichever provider was up. It would be a new
measurement, not a reproduction of the published one. Instead, a one-off script
ran the answer eval with a chat provider that only reads the old entries. For
each prompt it recomputed the old key for Gemini and for Groq and required
exactly one entry to exist. The front cache wrote each answer under its new key.
All 30 prompts found exactly one entry. The old entries were only read, and a
copy of the directory was taken first.

## Results

The published report, `20260918T144440Z-26b03ce9a1c2c1d4-answer-dense.json`, was
compared field by field with each replay: per question (answer, citations,
cited chunk lengths, provider, model, hit, context recall, parse error) and in
aggregate (hit rates, resolution rate, citation counts and lengths, providers,
prompt and corpus version).

| Run | Network calls | Compared with the published report |
| --- | --- | --- |
| keys unset | 0 | identical: 0.533, 0.696 (23), 51/52 resolved, Groq 29 / Gemini 1 |
| both keys set | 0 | identical |

The second row is the one ADR 0007 could not produce: a live Gemini key no
longer changes a recorded answer.

Three regressions were fixed in the frame before building, then planted one at a
time with no key:

| Planted change | Result |
| --- | --- |
| one character added to the system prompt | **refused**: `no cached answer for this prompt…`, exit 1, no request |
| `RAGEVAL_GROQ_CHAT_MODEL` changed | **refused**, no stale answer served |
| stored provider of q04 edited from `groq` to `gemini` | report credits q04 to `gemini`; the providers counter follows the stored value |

The builder has a regression test for the original failure: with both keys, an
answer recorded while Gemini returned 429 is replayed as Groq's on a second run
in which Gemini would answer, and the second run sends no request. It fails
against the per-provider layout.

## Consequences

- **The published answer number is reproducible offline.** It still cannot be
  reproduced in CI, because the 30 entries exist only in the local cache. Moving
  them to CI, and defining what counts as drift for an answer, is the answer
  gate's slice.
- **An outage at first write is frozen into the record.** If Gemini is down for
  the first run, Groq's answers remain the record until the cache is cleared.
  This is visible per question and in the README row.
- **Replays no longer exercise failover.** A cached question never reaches it.
  Failover keeps its unit tests, and any uncached prompt still goes through it.
- **The report header's `chat_model` is still failover's first model**
  (`gemini-3.5-flash`), although Groq wrote 29 answers. The per-question fields
  and the `providers` counter are correct. Fixing the header changes the
  report's content, so it waits for the answer gate, which cuts a new baseline
  anyway.
- Next: the answer gate, then a correctness judge, then abstention.
