# ADR 0012 — A static citation viewer, cut in Python, published on Pages

Date: 2026-09-18 · Slice: RAG-12 · Status: accepted

## Context

The project's central claim is that each citation resolves to exact characters
of its source: `resolve` finds the model's quote in the passage it cites and
records `[start_char, end_char)` in the document ([ADR 0007](0007-citations.md)).
Until now that claim was visible as one number (0.981 resolved) and a JSON
report. Someone reading the repository on GitHub could not see a span without
Docker, Python 3.13 and a key.

The README's roadmap named F7: a frontend where clicking a citation highlights
the span it came from. The scope was chosen on 2026-09-18: static, the thirty
golden questions, published on GitHub Pages.

## Decisions

### Static, from the gated reports — no live questions

The page is built from files already committed: the answer baseline CI
compares against, the three gated retrieval baselines, four recorded reports,
the golden set and `documents/`. It makes no request at runtime beyond its own
files, and a test over the built HTML fails if a page loads anything from
another origin.

Rejected: **free-form questions, live.** That needs pgvector and a Gemini key
behind a public endpoint — a server that outlives the experiment, a key open to
abuse, and a free tier that already rate-limits the author's own runs (Gemini
answered 1 of 30). Also rejected: a local live mode beside the static page. It
doubles the surface for a mode no visitor can reach.

### The exporter refuses what CI would refuse

`python -m rageval.site` recomputes the corpus version from `documents/` and the
chunk configuration, and the golden-set digest from the parsed questions
([ADR 0011](0011-answer-gate.md)). An answer report of another corpus, or
scored against another golden set, or recording no digest at all, is refused
with both values named. A retrieval report with no digest — the four written
before RAG-11 — is accepted only if it answers exactly the golden set's
questions, and it is shown as *recorded*, not *gated*. The page cannot show a
number the gate does not stand behind without saying so.

### Offsets are cut in Python, where they were measured

`start_char` is an index into a Python `str`: code points, over the normalised
text (BOM stripped, CRLF folded). JavaScript strings are UTF-16, so a naive
`slice()` in the browser shifts every highlight after the first astral
character. The exporter therefore splits each document into lines and segments
at every span boundary itself, and the page only renders text it is given. No
PEP in this corpus contains an astral character today; the unit suite plants
one before a span and checks the marked code points.

Rejected: **highlighting at runtime** (Range or the CSS Custom Highlight API).
It would convert code points in the browser and fail with JavaScript off, for
data that never changes.

### The source is shown as source

The document is shown as its markdown text, in a monospace, with lines styled
by kind (heading, fence, code) and nothing reflowed. A span is an offset into
that text, so showing it keeps every offset literally checkable: a reader can
count characters if they doubt the highlight.

Rejected: **rendered markdown with source maps.** Maps survive per block, so a
span inside a paragraph or across inline markup cannot be placed exactly.

### No JavaScript

The site is Astro, prerendered: 33 pages, 0 KB of script. A citation is a link
to a fragment; its first segment carries the anchor; `:target` and `:has()`
emphasise every segment of that span and the chip that points to it. The
expected span from the golden set is underlined beside the cited one, so a hit
reads as the two overlapping and a miss as two separate places.

Beside each document, a rail as tall as the document marks where every cited
and expected span sits, to scale. It began as a row of links; 4-pixel ticks
fail WCAG 2.5.8 target size, and overlapping spans cannot be spaced apart, so
it is a figure with a text description and the chips carry the navigation.

Rejected: **a Vite + React single-page app.** Around 60 KB of runtime and a
client router for thirty-one pages whose content never changes, plus a 404
redirect to make deep links work on Pages.

### Design

The palette takes python.org's two colours, where every PEP is published:
the blue for anything that acts and for the expected span, the yellow as a
highlighter over what was cited. Schibsted Grotesk sets the answer; Red Hat
Mono sets the source, because PEPs were written as fixed-width text wrapped at
79 columns. Both are self-hosted, 70 KB together. Every colour pair used for
text, the focus ring and the span marks clears WCAG AA in light and dark.

## What the check covers

`web/test/site.test.mjs` runs over the built HTML in CI. For each of the 51
resolved citations it rebuilds the document from the exported lines, slices
`[start_char, end_char)` by code point, and requires the text inside that
citation's marks to equal it; the one unresolved citation (q15, `not_in_chunk`)
must mark nothing and is shown with its reason. Shifting one citation's start
by a single character fails it. The same holds for the thirty expected spans.
`tests/test_site.py` holds the segmentation, the refusals, and the same 51/51
over the committed baseline without a browser.

Measured on the build: axe (WCAG 2.2 AA) reports no violation on five routes in
light and dark; no horizontal scroll from 320 px to 1440 px or at 200 % zoom.
Not measured: field Core Web Vitals — there is no traffic yet.

## Consequences

- A change to the answer baseline, the golden set or a gated report changes
  the page in the same pull request; a mismatch fails the `web` job before the
  build.
- `web/src/data/` is generated and not committed: run
  `uv run python -m rageval.site --out web/src/data` before `pnpm --dir web dev`.
- Pages must be enabled with GitHub Actions as its source for the `pages`
  workflow to deploy.
- A question page carries its whole source document, 15–58 KB compressed. Fine
  for thirty static pages; a larger corpus would load documents on demand.
