# ADR 0002 — Ingest: chunks with a span back into the source

Date: 2026-09-09 · Slice: F1 · Status: accepted

## Context

F3 has to publish a number, and F5 has to point at the exact characters in a
source document that support an answer. Both depend on what ingest produces, so
the shape of a chunk is decided here and everything after it inherits the
decision.

## Decisions

### A chunk is a character span, not a copy of text

Every chunk carries `start_char` and `end_char` into the normalised document
text, and `document.text[start_char:end_char] == chunk.text` is asserted for
every chunk of every fixture. The span is the thing F5 highlights; the text is a
convenience derived from it.

Rejected: storing the text alone and searching for it again when a citation has
to be rendered. The same sentence appears more than once in a real corpus, so
the search has no single right answer, and the cost lands in the slice that can
least afford to be vague.

The cost accepted: the offsets are only meaningful against the *normalised*
text, so normalisation is part of the contract — line endings become `\n` and a
byte-order mark is stripped before anything is measured or hashed.

### Character-based chunking, not token-based

`chunk_size` and `overlap` are counted in characters, at 1000 and 150.

Rejected: counting tokens. A token count is a property of a specific tokenizer,
which is a property of the embedding model — and the embedding model is chosen
in F2, not here. Pulling a tokenizer in now would fix the chunker to a model
that has not been picked, and would add a dependency to a slice that otherwise
needs none.

The numbers themselves are a starting point to be measured, not a claim. F3
gives them a baseline; F4 is where tuning them can be justified against it.

### Split by structure, then pack, then trim

Text is partitioned by descending separators — markdown heading, blank line,
newline, space — into contiguous pieces, and adjacent pieces are packed until
the next one would exceed `chunk_size`. Because the pieces are contiguous, the
packed chunk is a single span and the offset invariant survives packing.

Leading and trailing whitespace is removed by moving the offsets inward rather
than by editing the text, so a trimmed chunk is still an exact span.

A run longer than `chunk_size` with no separator inside it — a URL, a base64
blob — is emitted whole rather than cut mid-token. It is the one case where a
chunk exceeds the configured size, and losing the middle of an identifier is
worse than one oversized chunk.

Each chunk records the markdown heading path in force where its own text starts,
with the overlap excluded from that lookup — the overlap belongs to the previous
section and would otherwise mislabel the chunk. Heading detection ignores lines
inside a fenced code block: a `# install the client` comment in a Python sample
would otherwise register as an H1 and wipe the heading path for the rest of the
document, which in a corpus of technical documents is most of it.

Rejected: a fixed-width sliding window. Cheaper to write, and it cuts sentences
and headings in half, which shows up later as retrieved context that reads as
nonsense in an eval.

### Identity is content, never a path or a timestamp

`document.id` is the sha256 of the normalised text. Two files with identical
content are therefore one document, and the first path in sorted order is the
one recorded — a duplicate would otherwise produce colliding chunk ids, since a
chunk id is the hash of its document id and its offsets.

`corpus_version` is the sha256 of the sorted document ids plus the serialised
chunk config, truncated to 16 characters for use as a directory name. So a
changed document *or* a changed chunk parameter is a different corpus, and
`created_at` stays in the manifest and out of the hash — a re-run produces the
same version.

This is what makes a measurement quotable: a number in the README names the
corpus version that produced it, and no two configurations can hide behind one
name.

### The corpus stays on disk until F3

Ingest writes `data/corpus/<corpus_version>/` — `documents.jsonl`,
`chunks.jsonl`, `manifest.json` — and touches no database.

Rejected: writing chunks to Postgres now. The chunk table's vector column needs
an embedding dimension, and the embedding model is chosen in F2, so the schema
written today is a migration owed in F3. Keeping F1 on disk also keeps its whole
test suite in the default `pytest` run, with no Docker and no `integration`
marker.

`documents.jsonl` carries the document text alongside its metadata, so a corpus
resolves a span without reading the original files again. It costs roughly a
second copy of the source on disk, for a corpus that stays self-contained.

### Markdown and plain text only

`.md` and `.txt` are loaded; anything else in the directory is ignored, and a
file that is not valid UTF-8 raises an error naming its path rather than being
skipped quietly.

Rejected: PDF. Extracted PDF text has no stable mapping back to what a reader
sees on the page, so a character span in the extraction cannot be highlighted in
the document — the F5 promise would become approximate exactly where it is
supposed to be verifiable.

## Consequences

- `uv run python -m rageval.ingest <dir>` prints the corpus version, document
  count and chunk count, and exits non-zero naming the path when the directory
  is missing, empty of supported files, or the chunk config does not fit.
- `data/` is git-ignored: the corpus is derived, and its version proves it can be
  rebuilt.
- F2 may replace character counts with token counts once the embedding model is
  known. That changes `corpus_version`, which is the intended signal — the old
  number is no longer comparable.
