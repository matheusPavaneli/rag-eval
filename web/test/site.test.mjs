import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

const root = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const dist = join(root, 'dist');
const questionsDir = join(root, 'src', 'data', 'questions');

const ENTITIES = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'" };
const decode = (html) =>
  html
    .replace(/&(amp|lt|gt|quot|#39);/g, (entity) => ENTITIES[entity])
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCodePoint(Number.parseInt(code, 16)));

const pages = readdirSync(questionsDir)
  .filter((name) => name.endsWith('.json'))
  .sort()
  .map((name) => JSON.parse(readFileSync(join(questionsDir, name), 'utf8')));

const documentText = (view) =>
  view.lines.map((line) => line.segments.map((segment) => segment.text).join('')).join('\n');

const html = (path) => readFileSync(join(dist, path, 'index.html'), 'utf8');

function markedText(source, id) {
  const pattern = /<mark class="[^"]*" data-m="([^"]*)">([^<]*)<\/mark>/g;
  let text = '';
  for (const [, marks, body] of source.matchAll(pattern)) {
    if (marks.split(' ').includes(id)) text += decode(body);
  }
  return text;
}

test('every question, the index and the numbers page are built', () => {
  assert.equal(pages.length, 30);
  for (const page of pages) assert.ok(existsSync(join(dist, 'q', page.id, 'index.html')), page.id);
  assert.ok(existsSync(join(dist, 'index.html')));
  assert.ok(existsSync(join(dist, 'numbers', 'index.html')));
});

test('each resolved citation highlights exactly the characters of its span', () => {
  let exact = 0;
  for (const page of pages) {
    const source = html(join('q', page.id));
    page.citations.forEach((citation, index) => {
      const id = `c${index + 1}`;
      if (citation.status !== 'resolved') {
        assert.equal(markedText(source, id), '', `${page.id} ${id} is unresolved and marks nothing`);
        return;
      }
      const view = page.documents.find((candidate) => candidate.source_path === citation.source_path);
      assert.ok(view, `${page.id} ${id} has its document`);
      const span = Array.from(documentText(view))
        .slice(citation.start_char, citation.end_char)
        .join('');
      assert.equal(markedText(source, id), span.replaceAll('\n', ''), `${page.id} ${id}`);
      assert.match(source, new RegExp(`<span class="anchor" id="${id}"></span>`), `${page.id} ${id} anchor`);
      assert.match(source, new RegExp(`href="#${id}"`), `${page.id} ${id} link`);
      exact += 1;
    });
  }
  assert.equal(exact, 51);
});

test('each expected span is marked and linked', () => {
  for (const page of pages) {
    const source = html(join('q', page.id));
    page.supports.forEach((support, index) => {
      const view = page.documents.find((candidate) => candidate.source_path === support.source_path);
      const span = Array.from(documentText(view)).slice(support.start_char, support.end_char).join('');
      assert.equal(markedText(source, `e${index + 1}`), span.replaceAll('\n', ''), page.id);
    });
  }
});

test('no page loads anything from another origin', () => {
  const offsite = [
    /\ssrc="(?:https?:)?\/\//,
    /url\(\s*['"]?(?:https?:)?\/\//,
    /<link[^>]+rel="(?:stylesheet|preload|modulepreload|icon)"[^>]+href="(?:https?:)?\/\//,
    /<script[^>]*>/,
  ];
  const paths = [...pages.map((page) => join('q', page.id)), '', 'numbers'];
  for (const path of paths) {
    const source = html(path);
    for (const pattern of offsite) assert.doesNotMatch(source, pattern, `${path || 'index'} ${pattern}`);
  }
});
