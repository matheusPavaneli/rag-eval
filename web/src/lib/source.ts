import type { DocumentView, Mark } from './data';

const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (character) => ESCAPES[character] ?? character);
}

export function renderLines(document: DocumentView): string {
  const kinds = new Map(document.marks.map((mark) => [mark.id, mark.kind]));
  return document.lines
    .map((line) => {
      const body = line.segments
        .map((segment) => {
          const anchors = segment.anchors
            .map((id) => `<span class="anchor" id="${escapeHtml(id)}"></span>`)
            .join('');
          const text = escapeHtml(segment.text);
          if (segment.marks.length === 0) return anchors + text;
          const classes = ['m', ...new Set(segment.marks.map((id) => kinds.get(id)))].join(' ');
          return `${anchors}<mark class="${classes}" data-m="${escapeHtml(segment.marks.join(' '))}">${text}</mark>`;
        })
        .join('');
      return `<span class="l ${line.kind}">${body}</span>`;
    })
    .join('\n');
}

export function targetRules(marks: Mark[]): string {
  return marks
    .map((mark) => {
      const id = mark.id;
      return [
        `.doc:has(#${id}:target) mark[data-m~="${id}"]{outline:2px solid var(--act);outline-offset:1px;animation:arrive var(--duration-target) ease-out}`,
        `body:has(#${id}:target) [data-chip="${id}"]{border-color:var(--act);box-shadow:inset 3px 0 0 var(--act)}`,
        `body:has(#${id}:target) [data-tick="${id}"]{outline:2px solid var(--act);outline-offset:1px}`,
      ].join('\n');
    })
    .join('\n');
}
