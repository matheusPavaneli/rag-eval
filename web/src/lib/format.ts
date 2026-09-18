import type { UnresolvedReason } from './data';

const grouped = new Intl.NumberFormat('en-US');

export function interval(start: number, end: number): string {
  return `[${grouped.format(start)}, ${grouped.format(end)})`;
}

export function decimal(value: number): string {
  return value.toFixed(3);
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function characters(count: number): string {
  return `${grouped.format(count)} characters`;
}

export function pep(sourcePath: string): string {
  const match = /^pep-0*(\d+)\.md$/.exec(sourcePath);
  return match?.[1] === undefined ? sourcePath : `PEP ${match[1]}`;
}

const REASONS: Record<UnresolvedReason, string> = {
  not_in_chunk: 'The quote is not in the passage it cites, even after collapsing whitespace.',
  no_such_chunk: 'The citation names a passage number that was never sent to the model.',
  empty_quote: 'The citation quotes nothing.',
  case_mismatch: 'The quote is there only if letter case is ignored, and case is not forgiven.',
};

export function reason(value: UnresolvedReason): string {
  return REASONS[value];
}

export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path}`;
}
