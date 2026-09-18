export type LineKind = 'heading' | 'fence' | 'code' | 'blank' | 'text';
export type MarkKind = 'cited' | 'expected';
export type UnresolvedReason = 'no_such_chunk' | 'empty_quote' | 'case_mismatch' | 'not_in_chunk';

export interface Mark {
  id: string;
  kind: MarkKind;
  source_path: string;
  start_char: number;
  end_char: number;
}

export interface Segment {
  text: string;
  marks: string[];
  anchors: string[];
}

export interface Line {
  kind: LineKind;
  segments: Segment[];
}

export interface DocumentView {
  source_path: string;
  title: string;
  length: number;
  marks: Mark[];
  lines: Line[];
}

export interface ResolvedCitation {
  status: 'resolved';
  chunk: number;
  quote: string;
  source_path: string;
  start_char: number;
  end_char: number;
}

export interface UnresolvedCitation {
  status: 'unresolved';
  chunk: number;
  quote: string;
  reason: UnresolvedReason;
}

export type Citation = ResolvedCitation | UnresolvedCitation;

export interface Support {
  source_path: string;
  start_char: number;
  end_char: number;
}

export interface QuestionPage {
  id: string;
  question: string;
  expected_answer: string;
  answer: string | null;
  parse_error: string | null;
  provider: string;
  model: string;
  citation_hit: boolean;
  context_recall: number;
  citations: Citation[];
  supports: Support[];
  documents: DocumentView[];
}

export interface QuestionSummary {
  id: string;
  question: string;
  citation_hit: boolean;
  retrieved: boolean;
}

export interface QuestionScore {
  id: string;
  context_recall: number;
  reciprocal_rank: number;
}

export interface ConfigurationRow {
  label: string;
  report: string;
  gated: boolean;
  context_recall_at_k: number;
  mrr_at_k: number;
  results: QuestionScore[];
}

export interface SitePayload {
  corpus_version: string;
  golden_set_digest: string;
  answer_report: string;
  prompt_version: string;
  chat_chain: string[];
  providers: Record<string, number>;
  embedding_model: string | null;
  k: number;
  question_count: number;
  document_count: number;
  retrieved_count: number;
  citation_hit_rate: number;
  citation_hit_rate_retrieved: number;
  citation_count: number;
  resolved_count: number;
  resolution_rate: number;
  questions: QuestionSummary[];
  configurations: ConfigurationRow[];
}

const sites = import.meta.glob<SitePayload>('../data/site.json', { eager: true, import: 'default' });
const pages = import.meta.glob<QuestionPage>('../data/questions/*.json', {
  eager: true,
  import: 'default',
});

const MISSING = 'no exported data in web/src/data — run `pnpm data` (python -m rageval.site) first';

export function site(): SitePayload {
  const payload = sites['../data/site.json'];
  if (payload === undefined) throw new Error(MISSING);
  return payload;
}

export function questionPages(): QuestionPage[] {
  const all = Object.values(pages).sort((a, b) => a.id.localeCompare(b.id));
  if (all.length === 0) throw new Error(MISSING);
  return all;
}

export function questionPage(id: string): QuestionPage {
  const page = questionPages().find((candidate) => candidate.id === id);
  if (page === undefined) throw new Error(`no exported page for question ${id}`);
  return page;
}
