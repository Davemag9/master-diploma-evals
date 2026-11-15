import { chunkGenerator, type BlockKind, type ChunkParams, type RawChunk } from "./chunker.js";
import { cpLen } from "./text.js";

export type { BlockKind, ChunkParams } from "./chunker.js";
export { parseBlocks, type Block } from "./blocks.js";
export { splitCodeBlock, splitSentences } from "./chunker.js";
export { digitDensity } from "./text.js";

/**
 * Preset to use:
 * - `"qna"`  — reference documents / Q&A pages ("tight" budgets): β_prose=700,
 *   β_dense=350, r=2, τ=0.02, no section merging, β_code = β_dense.
 * - `"code"` — technical documentation with code: β_prose=1200, β_dense=800,
 *   β_code=900, r=4, τ=0.02, short sections merged.
 */
export type ChunkMode = "code" | "qna";

/** Embedding function: one vector per input text, same order. */
export type EmbedFn = (texts: string[]) => Promise<number[][]> | number[][];

export interface ChunkOptions {
  /** β_prose — character budget for narrative text (and the merge threshold). */
  proseMax?: number;
  /** β_dense — character budget for digit-dense blocks (ρ ≥ τ). */
  denseMax?: number;
  /** β_code — budget for fenced code blocks; longer blocks are split by lines. Defaults to `denseMax` in `qna` mode. */
  codeMax?: number;
  /** r — data rows per table chunk (header + separator are repeated in each). */
  rowsPerChunk?: number;
  /** τ — digit-density threshold above which a block gets the dense budget. */
  densityThreshold?: number;
  /** Lines of overlap carried between pieces of a split code block. */
  codeOverlapLines?: number;
  /** μ — merge short sections: a heading closes the chunk only if it is already ≥ proseMax. */
  mergeSmallSections?: boolean;
  /** Document title: the root of every breadcrumb (`[Title › H1 › H2]`). Omitted → breadcrumb starts at the first heading. */
  title?: string;
}

export interface AsyncChunkOptions extends ChunkOptions {
  /** Embedding function used to choose the semantic cut inside oversize paragraphs. */
  embed?: EmbedFn;
}

export interface Chunk {
  /** 0-based position of the chunk in the document. */
  index: number;
  /** Text to embed / index: `[breadcrumb]\n` + body (no prefix when the breadcrumb is empty). */
  text: string;
  /** The chunk content without the breadcrumb line. */
  body: string;
  /** Breadcrumb joined with " › ", e.g. `Doc › Fees › Passport`. */
  breadcrumb: string;
  /** Breadcrumb parts, e.g. `["Doc", "Fees", "Passport"]`. */
  path: string[];
  /** Length of `text` in Unicode code points (same unit as the budgets). */
  length: number;
  /** Block kinds that ended up in this chunk, in order of first appearance. */
  kinds: BlockKind[];
}

type PresetParams = Omit<ChunkParams, "codeMax" | "title"> & { codeMax?: number };

/** Parameter presets from the thesis experiments (see README). */
export const PRESETS: Readonly<Record<ChunkMode, Readonly<PresetParams>>> = Object.freeze({
  qna: Object.freeze({
    proseMax: 700,
    denseMax: 350,
    rowsPerChunk: 2,
    densityThreshold: 0.02,
    codeOverlapLines: 1,
    mergeSmallSections: false,
  }),
  code: Object.freeze({
    proseMax: 1200,
    denseMax: 800,
    codeMax: 900,
    rowsPerChunk: 4,
    densityThreshold: 0.02,
    codeOverlapLines: 1,
    mergeSmallSections: true,
  }),
});

/** Document title from a file name, as the reference implementation does (`"Fees.md"` → `"Fees"`). */
export function titleFromFileName(fileName: string): string {
  const base = fileName.split(/[\\/]/).pop() ?? fileName;
  const dot = base.lastIndexOf(".");
  return dot === -1 ? base : base.slice(0, dot);
}

/** Merge a mode preset with user overrides and validate the result. */
export function resolveParams(mode: ChunkMode, options: ChunkOptions = {}): ChunkParams {
  const preset = PRESETS[mode];
  if (!preset) throw new TypeError(`md-chunkit: unknown mode ${JSON.stringify(mode)} (expected "code" or "qna")`);
  const denseMax = options.denseMax ?? preset.denseMax;
  const params: ChunkParams = {
    proseMax: options.proseMax ?? preset.proseMax,
    denseMax,
    codeMax: options.codeMax ?? preset.codeMax ?? denseMax,
    rowsPerChunk: options.rowsPerChunk ?? preset.rowsPerChunk,
    densityThreshold: options.densityThreshold ?? preset.densityThreshold,
    codeOverlapLines: options.codeOverlapLines ?? preset.codeOverlapLines,
    mergeSmallSections: options.mergeSmallSections ?? preset.mergeSmallSections,
    title: options.title,
  };
  for (const k of ["proseMax", "denseMax", "codeMax", "rowsPerChunk"] as const) {
    if (!Number.isInteger(params[k]) || params[k] < 1) {
      throw new RangeError(`md-chunkit: ${k} must be a positive integer, got ${params[k]}`);
    }
  }
  if (!Number.isInteger(params.codeOverlapLines) || params.codeOverlapLines < 0) {
    throw new RangeError(`md-chunkit: codeOverlapLines must be a non-negative integer`);
  }
  if (typeof params.densityThreshold !== "number" || Number.isNaN(params.densityThreshold)) {
    throw new RangeError(`md-chunkit: densityThreshold must be a number`);
  }
  return params;
}

function finalize(raw: RawChunk[]): Chunk[] {
  return raw.map((c, index) => {
    const breadcrumb = c.path.join(" › ");
    const text = c.path.length > 0 ? `[${breadcrumb}]\n${c.body}` : c.body;
    return { index, text, body: c.body, breadcrumb, path: c.path, length: cpLen(text), kinds: c.kinds };
  });
}

/** Validate input and normalize line endings (CRLF / CR → LF), like Python's text-mode file reading. */
function normalizeInput(markdown: unknown): string {
  if (typeof markdown !== "string") throw new TypeError("md-chunkit: markdown must be a string");
  return markdown.replace(/\r\n?/g, "\n");
}

/**
 * Chunk a Markdown document synchronously. Oversize paragraphs are cut at the
 * last admissible sentence boundary (no embeddings). For semantic cuts use
 * {@link chunkItAsync} with an `embed` function.
 */
export function chunkIt(markdown: string, mode: ChunkMode, options: ChunkOptions = {}): Chunk[] {
  const md = normalizeInput(markdown);
  if ((options as AsyncChunkOptions).embed !== undefined) {
    throw new TypeError("md-chunkit: `embed` needs the async API — use chunkItAsync()");
  }
  const gen = chunkGenerator(md, resolveParams(mode, options), false);
  const r = gen.next();
  if (!r.done) throw new Error("md-chunkit: internal error (unexpected embedding request)");
  return finalize(r.value);
}

/**
 * Chunk a Markdown document; with `options.embed`, an oversize paragraph is
 * cut at the admissible sentence boundary with the lowest cosine similarity
 * between adjacent sentences. `embed` is called at most once per oversize
 * paragraph (with all its sentences). Without `embed` the result equals
 * {@link chunkIt}.
 */
export async function chunkItAsync(
  markdown: string,
  mode: ChunkMode,
  options: AsyncChunkOptions = {},
): Promise<Chunk[]> {
  const md = normalizeInput(markdown);
  const { embed, ...rest } = options;
  if (embed !== undefined && typeof embed !== "function") {
    throw new TypeError("md-chunkit: `embed` must be a function");
  }
  const gen = chunkGenerator(md, resolveParams(mode, rest), embed !== undefined);
  let r = gen.next();
  while (!r.done) {
    const texts = r.value;
    const vectors = await (embed as EmbedFn)(texts);
    if (!Array.isArray(vectors) || vectors.length !== texts.length || !vectors.every(Array.isArray)) {
      throw new TypeError(
        `md-chunkit: embed() must return one number[] per input text (got ${
          Array.isArray(vectors) ? vectors.length : typeof vectors
        } for ${texts.length})`,
      );
    }
    r = gen.next(vectors);
  }
  return finalize(r.value);
}
