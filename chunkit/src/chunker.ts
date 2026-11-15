import { CODE_FENCE_RE, parseBlocks } from "./blocks.js";
import { PY_WS, cpLen, digitDensity, pyStrip } from "./text.js";

/** Kinds of Markdown blocks a chunk can contain. */
export type BlockKind = "paragraph" | "list" | "code" | "table" | "heading";

/** Fully resolved chunker parameters (see `ChunkOptions` for documentation). */
export interface ChunkParams {
  proseMax: number;
  denseMax: number;
  codeMax: number;
  rowsPerChunk: number;
  densityThreshold: number;
  codeOverlapLines: number;
  mergeSmallSections: boolean;
  title: string | undefined;
}

export interface RawChunk {
  path: string[];
  body: string;
  kinds: BlockKind[];
}

/** Breadcrumb parts: [title, H1, H2, ...], skipping a heading equal (case-insensitively) to the previous part. */
export function breadcrumbParts(title: string | undefined, stack: ReadonlyArray<[number, string]>): string[] {
  const parts: string[] = title ? [title] : [];
  for (const [, heading] of stack) {
    const prev = parts[parts.length - 1];
    if (prev === undefined || pyStrip(heading).toLowerCase() !== pyStrip(prev).toLowerCase()) {
      parts.push(heading);
    }
  }
  return parts;
}

/**
 * Cut a fenced code block into budget-sized pieces at line boundaries, each
 * re-wrapped in its own fence (original opening fence line, plain closing
 * fence), carrying `overlapLines` trailing lines into the next piece
 * (port of `_split_code_block`).
 */
export function splitCodeBlock(content: string, budget: number, overlapLines = 1): string[] {
  const lines = content.split("\n");
  const fenceLine = lines.length > 0 && CODE_FENCE_RE.test(lines[0] as string) ? (lines[0] as string) : "```";
  let body = lines.slice(1);
  if (body.length > 0 && CODE_FENCE_RE.test(body[body.length - 1] as string)) body = body.slice(0, -1);
  if (body.length === 0) return [content];

  const wrapLen = cpLen(fenceLine) + 4; // fence line + closing fence + newlines
  const pieces: string[] = [];
  let current: string[] = [];
  let currentLen = wrapLen;
  const emitPiece = () => fenceLine + "\n" + current.join("\n") + "\n```";

  for (const line of body) {
    const lineLen = cpLen(line) + 1;
    if (current.length > 0 && currentLen + lineLen > budget) {
      pieces.push(emitPiece());
      current = overlapLines > 0 ? current.slice(-overlapLines) : [];
      currentLen = wrapLen + current.reduce((acc, l) => acc + cpLen(l) + 1, 0);
    }
    current.push(line);
    currentLen += lineLen;
  }
  if (current.length > 0) pieces.push(emitPiece());
  return pieces.length > 0 ? pieces : [content];
}

const SENTENCE_SPLIT_RE = new RegExp(`(?<=[.!?\u2026])[${PY_WS}]+(?=[\u0410-\u042f\u0490\u0404\u0406\u0407A-Z0-9])`, "u");

/** Split text into sentences: on blank lines, then after `.!?…` followed by an uppercase letter or digit. */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  for (const rawPara of text.split("\n\n")) {
    const para = pyStrip(rawPara);
    if (!para) continue;
    for (const s of para.split(SENTENCE_SPLIT_RE)) {
      const t = pyStrip(s);
      if (t) out.push(t);
    }
  }
  return out;
}

function normalize(vectors: number[][]): number[][] {
  return vectors.map((v) => {
    let sq = 0;
    for (const x of v) sq += x * x;
    const norm = Math.max(Math.sqrt(sq), 1e-9);
    return v.map((x) => x / norm);
  });
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

/**
 * The chunking algorithm (port of `adaptive_hybrid_chunks`), written as a
 * generator: whenever a semantic cut needs embeddings it yields the sentence
 * list and expects the embedding vectors back. When `semantic` is false it
 * never yields, so it can be driven synchronously.
 */
export function* chunkGenerator(
  markdown: string,
  p: ChunkParams,
  semantic: boolean,
): Generator<string[], RawChunk[], number[][]> {
  const blocks = parseBlocks(markdown);
  const stack: Array<[number, string]> = [];
  const chunks: RawChunk[] = [];
  let parts: string[] = [];
  let kinds: BlockKind[] = [];
  let currentLen = 0;
  // breadcrumb captured when the open chunk started
  let currentPath: string[] | null = null;

  const pathNow = () => breadcrumbParts(p.title, stack);
  const emit = (body: string, k: BlockKind[], path?: string[] | null) => {
    chunks.push({ path: path ?? pathNow(), body, kinds: k });
  };
  const noteKind = (k: BlockKind) => {
    if (!kinds.includes(k)) kinds.push(k);
  };
  const flush = () => {
    if (parts.length > 0) emit(parts.join("\n\n"), kinds, currentPath);
    parts = [];
    kinds = [];
    currentLen = 0;
    currentPath = null;
  };
  const add = (content: string, k: BlockKind) => {
    if (parts.length === 0) currentPath = pathNow();
    parts.push(content);
    noteKind(k);
    currentLen += cpLen(content);
  };
  const budgetFor = (content: string) =>
    digitDensity(content) >= p.densityThreshold ? p.denseMax : p.proseMax;

  for (const block of blocks) {
    if (block.kind === "heading") {
      // Merge mode: a heading closes the open chunk only once it is >= proseMax;
      // otherwise the heading is kept inline as text.
      if (p.mergeSmallSections && parts.length > 0 && currentLen < p.proseMax) {
        const marker = "#".repeat(block.level) + " " + block.title;
        parts.push(marker);
        noteKind("heading");
        currentLen += cpLen(marker);
      } else {
        flush();
      }
      while (stack.length > 0 && (stack[stack.length - 1] as [number, string])[0] >= block.level) stack.pop();
      stack.push([block.level, block.title]);
      continue;
    }

    if (block.kind === "table") {
      // Tables are always their own chunks: header + separator repeated over
      // each group of `rowsPerChunk` data rows.
      flush();
      const lines = block.lines;
      if (lines.length < 2) {
        emit(lines.join("\n"), ["table"]);
        continue;
      }
      const header = lines.slice(0, 2);
      const rows = lines.slice(2);
      if (rows.length === 0) {
        emit(header.join("\n"), ["table"]);
        continue;
      }
      for (let i = 0; i < rows.length; i += p.rowsPerChunk) {
        emit(header.concat(rows.slice(i, i + p.rowsPerChunk)).join("\n"), ["table"]);
      }
      continue;
    }

    let content = block.text;
    if (!pyStrip(content)) continue;

    if (block.kind === "code") {
      const pieces =
        cpLen(content) <= p.codeMax ? [content] : splitCodeBlock(content, p.codeMax, p.codeOverlapLines);
      if (p.mergeSmallSections) {
        // keep code together with the prose that introduces it
        for (const piece of pieces) {
          if (parts.length > 0 && currentLen + cpLen(piece) > p.proseMax) flush();
          add(piece, "code");
          if (currentLen >= p.proseMax) flush();
        }
      } else {
        flush();
        for (const piece of pieces) emit(piece, ["code"]);
      }
      continue;
    }

    const budget = budgetFor(content);
    const len = cpLen(content);

    if (block.kind === "list") {
      // Atomic: never split a list, even if it alone exceeds the budget.
      if (parts.length > 0 && currentLen + len > budget) flush();
      add(content, "list");
      if (currentLen > budget) flush();
      continue;
    }

    // Prose paragraph.
    if (parts.length > 0 && currentLen + len > budget) flush();
    if (len <= budget) {
      add(content, "paragraph");
      continue;
    }

    // A paragraph larger than its budget: cut it at semantic seams.
    let sentences = splitSentences(content);
    const allSentences = sentences;
    let offset = 0; // index of sentences[0] within allSentences
    let vectors: number[][] | null = null;
    while (cpLen(content) > budget && sentences.length > 1) {
      // admissible cut indices (same rule as `_semantic_split_point`)
      const candidates: number[] = [];
      let running = 0;
      sentences.forEach((s, i) => {
        running += cpLen(s);
        if (i > 0 && running <= budget) candidates.push(i);
      });
      if (candidates.length === 0) break;
      let splitAt = candidates[candidates.length - 1] as number;
      if (semantic && candidates.length > 1) {
        if (vectors === null) {
          const raw: number[][] = yield allSentences;
          vectors = normalize(raw);
        }
        let best = Infinity;
        for (const c of candidates) {
          const sim = dot(vectors[offset + c - 1] as number[], vectors[offset + c] as number[]);
          if (sim < best) {
            best = sim;
            splitAt = c;
          }
        }
      }
      emit(sentences.slice(0, splitAt).join(" "), ["paragraph"]);
      sentences = sentences.slice(splitAt);
      offset += splitAt;
      content = sentences.join(" ");
    }
    if (pyStrip(content)) add(content, "paragraph");
  }

  flush();
  return chunks;
}
