import { PY_WS, pyStrip } from "./text.js";

/** A structural Markdown block, in document order. */
export type Block =
  | { kind: "heading"; level: number; title: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; text: string }
  | { kind: "code"; text: string }
  | { kind: "table"; lines: string[] };

const S = `[${PY_WS}]`;
// Ports of the regexes in chunkers.py (Python `\s`, `\d` and `.` semantics).
export const HEADING_LINE_RE = new RegExp(`^(#{1,6})${S}*([^\\n]+)$`, "u");
export const LIST_ITEM_RE = new RegExp(`^${S}*(?:[-*+]${S}+|\\p{Nd}+[.)]${S}+)`, "u");
export const TABLE_ROW_RE = new RegExp(`^${S}*\\|`, "u");
export const CODE_FENCE_RE = new RegExp(`^${S}*\`\`\``, "u");
const BLANK_RE = new RegExp(`^${S}*$`, "u");

const isBlank = (line: string) => BLANK_RE.test(line);

/**
 * Split raw Markdown into headings, paragraphs, lists, tables and fenced code
 * blocks (port of `_split_into_blocks`).
 *
 * - a heading is a line starting with 1–6 `#`;
 * - a fenced code block runs from a ``` line to the next ``` line (or EOF);
 * - a table is a run of lines starting with `|`;
 * - a list starts at a `-`/`*`/`+`/`1.`/`1)` item and runs until a blank
 *   line, heading, table row or fence (continuation lines are included);
 * - everything else is a paragraph, ended by a blank line or any of the above.
 */
export function parseBlocks(markdown: string): Block[] {
  const lines = markdown.split("\n");
  const blocks: Block[] = [];
  const n = lines.length;
  let i = 0;
  const at = (k: number) => lines[k] as string;

  while (i < n) {
    const line = at(i);
    if (isBlank(line)) {
      i++;
      continue;
    }
    const m = HEADING_LINE_RE.exec(line);
    if (m) {
      blocks.push({ kind: "heading", level: (m[1] as string).length, title: pyStrip(m[2] as string) });
      i++;
      continue;
    }
    if (CODE_FENCE_RE.test(line)) {
      const code = [line];
      i++;
      while (i < n && !CODE_FENCE_RE.test(at(i))) code.push(at(i++));
      if (i < n) code.push(at(i++)); // closing fence
      blocks.push({ kind: "code", text: code.join("\n") });
      continue;
    }
    if (TABLE_ROW_RE.test(line)) {
      const rows: string[] = [];
      while (i < n && TABLE_ROW_RE.test(at(i))) rows.push(at(i++));
      blocks.push({ kind: "table", lines: rows });
      continue;
    }
    if (LIST_ITEM_RE.test(line)) {
      const items = [line];
      i++;
      while (
        i < n &&
        !isBlank(at(i)) &&
        !HEADING_LINE_RE.test(at(i)) &&
        !TABLE_ROW_RE.test(at(i)) &&
        !CODE_FENCE_RE.test(at(i))
      ) {
        items.push(at(i++));
      }
      blocks.push({ kind: "list", text: items.join("\n") });
      continue;
    }
    const para: string[] = [];
    while (
      i < n &&
      !isBlank(at(i)) &&
      !HEADING_LINE_RE.test(at(i)) &&
      !TABLE_ROW_RE.test(at(i)) &&
      !LIST_ITEM_RE.test(at(i)) &&
      !CODE_FENCE_RE.test(at(i))
    ) {
      para.push(at(i++));
    }
    blocks.push({ kind: "paragraph", text: para.join("\n") });
  }
  return blocks;
}
