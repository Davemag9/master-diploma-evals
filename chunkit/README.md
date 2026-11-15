# md-chunkit

Adaptive **structural-semantic** Markdown chunker for RAG (retrieval-augmented generation).

`md-chunkit` turns a Markdown document into retrieval-ready chunks. It follows the document's structure
instead of cutting at fixed character offsets:

- every chunk is prefixed with its **heading breadcrumb**: `[Doc › Section › Subsection]`
- **lists are atomic**: a list is never split across chunks
- **tables** are split into groups of rows, and each group **repeats the header row**
- **digit-dense blocks** (fees, deadlines, phone numbers, tariffs) get a **tighter budget** than narrative prose
- an over-long paragraph is cut at a sentence boundary, optionally at the **semantic seam**: the point where two adjacent sentences are least similar (you supply the embedding function)
- **fenced code** is split by lines with its fences re-added; in documentation mode, short sections and their code are **merged** so each chunk holds enough to answer a question

It is a TypeScript port of the reference Python implementation from Oleh Lozovyi's master's thesis,
*"A method of adaptive structural-semantic segmentation of text documents in Retrieval-Augmented Generation systems"*
(«Метод адаптивної структурно-семантичної сегментації текстових документів у системах Retrieval-Augmented Generation»). On the thesis corpora it produces output identical to the Python version (see [Parity](#parity-with-the-reference-implementation)).

Zero runtime dependencies. ESM + CommonJS. Types included. Requires Node ≥ 18. The source uses no Node-specific APIs, so it should also run in Deno, Bun and browsers, but those runtimes are not tested.

## Install

```bash
npm install md-chunkit
```

## Quick start

```ts
import { chunkIt } from "md-chunkit";

const markdown = `
# Passport

## Fees

| Service        | Fee, EUR | Days |
|----------------|----------|------|
| Standard       | 30       | 20   |
| Urgent         | 60       | 7    |
| Lost passport  | 45       | 20   |

## How to apply

- Book an appointment online
- Bring the old passport and two photos
`;

const chunks = chunkIt(markdown, "qna", { title: "Consulate" });
for (const c of chunks) console.log(c.text, "\n---");
```

```text
[Consulate › Passport › Fees]
| Service        | Fee, EUR | Days |
|----------------|----------|------|
| Standard       | 30       | 20   |
| Urgent         | 60       | 7    |
---
[Consulate › Passport › Fees]
| Service        | Fee, EUR | Days |
|----------------|----------|------|
| Lost passport  | 45       | 20   |
---
[Consulate › Passport › How to apply]
- Book an appointment online
- Bring the old passport and two photos
---
```

### Technical documentation with code

```ts
import { readFileSync } from "node:fs";
import { chunkIt, titleFromFileName } from "md-chunkit";

const file = "docs/Background Tasks.md";
const chunks = chunkIt(readFileSync(file, "utf8"), "code", {
  title: titleFromFileName(file), // "Background Tasks"
});
```

In `"code"` mode, short sections are merged. A heading closes the open chunk only once that chunk has reached
`proseMax` characters. Otherwise the heading stays inline (`## Run`) and the next section follows it in the same chunk.
A code block stays with the prose that introduces it.

### Semantic cuts with your own embeddings (async)

```ts
import { chunkItAsync } from "md-chunkit";
import OpenAI from "openai";

const openai = new OpenAI();
const embed = async (texts: string[]) => {
  const res = await openai.embeddings.create({ model: "text-embedding-3-small", input: texts });
  return res.data.map((d) => d.embedding);
};

const chunks = await chunkItAsync(markdown, "qna", { title: "Consulate", embed });
```

Any function with the signature `(texts: string[]) => Promise<number[][]>` works. Vectors don't need to be normalised.

### Sync or async?

| Function | Returns | Embeddings | Oversize-paragraph cut |
|---|---|---|---|
| `chunkIt(md, mode, options?)` | `Chunk[]` | not accepted (throws if `embed` is passed) | last admissible sentence boundary |
| `chunkItAsync(md, mode, options?)` | `Promise<Chunk[]>` | optional `options.embed` | least-similar admissible boundary when `embed` is given; otherwise the same as `chunkIt` |

`embed` is only called for paragraphs longer than their budget that have more than one admissible cut. It is called **at
most once per such paragraph**, with all of that paragraph's sentences. Most documents need few or no embedding calls.

## Output

```ts
interface Chunk {
  index: number;       // 0-based position in the document
  text: string;        // "[breadcrumb]\n" + body: the string to embed and index
  body: string;        // chunk content without the breadcrumb line
  breadcrumb: string;  // "Consulate › Passport › Fees"
  path: string[];      // ["Consulate", "Passport", "Fees"]
  length: number;      // length of `text` in Unicode code points
  kinds: ("paragraph" | "list" | "code" | "table" | "heading")[]; // block kinds inside
}
```

If you don't pass a `title`, the breadcrumb starts at the first heading. A chunk that comes before any heading has an
empty breadcrumb, and its `text` equals its `body` with no `[]` prefix.

## Modes (presets)

| Parameter | Thesis symbol | `"qna"`: reference docs / Q&A | `"code"`: technical docs + code |
|---|---|---|---|
| `proseMax` | β_prose | 700 | 1200 |
| `denseMax` | β_dense | 350 | 800 |
| `codeMax` | β_code | = `denseMax` (350) | 900 |
| `rowsPerChunk` | r | 2 | 4 |
| `densityThreshold` | τ | 0.02 | 0.02 |
| `mergeSmallSections` | μ | `false` | `true` |
| `codeOverlapLines` | — | 1 | 1 |

`"qna"` is the *tight* configuration tuned on a corpus of Ukrainian consular reference pages (fees, deadlines,
procedures). `"code"` is the documentation configuration tuned on the FastAPI documentation.

## Options

All parameters are optional and override the preset of the chosen mode.

| Option | Type | Description |
|---|---|---|
| `title` | `string` | Document title, the root of every breadcrumb. `titleFromFileName("Fees.md")` returns `"Fees"`. |
| `proseMax` | `number` | Character budget for narrative blocks. In merge mode it is also the length at which a heading closes the open chunk. |
| `denseMax` | `number` | Budget for digit-dense blocks, used when digit density ρ ≥ `densityThreshold`. |
| `codeMax` | `number` | Budget for a fenced code block. A longer block is split at line boundaries. Defaults to `denseMax` in `qna` mode. |
| `rowsPerChunk` | `number` | Data rows per table chunk. The header row and separator are repeated in every group. |
| `densityThreshold` | `number` | τ, the digit-density threshold (digits ÷ characters). |
| `mergeSmallSections` | `boolean` | μ. Merges short sections: a heading closes the chunk only when the chunk is ≥ `proseMax`. |
| `codeOverlapLines` | `number` | Lines repeated between consecutive pieces of a split code block. |
| `embed` | `(texts) => Promise<number[][]>` | `chunkItAsync` only. Enables the semantic cut. |

Budgets are measured in Unicode code points, like Python's `len()`, so emoji count as one character. CRLF and CR line
endings are normalised to LF.

Lower-level helpers are also exported: `parseBlocks`, `splitSentences`, `splitCodeBlock`, `digitDensity`,
`resolveParams` and `PRESETS`.

## How it maps to the thesis method

1. **Block parsing.** The Markdown is split into headings (`#`–`######`), paragraphs, lists (`-`, `*`, `+`, `1.`, `1)`
   with continuation lines), tables (runs of lines starting with `|`) and fenced ```` ``` ```` code blocks.
2. **Heading stack → breadcrumb.** A heading pops every stack entry of equal or deeper level, then pushes itself. Each
   chunk is labelled `[Doc › H1 › H2]` with the breadcrumb that was current **when the chunk was opened**. A heading
   that equals the previous breadcrumb link (case-insensitive) is not repeated.
3. **Adaptive budget.** For each block, ρ(b) = digits / length. The budget β(b) is β_dense if ρ(b) ≥ τ, otherwise
   β_prose. If adding a block to the open chunk would exceed β(b), the chunk is closed first.
4. **Lists are atomic.** A list that doesn't fit closes the open chunk and goes whole into the next one. A list longer
   than its budget becomes a chunk by itself.
5. **Tables.** A table always gets its own chunks: header + separator + `r` data rows per chunk.
6. **Oversize paragraphs.** The paragraph is split into sentences (after `.`, `!`, `?` or `…` followed by an
   uppercase Latin or Cyrillic letter or a digit). The admissible cut indices A are those where the prefix stays
   within the budget. Without embeddings, the last admissible index is used. With embeddings, the cut goes at the
   index whose two adjacent sentences have the lowest cosine similarity. This repeats until the remainder fits.
7. **Code.** Blocks longer than β_code are cut at line boundaries. Each piece gets the original opening fence (with the
   language tag) and a closing fence, plus a 1-line overlap. With μ off, code pieces are separate chunks. With μ on,
   pieces join the open chunk until it reaches β_prose.
8. **Merging short sections (μ).** Close the chunk at a heading ⟺ |c| ≥ β_prose or μ = 0. Otherwise the heading is
   appended as a `## Heading` line.

### Faithful to the reference code, not the formula

The port reproduces the Python reference exactly, including two details where the code is stricter than the
formula in the thesis text:

- **Admissible cut indices.** The thesis defines A = { i : Σ_{j<i} |s_j| ≤ β }. The reference code also counts
  sentence *i* itself: it requires Σ_{j≤i} |s_j| ≤ β. As a result, if the first two sentences of an oversize
  paragraph already exceed the budget, the paragraph is not split. The lengths also leave out the joining spaces.
- **β_code in the Q&A preset.** The thesis table lists β_code as "not applicable" for the tight preset. In the code,
  `code_max=None` falls back to β_dense (350), and the port does the same.

## Parity with the reference implementation

`scripts/parity_py.py` runs the reference `chunkers.py`, and `scripts/parity.mjs` compares its output with this
package. They were run on both thesis corpora: 34 consular pages and 39 FastAPI pages. Each corpus was chunked in
both modes, with and without a deterministic test embedder, plus three synthetic edge-case documents. Across 304
runs and 4,660 chunks, the output was **byte-for-byte identical**.

## License

MIT © Oleh Lozovyi
