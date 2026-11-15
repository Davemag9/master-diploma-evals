import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import {
  PRESETS,
  chunkIt,
  chunkItAsync,
  digitDensity,
  parseBlocks,
  resolveParams,
  splitCodeBlock,
  splitSentences,
  titleFromFileName,
} from "../src/index.js";

const fixture = (name: string) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
const bodies = (md: string, mode: "qna" | "code", opts = {}) => chunkIt(md, mode, opts).map((c) => c.body);

describe("parseBlocks", () => {
  it("recognises headings, paragraphs, lists, tables and fenced code", () => {
    const md = [
      "# Title",
      "",
      "Para line 1",
      "para line 2",
      "- a",
      "  continued",
      "- b",
      "| h |",
      "|---|",
      "| 1 |",
      "```js",
      "x()",
      "",
      "```",
    ].join("\n");
    expect(parseBlocks(md).map((b) => b.kind)).toEqual(["heading", "paragraph", "list", "table", "code"]);
    const code = parseBlocks(md)[4] as { text: string };
    expect(code.text).toBe("```js\nx()\n\n```");
  });
});

describe("breadcrumbs", () => {
  it("prefixes every chunk with [Doc › H1 › H2] and a newline", () => {
    const md = "# Visa\n\nIntro text.\n\n## Fees\n\nThe fee is paid in euros.\n\n# Other\n\nMore.";
    const chunks = chunkIt(md, "qna", { title: "Consulate" });
    expect(chunks.map((c) => c.text)).toEqual([
      "[Consulate › Visa]\nIntro text.",
      "[Consulate › Visa › Fees]\nThe fee is paid in euros.",
      "[Consulate › Other]\nMore.",
    ]);
    expect(chunks[1]).toMatchObject({
      index: 1,
      breadcrumb: "Consulate › Visa › Fees",
      path: ["Consulate", "Visa", "Fees"],
      body: "The fee is paid in euros.",
      kinds: ["paragraph"],
    });
    expect(chunks[1]!.length).toBe(chunks[1]!.text.length);
  });

  it("does not repeat a heading equal (case-insensitively) to the previous link", () => {
    const md = "# consulate\n\n## CONSULATE\n\nText.";
    expect(chunkIt(md, "qna", { title: "Consulate" })[0]!.text).toBe("[Consulate]\nText.");
  });

  it("pops deeper/equal levels from the heading stack", () => {
    const md = "# A\n\n### C\n\nx\n\n## B\n\ny";
    expect(chunkIt(md, "qna", { title: "D" }).map((c) => c.breadcrumb)).toEqual(["D › A › C", "D › A › B"]);
  });

  it("omits the prefix entirely when there is no title and no heading", () => {
    const [c] = chunkIt("Just text.", "qna");
    expect(c!.text).toBe("Just text.");
    expect(c!.breadcrumb).toBe("");
    expect(c!.path).toEqual([]);
  });

  it("takes the breadcrumb when the chunk is opened (merge mode spans sections)", () => {
    const md = "# A\n\nShort one.\n\n# B\n\nShort two.";
    const chunks = chunkIt(md, "code", { title: "Doc" });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toBe("[Doc › A]\nShort one.\n\n# B\n\nShort two.");
    expect(chunks[0]!.kinds).toEqual(["paragraph", "heading"]);
  });

  it("derives a title from a file name like the reference code", () => {
    expect(titleFromFileName("docs/Довідки.md")).toBe("Довідки");
    expect(titleFromFileName("a.b.md")).toBe("a.b");
    expect(titleFromFileName("README")).toBe("README");
  });
});

describe("lists are atomic", () => {
  const list = Array.from({ length: 30 }, (_, i) => `- list item number ${"x".repeat(i % 5)} words words`).join("\n");

  it("never splits a list even when it exceeds the budget", () => {
    const chunks = chunkIt(`${list}`, "qna");
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.body).toBe(list);
    expect(chunks[0]!.length).toBeGreaterThan(PRESETS.qna.proseMax);
  });

  it("closes the open chunk and moves the list to the next one when it does not fit", () => {
    const para = "Intro sentence about the list. ".repeat(10).trim();
    const chunks = chunkIt(`${para}\n\n${list}\n\nAfter.`, "qna");
    expect(chunks.map((c) => c.body)).toEqual([para, list, "After."]);
  });

  it("packs a small list together with the preceding paragraph", () => {
    expect(bodies("Before.\n\n- a\n- b\n\nAfter.", "qna")).toEqual(["Before.\n\n- a\n- b\n\nAfter."]);
  });
});

describe("tables", () => {
  const header = "| Service | Fee |\n|---|---|";
  const rows = Array.from({ length: 5 }, (_, i) => `| S${i} | ${i * 10} |`);
  const table = [header, ...rows].join("\n");

  it("groups r data rows per chunk and repeats header + separator (qna: r=2)", () => {
    const chunks = chunkIt(`# T\n\n${table}`, "qna", { title: "Doc" });
    expect(chunks.map((c) => c.body)).toEqual([
      [header, rows[0], rows[1]].join("\n"),
      [header, rows[2], rows[3]].join("\n"),
      [header, rows[4]].join("\n"),
    ]);
    expect(chunks.every((c) => c.text.startsWith("[Doc › T]\n| Service | Fee |\n|---|---|"))).toBe(true);
    expect(chunks.every((c) => c.kinds.join() === "table")).toBe(true);
  });

  it("uses r=4 in code mode and honours a rowsPerChunk override", () => {
    expect(chunkIt(table, "code")).toHaveLength(2);
    expect(chunkIt(table, "code", { rowsPerChunk: 1 })).toHaveLength(5);
  });

  it("closes the open chunk before a table and handles header-only tables", () => {
    expect(bodies(`Before.\n\n${header}\n\n| lonely |`, "qna")).toEqual(["Before.", header, "| lonely |"]);
  });
});

describe("adaptive budget", () => {
  it("computes digit density like Python str.isdigit", () => {
    expect(digitDensity("")).toBe(0);
    expect(digitDensity("ab12")).toBe(0.5);
    expect(digitDensity("x²")).toBe(0.5); // superscript counts, as in Python
    expect(digitDensity("😀1")).toBe(0.5); // length in code points
  });

  it("gives digit-dense blocks the tight β_dense budget", () => {
    // ~500 chars: under β_prose=700 but over β_dense=350
    const prose = "This sentence has no numbers in it at all. ".repeat(11).trim();
    const dense = "Fee 30 EUR, 45 EUR, 60 EUR within 10 days. ".repeat(11).trim();
    expect(prose.length).toBeGreaterThan(350);
    expect(prose.length).toBeLessThan(700);
    expect(digitDensity(dense)).toBeGreaterThanOrEqual(0.02);
    expect(chunkIt(prose, "qna")).toHaveLength(1);
    const denseChunks = chunkIt(dense, "qna");
    expect(denseChunks.length).toBeGreaterThan(1);
    expect(denseChunks.every((c) => c.length <= 350 + 10)).toBe(true);
  });

  it("closes the open chunk when a dense block would push it over β_dense", () => {
    const a = "Plain words only here. ".repeat(10).trim(); // ~230 chars
    const b = "Call 044 123 45 67 or 044 765 43 21. ".repeat(4).trim(); // dense, ~150
    expect(bodies(`${a}\n\n${b}`, "qna")).toEqual([a, b]);
    // with a looser dense budget they fit together
    expect(bodies(`${a}\n\n${b}`, "qna", { denseMax: 700 })).toEqual([`${a}\n\n${b}`]);
  });

  it("respects densityThreshold overrides", () => {
    const dense = "Fee 30 EUR, 45 EUR, 60 EUR within 10 days. ".repeat(11).trim();
    expect(chunkIt(dense, "qna", { densityThreshold: 0.9 })).toHaveLength(1);
  });
});

describe("oversize paragraphs", () => {
  // three topics, 5 sentences each; topic A and B sentences are written so a
  // fake embedder can tell them apart
  const A = Array.from({ length: 5 }, (_, i) => `Alpha topic sentence number ${"one two three four"} item ${i}.`);
  const B = Array.from({ length: 5 }, (_, i) => `Beta subject statement variant ${"five six seven eight"} entry ${i}.`);
  const para = [...A, ...B].join(" ");

  it("splits sentences on terminal punctuation followed by an uppercase letter or digit", () => {
    expect(splitSentences("One. Two! три? Чотири… 5 items.\n\nNew para.")).toEqual([
      "One.",
      "Two! три?",
      "Чотири…",
      "5 items.",
      "New para.",
    ]);
  });

  it("without embeddings cuts at the last admissible index (fill the budget)", () => {
    const chunks = chunkIt(para, "qna", { proseMax: 300 });
    expect(chunks.length).toBeGreaterThan(1);
    // the first cut is the furthest sentence index whose running length (incl.
    // the next sentence) is within budget — same rule as chunkers.py
    const sentences = splitSentences(para);
    let running = 0;
    let last = 0;
    sentences.forEach((s, i) => {
      running += s.length;
      if (i > 0 && running <= 300) last = i;
    });
    expect(chunks[0]!.body).toBe(sentences.slice(0, last).join(" "));
    expect(chunks.map((c) => c.body).join(" ")).toBe(sentences.join(" "));
  });

  it("with an embed function cuts at the least similar adjacent pair", async () => {
    const embed = vi.fn(async (texts: string[]) =>
      texts.map((t) => (t.startsWith("Alpha") ? [1, 0] : [0, 1])),
    );
    const chunks = await chunkItAsync(para, "qna", { proseMax: 400, embed });
    expect(chunks.map((c) => c.body)).toEqual([A.join(" "), B.join(" ")]);
    expect(embed).toHaveBeenCalledTimes(1); // once per oversize paragraph
    expect(embed.mock.calls[0]![0]).toEqual([...A, ...B]);
    // without embeddings the same input is cut elsewhere
    expect(chunkIt(para, "qna", { proseMax: 400 }).map((c) => c.body)).not.toEqual([A.join(" "), B.join(" ")]);
  });

  it("chunkItAsync without embed equals chunkIt", async () => {
    const md = fixture("synthetic.md");
    expect(await chunkItAsync(md, "qna", { title: "s" })).toEqual(chunkIt(md, "qna", { title: "s" }));
  });

  it("does not call embed when there is only one admissible cut", async () => {
    const embed = vi.fn(async (texts: string[]) => texts.map(() => [1]));
    await chunkItAsync("Short. Paragraph.", "qna", { embed });
    expect(embed).not.toHaveBeenCalled();
  });

  it("validates the embed output", async () => {
    const bad = async () => [[1, 0]];
    await expect(chunkItAsync(para, "qna", { proseMax: 400, embed: bad })).rejects.toThrow(/one number\[\] per input/);
  });

  it("rejects embed in the sync API", () => {
    // @ts-expect-error embed is only accepted by chunkItAsync
    expect(() => chunkIt(para, "qna", { embed: async () => [] })).toThrow(/chunkItAsync/);
  });
});

describe("code blocks", () => {
  const lines = Array.from({ length: 60 }, (_, i) => `    value_${i} = compute(${i})  # step ${i}`);
  const code = ["```python", ...lines, "```"].join("\n");

  it("splits oversize code at line boundaries with re-added fences and 1-line overlap", () => {
    const pieces = splitCodeBlock(code, 900, 1);
    expect(pieces.length).toBeGreaterThan(1);
    for (const p of pieces) {
      expect(p.startsWith("```python\n")).toBe(true);
      expect(p.endsWith("\n```")).toBe(true);
      expect(p.length).toBeLessThanOrEqual(900);
    }
    const bodyLines = (p: string) => p.split("\n").slice(1, -1);
    expect(bodyLines(pieces[1]!)[0]).toBe(bodyLines(pieces[0]!).at(-1)); // overlap
    expect(splitCodeBlock(code, 900, 0).flatMap(bodyLines)).toEqual(lines);
  });

  it("qna mode: code is emitted as its own chunks with β_code = β_dense (350)", () => {
    const chunks = chunkIt(`Intro.\n\n${code}\n\nOutro.`, "qna");
    expect(chunks[0]!.body).toBe("Intro.");
    expect(chunks.at(-1)!.body).toBe("Outro.");
    const codeChunks = chunks.slice(1, -1);
    expect(codeChunks.every((c) => c.kinds.join() === "code" && c.body.length <= 350)).toBe(true);
    expect(resolveParams("qna").codeMax).toBe(350);
    expect(resolveParams("qna", { denseMax: 500 }).codeMax).toBe(500);
  });

  it("code mode (merge on): code stays with the prose that introduces it", () => {
    const md = "# Setup\n\nInstall the package:\n\n```bash\npip install fastapi\n```\n\n## Run\n\nStart it.";
    const chunks = chunkIt(md, "code", { title: "Doc" });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toBe(
      "[Doc › Setup]\nInstall the package:\n\n```bash\npip install fastapi\n```\n\n## Run\n\nStart it.",
    );
    expect(chunks[0]!.kinds).toEqual(["paragraph", "code", "heading"]);
  });

  it("code mode: pieces of a long block fill chunks up to β_prose", () => {
    const chunks = chunkIt(`Intro.\n\n${code}`, "code");
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0]!.body.startsWith("Intro.\n\n```python\n")).toBe(true);
    expect(chunks.every((c) => c.body.length <= 1200)).toBe(true);
  });

  it("merge mode: a heading closes the chunk only once it reaches β_prose", () => {
    const p600 = "a".repeat(599) + ".";
    const p500 = "b".repeat(499) + ".";
    // 600 + 600 = 1200 >= β_prose → heading B starts a new chunk
    const closed = chunkIt(`# A\n\n${p600}\n\n${p600}\n\n# B\n\nTail.`, "code", { title: "D" });
    expect(closed.map((c) => c.breadcrumb)).toEqual(["D › A", "D › B"]);
    // 600 + 500 = 1100 < β_prose → heading B is kept inline as text
    const merged = chunkIt(`# A\n\n${p600}\n\n${p500}\n\n# B\n\nTail.`, "code", { title: "D" });
    expect(merged).toHaveLength(1);
    expect(merged[0]!.body.endsWith("\n\n# B\n\nTail.")).toBe(true);
    // merge off → every heading closes the chunk
    expect(chunkIt("# A\n\nx\n\n# B\n\ny", "code", { mergeSmallSections: false })).toHaveLength(2);
  });
});

describe("presets", () => {
  it("match the thesis / chunkers.py values", () => {
    expect(resolveParams("qna")).toEqual({
      proseMax: 700,
      denseMax: 350,
      codeMax: 350,
      rowsPerChunk: 2,
      densityThreshold: 0.02,
      codeOverlapLines: 1,
      mergeSmallSections: false,
      title: undefined,
    });
    expect(resolveParams("code")).toEqual({
      proseMax: 1200,
      denseMax: 800,
      codeMax: 900,
      rowsPerChunk: 4,
      densityThreshold: 0.02,
      codeOverlapLines: 1,
      mergeSmallSections: true,
      title: undefined,
    });
  });

  it("produce different chunkings of the same document", () => {
    const md = fixture("synthetic.md");
    const qna = chunkIt(md, "qna", { title: "s" });
    const code = chunkIt(md, "code", { title: "s" });
    expect(qna.length).toBeGreaterThan(code.length);
    expect(qna.map((c) => c.index)).toEqual(qna.map((_, i) => i));
  });

  it("validates mode and options", () => {
    // @ts-expect-error invalid mode
    expect(() => chunkIt("x", "prose")).toThrow(/unknown mode/);
    expect(() => chunkIt("x", "qna", { proseMax: 0 })).toThrow(RangeError);
    expect(() => chunkIt("x", "qna", { rowsPerChunk: 1.5 })).toThrow(RangeError);
    // @ts-expect-error non-string input
    expect(() => chunkIt(42, "qna")).toThrow(TypeError);
  });

  it("normalises CRLF line endings", () => {
    expect(chunkIt(fixture("synthetic_crlf.md"), "qna")).toEqual(chunkIt(fixture("synthetic.md"), "qna"));
  });

  it("returns no chunks for empty input", () => {
    expect(chunkIt("", "qna")).toEqual([]);
    expect(chunkIt("\n\n  \n", "code")).toEqual([]);
  });
});
