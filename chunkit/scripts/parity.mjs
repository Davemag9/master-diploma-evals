// Compare the JS port against the Python reference output from parity_py.py.
// Usage: npm run build && node scripts/parity.mjs <py_out.json>
import { readFileSync } from "node:fs";
import { chunkIt, chunkItAsync, titleFromFileName } from "../dist/index.js";

const [pyOut] = process.argv.slice(2);
const runs = JSON.parse(readFileSync(pyOut, "utf8"));

// mirror of FakeEmbedder in parity_py.py
const DIM = 8;
const fakeEmbed = async (texts) =>
  texts.map((t) => {
    const v = new Array(DIM).fill(0);
    for (const ch of t) {
      const c = ch.codePointAt(0);
      v[c % DIM] += 1.0 + (c % 7) * 0.1;
    }
    return v;
  });

let same = 0, diff = 0, pyTotal = 0, jsTotal = 0;
const summary = {};
for (const run of runs) {
  const md = readFileSync(run.path, "utf8");
  const opts = { title: titleFromFileName(run.file) };
  const js = run.semantic
    ? (await chunkItAsync(md, run.mode, { ...opts, embed: fakeEmbed })).map((c) => c.text)
    : chunkIt(md, run.mode, opts).map((c) => c.text);
  const key = `${run.folder}/${run.mode}/${run.semantic ? "semantic" : "no-embed"}`;
  summary[key] ??= { docs: 0, identical: 0, pyChunks: 0, jsChunks: 0 };
  const s = summary[key];
  s.docs++; s.pyChunks += run.chunks.length; s.jsChunks += js.length;
  pyTotal += run.chunks.length; jsTotal += js.length;
  const ok = js.length === run.chunks.length && js.every((t, i) => t === run.chunks[i]);
  if (ok) { same++; s.identical++; }
  else {
    diff++;
    const i = js.findIndex((t, k) => t !== run.chunks[k]);
    console.log(`DIFF ${key} ${run.file}: py=${run.chunks.length} js=${js.length}, first mismatch at #${i}`);
    console.log("  py:", JSON.stringify(run.chunks[i]?.slice(0, 200)));
    console.log("  js:", JSON.stringify(js[i]?.slice(0, 200)));
  }
}
console.table(summary);
console.log(`runs identical: ${same}/${same + diff}; chunks py=${pyTotal} js=${jsTotal}`);
process.exit(diff ? 1 : 0);
