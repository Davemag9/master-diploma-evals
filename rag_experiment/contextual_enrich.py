"""Contextual chunk augmentation: an LLM writes one short sentence situating
each chunk inside its document, and that sentence is prepended to the chunk
BEFORE embedding.

Motivation, from the failures this corpus produced: for several questions the
retriever pulled the right *document* but the wrong *fragment* -- the fact was
inside a code block whose own text ("ALGORITHM = \"HS256\"") shares almost no
vocabulary with the question ("which signing algorithm..."). A code fragment
carries the fact but not the words a user would search with. A one-line
context sentence restores exactly those words.

This is the standard "contextual retrieval" recipe, and it is the same class
of technique as RAPTOR's node summaries -- both spend LLM calls at index time
-- so comparing an enriched chunker against RAPTOR stays fair.

Costs one LLM call per chunk, so:
  - it runs on BULK_MODEL (the light model with the large free-tier quota),
    exactly like the RAPTOR summaries, never on the answer model;
  - every context line is cached on disk by content hash, so a run that hits
    the daily quota can simply be resumed the next day.

Usage:
    python contextual_enrich.py <config_name>     # e.g. adaptive_tight
"""
import hashlib
import json
import sys
from pathlib import Path

from chunkers import build_chunks
from gen_client import (
    GenerationClient,
    AllKeysExhausted,
    BULK_MODEL,
    BULK_DAILY_CAP,
    BULK_MIN_INTERVAL,
)

CACHE_PATH = Path(__file__).parent / "data" / "cache" / "chunk_context.json"

PROMPT = """<document>
{doc_excerpt}
</document>

Here is a fragment taken from that document:
<fragment>
{chunk}
</fragment>

Write ONE short sentence (max 25 words) that situates this fragment within the document, so it can be found by search. Name the feature, class, parameter or setting it is about, using the words a developer would search with. Answer with the sentence only, nothing else."""

DOC_EXCERPT_CHARS = 4000


def _key(doc_name, chunk_text):
    return hashlib.sha256(f"{doc_name}\x00{chunk_text}".encode("utf-8")).hexdigest()


class ContextCache:
    def __init__(self, path=CACHE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._dirty = 0

    def get(self, doc_name, chunk_text):
        return self._data.get(_key(doc_name, chunk_text))

    def put(self, doc_name, chunk_text, context):
        self._data[_key(doc_name, chunk_text)] = context
        self._dirty += 1
        if self._dirty >= 5:
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        self._dirty = 0

    def __len__(self):
        return len(self._data)


def enrich_chunks(chunks, docs, cache, gen=None, on_progress=None):
    """Return a new chunk list whose text is 'context sentence + original'.

    Chunks whose context is not cached yet are generated with `gen`; if `gen`
    is None (or its quota runs out) the chunk is returned unchanged, so an
    incomplete run still produces a usable, if partly un-enriched, index.
    """
    out = []
    exhausted = False
    for i, chunk in enumerate(chunks):
        context = cache.get(chunk["doc"], chunk["text"])
        if context is None and gen is not None and not exhausted:
            doc_text = docs.get(chunk["doc"], "")[:DOC_EXCERPT_CHARS]
            prompt = PROMPT.format(doc_excerpt=doc_text, chunk=chunk["text"])
            try:
                context = gen.generate(prompt).strip().replace("\n", " ")
                cache.put(chunk["doc"], chunk["text"], context)
            except AllKeysExhausted as e:
                print(f"STOP (context generation quota): {e}", file=sys.stderr)
                exhausted = True
                context = None
            except RuntimeError as e:
                print(f"  skip {chunk['doc']}: {e}", file=sys.stderr)
                context = None
        new = dict(chunk)
        if context:
            new["text"] = f"{context}\n{chunk['text']}"
            new["context"] = context
        out.append(new)
        if on_progress:
            on_progress(i + 1, len(chunks))
    cache.save()
    return out


# The light model's real free-tier ceiling is far above the conservative cap
# the RAPTOR build used. The client detects a real "PerDay" 429 and rotates
# keys on its own, so a generous local cap only avoids stopping early.
BULK_CAP_FOR_CONTEXT = 400


# The documentation preset found by tune_code_chunking.py: wider budgets and
# merged small sections, so a chunk is a whole answerable section rather than
# a two-sentence fragment.
DOCS_PRESET = dict(prose_max=1200, dense_max=800, rows_per_chunk=4,
                   code_max=900, merge_small_sections=True)
TIGHT_PRESET = dict(prose_max=700, dense_max=350, rows_per_chunk=2, code_max=700)


def adaptive_chunks_for(name, text, preset="docs"):
    from chunkers import adaptive_hybrid_chunks
    kwargs = DOCS_PRESET if preset == "docs" else TIGHT_PRESET
    return adaptive_hybrid_chunks(name, text, embedder=None, **kwargs)


def main():
    from retrieval_sweep_fastapi import RAW_DIR

    preset = sys.argv[1] if len(sys.argv) > 1 else "docs"
    config_name = f"adaptive_{preset}"
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}

    chunks = []
    for name, text in docs.items():
        chunks.extend(adaptive_chunks_for(name, text, preset))

    cache = ContextCache()
    gen = GenerationClient(model=BULK_MODEL, daily_cap=BULK_CAP_FOR_CONTEXT,
                           min_interval=BULK_MIN_INTERVAL, round_robin=True)
    todo = sum(1 for c in chunks if cache.get(c["doc"], c["text"]) is None)
    print(f"{config_name}: {len(chunks)} chunks, {todo} need a context line "
          f"({len(cache)} cached, model {BULK_MODEL}, quota left {gen.remaining_today()})")

    def progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"  [{done}/{total}] contexts ready", file=sys.stderr)

    enrich_chunks(chunks, docs, cache, gen=gen, on_progress=progress)
    still = sum(1 for c in chunks if cache.get(c["doc"], c["text"]) is None)
    print(f"done: {len(chunks) - still}/{len(chunks)} chunks have a context line")
    if still:
        print("re-run this script after the quota resets to finish the rest")


if __name__ == "__main__":
    main()
