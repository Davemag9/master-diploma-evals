"""Run the reference Python chunker (rag_experiment/chunkers.py) over the
corpus folders and dump chunk texts as JSON, for parity comparison with the
JS port.  Read-only: does not write into rag_experiment (no .pyc files).

Usage: PYTHONDONTWRITEBYTECODE=1 <venv python> parity_py.py <rag_experiment dir> <out.json> [extra_md_dir ...]
"""
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from chunkers import adaptive_hybrid_chunks  # noqa: E402

QNA = dict(prose_max=700, dense_max=350, rows_per_chunk=2, density_threshold=0.02)  # CHUNKING_CONFIGS["adaptive_tight"]
CODE = dict(prose_max=1200, dense_max=800, rows_per_chunk=4, code_max=900,
            merge_small_sections=True)  # contextual_enrich.DOCS_PRESET / CHUNKING_CONFIGS["adaptive_docs"]


class FakeEmbedder:
    """Deterministic toy embedder; mirrored exactly in parity.mjs."""
    DIM = 8

    def embed_many(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for i, ch in enumerate(t):
                c = ord(ch)
                v[c % self.DIM] += 1.0 + (c % 7) * 0.1
            out.append(v)
        return out


result = []
folders = [root / "data" / "raw", root / "data" / "raw_fastapi"] + [Path(d) for d in sys.argv[3:]]
for folder in folders:
    for mode, preset in (("qna", QNA), ("code", CODE)):
        for p in sorted(folder.glob("*.md")):
            text = p.read_text(encoding="utf-8")
            for sem in (False, True):
                emb = FakeEmbedder() if sem else None
                chunks = adaptive_hybrid_chunks(p.name, text, embedder=emb, **preset)
                result.append({"folder": folder.name, "path": str(p), "file": p.name, "mode": mode,
                               "semantic": sem, "chunks": [c["text"] for c in chunks]})
Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
print(f"{len(result)} runs written")
