"""Experimentally find the best chunking parameters for RAG retrieval.

For every chunking config (whole-document baseline, several fixed-size
grids, and structure-aware chunkers), we:
  1. split every document into chunks,
  2. embed every chunk with gemini-embedding-001 (cached, rate-limited),
  3. build a FAISS cosine-similarity index over the chunks,
  4. for every one of the 107 QA questions, retrieve the top-k chunks and
     check whether the fact-bearing chunk was actually retrieved,
  5. aggregate recall@1/3/5 and how much "noisy" context (chars) would be
     fed to the LLM if we used that config.

This uses ONLY the embedding API (generous free-tier quota), never the
tightly-limited gemini-flash-latest chat model, so it can run over the
whole 107-question benchmark instead of a tiny sample.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import faiss

from chunkers import build_chunks, CHUNKING_CONFIGS, EMBEDDER_REQUIRED_KINDS
from embed_client import EmbeddingClient, AllKeysExhausted
from qa_dataset import QA_ITEMS

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_PATH = Path(__file__).parent / "data" / "results" / "sweep_results.json"

TOP_KS = [1, 3, 5]

# One representative config per FAMILY, not the full grid — the embedding
# API turned out to cap at 1000 requests/day/key, so an exhaustive grid
# search over every size x overlap combination isn't realistic. This set
# still tells the story: naive fixed-size vs structure-aware vs
# hierarchical-breadcrumb vs table-aware vs semantic, plus the whole-doc
# baseline that "beremo tsili faily" argues against.
FINAL_CONFIGS = [
    # --- baselines from the first wave of the experiment ---
    "whole_doc",
    "fixed_400_0",
    "fixed_400_100",
    "fixed_1500_0",
    "structural_700",
    "hierarchical_800",
    "table_aware_3row",
    "semantic_600",
    # --- external / industry baselines (LangChain) ---
    "lc_recursive_400",
    "lc_recursive_800",
    "lc_recursive_1500",
    "lc_markdown_500",
    "lc_markdown_800",
    "lc_token_512",
    # --- proposed method: adaptive structural-semantic segmentation ---
    "adaptive_nosem",   # ablation: structural rules only, no semantic split point
    "adaptive_hybrid",  # full method
    "adaptive_tight",
    "adaptive_wide",
    # --- external baseline: RAPTOR tree (collapsed-tree retrieval) ---
    "raptor",
]


def load_docs():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}


def is_relevant_chunk(chunk, item):
    # A RAPTOR summary node can span several source documents, so match against
    # the node's full document set rather than a single "doc" label.
    chunk_docs = chunk.get("docs") or [chunk["doc"]]
    if item["gold_doc"] not in chunk_docs:
        return False
    text_lower = chunk["text"].lower()
    return any(kw.lower() in text_lower for kw in item["fact_keywords"])


def progress_printer(label):
    start = time.time()

    def cb(done, total):
        if done == total or done % 25 == 0:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{label}] {done}/{total} embedded "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                file=sys.stderr,
            )

    return cb


def build_faiss_index(vectors):
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


def main():
    docs = load_docs()
    embedder = EmbeddingClient()

    results = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {len(QA_ITEMS)} questions...", file=sys.stderr)
    q_texts = [item["question"] for item in QA_ITEMS]
    try:
        q_vectors = embedder.embed_many(q_texts, on_progress=progress_printer("questions"))
    except AllKeysExhausted as e:
        print(f"STOP (before starting): {e}")
        return

    config_names = [c for c in FINAL_CONFIGS if c not in results]
    print(f"Already done: {[c for c in FINAL_CONFIGS if c in results]}")
    print(f"To run: {config_names}")
    for ci, config_name in enumerate(config_names, 1):
        print(f"\n[{ci}/{len(config_names)}] config = {config_name}", file=sys.stderr)

        if config_name == "raptor":
            # Collapsed-tree retrieval: leaves + every summary node, one index.
            from generation_compare_v2 import build_method_chunks
            all_chunks = build_method_chunks("raptor", docs, embedder)
            needs_embedder = False
        else:
            needs_embedder = CHUNKING_CONFIGS[config_name]["kind"] in EMBEDDER_REQUIRED_KINDS
            all_chunks = []
        try:
            for doc_name, text in (docs.items() if config_name != "raptor" else []):
                kwargs = {"embedder": embedder} if needs_embedder else {}
                all_chunks.extend(build_chunks(config_name, doc_name, text, **kwargs))
        except AllKeysExhausted as e:
            print(f"STOP (while building chunks): {e}")
            print("Progress so far saved to embeddings cache; re-run this script later to continue.")
            break
        print(f"  {len(all_chunks)} chunks to embed", file=sys.stderr)

        chunk_texts = [c["text"] for c in all_chunks]
        try:
            chunk_vectors = embedder.embed_many(chunk_texts, on_progress=progress_printer(config_name))
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            print("Progress so far saved to embeddings cache; re-run this script later to continue.")
            break

        index = build_faiss_index(chunk_vectors)

        q_arr = np.array(q_vectors, dtype="float32")
        faiss.normalize_L2(q_arr)
        max_k = max(TOP_KS)
        _, top_idx = index.search(q_arr, min(max_k, len(all_chunks)))

        hits = {k: 0 for k in TOP_KS}
        context_chars_at3 = []
        for qi, item in enumerate(QA_ITEMS):
            retrieved = [all_chunks[idx] for idx in top_idx[qi]]
            for k in TOP_KS:
                if any(is_relevant_chunk(c, item) for c in retrieved[:k]):
                    hits[k] += 1
            context_chars_at3.append(sum(len(c["text"]) for c in retrieved[:3]))

        n = len(QA_ITEMS)
        results[config_name] = {
            "num_chunks": len(all_chunks),
            "avg_chunk_chars": round(sum(len(t) for t in chunk_texts) / len(chunk_texts), 1),
            "recall_at_1": round(hits[1] / n, 3),
            "recall_at_3": round(hits[3] / n, 3),
            "recall_at_5": round(hits[5] / n, 3),
            "avg_context_chars_top3": round(sum(context_chars_at3) / n, 1),
        }
        r = results[config_name]
        print(
            f"  recall@1={r['recall_at_1']} recall@3={r['recall_at_3']} "
            f"recall@5={r['recall_at_5']} avg_ctx_chars@3={r['avg_context_chars_top3']}",
            file=sys.stderr,
        )
        OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY (sorted by recall@3) ===")
    header = f"{'config':20s} {'chunks':>7s} {'avg_len':>8s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'ctx@3(chars)':>13s}"
    print(header)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["recall_at_3"]):
        print(
            f"{name:20s} {r['num_chunks']:7d} {r['avg_chunk_chars']:8.0f} "
            f"{r['recall_at_1']:6.2f} {r['recall_at_3']:6.2f} {r['recall_at_5']:6.2f} "
            f"{r['avg_context_chars_top3']:13.0f}"
        )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
