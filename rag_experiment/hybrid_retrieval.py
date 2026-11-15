"""Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with RRF.

Motivation from the sweep results: the best chunkers all saturate at
recall@3 = 0.972 and recall@5 = 1.000, so better *segmentation* alone can no
longer move recall@3 on this benchmark. The remaining errors are a retrieval
problem, not a chunking problem -- and they are concentrated on exactly the
kind of query where dense embeddings are weakest: short, literal facts
(a phone number, a fee in euros, a decree number). Lexical BM25 matches
those tokens exactly.

So this module combines the thesis's structural chunking with the standard
industry recipe for fact-heavy corpora:

    Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)
        score(d) = sum over retrievers of  1 / (rrf_k + rank_r(d))

RRF needs no score normalisation between the two very differently-scaled
retrievers, which is why it is the default fusion method in Elasticsearch,
Weaviate, Qdrant and LangChain's EnsembleRetriever.

Costs zero additional API calls: BM25 is local, and the dense vectors for
every chunker are already cached from the sweep.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from chunkers import build_chunks, CHUNKING_CONFIGS, EMBEDDER_REQUIRED_KINDS
from embed_client import EmbeddingClient
from qa_dataset import QA_ITEMS
from retrieval_sweep import RAW_DIR, is_relevant_chunk, build_faiss_index

OUT_PATH = Path(__file__).parent / "data" / "results" / "hybrid_results.json"

TOP_KS = [1, 3, 5]
RRF_K = 60  # standard constant from the RRF paper
CANDIDATE_DEPTH = 20  # how deep each retriever's ranking is fused

# Chunkers to test hybrid retrieval on: the top structural performers plus
# the strongest external baseline, so the comparison stays honest.
METHODS = [
    "adaptive_tight",
    "table_aware_3row",
    "hierarchical_800",
    "lc_recursive_1500",
]

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Lowercase word/number tokens. Numbers survive intact, which is the
    whole point for a corpus of fees, deadlines and phone numbers."""
    return TOKEN_RE.findall(text.lower())


def load_docs():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}


def rrf_fuse(rankings, rrf_k=RRF_K, weights=None):
    """rankings: list of lists of chunk indices, best-first. Returns indices
    sorted by fused (optionally weighted) RRF score."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + weight / (rrf_k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def cascade_rerank(dense_ranking, bm25_ranking, depth, bm25_weight):
    """Dense-first cascade: keep only the dense retriever's top-`depth`
    candidates (which already contain the gold chunk ~100% of the time at
    depth=5), then gently reorder them using the lexical ranking.

    Unlike full fusion this can never pull in a chunk the dense retriever
    did not rank highly, so a weak BM25 can reorder but not derail.
    """
    candidates = dense_ranking[:depth]
    candidate_set = set(candidates)
    bm25_within = [i for i in bm25_ranking if i in candidate_set]
    # any candidate BM25 never scored goes to the back of its ranking
    bm25_within += [i for i in candidates if i not in set(bm25_within)]
    fused = rrf_fuse([candidates, bm25_within], weights=[1.0, bm25_weight])
    return fused + [i for i in dense_ranking if i not in candidate_set]


def evaluate(chunks, dense_ranks, bm25_ranks, mode, bm25_weight=1.0, cascade_depth=5):
    """mode: 'dense' | 'bm25' | 'hybrid' (full weighted RRF) | 'cascade'"""
    hits = {k: 0 for k in TOP_KS}
    ctx_chars = []
    for qi, item in enumerate(QA_ITEMS):
        if mode == "dense":
            ranking = dense_ranks[qi]
        elif mode == "bm25":
            ranking = bm25_ranks[qi]
        elif mode == "cascade":
            ranking = cascade_rerank(dense_ranks[qi], bm25_ranks[qi], cascade_depth, bm25_weight)
        else:
            ranking = rrf_fuse([dense_ranks[qi], bm25_ranks[qi]], weights=[1.0, bm25_weight])
        retrieved = [chunks[i] for i in ranking[: max(TOP_KS)]]
        for k in TOP_KS:
            if any(is_relevant_chunk(c, item) for c in retrieved[:k]):
                hits[k] += 1
        ctx_chars.append(sum(len(c["text"]) for c in retrieved[:3]))
    n = len(QA_ITEMS)
    return {
        "recall_at_1": round(hits[1] / n, 3),
        "recall_at_3": round(hits[3] / n, 3),
        "recall_at_5": round(hits[5] / n, 3),
        "avg_context_chars_top3": round(sum(ctx_chars) / n, 1),
    }


def main():
    docs = load_docs()
    embedder = EmbeddingClient()

    print(f"Embedding {len(QA_ITEMS)} questions (cached)...", file=sys.stderr)
    q_vectors = embedder.embed_many([item["question"] for item in QA_ITEMS])
    q_arr = np.array(q_vectors, dtype="float32")
    faiss.normalize_L2(q_arr)

    results = {}
    for method in METHODS:
        needs_embedder = CHUNKING_CONFIGS[method]["kind"] in EMBEDDER_REQUIRED_KINDS
        chunks = []
        for name, text in docs.items():
            kwargs = {"embedder": embedder} if needs_embedder else {}
            chunks.extend(build_chunks(method, name, text, **kwargs))
        print(f"\n{method}: {len(chunks)} chunks", file=sys.stderr)

        # dense ranking (embeddings all cached from the sweep)
        vectors = embedder.embed_many([c["text"] for c in chunks])
        index = build_faiss_index(vectors)
        depth = min(CANDIDATE_DEPTH, len(chunks))
        _, dense_idx = index.search(q_arr, depth)
        dense_ranks = [list(row) for row in dense_idx]

        # lexical ranking
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        bm25_ranks = []
        for item in QA_ITEMS:
            scores = bm25.get_scores(tokenize(item["question"]))
            bm25_ranks.append(list(np.argsort(scores)[::-1][:depth]))

        method_results = {
            "dense": evaluate(chunks, dense_ranks, bm25_ranks, "dense"),
            "bm25": evaluate(chunks, dense_ranks, bm25_ranks, "bm25"),
        }
        # equal-weight fusion (the naive/default recipe) plus a weight sweep,
        # and the dense-first cascade at a few depths
        for w in (1.0, 0.5, 0.3, 0.15):
            label = "hybrid_rrf" if w == 1.0 else f"hybrid_rrf_w{w}"
            method_results[label] = evaluate(chunks, dense_ranks, bm25_ranks, "hybrid", bm25_weight=w)
        for depth in (5, 10):
            for w in (0.5, 0.3, 0.15):
                method_results[f"cascade_d{depth}_w{w}"] = evaluate(
                    chunks, dense_ranks, bm25_ranks, "cascade", bm25_weight=w, cascade_depth=depth
                )
        results[method] = method_results
        for mode, r in method_results.items():
            print(f"  {mode:18s} R@1={r['recall_at_1']:.3f} R@3={r['recall_at_3']:.3f} "
                  f"R@5={r['recall_at_5']:.3f} ctx@3={r['avg_context_chars_top3']:.0f}",
                  file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== HYBRID RETRIEVAL (BM25 + dense) ===")
    print(f"{'method':18s} {'mode':18s} {'R@1':>7s} {'R@3':>7s} {'R@5':>7s} {'ctx@3':>8s}")
    for method, modes in results.items():
        base = modes["dense"]
        for mode, r in modes.items():
            flag = ""
            if mode != "dense":
                d1 = r["recall_at_1"] - base["recall_at_1"]
                d3 = r["recall_at_3"] - base["recall_at_3"]
                if d3 > 0 or (d3 == 0 and d1 > 0):
                    flag = "  <-- beats pure dense"
                elif d3 < 0:
                    flag = "  (worse)"
            print(f"{method:18s} {mode:18s} {r['recall_at_1']:7.3f} {r['recall_at_3']:7.3f} "
                  f"{r['recall_at_5']:7.3f} {r['avg_context_chars_top3']:8.0f}{flag}")
        print()
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
