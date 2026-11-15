"""RAPTOR ("RAG tree") baseline -- Recursive Abstractive Processing for
Tree-Organized Retrieval (Sarthi et al., ICLR 2024).

This is the strongest published "smart chunking" competitor to the thesis's
method, and it works on a completely different principle: instead of trying
to cut the document at better boundaries, it BUILDS A TREE of increasingly
abstract LLM-written summaries on top of ordinary chunks, then retrieves
over every level at once.

Pipeline:
  1. leaves      -- ordinary chunks (we use LangChain's RecursiveCharacter
                    splitter, i.e. the standard setup a practitioner would
                    pair RAPTOR with -- so this is a fair external baseline,
                    not "RAPTOR on top of our method")
  2. embed       -- gemini-embedding-001
  3. cluster     -- UMAP dimensionality reduction + Gaussian Mixture, with
                    BIC choosing the cluster count (as in the paper)
  4. summarize   -- one LLM call per cluster produces a parent node
  5. recurse     -- repeat on the summaries until few enough nodes remain
  6. retrieval   -- "collapsed tree": leaves AND all summary nodes go into a
                    single flat index, so a query can match either a raw
                    passage or an abstract summary

Tree construction costs LLM calls (one per cluster per level), which is the
expensive part on a free-tier quota, so the built tree is cached to disk and
the build is resumable: already-written summaries are never regenerated.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from chunkers import build_chunks
from embed_client import EmbeddingClient, AllKeysExhausted
from gen_client import (
    GenerationClient,
    AllKeysExhausted as GenExhausted,
    BULK_MODEL,
    BULK_DAILY_CAP,
    BULK_MIN_INTERVAL,
)

RAW_DIR = Path(__file__).parent / "data" / "raw"
TREE_PATH = Path(__file__).parent / "data" / "cache" / "raptor_tree.json"

LEAF_CONFIG = "lc_recursive_800"  # standard practitioner leaf chunking
MAX_LEVELS = 3
MIN_NODES_TO_CLUSTER = 8  # stop recursing once the level is this small
TARGET_CLUSTER_SIZE = 8  # aim for ~this many nodes per cluster
RANDOM_STATE = 42

SUMMARY_PROMPT = """Нижче наведено кілька фрагментів з консульських сторінок Посольства України в Латвії.

Напиши стислий структурований підсумок цих фрагментів українською мовою. КРИТИЧНО ВАЖЛИВО: збережи ВСІ конкретні факти дослівно — суми зборів, валюти, терміни, дати, номери телефонів, назви документів, номери постанов. Не узагальнюй числа і не пропускай їх.

ФРАГМЕНТИ:
{chunks}

СТИСЛИЙ ПІДСУМОК (зі збереженням усіх конкретних фактів):"""


def load_docs():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}


def build_leaves():
    docs = load_docs()
    leaves = []
    for name, text in docs.items():
        leaves.extend(build_chunks(LEAF_CONFIG, name, text))
    return leaves


def reduce_dims(vectors, n_components=10):
    """UMAP reduction, as in the RAPTOR paper (GMM on raw 3072-dim embeddings
    is both slow and degenerate)."""
    import umap

    n = len(vectors)
    n_components = min(n_components, max(2, n - 2))
    n_neighbors = min(15, max(2, n - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(np.array(vectors, dtype="float32"))


def pick_clusters(reduced, target_size=TARGET_CLUSTER_SIZE, max_candidates=12):
    """Choose the cluster count by BIC, around a target average cluster size."""
    n = len(reduced)
    guess = max(2, round(n / target_size))
    candidates = sorted({
        c for c in range(max(2, guess - 3), min(n - 1, guess + 4)) if 2 <= c < n
    })[:max_candidates]
    if not candidates:
        return np.zeros(n, dtype=int), 1

    best_labels, best_bic, best_k = None, np.inf, 1
    for k in candidates:
        gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, covariance_type="full")
        try:
            labels = gm.fit_predict(reduced)
            bic = gm.bic(reduced)
        except Exception:
            continue
        if bic < best_bic:
            best_labels, best_bic, best_k = labels, bic, k
    if best_labels is None:
        return np.zeros(n, dtype=int), 1
    return best_labels, best_k


def load_tree():
    if TREE_PATH.exists():
        return json.loads(TREE_PATH.read_text(encoding="utf-8"))
    return {"leaf_config": LEAF_CONFIG, "levels": {}}


def save_tree(tree):
    TREE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TREE_PATH.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")


def get_all_nodes(tree):
    """Collapsed-tree retrieval set: leaves + every summary node."""
    nodes = []
    for level in sorted(tree["levels"], key=int):
        nodes.extend(tree["levels"][level])
    return nodes


def main():
    embedder = EmbeddingClient()
    # Cluster summarisation is bulk offline indexing, so it runs on the lighter
    # model (its own, much larger free-tier quota) and leaves the answer
    # model's tiny daily budget untouched for the actual QA evaluation.
    gen = GenerationClient(
        model=BULK_MODEL, daily_cap=BULK_DAILY_CAP, min_interval=BULK_MIN_INTERVAL
    )
    tree = load_tree()
    print(f'summarisation model: {BULK_MODEL} (quota left today: {gen.remaining_today()})')

    # --- level 0: leaves ---
    if "0" not in tree["levels"]:
        leaves = build_leaves()
        tree["levels"]["0"] = [
            {"text": c["text"], "doc": c["doc"], "level": 0} for c in leaves
        ]
        save_tree(tree)
    print(f"level 0: {len(tree['levels']['0'])} leaf chunks ({LEAF_CONFIG})")

    def on_wait(seconds):
        print(f"    [rate limit] waiting {seconds:.0f}s...", file=sys.stderr)

    # --- recursive summarisation ---
    for level in range(1, MAX_LEVELS + 1):
        prev = tree["levels"][str(level - 1)]
        if len(prev) < MIN_NODES_TO_CLUSTER:
            print(f"level {level}: only {len(prev)} nodes below -- stopping recursion")
            break

        key = str(level)
        pending_key = f"pending_{level}"

        # Cluster assignment is deterministic, so compute it once and cache it.
        if pending_key not in tree:
            print(f"level {level}: embedding {len(prev)} nodes for clustering...")
            try:
                vectors = embedder.embed_many([n["text"] for n in prev])
            except AllKeysExhausted as e:
                print(f"STOP (embeddings): {e}")
                return
            reduced = reduce_dims(vectors)
            labels, k = pick_clusters(reduced)
            clusters = {}
            for idx, lab in enumerate(labels):
                clusters.setdefault(str(int(lab)), []).append(idx)
            tree[pending_key] = clusters
            save_tree(tree)
            print(f"level {level}: {k} clusters (avg {len(prev)/max(k,1):.1f} nodes each)")

        clusters = tree[pending_key]
        tree["levels"].setdefault(key, [])
        done_ids = {n.get("cluster_id") for n in tree["levels"][key]}

        todo = [cid for cid in sorted(clusters, key=int) if cid not in done_ids]
        print(f"level {level}: {len(todo)} cluster summaries to generate "
              f"({len(done_ids)} already cached)")

        for cid in todo:
            member_idx = clusters[cid]
            members = [prev[i] for i in member_idx]
            joined = "\n\n---\n\n".join(m["text"] for m in members)
            # keep the prompt within a sane size
            if len(joined) > 20000:
                joined = joined[:20000]
            try:
                summary = gen.generate(SUMMARY_PROMPT.format(chunks=joined), on_wait=on_wait)
            except GenExhausted as e:
                print(f"STOP (generation quota): {e}")
                print("Tree build is resumable -- re-run this script when quota resets.")
                save_tree(tree)
                return
            docs_in_cluster = sorted({m["doc"] for m in members})
            tree["levels"][key].append({
                "text": summary,
                "doc": docs_in_cluster[0] if len(docs_in_cluster) == 1 else "MULTI",
                "docs": docs_in_cluster,
                "level": level,
                "cluster_id": cid,
                "n_members": len(members),
            })
            save_tree(tree)
            print(f"  [{len(tree['levels'][key])}/{len(clusters)}] cluster {cid}: "
                  f"{len(members)} nodes -> {len(summary)} chars")

        print(f"level {level}: complete ({len(tree['levels'][key])} summary nodes)")

    total = sum(len(v) for v in tree["levels"].values())
    print(f"\nRAPTOR tree complete: {total} nodes across "
          f"{len(tree['levels'])} levels -> {TREE_PATH}")
    for lvl in sorted(tree["levels"], key=int):
        print(f"  level {lvl}: {len(tree['levels'][lvl])} nodes")


if __name__ == "__main__":
    main()
