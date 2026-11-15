"""Generation comparison for arbitrary retrieval methods, including RAPTOR.

Generalises generation_compare_multi.py: any method named on the command
line is evaluated on all 107 questions and stored under
    generation_full.json -> <question_id> -> "chunked_variants" -> <method>
so it sits alongside the methods already measured, and the whole top-N
table can be recomputed from one file.

A method name is either:
  - a key in chunkers.CHUNKING_CONFIGS (normal flat chunker), or
  - "raptor" -- collapsed-tree retrieval over the cached RAPTOR tree
    (leaves + every LLM-written summary node in a single flat index)

Usage:
    python generation_compare_v2.py adaptive_hybrid raptor lc_markdown_800
    python generation_compare_v2.py --limit 60 adaptive_hybrid

Fully resumable: any (question, method) pair already present is skipped, so
re-running after the daily free-tier quota resets just continues.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import faiss

from chunkers import build_chunks, CHUNKING_CONFIGS, EMBEDDER_REQUIRED_KINDS
from embed_client import EmbeddingClient, AllKeysExhausted as EmbedExhausted
from gen_client import GenerationClient, AllKeysExhausted
from qa_dataset import QA_ITEMS, check_answer_precise

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_PATH = Path(__file__).parent / "data" / "results" / "generation_full.json"
RAPTOR_TREE_PATH = Path(__file__).parent / "data" / "cache" / "raptor_tree.json"

TOP_K = 3

PROMPT_TEMPLATE = """Ти консульський асистент. Дай КОРОТКУ і ТОЧНУ відповідь на питання громадянина, спираючись ВИКЛЮЧНО на контекст нижче. Якщо в контексті немає точної відповіді - чесно скажи, що інформації недостатньо. Не вигадуй фактів.

КОНТЕКСТ:
{context}

ПИТАННЯ: {question}

ВІДПОВІДЬ:"""


def load_docs():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}


def build_index(vectors):
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


def retrieve_top(index, chunks, query_vec, k, bm25=None, question=None,
                 cascade_depth=10, bm25_weight=0.5):
    """Dense top-k, optionally reordered by the BM25 cascade.

    When `bm25` is given, the dense retriever's top-`cascade_depth` candidates
    are gently reordered using the lexical ranking (RRF, dense weighted 1.0 vs
    BM25 `bm25_weight`). The cascade can reorder but never introduce a chunk
    the dense retriever did not already rank highly -- which is what made it
    help where equal-weight fusion hurt.
    """
    q = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(q)
    depth = max(k, cascade_depth) if bm25 is not None else k
    _, idx = index.search(q, min(depth, len(chunks)))
    dense_ranking = list(idx[0])

    if bm25 is None:
        return [chunks[i] for i in dense_ranking[:k]]

    from hybrid_retrieval import cascade_rerank, tokenize
    scores = bm25.get_scores(tokenize(question))
    bm25_ranking = list(np.argsort(scores)[::-1][: max(depth, 20)])
    ranking = cascade_rerank(dense_ranking, bm25_ranking, cascade_depth, bm25_weight)
    return [chunks[i] for i in ranking[:k]]


def parse_method(spec):
    """'adaptive_tight+cascade' -> ('adaptive_tight', True)."""
    if spec.endswith("+cascade"):
        return spec[: -len("+cascade")], True
    return spec, False


def build_method_chunks(method, docs, embedder):
    """Return the list of retrievable units for a method."""
    if method == "raptor":
        if not RAPTOR_TREE_PATH.exists():
            raise RuntimeError(
                f"{RAPTOR_TREE_PATH} not found -- run raptor_index.py first"
            )
        tree = json.loads(RAPTOR_TREE_PATH.read_text(encoding="utf-8"))
        nodes = []
        for level in sorted(tree["levels"], key=int):
            for node in tree["levels"][level]:
                nodes.append({
                    "doc": node["doc"],
                    "docs": node.get("docs", [node["doc"]]),
                    "text": node["text"],
                    "level": node["level"],
                })
        return nodes

    if method not in CHUNKING_CONFIGS:
        raise ValueError(f"unknown method: {method}")
    needs_embedder = CHUNKING_CONFIGS[method]["kind"] in EMBEDDER_REQUIRED_KINDS
    chunks = []
    for name, text in docs.items():
        kwargs = {"embedder": embedder} if needs_embedder else {}
        chunks.extend(build_chunks(method, name, text, **kwargs))
    return chunks


def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results):
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = [a for a in sys.argv[1:]]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i : i + 2]
    methods = args
    if not methods:
        print(__doc__)
        print("ERROR: name at least one method to evaluate.")
        return

    docs = load_docs()
    embedder = EmbeddingClient()
    gen = GenerationClient()

    results = load_results()
    for row in results.values():
        row.setdefault("chunked_variants", {})

    print(f"Methods: {methods}")
    print(f"Gen quota remaining today (all keys): {gen.remaining_today()}")

    # Build + index every requested method (embeddings are cached, so this is
    # fast/free for anything already swept).
    method_data = {}
    for spec in methods:
        base, use_cascade = parse_method(spec)
        chunks = build_method_chunks(base, docs, embedder)
        try:
            vectors = embedder.embed_many([c["text"] for c in chunks])
        except EmbedExhausted as e:
            print(f"STOP (embedding quota) while indexing {spec}: {e}")
            return
        bm25 = None
        if use_cascade:
            from rank_bm25 import BM25Okapi
            from hybrid_retrieval import tokenize
            bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        method_data[spec] = (chunks, build_index(vectors), bm25)
        print(f"{spec}: {len(chunks)} units indexed"
              f"{' (+BM25 cascade rerank)' if use_cascade else ''}")

    def on_wait(seconds):
        print(f"    [rate limit] waiting {seconds:.0f}s...", file=sys.stderr)

    todo = []
    for item in QA_ITEMS:
        row = results.setdefault(item["id"], {
            "id": item["id"], "question": item["question"],
            "gold_doc": item["gold_doc"], "chunked_variants": {},
        })
        for method in methods:
            if row["chunked_variants"].get(method) is None:
                todo.append((item, method))
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} (question, method) pairs left to process this run")

    processed = 0
    for item, method in todo:
        chunks, index, bm25 = method_data[method]
        q_vec = embedder.embed(item["question"])
        top = retrieve_top(index, chunks, q_vec, TOP_K,
                           bm25=bm25, question=item["question"])
        context = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top)
        prompt = PROMPT_TEMPLATE.format(context=context, question=item["question"])
        answer = None
        quota_exhausted = False
        # Outer resilience layer: a sustained model-wide 503 "high demand"
        # spike can outlast the client's own per-call retry budget. Rather
        # than crash the whole run (losing the rest of the day's quota
        # window to a script that needs to be manually restarted), wait
        # longer and retry the SAME item a few more times before giving up.
        for outer_attempt in range(5):
            try:
                answer = gen.generate(prompt, on_wait=on_wait)
                break
            except AllKeysExhausted as e:
                print(f"STOP: {e}")
                quota_exhausted = True
                break
            except RuntimeError as e:
                if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                    raise
                wait_s = 60 * (outer_attempt + 1)
                print(f"  sustained overload on {item['id']}::{method}, "
                      f"waiting {wait_s}s before outer retry {outer_attempt + 1}/5",
                      file=sys.stderr)
                time.sleep(wait_s)
        if quota_exhausted:
            break
        if answer is None:
            print(f"SKIP (still overloaded after outer retries): {item['id']}::{method}")
            continue
        row = results[item["id"]]
        entry = {
            "retrieved_docs": [c["doc"] for c in top],
            "context_chars": len(context),
            "answer": answer,
            "answer_chars": len(answer),
            "precise": check_answer_precise(answer, item),
        }
        if parse_method(method)[0] == "raptor":
            entry["retrieved_levels"] = [c.get("level") for c in top]
        row["chunked_variants"][method] = entry
        processed += 1
        save_results(results)
        print(f"[{processed}/{len(todo)}] {item['id']} :: {method} precise={entry['precise']}")

    for method in methods:
        done = sum(
            1 for row in results.values()
            if row.get("chunked_variants", {}).get(method) is not None
        )
        print(f"{method}: {done}/{len(QA_ITEMS)} questions done")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
