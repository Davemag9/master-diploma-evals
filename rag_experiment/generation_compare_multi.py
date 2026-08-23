"""Extend the generation comparison to the full top-5 chunking methods
(ranked by recall@3 in the retrieval sweep):
    1. hierarchical_800
    2. table_aware_3row   (already done in generation_full.json as "chunked")
    3. whole_doc          (already done in generation_full.json as "whole_doc")
    4. fixed_1500_0
    5. structural_700

This script only needs to generate answers for the NEW configs
(hierarchical_800, fixed_1500_0, structural_700) -- the whole_doc baseline
and table_aware_3row are reused as-is from the earlier run. All of these
configs were already embedded during retrieval_sweep.py, so building their
chunk indexes here costs zero new embedding-API calls -- only the tightly
rate-limited chat model is spent, on 107 questions x 3 new configs = 321
calls.

Results are stored per-question under "chunked_variants": {config_name:
{...}} in the same data/results/generation_full.json file, alongside the
existing "whole_doc" / "chunked" fields. Fully resumable: skips any
(question, config) pair already present.
"""
import json
import sys
from pathlib import Path

import numpy as np
import faiss

from chunkers import build_chunks
from embed_client import EmbeddingClient
from gen_client import GenerationClient, AllKeysExhausted
from qa_dataset import QA_ITEMS, check_answer_precise

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_PATH = Path(__file__).parent / "data" / "results" / "generation_full.json"

NEW_CONFIGS = ["hierarchical_800", "fixed_1500_0", "structural_700"]

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


def retrieve_top(index, chunks, query_vec, k):
    q = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(q)
    _, idx = index.search(q, min(k, len(chunks)))
    return [chunks[i] for i in idx[0]]


def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results):
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    docs = load_docs()
    embedder = EmbeddingClient()
    gen = GenerationClient()

    results = load_results()
    for row in results.values():
        row.setdefault("chunked_variants", {})

    print(f"Gen quota remaining today (all keys): {gen.remaining_today()}")

    def on_wait(seconds):
        print(f"    [rate limit] waiting {seconds:.0f}s...", file=sys.stderr)

    # Build (chunks, index) per new config -- all embeddings already cached
    # from retrieval_sweep.py, so this is essentially free / fast.
    config_data = {}
    for config_name in NEW_CONFIGS:
        chunks = []
        for name, text in docs.items():
            chunks.extend(build_chunks(config_name, name, text))
        vectors = embedder.embed_many([c["text"] for c in chunks])
        index = build_index(vectors)
        config_data[config_name] = (chunks, index)
        print(f"{config_name}: {len(chunks)} chunks indexed (embeddings from cache)")

    # Build the full todo list: every (question, config) pair not yet done.
    todo = []
    for item in QA_ITEMS:
        row = results.setdefault(item["id"], {
            "id": item["id"], "question": item["question"], "gold_doc": item["gold_doc"],
            "chunked_variants": {},
        })
        for config_name in NEW_CONFIGS:
            if row["chunked_variants"].get(config_name) is None:
                todo.append((item, config_name))

    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if max_new is not None:
        todo = todo[:max_new]
    print(f"{len(todo)} (question, config) pairs left to process this run")

    stopped = False
    processed = 0
    for item, config_name in todo:
        if stopped:
            break
        chunks, index = config_data[config_name]
        q_vec = embedder.embed(item["question"])
        top = retrieve_top(index, chunks, q_vec, k=3)
        context = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top)
        try:
            answer = gen.generate(
                PROMPT_TEMPLATE.format(context=context, question=item["question"]), on_wait=on_wait
            )
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            stopped = True
            break
        row = results[item["id"]]
        row["chunked_variants"][config_name] = {
            "retrieved_docs": [c["doc"] for c in top],
            "context_chars": len(context),
            "answer": answer,
            "answer_chars": len(answer),
            "precise": check_answer_precise(answer, item),
        }
        processed += 1
        save_results(results)
        print(f"[{processed}/{len(todo)}] {item['id']} :: {config_name} "
              f"precise={row['chunked_variants'][config_name]['precise']}")

    total_pairs_done = sum(
        1 for row in results.values() for c in NEW_CONFIGS if row["chunked_variants"].get(c) is not None
    )
    print(f"\nProgress: {total_pairs_done}/{len(QA_ITEMS) * len(NEW_CONFIGS)} "
          f"(question, config) pairs done across {NEW_CONFIGS}.")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
