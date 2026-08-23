"""Full-scale version of generation_compare.py: whole-document RAG vs
best-chunking RAG, answered by gemini-flash-latest, across ALL 107
QA_ITEMS (214 calls total).

Free-tier daily quota is still only ~18 safe calls PER key, so even with 4
rotating keys (~72/day) this will very likely take multiple days. The
script is fully resumable: results are saved incrementally to
data/results/generation_full.json, and any question/condition pair already
present there is skipped on the next run. Just re-run this script again
(e.g. tomorrow, once quota resets) to keep making progress -- no need to
track what's left manually.
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
SWEEP_PATH = Path(__file__).parent / "data" / "results" / "sweep_results.json"
OUT_PATH = Path(__file__).parent / "data" / "results" / "generation_full.json"

CHUNKED_CONFIG = "table_aware_3row"  # winner from the retrieval sweep

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
    print(f"Already have {len(results)}/{len(QA_ITEMS)} questions done "
          f"(each needs whole_doc + chunked = 2 calls)")
    print(f"Gen quota remaining today (all keys): {gen.remaining_today()}")

    # whole-doc index
    whole_chunks = []
    for name, text in docs.items():
        whole_chunks.extend(build_chunks("whole_doc", name, text))
    whole_vectors = embedder.embed_many([c["text"] for c in whole_chunks])
    whole_index = build_index(whole_vectors)

    # best chunked config index
    best_chunks = []
    for name, text in docs.items():
        best_chunks.extend(build_chunks(CHUNKED_CONFIG, name, text))
    best_vectors = embedder.embed_many([c["text"] for c in best_chunks])
    best_index = build_index(best_vectors)

    def on_wait(seconds):
        print(f"    [rate limit] waiting {seconds:.0f}s...", file=sys.stderr)

    todo = [item for item in QA_ITEMS if results.get(item["id"], {}).get("chunked") is None]
    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if max_new is not None:
        todo = todo[:max_new]
        print(f"Limiting this run to {max_new} new questions (of {len(todo)} available)")
    print(f"{len(todo)} questions left to process this run")

    stopped = False
    for item in todo:
        if stopped:
            break
        q_vec = embedder.embed(item["question"])
        row = {"id": item["id"], "question": item["question"], "gold_doc": item["gold_doc"]}

        top_whole = retrieve_top(whole_index, whole_chunks, q_vec, k=1)
        context_whole = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top_whole)
        try:
            answer_whole = gen.generate(
                PROMPT_TEMPLATE.format(context=context_whole, question=item["question"]), on_wait=on_wait
            )
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            stopped = True
            break
        row["whole_doc"] = {
            "retrieved_doc": top_whole[0]["doc"],
            "context_chars": len(context_whole),
            "answer": answer_whole,
            "answer_chars": len(answer_whole),
            "precise": check_answer_precise(answer_whole, item),
        }

        top_chunked = retrieve_top(best_index, best_chunks, q_vec, k=3)
        context_chunked = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top_chunked)
        try:
            answer_chunked = gen.generate(
                PROMPT_TEMPLATE.format(context=context_chunked, question=item["question"]), on_wait=on_wait
            )
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            row["chunked"] = None
            results[item["id"]] = row
            save_results(results)
            stopped = True
            break
        row["chunked"] = {
            "config": CHUNKED_CONFIG,
            "retrieved_docs": [c["doc"] for c in top_chunked],
            "context_chars": len(context_chunked),
            "answer": answer_chunked,
            "answer_chars": len(answer_chunked),
            "precise": check_answer_precise(answer_chunked, item),
        }

        results[item["id"]] = row
        save_results(results)
        print(f"[{len(results)}/{len(QA_ITEMS)}] {item['id']}: "
              f"whole_doc precise={row['whole_doc']['precise']}, chunked precise={row['chunked']['precise']}")

    done = sum(1 for r in results.values() if r.get("chunked") is not None)
    print(f"\nProgress: {done}/{len(QA_ITEMS)} questions fully done, "
          f"{len(QA_ITEMS) - done} remaining. Re-run tomorrow (or later today with fresh quota) to continue.")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
