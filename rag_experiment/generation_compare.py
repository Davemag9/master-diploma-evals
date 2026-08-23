"""Final demo: whole-document RAG vs best-chunking RAG, answered by
gemini-flash-latest, on a small sample of questions.

The retrieval-parameter sweep (retrieval_sweep.py) runs over all 107
questions using ONLY the generous-quota embedding API. This script is
different: it spends the tightly-limited free-tier chat model quota
(RPD=20 on gemini-flash-latest), so it only runs on a small, deliberately
diverse sample of questions -- including one "trap" question where a
whole-document retriever is likely to grab the wrong (topically similar)
document while chunk-level retrieval finds the right paragraph.

For each sample question we build TWO contexts:
  - "whole_doc": the single most similar WHOLE document (no sub-splitting)
  - "chunked":   the top-3 chunks under the best config found by the sweep
and ask the same question against each, then check whether the answer
actually contains the required fact.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import faiss

from chunkers import build_chunks
from embed_client import EmbeddingClient
from gen_client import GenerationClient, AllKeysExhausted
from qa_dataset import QA_ITEMS, check_answer_precise

RAW_DIR = Path(__file__).parent / "data" / "raw"
SWEEP_PATH = Path(__file__).parent / "data" / "results" / "sweep_results.json"
OUT_PATH = Path(__file__).parent / "data" / "results" / "generation_comparison.json"

SAMPLE_QUESTION_IDS = [
    "certs_01",     # apostille-on-embassy-documents "trap": similarly-titled doc exists
    "tariff_02",    # price fact buried in a large tariff table
    "emergency_01", # phone number buried in a long emergency-procedures page
    "rnokpp_01",    # duration fact in a long procedural page
    "registry_01",  # price fact among several similar price lines
    "passport_01",  # short recent-update fact at the top of a long page
]

PROMPT_TEMPLATE = """Ти консульський асистент. Дай КОРОТКУ і ТОЧНУ відповідь на питання громадянина, спираючись ВИКЛЮЧНО на контекст нижче. Якщо в контексті немає точної відповіді - чесно скажи, що інформації недостатньо. Не вигадуй фактів.

КОНТЕКСТ:
{context}

ПИТАННЯ: {question}

ВІДПОВІДЬ:"""


def load_docs():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(RAW_DIR.glob("*.md"))}


def pick_best_chunked_config():
    if not SWEEP_PATH.exists():
        raise RuntimeError(f"{SWEEP_PATH} not found -- run retrieval_sweep.py first")
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    candidates = {k: v for k, v in sweep.items() if k != "whole_doc"}
    best_name = max(candidates, key=lambda k: (candidates[k]["recall_at_3"], -candidates[k]["avg_chunk_chars"]))
    return best_name, sweep


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


def main():
    docs = load_docs()
    best_config, sweep = pick_best_chunked_config()
    print(f"Best chunked config from sweep: {best_config} "
          f"(recall@3={sweep[best_config]['recall_at_3']}, "
          f"whole_doc recall@3={sweep['whole_doc']['recall_at_3']})")

    embedder = EmbeddingClient()
    gen = GenerationClient()
    print(f"Gen quota remaining today: {gen.remaining_today()}")

    sample = [item for item in QA_ITEMS if item["id"] in SAMPLE_QUESTION_IDS]
    needed_calls = len(sample) * 2
    if gen.remaining_today() < needed_calls:
        print(f"WARNING: need {needed_calls} calls but only {gen.remaining_today()} left today. "
              f"Will run as many as the quota allows.")

    # whole-doc chunks (1 per doc)
    whole_chunks = []
    for name, text in docs.items():
        whole_chunks.extend(build_chunks("whole_doc", name, text))
    whole_texts = [c["text"] for c in whole_chunks]
    whole_vectors = embedder.embed_many(whole_texts)
    whole_index = build_index(whole_vectors)

    # best chunked config
    best_chunks = []
    for name, text in docs.items():
        best_chunks.extend(build_chunks(best_config, name, text))
    best_texts = [c["text"] for c in best_chunks]
    best_vectors = embedder.embed_many(best_texts)
    best_index = build_index(best_vectors)

    results = []

    def on_wait(seconds):
        print(f"    [rate limit] waiting {seconds:.0f}s...", file=sys.stderr)

    total_steps = len(sample) * 2
    step = 0
    for item in sample:
        q_vec = embedder.embed(item["question"])
        row = {"id": item["id"], "question": item["question"], "gold_doc": item["gold_doc"]}

        # --- condition A: whole document ---
        top_whole = retrieve_top(whole_index, whole_chunks, q_vec, k=1)
        context_whole = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top_whole)
        step += 1
        print(f"[{step}/{total_steps}] {item['id']} :: whole_doc "
              f"(retrieved: {top_whole[0]['doc']}, {len(context_whole)} chars)")
        try:
            answer_whole = gen.generate(PROMPT_TEMPLATE.format(context=context_whole, question=item["question"]), on_wait=on_wait)
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            break
        row["whole_doc"] = {
            "retrieved_doc": top_whole[0]["doc"],
            "context_chars": len(context_whole),
            "answer": answer_whole,
            "answer_chars": len(answer_whole),
            "precise": check_answer_precise(answer_whole, item),
        }

        # --- condition B: best chunking config ---
        top_chunked = retrieve_top(best_index, best_chunks, q_vec, k=3)
        context_chunked = "\n\n---\n\n".join(f"[{c['doc']}]\n{c['text']}" for c in top_chunked)
        step += 1
        print(f"[{step}/{total_steps}] {item['id']} :: chunked/{best_config} "
              f"(retrieved docs: {[c['doc'] for c in top_chunked]}, {len(context_chunked)} chars)")
        try:
            answer_chunked = gen.generate(PROMPT_TEMPLATE.format(context=context_chunked, question=item["question"]), on_wait=on_wait)
        except AllKeysExhausted as e:
            print(f"STOP: {e}")
            results.append(row)
            break
        row["chunked"] = {
            "config": best_config,
            "retrieved_docs": [c["doc"] for c in top_chunked],
            "context_chars": len(context_chunked),
            "answer": answer_chunked,
            "answer_chars": len(answer_chunked),
            "precise": check_answer_precise(answer_chunked, item),
        }

        results.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"best_config": best_config, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for row in results:
        wd = row.get("whole_doc", {})
        ch = row.get("chunked", {})
        print(f"\n[{row['id']}] {row['question']}")
        if wd:
            print(f"  whole_doc : precise={wd['precise']} ctx={wd['context_chars']}ch ans={wd['answer_chars']}ch retrieved={wd['retrieved_doc']}")
        if ch:
            print(f"  chunked   : precise={ch['precise']} ctx={ch['context_chars']}ch ans={ch['answer_chars']}ch retrieved={ch['retrieved_docs']}")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
