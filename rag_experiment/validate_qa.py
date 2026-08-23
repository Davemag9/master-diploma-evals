"""Validate that every QA item's fact_keywords actually occur (case/whitespace
insensitively) in the claimed source document. Reports failures with context
so they can be fixed."""
import re
from pathlib import Path

from qa_dataset import QA_ITEMS

RAW_DIR = Path(__file__).parent / "data" / "raw"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main():
    docs = {p.name: p.read_text(encoding="utf-8") for p in RAW_DIR.glob("*.md")}
    n_fail = 0
    ids = set()
    for item in QA_ITEMS:
        if item["id"] in ids:
            print(f"DUPLICATE ID: {item['id']}")
        ids.add(item["id"])

        doc_text = docs.get(item["gold_doc"])
        if doc_text is None:
            print(f"[{item['id']}] MISSING DOC: {item['gold_doc']}")
            n_fail += 1
            continue
        norm_doc = normalize_ws(doc_text).lower()
        for kw in item["fact_keywords"]:
            norm_kw = normalize_ws(kw).lower()
            if norm_kw not in norm_doc:
                print(f"[{item['id']}] KEYWORD NOT FOUND: {kw!r} in {item['gold_doc']}")
                # try to help find the closest match
                first_word = norm_kw.split()[0] if norm_kw.split() else norm_kw
                idx = norm_doc.find(first_word)
                if idx >= 0:
                    print(f"    context: ...{norm_doc[max(0,idx-40):idx+60]}...")
                n_fail += 1

    print(f"\n{len(QA_ITEMS)} items checked, {n_fail} keyword failures")


if __name__ == "__main__":
    main()
