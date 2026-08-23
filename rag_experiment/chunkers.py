"""Chunking strategies compared in the experiment.

Each chunker takes (doc_name, text) and returns a list of dicts:
    {"doc": doc_name, "text": chunk_text}

- whole_doc_chunks: baseline — "беремо цілі файли" (no sub-document splitting).
- fixed_size_chunks: naive fixed-length windows with overlap, snapped to
  whitespace so words aren't cut in half.
- structural_chunks: splits on markdown headings + paragraph breaks first,
  then packs paragraphs into chunks up to max_chars, keeping the nearest
  heading as context.
- hierarchical_chunks: like structural, but (a) tracks the FULL heading
  path (# > ## > ###, not just the nearest one) and prefixes every chunk
  with it, and (b) treats markdown lists as atomic units that are never
  split across chunk boundaries, even if that makes a chunk exceed
  max_chars.
- table_aware_chunks: markdown tables (like the tariff price table) are
  kept as (heading, column header, separator) + a small group of data
  rows per chunk, instead of being cut at an arbitrary character offset.
- semantic_chunks: splits into sentences, embeds each one, and cuts where
  cosine similarity between consecutive sentences drops sharply — the
  "structural-semantic coherence" idea the thesis is built around.
"""
import re

import numpy as np

HEADING_RE = re.compile(r"^(#{1,6})\s*(.+)$", re.MULTILINE)
HEADING_LINE_RE = re.compile(r"^(#{1,6})\s*(.+)$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
TABLE_ROW_RE = re.compile(r"^\s*\|")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯҐЄІЇA-Z0-9])")


def _breadcrumb(doc_title, heading_stack):
    parts = [doc_title]
    for _, title in heading_stack:
        if title.strip().lower() != parts[-1].strip().lower():
            parts.append(title)
    return " › ".join(parts)


def whole_doc_chunks(doc_name, text):
    return [{"doc": doc_name, "text": text}]


def fixed_size_chunks(doc_name, text, size, overlap):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    n = len(text)
    start = 0
    step = size - overlap
    while start < n:
        end = min(start + size, n)
        if end < n:
            # snap forward to the next whitespace so we don't cut a word
            snap = text.find(" ", end)
            if snap != -1 and snap - end < 40:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"doc": doc_name, "text": chunk})
        if end >= n:
            break
        start += step
    return chunks


def structural_chunks(doc_name, text, max_chars):
    # Split the doc into (heading, body) sections.
    sections = []
    last_heading = doc_name
    last_pos = 0
    heading_matches = list(HEADING_RE.finditer(text))
    if not heading_matches:
        sections.append((doc_name, text))
    else:
        first_start = heading_matches[0].start()
        if first_start > 0:
            sections.append((doc_name, text[:first_start]))
        for i, m in enumerate(heading_matches):
            heading_text = m.group(2).strip()
            body_start = m.end()
            body_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
            sections.append((heading_text, text[body_start:body_end]))

    chunks = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) > max_chars and current:
                chunks.append({"doc": doc_name, "text": f"[{heading}]\n{current}"})
                current = para
            else:
                current = candidate
        if current:
            chunks.append({"doc": doc_name, "text": f"[{heading}]\n{current}"})
    return chunks


def _split_into_blocks(text):
    """Split raw markdown into a flat list of ('heading', level, title) /
    ('list', text) / ('para', text) blocks, in document order."""
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = HEADING_LINE_RE.match(line)
        if m:
            blocks.append(("heading", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if TABLE_ROW_RE.match(line):
            table_lines = []
            while i < n and TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", table_lines))
            continue
        if LIST_ITEM_RE.match(line):
            list_lines = [line]
            i += 1
            while i < n and lines[i].strip() and not HEADING_LINE_RE.match(lines[i]) and not TABLE_ROW_RE.match(lines[i]):
                # list item continuation or a new item at the same/deeper level
                list_lines.append(lines[i])
                i += 1
            blocks.append(("list", "\n".join(list_lines)))
            continue
        para_lines = []
        while i < n and lines[i].strip() and not HEADING_LINE_RE.match(lines[i]) and not TABLE_ROW_RE.match(lines[i]) and not LIST_ITEM_RE.match(lines[i]):
            para_lines.append(lines[i])
            i += 1
        blocks.append(("para", "\n".join(para_lines)))
    return blocks


def hierarchical_chunks(doc_name, text, max_chars):
    """Structure-aware chunker with a FULL heading breadcrumb per chunk and
    atomic (never-split) lists — per the thesis's structural-semantic idea."""
    blocks = _split_into_blocks(text)
    doc_title = doc_name.rsplit(".", 1)[0]
    heading_stack = []  # list of (level, title)
    chunks = []
    current_parts = []
    current_len = 0

    def path_str():
        return _breadcrumb(doc_title, heading_stack)

    def flush():
        nonlocal current_parts, current_len
        if current_parts:
            body = "\n\n".join(current_parts)
            chunks.append({"doc": doc_name, "text": f"[{path_str()}]\n{body}"})
        current_parts = []
        current_len = 0

    for block in blocks:
        if block[0] == "heading":
            flush()
            level, title = block[1], block[2]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        content = "\n".join(block[1]) if block[0] == "table" else block[1]

        if block[0] == "list":
            # Atomic: a list is never split across two chunks. If it doesn't
            # fit in what's left, flush first — but never break the list itself.
            if current_parts and current_len + len(content) > max_chars:
                flush()
            current_parts.append(content)
            current_len += len(content)
            if current_len > max_chars:
                flush()
        else:
            if current_parts and current_len + len(content) > max_chars:
                flush()
            current_parts.append(content)
            current_len += len(content)

    flush()
    return chunks


def table_aware_chunks(doc_name, text, rows_per_chunk=3, max_chars=700):
    """Keeps markdown table rows grouped with their header/separator row
    instead of letting a fixed-size window cut a table mid-row. Non-table
    content (most of the corpus) is packed like hierarchical_chunks, up to
    max_chars, so this stays comparable to the other chunkers instead of
    degenerating into one chunk per paragraph."""
    blocks = _split_into_blocks(text)
    doc_title = doc_name.rsplit(".", 1)[0]
    heading_stack = []
    chunks = []
    current_parts = []
    current_len = 0

    def path_str():
        return _breadcrumb(doc_title, heading_stack)

    def flush():
        nonlocal current_parts, current_len
        if current_parts:
            body = "\n\n".join(current_parts)
            chunks.append({"doc": doc_name, "text": f"[{path_str()}]\n{body}"})
        current_parts = []
        current_len = 0

    for block in blocks:
        if block[0] == "heading":
            flush()
            level, title = block[1], block[2]
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if block[0] == "table":
            flush()  # tables are always their own atomic chunk group
            lines = block[1]
            if len(lines) < 2:
                chunks.append({"doc": doc_name, "text": f"[{path_str()}]\n" + "\n".join(lines)})
                continue
            header_lines = lines[:2]  # column header + separator row
            data_rows = lines[2:]
            for i in range(0, len(data_rows), rows_per_chunk) if data_rows else [0]:
                group = data_rows[i : i + rows_per_chunk] if data_rows else []
                table_text = "\n".join(header_lines + group)
                chunks.append({"doc": doc_name, "text": f"[{path_str()}]\n{table_text}"})
        else:
            content = block[1]
            if content.strip():
                if current_parts and current_len + len(content) > max_chars:
                    flush()
                current_parts.append(content)
                current_len += len(content)

    flush()
    return chunks


def _split_sentences(text):
    sentences = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for s in SENTENCE_SPLIT_RE.split(para):
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def semantic_chunks(doc_name, text, embedder, max_chars, breakpoint_percentile=30):
    """Embeds each sentence and cuts where similarity between consecutive
    sentences drops sharply (a semantic "topic boundary"), capped at
    max_chars per chunk."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [{"doc": doc_name, "text": text}]

    vectors = np.array(embedder.embed_many(sentences), dtype="float32")
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = (norms[:-1] * norms[1:]).sum(axis=1)
    threshold = np.percentile(sims, breakpoint_percentile)

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sent = sentences[i]
        is_breakpoint = sims[i - 1] < threshold
        if current and (is_breakpoint or current_len + len(sent) > max_chars):
            chunks.append({"doc": doc_name, "text": " ".join(current)})
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append({"doc": doc_name, "text": " ".join(current)})
    return chunks


CHUNKING_CONFIGS = {
    "whole_doc": {"kind": "whole"},
    "fixed_400_0": {"kind": "fixed", "size": 400, "overlap": 0},
    "fixed_400_100": {"kind": "fixed", "size": 400, "overlap": 100},
    "fixed_800_0": {"kind": "fixed", "size": 800, "overlap": 0},
    "fixed_800_100": {"kind": "fixed", "size": 800, "overlap": 100},
    "fixed_1500_0": {"kind": "fixed", "size": 1500, "overlap": 0},
    "fixed_1500_100": {"kind": "fixed", "size": 1500, "overlap": 100},
    "structural_500": {"kind": "structural", "max_chars": 500},
    "structural_700": {"kind": "structural", "max_chars": 700},
    "structural_1000": {"kind": "structural", "max_chars": 1000},
    "hierarchical_500": {"kind": "hierarchical", "max_chars": 500},
    "hierarchical_800": {"kind": "hierarchical", "max_chars": 800},
    "hierarchical_1200": {"kind": "hierarchical", "max_chars": 1200},
    "table_aware_1row": {"kind": "table_aware", "rows_per_chunk": 1},
    "table_aware_3row": {"kind": "table_aware", "rows_per_chunk": 3},
    "semantic_600": {"kind": "semantic", "max_chars": 600, "breakpoint_percentile": 30},
    "semantic_1000": {"kind": "semantic", "max_chars": 1000, "breakpoint_percentile": 20},
}

# configs whose kind needs an embedder passed into build_chunks()
EMBEDDER_REQUIRED_KINDS = {"semantic"}


def build_chunks(config_name, doc_name, text, embedder=None):
    cfg = CHUNKING_CONFIGS[config_name]
    if cfg["kind"] == "whole":
        return whole_doc_chunks(doc_name, text)
    if cfg["kind"] == "fixed":
        return fixed_size_chunks(doc_name, text, cfg["size"], cfg["overlap"])
    if cfg["kind"] == "structural":
        return structural_chunks(doc_name, text, cfg["max_chars"])
    if cfg["kind"] == "hierarchical":
        return hierarchical_chunks(doc_name, text, cfg["max_chars"])
    if cfg["kind"] == "table_aware":
        return table_aware_chunks(doc_name, text, cfg["rows_per_chunk"])
    if cfg["kind"] == "semantic":
        if embedder is None:
            raise ValueError(f"config {config_name!r} needs an embedder")
        return semantic_chunks(doc_name, text, embedder, cfg["max_chars"], cfg["breakpoint_percentile"])
    raise ValueError(f"unknown chunker kind: {cfg['kind']}")
