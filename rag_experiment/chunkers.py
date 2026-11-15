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
CODE_FENCE_RE = re.compile(r"^\s*```")
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
        if CODE_FENCE_RE.match(line):
            code_lines = [line]
            i += 1
            while i < n and not CODE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < n:  # consume the closing fence
                code_lines.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code_lines)))
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
            while i < n and lines[i].strip() and not HEADING_LINE_RE.match(lines[i]) and not TABLE_ROW_RE.match(lines[i]) and not CODE_FENCE_RE.match(lines[i]):
                # list item continuation or a new item at the same/deeper level
                list_lines.append(lines[i])
                i += 1
            blocks.append(("list", "\n".join(list_lines)))
            continue
        para_lines = []
        while i < n and lines[i].strip() and not HEADING_LINE_RE.match(lines[i]) and not TABLE_ROW_RE.match(lines[i]) and not LIST_ITEM_RE.match(lines[i]) and not CODE_FENCE_RE.match(lines[i]):
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

        if block[0] in ("list", "code"):
            # Atomic: a list/code block is never split across two chunks. If
            # it doesn't fit in what's left, flush first -- but never break
            # the block itself (code especially: splitting mid-function loses
            # the very facts, like a status code or default value, it holds).
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


def _split_code_block(content, budget, overlap_lines=1):
    """Cut a fenced code block into budget-sized pieces at LINE boundaries.

    Keeping a whole code block atomic sounds right, but a 4000-char snippet
    embeds as one averaged vector in which a single fact line (a status code,
    a default value, a version pin) is drowned out by imports and boilerplate.
    Cutting at line boundaries keeps every statement intact while giving the
    retriever small, specific vectors. Each piece is re-wrapped in its own
    fence (with the original language tag) so it still reads as valid code,
    and `overlap_lines` carries the tail of one piece into the next so a fact
    sitting on a seam is not lost.
    """
    lines = content.split("\n")
    fence_line = lines[0] if lines and CODE_FENCE_RE.match(lines[0]) else "```"
    body = lines[1:]
    if body and CODE_FENCE_RE.match(body[-1]):
        body = body[:-1]
    if not body:
        return [content]

    wrap_len = len(fence_line) + 4  # fence line + closing fence + newlines
    pieces = []
    current = []
    current_len = wrap_len

    def emit_piece():
        return fence_line + "\n" + "\n".join(current) + "\n```"

    for line in body:
        line_len = len(line) + 1
        if current and current_len + line_len > budget:
            pieces.append(emit_piece())
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = wrap_len + sum(len(l) + 1 for l in current)
        current.append(line)
        current_len += line_len
    if current:
        pieces.append(emit_piece())
    return pieces or [content]


def _fact_density(text):
    """Rough 'fact density' of a block: share of digit characters.

    Tariff rows, deadlines, phone numbers and fees are digit-heavy; narrative
    procedure text is not. This is what the adaptive chunker keys off when it
    decides how tight a chunk should be.
    """
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def _semantic_split_point(sentences, budget, embedder):
    """Pick the index to split a sentence run at: among the split points that
    keep the first part within budget, choose the one where consecutive
    sentences are LEAST similar (i.e. the natural topic boundary).

    Returns an index into `sentences` (the start of the second part), or None
    if no in-budget split point exists.
    """
    candidates = []
    running = 0
    for i, sent in enumerate(sentences):
        running += len(sent)
        if i > 0 and running <= budget:
            candidates.append(i)
    if not candidates:
        return None
    if embedder is None or len(candidates) == 1:
        return candidates[-1]  # no embedder: fall back to "fill the budget"

    vectors = np.array(embedder.embed_many(sentences), dtype="float32")
    norms = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    # similarity between sentence i-1 and i == the seam we'd cut at index i
    return min(candidates, key=lambda i: float(norms[i - 1] @ norms[i]))


def adaptive_hybrid_chunks(
    doc_name,
    text,
    prose_max=900,
    dense_max=450,
    rows_per_chunk=3,
    density_threshold=0.02,
    embedder=None,
    code_max=None,
    code_overlap_lines=1,
    merge_small_sections=False,
):
    """The thesis's proposed method: adaptive structural-semantic segmentation.

    It combines exactly the ingredients that won their respective comparisons
    in this experiment, instead of picking one strategy for the whole corpus:

    - full heading breadcrumb on every chunk        (from hierarchical_800 -- best precision)
    - lists kept atomic, never split across chunks  (from hierarchical_800)
    - table rows grouped under a repeated header    (from table_aware_3row -- best context efficiency)
    - ADAPTIVE chunk budget per block: fact-dense blocks (digit-heavy: fees,
      deadlines, phone numbers) get a tight budget so a retrieved chunk is
      mostly signal, while narrative prose gets a larger budget so procedural
      context isn't fragmented
    - SEMANTIC split point: when a prose run must be broken, it is cut at the
      weakest inter-sentence similarity rather than at an arbitrary offset
      (only used as a tie-breaker, so it costs a fraction of the embedding
      calls that full semantic chunking needs)
    - MERGE_SMALL_SECTIONS (for reference documentation): technical docs are
      written as many two-sentence sections, so cutting at every heading
      yields fragments too small to answer anything. With this on, a heading
      only ends a chunk once the chunk is worth ending; shorter sibling
      sections are packed together, their headings kept inline as text.
    """
    blocks = _split_into_blocks(text)
    doc_title = doc_name.rsplit(".", 1)[0]
    heading_stack = []
    chunks = []
    current_parts = []
    current_len = 0
    if code_max is None:
        code_max = dense_max
    # breadcrumb captured when the open chunk started, so a chunk spanning
    # several small sections is still labelled by the section it begins in
    current_path = None

    def path_str():
        return _breadcrumb(doc_title, heading_stack)

    def emit(body, path=None):
        chunks.append({"doc": doc_name, "text": f"[{path or path_str()}]\n{body}"})

    def flush():
        nonlocal current_parts, current_len, current_path
        if current_parts:
            emit("\n\n".join(current_parts), current_path)
        current_parts = []
        current_len = 0
        current_path = None

    def add(content):
        """Append a block to the open chunk, remembering where it started."""
        nonlocal current_parts, current_len, current_path
        if not current_parts:
            current_path = path_str()
        current_parts.append(content)
        current_len += len(content)

    def budget_for(content):
        return dense_max if _fact_density(content) >= density_threshold else prose_max

    for block in blocks:
        kind = block[0]

        if kind == "heading":
            level, title = block[1], block[2]
            # With section merging on, a heading only closes the open chunk
            # once that chunk has enough in it to stand alone; otherwise the
            # heading is kept inline and the next section packs in behind it.
            if merge_small_sections and current_parts and current_len < prose_max:
                heading_marker = "#" * level + " " + title
                current_parts.append(heading_marker)
                current_len += len(heading_marker)
            else:
                flush()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if kind == "table":
            # Tables are always their own chunks: header + separator repeated
            # over each small group of data rows.
            flush()
            lines = block[1]
            if len(lines) < 2:
                emit("\n".join(lines))
                continue
            header_lines, data_rows = lines[:2], lines[2:]
            if not data_rows:
                emit("\n".join(header_lines))
                continue
            for i in range(0, len(data_rows), rows_per_chunk):
                emit("\n".join(header_lines + data_rows[i : i + rows_per_chunk]))
            continue

        content = block[1]
        if not content.strip():
            continue

        if kind == "code":
            # Code blocks carry this corpus's facts (a status code, a default
            # value, a version pin). A block over budget is cut at LINE
            # boundaries, never mid-statement.
            pieces = ([content] if len(content) <= code_max
                      else _split_code_block(content, code_max, code_overlap_lines))
            if merge_small_sections:
                # Keep code with the prose that introduces it: that prose is
                # what carries the words a question is phrased in, while the
                # code carries the value being asked about.
                for piece in pieces:
                    if current_parts and current_len + len(piece) > prose_max:
                        flush()
                    add(piece)
                    if current_len >= prose_max:
                        flush()
            else:
                flush()
                for piece in pieces:
                    emit(piece)
            continue

        budget = budget_for(content)

        if kind == "list":
            # Atomic: never split a list across chunks, even if oversized.
            if current_parts and current_len + len(content) > budget:
                flush()
            add(content)
            if current_len > budget:
                flush()
            continue

        # Prose paragraph.
        if current_parts and current_len + len(content) > budget:
            flush()

        if len(content) <= budget:
            add(content)
            continue

        # A single paragraph larger than its budget: break it at semantic seams.
        sentences = _split_sentences(content)
        while len(content) > budget and len(sentences) > 1:
            split_at = _semantic_split_point(sentences, budget, embedder)
            if not split_at:
                break
            emit(" ".join(sentences[:split_at]))
            sentences = sentences[split_at:]
            content = " ".join(sentences)
        if content.strip():
            add(content)

    flush()
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
    # --- external / industry baselines (LangChain) ---
    "lc_recursive_800": {"kind": "lc_recursive", "chunk_size": 800, "chunk_overlap": 100},
    "lc_recursive_400": {"kind": "lc_recursive", "chunk_size": 400, "chunk_overlap": 50},
    "lc_recursive_1500": {"kind": "lc_recursive", "chunk_size": 1500, "chunk_overlap": 200},
    "lc_markdown_800": {"kind": "lc_markdown", "chunk_size": 800, "chunk_overlap": 0},
    "lc_markdown_500": {"kind": "lc_markdown", "chunk_size": 500, "chunk_overlap": 0},
    "lc_token_512": {"kind": "lc_token", "chunk_size": 512, "chunk_overlap": 64},
    # --- proposed method: adaptive structural-semantic segmentation ---
    # ablation: same structural rules, but WITHOUT the semantic split point
    "adaptive_nosem": {
        "kind": "adaptive", "prose_max": 900, "dense_max": 450,
        "rows_per_chunk": 3, "density_threshold": 0.02, "use_semantic": False,
    },
    # full method (structural rules + semantic split point)
    "adaptive_hybrid": {
        "kind": "adaptive", "prose_max": 900, "dense_max": 450,
        "rows_per_chunk": 3, "density_threshold": 0.02, "use_semantic": True,
    },
    # tighter / looser budget variants, to show the adaptive budget matters
    "adaptive_tight": {
        "kind": "adaptive", "prose_max": 700, "dense_max": 350,
        "rows_per_chunk": 2, "density_threshold": 0.02, "use_semantic": True,
    },
    "adaptive_wide": {
        "kind": "adaptive", "prose_max": 1200, "dense_max": 600,
        "rows_per_chunk": 4, "density_threshold": 0.02, "use_semantic": True,
    },
    # The same method retuned for reference documentation (prose + code):
    # wider budgets, code cut at line boundaries, and the many two-sentence
    # sections such docs are written in merged into answerable chunks.
    "adaptive_docs": {
        "kind": "adaptive", "prose_max": 1200, "dense_max": 800,
        "rows_per_chunk": 4, "density_threshold": 0.02, "use_semantic": True,
        "code_max": 900, "merge_small_sections": True,
    },
}

# configs whose kind needs an embedder passed into build_chunks()
EMBEDDER_REQUIRED_KINDS = {"semantic", "adaptive"}


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
    if cfg["kind"] == "adaptive":
        if cfg["use_semantic"] and embedder is None:
            raise ValueError(f"config {config_name!r} needs an embedder")
        return adaptive_hybrid_chunks(
            doc_name, text,
            prose_max=cfg["prose_max"],
            dense_max=cfg["dense_max"],
            rows_per_chunk=cfg["rows_per_chunk"],
            density_threshold=cfg["density_threshold"],
            embedder=embedder if cfg["use_semantic"] else None,
            code_max=cfg.get("code_max"),
            code_overlap_lines=cfg.get("code_overlap_lines", 1),
            merge_small_sections=cfg.get("merge_small_sections", False),
        )
    if cfg["kind"] == "lc_recursive":
        from chunkers_baselines import lc_recursive_chunks
        return lc_recursive_chunks(doc_name, text, cfg["chunk_size"], cfg["chunk_overlap"])
    if cfg["kind"] == "lc_markdown":
        from chunkers_baselines import lc_markdown_header_chunks
        return lc_markdown_header_chunks(doc_name, text, cfg["chunk_size"], cfg["chunk_overlap"])
    if cfg["kind"] == "lc_token":
        from chunkers_baselines import lc_token_chunks
        return lc_token_chunks(doc_name, text, cfg["chunk_size"], cfg["chunk_overlap"])
    raise ValueError(f"unknown chunker kind: {cfg['kind']}")
