"""Off-the-shelf, production-grade chunkers used as external baselines.

These are the splitters people actually reach for in real RAG systems, so
comparing the thesis's method against them (rather than only against
hand-rolled fixed-size windows) is what makes the evaluation credible:

- RecursiveCharacterTextSplitter -- LangChain's default and by far the most
  widely used splitter in production RAG. Splits on a priority list of
  separators ("\\n\\n", "\\n", " ", "") so it prefers paragraph, then line,
  then word boundaries, packing up to chunk_size.
- MarkdownHeaderTextSplitter (+ recursive pass) -- LangChain's recommended
  recipe for markdown: first split on headers (keeping them as metadata),
  then size-limit each section. This is the closest industry equivalent to
  the thesis's structural chunkers, so it's the key head-to-head.
- TokenTextSplitter -- splits by LLM tokens rather than characters, which
  is what actually matters for context budgets.

Each function matches the project's chunker interface:
    (doc_name, text, ...) -> [{"doc": ..., "text": ...}, ...]
"""
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def lc_recursive_chunks(doc_name, text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return [
        {"doc": doc_name, "text": piece}
        for piece in splitter.split_text(text)
        if piece.strip()
    ]


def lc_markdown_header_chunks(doc_name, text, chunk_size, chunk_overlap=0):
    """LangChain's standard markdown recipe: header-split, then size-limit.

    Header metadata is prepended back onto the chunk text so the retriever
    sees it -- this is what LangChain users do in practice, and it keeps the
    comparison against our own breadcrumb chunkers fair.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    try:
        sections = header_splitter.split_text(text)
    except Exception:
        sections = []
    if not sections:
        return lc_recursive_chunks(doc_name, text, chunk_size, chunk_overlap)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = []
    for section in sections:
        header_path = " › ".join(
            section.metadata[key] for key in ("h1", "h2", "h3", "h4") if key in section.metadata
        )
        for piece in size_splitter.split_text(section.page_content):
            if not piece.strip():
                continue
            body = f"[{header_path}]\n{piece}" if header_path else piece
            chunks.append({"doc": doc_name, "text": body})
    return chunks


def lc_token_chunks(doc_name, text, chunk_size, chunk_overlap):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {"doc": doc_name, "text": piece}
        for piece in splitter.split_text(text)
        if piece.strip()
    ]
