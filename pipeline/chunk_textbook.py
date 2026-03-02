#!/usr/bin/env python3
"""
Content-type-aware chunker for Milvus vector DB ingestion.

Reads PageBlock JSON from postprocess_docstrange.py and produces chunk
dicts ready for embedding + insertion.  No vectors are computed here —
that is handled by ingest_milvus.py.

Chunking rules (per content_type):
  table / clinical_criteria / clinical_scale / image_desc
      → 1 atomic chunk (never split)
  exam_technique_flattened
      → 1 chunk per PageBlock (already page-scoped)
  table_footnote
      → merged into the immediately preceding table chunk
  list
      → atomic if ≤ 400 tokens; split at item boundaries otherwise
  prose
      → target 300–400 tokens, sentence-boundary splits, 1–2 sentence overlap

Token counting uses the PubMedBERT tokenizer to match the embedding model.

Usage:
    python chunk_textbook.py <pageblocks_json> <meta_json> [--output FILE]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizer = None

# Embedding model used for dense vectors — tokenizer must match.
# S-PubMedBert-MS-MARCO: PubMedBERT fine-tuned on MS-MARCO for retrieval.
# 768-dim, mean pooling, max_seq_length=350.
TOKENIZER_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def _get_tokenizer():
    """Lazy-load the PubMedBERT tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            print(f"[chunker] Loaded tokenizer: {TOKENIZER_MODEL}")
        except ImportError:
            print(
                "[chunker] WARNING: transformers not installed. "
                "Falling back to word-based token estimation."
            )
            _tokenizer = "fallback"
        except OSError:
            print(
                f"[chunker] WARNING: Could not load {TOKENIZER_MODEL}. "
                f"Falling back to word-based token estimation."
            )
            _tokenizer = "fallback"
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using the PubMedBERT tokenizer (or word estimate)."""
    tok = _get_tokenizer()
    if tok == "fallback":
        # Rough approximation: ~1.3 tokens per whitespace-delimited word
        return int(len(text.split()) * 1.3)
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

# Target token range for prose chunks
PROSE_TARGET_MIN = 300
PROSE_TARGET_MAX = 400

# Minimum prose block size before attempting merge with next.
# Blocks with fewer tokens than this will be merged forward into the next
# prose block on the same page, even across section boundaries.
PROSE_MERGE_THRESHOLD = 30

# Maximum tokens for a list to remain atomic
LIST_ATOMIC_MAX = 400

# Heading-only prose chunks below this token count are dropped as noise.
# Section names are already captured in section/subsection metadata fields.
HEADING_ONLY_DROP_MAX = 20

# Overlap: number of sentences to carry forward between prose chunks
PROSE_OVERLAP_SENTENCES = 2


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Sentence boundary regex: split after period/question/exclamation followed by
# whitespace and a capital letter.  Avoids splitting on abbreviations like
# "Dr.", "Fig.", "vs.", "e.g." by requiring the next char to be uppercase.
RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = RE_SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# List item splitting
# ---------------------------------------------------------------------------

RE_LIST_ITEM = re.compile(
    r"^(\s*(?:[\*\-\u2022\u25CB\u25A0]|\d+[\.\)])\s+)",
    re.MULTILINE,
)


def split_list_items(text: str) -> list[str]:
    """Split a list block into individual items, preserving markers."""
    # Find all item start positions
    matches = list(RE_LIST_ITEM.finditer(text))
    if not matches:
        return [text]

    items = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        items.append(text[start:end].rstrip())

    # Include any text before the first list item (heading, etc.)
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            items.insert(0, prefix)

    return items


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------


def _make_chunk(
    block: dict[str, Any],
    text: str,
    chunk_index: int,
    book_slug: str,
) -> dict[str, Any]:
    """Assemble a Milvus-ready chunk dict from a PageBlock and text."""
    chapter_num = block.get("chapter_num") or 0
    page_start = block.get("book_page_start") or 0
    section_path = block.get("section_path") or []

    chunk_id = f"{book_slug}_ch{chapter_num:02d}_p{page_start:04d}_c{chunk_index:04d}"

    return {
        "id": chunk_id,
        "text": text.strip(),
        "book": book_slug,
        "chapter_num": chapter_num,
        "chapter_title": block.get("chapter_title") or "",
        "section": section_path[1] if len(section_path) > 1 else "",
        "subsection": section_path[2] if len(section_path) > 2 else "",
        "page_start": page_start,
        "page_end": block.get("book_page_end") or page_start,
        "chunk_index": chunk_index,
        "content_type": block.get("content_type") or "prose",
    }


# ---------------------------------------------------------------------------
# Content-type chunking strategies
# ---------------------------------------------------------------------------


def chunk_atomic(
    block: dict[str, Any],
    chunk_index: int,
    book_slug: str,
) -> tuple[list[dict[str, Any]], int]:
    """Atomic chunk: one chunk per block (tables, criteria, scales, images, exam)."""
    text = block["content"]
    chunk = _make_chunk(block, text, chunk_index, book_slug)
    return [chunk], chunk_index + 1


def chunk_list(
    block: dict[str, Any],
    chunk_index: int,
    book_slug: str,
) -> tuple[list[dict[str, Any]], int]:
    """Chunk a list block: atomic if ≤ 400 tokens, split at item boundaries."""
    text = block["content"]
    tokens = count_tokens(text)

    if tokens <= LIST_ATOMIC_MAX:
        chunk = _make_chunk(block, text, chunk_index, book_slug)
        return [chunk], chunk_index + 1

    # Split at list item boundaries
    items = split_list_items(text)
    if len(items) <= 1:
        # Can't split further — keep atomic
        chunk = _make_chunk(block, text, chunk_index, book_slug)
        return [chunk], chunk_index + 1

    # Group items into chunks targeting LIST_ATOMIC_MAX tokens
    chunks = []
    current_items: list[str] = []
    current_tokens = 0

    for item in items:
        item_tokens = count_tokens(item)
        if current_items and current_tokens + item_tokens > LIST_ATOMIC_MAX:
            # Emit current group
            chunk_text = "\n".join(current_items)
            chunk = _make_chunk(block, chunk_text, chunk_index, book_slug)
            chunks.append(chunk)
            chunk_index += 1
            current_items = []
            current_tokens = 0

        current_items.append(item)
        current_tokens += item_tokens

    # Emit remaining items
    if current_items:
        chunk_text = "\n".join(current_items)
        chunk = _make_chunk(block, chunk_text, chunk_index, book_slug)
        chunks.append(chunk)
        chunk_index += 1

    return chunks, chunk_index


def chunk_exam_technique(
    blocks: list[dict[str, Any]],
    start_idx: int,
    chunk_index: int,
    book_slug: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge consecutive exam_technique_flattened blocks from the same page.

    The postprocessor splits exam pages into many small sub-blocks (headings,
    bullet items, etc.).  The spec says exam technique should be "1 chunk per
    PageBlock (already page-scoped)", so we re-merge blocks from the same
    source page into a single chunk.

    Returns (chunks, next_block_index, next_chunk_index).
    """
    chunks = []
    i = start_idx
    source_block = blocks[i]
    page = source_block.get("book_page_start")

    # Collect all exam_technique blocks from this page
    parts: list[str] = []
    while i < len(blocks):
        block = blocks[i]
        if block["content_type"] != "exam_technique_flattened":
            break
        if block.get("book_page_start") != page:
            break
        parts.append(block["content"])
        i += 1

    merged_text = "\n\n".join(parts)
    chunk = _make_chunk(source_block, merged_text, chunk_index, book_slug)
    chunks.append(chunk)
    chunk_index += 1

    return chunks, i, chunk_index


def chunk_prose(
    blocks: list[dict[str, Any]],
    start_idx: int,
    chunk_index: int,
    book_slug: str,
) -> tuple[list[dict[str, Any]], int, int]:
    """Chunk prose blocks with greedy same-page merging and sentence splitting.

    Greedily collects consecutive prose blocks on the same page into one
    buffer, then emits chunks in the 300–400 token target range with
    sentence-boundary splitting and overlap.  This prevents section headings
    and short paragraphs from becoming isolated tiny chunks.

    Returns (chunks, next_block_index, next_chunk_index).
    """
    chunks = []
    i = start_idx

    # ---- Phase 1: greedily collect all consecutive prose on this page ----
    page = blocks[i].get("book_page_start")
    collected_parts: list[str] = []
    source_block = blocks[i]  # metadata donor

    while i < len(blocks):
        block = blocks[i]
        if block["content_type"] != "prose":
            break
        if block.get("book_page_start") != page:
            break
        collected_parts.append(block["content"])
        i += 1

    merged_text = "\n\n".join(collected_parts)

    # ---- Phase 2: split the collected text into target-sized chunks ----
    total_tokens = count_tokens(merged_text)

    if total_tokens <= PROSE_TARGET_MAX:
        # Small enough for a single chunk
        chunk = _make_chunk(source_block, merged_text, chunk_index, book_slug)
        chunks.append(chunk)
        chunk_index += 1
        return chunks, i, chunk_index

    # Split at sentence boundaries with overlap
    sentences = split_sentences(merged_text)
    if len(sentences) <= 1:
        # Can't split — keep as single chunk
        chunk = _make_chunk(source_block, merged_text, chunk_index, book_slug)
        chunks.append(chunk)
        chunk_index += 1
        return chunks, i, chunk_index

    current_sentences: list[str] = []
    current_tokens = 0

    for sent_idx, sentence in enumerate(sentences):
        sent_tokens = count_tokens(sentence)

        if current_sentences and current_tokens + sent_tokens > PROSE_TARGET_MAX:
            # Emit current chunk
            chunk_text = " ".join(current_sentences)
            chunk = _make_chunk(source_block, chunk_text, chunk_index, book_slug)
            chunks.append(chunk)
            chunk_index += 1

            # Overlap: carry forward last N sentences
            overlap = current_sentences[-PROSE_OVERLAP_SENTENCES:]
            current_sentences = list(overlap)
            current_tokens = sum(count_tokens(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Emit remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        # Don't emit if this is just overlap from the last chunk
        if chunk_text.strip():
            chunk = _make_chunk(source_block, chunk_text, chunk_index, book_slug)
            chunks.append(chunk)
            chunk_index += 1

    return chunks, i, chunk_index


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------


def chunk_pageblocks(
    blocks: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Chunk all PageBlocks into Milvus-ready dicts."""
    book_slug = meta.get("book", "bates")
    all_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    collisions = 0

    # Atomic content types (1 block → 1 chunk, never split)
    ATOMIC_TYPES = {
        "table",
        "clinical_criteria",
        "clinical_scale",
        "image_desc",
    }

    # Track last table chunk for footnote merging
    last_table_chunk_idx: int | None = None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        content_type = block.get("content_type", "prose")
        page_start = block.get("book_page_start") or 0

        # Determine chunk_index: sequential within each page
        # Count existing chunks for this page to get the next index
        chunk_index = sum(1 for c in all_chunks if c.get("page_start") == page_start)

        if content_type == "table_footnote":
            # Merge into the last table chunk
            if last_table_chunk_idx is not None:
                prev = all_chunks[last_table_chunk_idx]
                prev["text"] = prev["text"] + "\n\n" + block["content"]
                i += 1
                continue
            else:
                # No preceding table — emit as standalone prose
                chunks, next_ci = chunk_atomic(block, chunk_index, book_slug)
                for c in chunks:
                    c["content_type"] = "prose"

        elif content_type in ATOMIC_TYPES:
            chunks, next_ci = chunk_atomic(block, chunk_index, book_slug)
            if content_type == "table":
                last_table_chunk_idx = len(all_chunks)
            else:
                last_table_chunk_idx = None

        elif content_type == "exam_technique_flattened":
            # Merge all exam blocks from this page into one chunk
            chunks, i, next_ci = chunk_exam_technique(blocks, i, chunk_index, book_slug)
            last_table_chunk_idx = None
            # chunk_exam_technique advances i itself
            for chunk in chunks:
                chunk_id = chunk["id"]
                while chunk_id in seen_ids:
                    next_ci_bump = int(chunk_id.split("_c")[-1]) + 1
                    chunk_id = chunk_id.rsplit("_c", 1)[0] + f"_c{next_ci_bump:04d}"
                    collisions += 1
                chunk["id"] = chunk_id
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)
            continue  # i already advanced

        elif content_type == "list":
            chunks, next_ci = chunk_list(block, chunk_index, book_slug)
            last_table_chunk_idx = None

        elif content_type == "prose":
            chunks, i, next_ci = chunk_prose(blocks, i, chunk_index, book_slug)
            last_table_chunk_idx = None
            # chunk_prose advances i itself, so skip the i += 1 below
            for chunk in chunks:
                chunk_id = chunk["id"]
                while chunk_id in seen_ids:
                    next_ci_bump = int(chunk_id.split("_c")[-1]) + 1
                    chunk_id = chunk_id.rsplit("_c", 1)[0] + f"_c{next_ci_bump:04d}"
                    collisions += 1
                chunk["id"] = chunk_id
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)
            continue  # i already advanced by chunk_prose

        else:
            # Unknown type — treat as atomic
            chunks, next_ci = chunk_atomic(block, chunk_index, book_slug)
            last_table_chunk_idx = None

        # Deduplicate IDs
        for chunk in chunks:
            chunk_id = chunk["id"]
            while chunk_id in seen_ids:
                next_ci_bump = int(chunk_id.split("_c")[-1]) + 1
                chunk_id = chunk_id.rsplit("_c", 1)[0] + f"_c{next_ci_bump:04d}"
                collisions += 1
            chunk["id"] = chunk_id
            seen_ids.add(chunk_id)
            all_chunks.append(chunk)

        i += 1

    if collisions:
        print(f"[chunker] Resolved {collisions} ID collision(s)")

    # ---- Post-filter: drop heading-only prose chunks below threshold ----
    re_heading_line = re.compile(r"^#{1,6}\s+")
    before = len(all_chunks)
    filtered: list[dict[str, Any]] = []
    for chunk in all_chunks:
        if chunk["content_type"] == "prose":
            tokens = count_tokens(chunk["text"])
            if tokens < HEADING_ONLY_DROP_MAX:
                lines = [l.strip() for l in chunk["text"].split("\n") if l.strip()]
                if lines and all(re_heading_line.match(l) for l in lines):
                    continue  # drop heading-only noise chunk
        filtered.append(chunk)
    dropped = before - len(filtered)
    if dropped:
        print(
            f"[chunker] Dropped {dropped} heading-only prose chunk(s) < {HEADING_ONLY_DROP_MAX} tokens"
        )

    return filtered


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_stats(chunks: list[dict[str, Any]]) -> None:
    """Print summary statistics about the chunks."""
    if not chunks:
        print("[chunker] No chunks produced")
        return

    # Content type distribution
    type_counts: dict[str, int] = {}
    for c in chunks:
        ct = c["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    print(f"\n[chunker] Total chunks: {len(chunks)}")
    print(f"[chunker] Content types:")
    for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:>30}: {count:>4}")

    # Token statistics
    token_counts = [count_tokens(c["text"]) for c in chunks]
    token_counts.sort()
    n = len(token_counts)
    print(f"\n[chunker] Token distribution:")
    print(f"  Min:    {token_counts[0]}")
    print(f"  P25:    {token_counts[int(n * 0.25)]}")
    print(f"  Median: {token_counts[n // 2]}")
    print(f"  Mean:   {sum(token_counts) // n}")
    print(f"  P75:    {token_counts[int(n * 0.75)]}")
    print(f"  P95:    {token_counts[int(n * 0.95)]}")
    print(f"  Max:    {token_counts[-1]}")

    # Bucket distribution
    buckets = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 10000)]
    print(f"\n[chunker] Token buckets:")
    for lo, hi in buckets:
        count = sum(1 for t in token_counts if lo <= t < hi)
        pct = count / n * 100
        print(f"  {lo:>5}-{hi:<5}: {count:>4} ({pct:.1f}%)")

    # Chapter coverage
    chapters = set(c["chapter_num"] for c in chunks)
    print(f"\n[chunker] Chapters: {sorted(chapters)}")

    # Unique IDs check
    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        dupes = len(ids) - len(set(ids))
        print(f"\n[chunker] WARNING: {dupes} duplicate IDs!")
    else:
        print(f"[chunker] All {len(ids)} IDs are unique")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Chunk PageBlocks for Milvus vector DB ingestion"
    )
    parser.add_argument(
        "pageblocks", help="Path to pageblocks.json from postprocess_docstrange.py"
    )
    parser.add_argument("meta", help="Path to meta.json from extract_toc.py")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: <pageblocks_stem>.chunks.json)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Print chunk statistics (default: True)",
    )
    args = parser.parse_args()

    pb_path = Path(args.pageblocks)
    meta_path = Path(args.meta)

    if not pb_path.exists():
        print(f"Error: PageBlocks file not found: {pb_path}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"Error: Meta file not found: {meta_path}")
        sys.exit(1)

    blocks = json.loads(pb_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"[chunker] Input: {pb_path.name} ({len(blocks)} blocks)")
    print(f"[chunker] Book: {meta.get('book', 'unknown')}")

    # Initialize tokenizer eagerly
    _get_tokenizer()

    chunks = chunk_pageblocks(blocks, meta)

    # Output
    output_path = (
        Path(args.output) if args.output else pb_path.with_suffix(".chunks.json")
    )
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[chunker] Output: {output_path}")

    if args.stats:
        print_stats(chunks)


if __name__ == "__main__":
    main()
