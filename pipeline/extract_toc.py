#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pymupdf>=1.24.0",
# ]
# ///
"""
Extract TOC and metadata from textbook PDF.

Usage:
    uv run extract_toc.py <pdf_path> [--output-dir DIR]

Outputs:
    - <bookname>.toc.json   - Enriched TOC entries with page ranges
    - <bookname>.meta.json  - Book metadata including page offset
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


BATES_CHAPTER_ANCHORS: dict[str, int] = {
    "Foundations for Clinical Proficiency": 1,
    "Evaluating Clinical Evidence": 29,
    "Interviewing and the Health History": 45,
    "Beginning the Physical Examination": 65,
    "Behavior and Mental Status": 87,
    "The Skin, Hair, and Nails": 99,
    "The Head and Neck": 129,
    "The Thorax and Lungs": 159,
    "The Cardiovascular System": 181,
    "The Breasts and Axillae": 203,
    "The Abdomen": 215,
    "The Peripheral Vascular System": 235,
    "Male Genitalia and Hernias": 249,
    "Female Genitalia": 265,
    "The Anus, Rectum, and Prostate": 283,
    "The Musculoskeletal System": 293,
    "The Nervous System": 331,
    "Assessing Children: Infancy through Adolescence": 369,
    "The Pregnant Woman": 403,
    "The Older Adult": 419,
}

BATES_CONTENT_END_PDF: int = 460


def parse_chapter_num(title: str) -> int | None:
    m = re.match(r"Chapter\s+(\d+)", title, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)\s", title)
    if m:
        return int(m.group(1))
    return None


def detect_page_offset_from_anchors(
    toc: list[tuple[int, str, int]], anchors: dict[str, int]
) -> int:
    """Detect page offset by matching TOC entries against known anchor titles.

    Used for books like Bates where we have hardcoded chapter → book page
    mappings.  Returns 0 if no anchors match.
    """
    from collections import Counter

    offsets: list[int] = []
    for entry in toc:
        level, title, pdf_page = entry[0], entry[1], entry[2]
        if level != 1:
            continue
        clean_title = re.sub(r"^\d+\s*", "", title).strip()
        for anchor_title, book_page in anchors.items():
            if (
                anchor_title.lower() in clean_title.lower()
                or clean_title.lower() in anchor_title.lower()
            ):
                offsets.append(pdf_page - book_page)
                break

    if not offsets:
        return 0

    offset_counts = Counter(offsets)
    best_offset, count = offset_counts.most_common(1)[0]
    if count < len(offsets):
        mismatches = [(o, c) for o, c in offset_counts.items() if o != best_offset]
        print(
            f"[toc] WARNING: inconsistent offsets detected. "
            f"Using {best_offset} ({count}/{len(offsets)} matches). "
            f"Outliers: {mismatches}"
        )
    return best_offset


def detect_page_offset_from_toc(toc: list[tuple[int, str, int]]) -> int:
    """Detect page offset from an embedded TOC by finding the first numbered chapter.

    Looks for entries like "1 The Practice of Medicine" at a known PDF page.
    The chapter's book page is assumed to be its number (chapter 1 → book page 1).
    Offset = pdf_page - book_page.

    Heuristic: find the first L2 (or L1) entry whose title starts with "1 "
    or "PART 1".  For books where Part 1 / Chapter 1 starts on book page 1,
    the offset = pdf_page_of_that_entry - 1.

    Falls back to examining multiple low-numbered chapters and taking the mode.
    """
    from collections import Counter

    offsets: list[int] = []
    for level, title, pdf_page in toc:
        m = re.match(r"^(\d+)\s+\S", title)
        if m:
            chapter_num = int(m.group(1))
            if chapter_num > 10:
                continue  # only use early chapters for offset detection
            # Assumption: chapter N starts on book page N only for chapter 1.
            # For others, we can't know the book page from the chapter number.
            # But we CAN use the pattern: for most textbooks, the first chapter
            # in the TOC (chapter 1 or Part 1) starts on book page 1.
            if chapter_num == 1:
                offsets.append(pdf_page - 1)

        # Also check for "PART 1" patterns — these often start on page 1
        if re.match(r"^PART\s+1\b", title, re.IGNORECASE) and level == 1:
            offsets.append(pdf_page - 1)

    if not offsets:
        return 0

    offset_counts = Counter(offsets)
    best_offset, count = offset_counts.most_common(1)[0]
    return best_offset


def classify_entry(book_page_start: int | None, title: str) -> str:
    """Classify a TOC entry as front_matter, back_matter, or chapter."""
    if book_page_start is None or book_page_start < 1:
        return "front_matter"
    lower_title = title.lower()
    if any(x in lower_title for x in ["index", "appendix", "references", "glossary"]):
        return "back_matter"
    return "chapter"


def generate_toc_from_anchors(
    anchors: dict[str, int],
    page_offset: int,
    total_pdf_pages: int,
    content_end_pdf: int | None = None,
) -> list[dict[str, Any]]:
    chapters = sorted(anchors.items(), key=lambda x: x[1])
    enriched = []
    effective_end = content_end_pdf if content_end_pdf else total_pdf_pages
    for i, (title, book_page_start) in enumerate(chapters):
        pdf_page_start = book_page_start + page_offset
        if i + 1 < len(chapters):
            pdf_page_end = chapters[i + 1][1] + page_offset - 1
        else:
            pdf_page_end = effective_end
        book_page_end = pdf_page_end - page_offset
        chapter_num = parse_chapter_num(title) or (i + 1)
        enriched.append(
            {
                "level": 1,
                "title": title,
                "chapter_num": chapter_num,
                "pdf_page_start": pdf_page_start,
                "pdf_page_end": pdf_page_end,
                "book_page_start": book_page_start,
                "book_page_end": book_page_end,
                "entry_type": "chapter",
            }
        )
    return enriched


def extract_toc(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    total_pdf_pages = len(doc)
    raw_toc = doc.get_toc(simple=True)
    doc.close()

    book_name = pdf_path.stem.lower()
    # Default offset; will be overridden by TOC detection if possible.
    # Bates has no embedded TOC, so we keep its known offset as fallback.
    page_offset = 17 if book_name == "bates" else 0

    # Ensure TOC entries are sorted by pdf_page (PyMuPDF normally returns
    # document order, but malformed PDFs can produce unsorted entries).
    if raw_toc:
        raw_toc = sorted(raw_toc, key=lambda e: e[2])

        # For Bates: try anchor-based detection (known chapter → book page mapping).
        # For other books: use embedded TOC heuristic (first chapter at PDF page N → offset).
        # Only try Bates anchors for Bates PDFs to avoid false substring matches.
        detected_offset = 0
        offset_source = "default"

        if book_name == "bates":
            detected_offset = detect_page_offset_from_anchors(
                raw_toc, BATES_CHAPTER_ANCHORS
            )
            if detected_offset > 0:
                offset_source = "anchor matching"

        if detected_offset == 0:
            detected_offset = detect_page_offset_from_toc(raw_toc)
            if detected_offset > 0:
                offset_source = "embedded TOC"

        if detected_offset > 0:
            page_offset = detected_offset

        print(f"[toc] Using page offset: {page_offset} (from {offset_source})")
    else:
        print(
            f"[warn] No embedded TOC in {pdf_path.name}, generating from known chapter anchors"
        )
        if book_name == "bates" and BATES_CHAPTER_ANCHORS:
            print(f"[toc] Using page offset: {page_offset} (hardcoded for BATES)")
            enriched = generate_toc_from_anchors(
                BATES_CHAPTER_ANCHORS,
                page_offset,
                total_pdf_pages,
                content_end_pdf=BATES_CONTENT_END_PDF,
            )
        else:
            print(
                f"[error] No embedded TOC and no hardcoded anchors for '{book_name}'. "
                f"Add chapter anchors to the script or use a PDF with an embedded TOC."
            )
            return [], {}

        meta = {
            "book": book_name,
            "source_pdf": pdf_path.name,
            "total_pdf_pages": total_pdf_pages,
            "page_offset": page_offset,
            "content_pdf_page_start": enriched[0]["pdf_page_start"]
            if enriched
            else None,
            "content_pdf_page_end": enriched[-1]["pdf_page_end"] if enriched else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return enriched, meta

    print(f"[toc] Detected page offset: {page_offset}")

    enriched = []
    for i, entry in enumerate(raw_toc):
        level, title, pdf_page_start = entry[0], entry[1], entry[2]

        if i + 1 < len(raw_toc):
            pdf_page_end = raw_toc[i + 1][2] - 1
        else:
            pdf_page_end = total_pdf_pages

        # Container entries (Parts, Sections) may start on the same page as
        # their first child, giving pdf_page_end < pdf_page_start.  Clamp.
        if pdf_page_end < pdf_page_start:
            pdf_page_end = pdf_page_start

        book_page_start = pdf_page_start - page_offset
        book_page_end = pdf_page_end - page_offset

        chapter_num = parse_chapter_num(title)
        entry_type = classify_entry(book_page_start, title)

        enriched.append(
            {
                "level": level,
                "title": title,
                "chapter_num": chapter_num,
                "pdf_page_start": pdf_page_start,
                "pdf_page_end": pdf_page_end,
                "book_page_start": book_page_start if book_page_start > 0 else None,
                "book_page_end": book_page_end if book_page_end > 0 else None,
                "entry_type": entry_type,
            }
        )

    content_pdf_page_start = None
    content_pdf_page_end = None
    for entry in enriched:
        if entry["entry_type"] == "chapter":
            if content_pdf_page_start is None:
                content_pdf_page_start = entry["pdf_page_start"]
            content_pdf_page_end = entry["pdf_page_end"]

    meta = {
        "book": book_name,
        "source_pdf": pdf_path.name,
        "total_pdf_pages": total_pdf_pages,
        "page_offset": page_offset,
        "content_pdf_page_start": content_pdf_page_start,
        "content_pdf_page_end": content_pdf_page_end,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return enriched, meta


def main():
    parser = argparse.ArgumentParser(
        description="Extract TOC and metadata from textbook PDF"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory (default: current dir)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[toc] Extracting TOC from: {pdf_path}")
    toc, meta = extract_toc(pdf_path)

    if not toc:
        print("[toc] No TOC entries extracted")
        sys.exit(1)

    book_name = meta["book"]
    toc_path = output_dir / f"{book_name}.toc.json"
    meta_path = output_dir / f"{book_name}.meta.json"

    toc_path.write_text(json.dumps(toc, indent=2, ensure_ascii=False))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[toc] Wrote {len(toc)} entries to {toc_path}")
    print(f"[toc] Wrote metadata to {meta_path}")
    print(
        f"[toc] Content pages: PDF {meta['content_pdf_page_start']}-{meta['content_pdf_page_end']}"
    )


if __name__ == "__main__":
    main()
