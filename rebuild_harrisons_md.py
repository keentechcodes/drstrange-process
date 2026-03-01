#!/usr/bin/env python3
"""rebuild_harrisons_md.py — Fix structural issues in Harrison's OCR output.

Reads individual chunk files from results_harrisons4/chunk_00N/ and rebuilds
the concatenated HARRISONS21ST.md with the following fixes:

1. Correct chunk ordering (was wrong due to as_completed() in parallel mode)
2. Renumber page markers (# Page N and ## Page N) from chunk-relative to
   absolute book page numbers
3. Fix image references (add c{chunk}_  prefix to image paths)
4. Split merged pages (where two pages were combined under one ## Page header)

Usage:
    python rebuild_harrisons_md.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path("results_harrisons4")
ORIGINAL_META = Path("harrisons21st.meta.json")
OUTPUT_FILE = RESULTS_DIR / "HARRISONS21ST.md"
BACKUP_FILE = RESULTS_DIR / "HARRISONS21ST.md.bak"


def load_chunk(chunk_num: int) -> tuple[str, dict]:
    """Load a chunk's markdown and meta."""
    chunk_dir = RESULTS_DIR / f"chunk_{chunk_num:03d}"
    meta_path = chunk_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    md_files = list(chunk_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md file in {chunk_dir}")
    md_text = md_files[0].read_text(encoding="utf-8")
    return md_text, meta


def renumber_pages(md_text: str, original_pdf_start: int, page_offset: int) -> str:
    """Renumber both # Page N and ## Page N markers to absolute book pages.

    book_page = original_pdf_start - page_offset
    shift = book_page - 1  (since chunk pages start at 1)
    """
    book_page_first = original_pdf_start - page_offset
    shift = book_page_first - 1

    if shift == 0:
        return md_text

    def _shift(match: re.Match) -> str:
        hashes = match.group(1)
        old_num = int(match.group(2))
        return f"{hashes} Page {old_num + shift}"

    # Match both "# Page N" and "## Page N" at line start
    return re.sub(r"^(#{1,2}) Page (\d+)", _shift, md_text, flags=re.MULTILINE)


def fix_image_refs(md_text: str, chunk_num: int) -> str:
    """Add chunk prefix to image paths in markdown references.

    Transforms:
        ![alt](images/page31_img2.png)  ->  ![alt](images/c001_page31_img2.png)
        <img src="images/page31_img2.png">  ->  <img src="images/c001_page31_img2.png">
    """
    prefix = f"c{chunk_num:03d}_"

    # Fix markdown image syntax: ![...](images/filename)
    md_text = re.sub(
        r"!\[([^\]]*)\]\(images/([^)]+)\)",
        lambda m: f"![{m.group(1)}](images/{prefix}{m.group(2)})",
        md_text,
    )

    # Fix HTML img tags: <img src="images/filename">
    md_text = re.sub(
        r'(<img[^>]*\bsrc=["\'])images/([^"\']+)(["\'])',
        lambda m: f"{m.group(1)}images/{prefix}{m.group(2)}{m.group(3)}",
        md_text,
    )

    return md_text


def split_merged_pages(md_text: str) -> str:
    """Split pages where two pages got merged under one ## Page header.

    Detects patterns like:
        ## Page 911
        ... content of page 911 ...
        ---
        <page_number>912</page_number>
        ... content of page 912 ...
        ## Page 913

    The <page_number> tag for the NEXT page within a page's content indicates
    a merged page. We split by inserting a ## Page N header.
    """
    # Strategy: Find <page_number>N</page_number> tags that appear between
    # two ## Page headers where N doesn't match either header's number.
    # These indicate a page boundary that should have its own ## Page header.

    lines = md_text.split("\n")
    output_lines = []
    i = 0
    current_page = None
    splits_made = 0

    while i < len(lines):
        line = lines[i]

        # Track current page
        page_match = re.match(r"^(#{1,2}) Page (\d+)$", line)
        if page_match:
            current_page = int(page_match.group(2))
            output_lines.append(line)
            i += 1
            continue

        # Check for <page_number>N</page_number> that indicates a merged page
        pn_match = re.match(r"^\s*<page_number>\s*(\d+)\s*</page_number>\s*$", line)
        if pn_match and current_page is not None:
            embedded_page = int(pn_match.group(1))
            # If the embedded page number is current_page + 1, this is a
            # page boundary — insert a ## Page header before it
            if embedded_page == current_page + 1:
                # Look back for a --- separator to place the split before it
                # Check if previous non-empty line is ---
                insert_pos = len(output_lines)
                for j in range(
                    len(output_lines) - 1, max(len(output_lines) - 4, -1), -1
                ):
                    if output_lines[j].strip() == "---":
                        insert_pos = j
                        break
                    elif output_lines[j].strip() == "":
                        continue
                    else:
                        break

                # Insert ## Page N header
                if insert_pos < len(output_lines):
                    # Replace the --- with a page break
                    output_lines.insert(insert_pos, "")
                    output_lines.insert(insert_pos + 1, f"## Page {embedded_page}")
                    output_lines.insert(insert_pos + 2, "")
                else:
                    output_lines.append("")
                    output_lines.append(f"## Page {embedded_page}")
                    output_lines.append("")

                current_page = embedded_page
                splits_made += 1
                # Keep the <page_number> tag line too
                output_lines.append(line)
                i += 1
                continue

        output_lines.append(line)
        i += 1

    if splits_made > 0:
        print(f"  Split {splits_made} merged pages")

    return "\n".join(output_lines)


def main():
    # Load original meta for page_offset
    if not ORIGINAL_META.exists():
        print(f"ERROR: {ORIGINAL_META} not found", file=sys.stderr)
        sys.exit(1)

    original_meta = json.loads(ORIGINAL_META.read_text())
    page_offset = original_meta["page_offset"]

    print(f"Book page_offset: {page_offset}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    # Backup original
    if OUTPUT_FILE.exists():
        print(f"Backing up original to {BACKUP_FILE}")
        import shutil

        shutil.copy2(str(OUTPUT_FILE), str(BACKUP_FILE))

    # Process each chunk in order
    all_chunks = []
    for chunk_num in range(1, 9):
        print(f"Processing chunk {chunk_num:03d}...")

        md_text, meta = load_chunk(chunk_num)
        original_pdf_start = meta["_original_pdf_start"]
        original_pdf_end = meta["_original_pdf_end"]
        book_page_first = original_pdf_start - page_offset
        book_page_last = original_pdf_end - page_offset

        print(
            f"  PDF pages {original_pdf_start}-{original_pdf_end} -> book pages {book_page_first}-{book_page_last}"
        )

        # Step 1: Renumber pages
        md_text = renumber_pages(md_text, original_pdf_start, page_offset)

        # Step 2: Fix image references
        md_text = fix_image_refs(md_text, chunk_num)

        # Step 3: Split merged pages
        md_text = split_merged_pages(md_text)

        # Verify page range
        page_nums = [
            int(m.group(1))
            for m in re.finditer(r"^## Page (\d+)", md_text, re.MULTILINE)
        ]
        if page_nums:
            print(
                f"  Page range: {min(page_nums)}-{max(page_nums)} ({len(page_nums)} pages)"
            )
        else:
            print(f"  WARNING: No ## Page markers found!")

        all_chunks.append(md_text.rstrip())

    # Concatenate in order
    print(f"\nConcatenating {len(all_chunks)} chunks in order...")
    final_md = "\n\n".join(all_chunks) + "\n"

    # Write output
    OUTPUT_FILE.write_text(final_md, encoding="utf-8")
    print(f"Output: {OUTPUT_FILE} ({len(final_md):,} chars)")

    # Validate
    print("\n=== Validation ===")
    all_pages = [
        int(m.group(1)) for m in re.finditer(r"^## Page (\d+)", final_md, re.MULTILINE)
    ]
    all_pages_sorted = sorted(all_pages)

    print(f"Total ## Page markers: {len(all_pages)}")
    print(f"Unique pages: {len(set(all_pages))}")
    print(f"Page range: {all_pages_sorted[0]}-{all_pages_sorted[-1]}")

    # Check duplicates
    from collections import Counter

    counts = Counter(all_pages)
    dupes = {p: c for p, c in counts.items() if c > 1}
    if dupes:
        print(f"DUPLICATE pages: {dupes}")
    else:
        print("Duplicate pages: none")

    # Check gaps
    gaps = []
    for i in range(1, len(all_pages_sorted)):
        if all_pages_sorted[i] != all_pages_sorted[i - 1] + 1:
            gap_size = all_pages_sorted[i] - all_pages_sorted[i - 1] - 1
            gaps.append((all_pages_sorted[i - 1], all_pages_sorted[i], gap_size))

    total_missing = sum(g[2] for g in gaps)
    print(f"Gaps: {len(gaps)} ({total_missing} missing pages)")
    if gaps:
        for prev, nxt, size in gaps:
            print(f"  {prev} -> {nxt} (missing {size})")

    # Check ordering
    ordered = all(all_pages[i] <= all_pages[i + 1] for i in range(len(all_pages) - 1))
    print(f"Pages in order: {'yes' if ordered else 'NO — still out of order!'}")

    # Check image references
    broken_refs = 0
    total_refs = 0
    images_dir = RESULTS_DIR / "images"
    for m in re.finditer(r"!\[[^\]]*\]\(images/([^)]+)\)", final_md):
        total_refs += 1
        img_path = images_dir / m.group(1)
        if not img_path.exists():
            broken_refs += 1

    print(f"Image references: {total_refs} total, {broken_refs} broken")


if __name__ == "__main__":
    main()
