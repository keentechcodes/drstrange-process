#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beautifulsoup4>=4.12.0",
#     "lxml>=5.0.0",
# ]
# ///
"""
Post-process DocStrange markdown output into structured PageBlocks.

Six sequential passes transform raw markdown into clean, classified,
chapter-annotated blocks ready for the chunker (chunk_textbook.py).

Usage:
    python postprocess_docstrange.py <markdown_file> <toc_json> <meta_json> [--output FILE]

Inputs:
    - DocStrange markdown (.md) with ## Page N headers
    - bates.toc.json from extract_toc.py
    - bates.meta.json from extract_toc.py

Output:
    - bates.pageblocks.json (list of PageBlock dicts)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def make_raw_page_block(
    pdf_page: int,
    book_page_start: int | None,
    book_page_end: int | None,
    raw_cleaned: str,
) -> dict[str, Any]:
    """Create a RawPageBlock dict (Pass 1 output)."""
    return {
        "pdf_page": pdf_page,
        "book_page_start": book_page_start,
        "book_page_end": book_page_end,
        "raw_cleaned": raw_cleaned,
        # Populated by later passes
        "chapter_num": None,
        "chapter_title": None,
        "section_path": [],
        "is_exam_page": False,
        "merged_from_pdf_pages": [pdf_page],
    }


# ---------------------------------------------------------------------------
# Pass 1 — Page Splitter
# ---------------------------------------------------------------------------

# Patterns for page boundaries and boilerplate
RE_PAGE_HEADER = re.compile(r"^#{1,2}\s+Page\s+(\d+)\s*$", re.MULTILINE)
RE_PAGE_NUMBER_TAG = re.compile(r"<page_number>\s*(\d+)\s*</page_number>")
RE_RUNNING_HEADER_CHAPTER = re.compile(r"^Chapter\s+\d+\s*\|\s*.+$", re.MULTILINE)
RE_RUNNING_HEADER_BOOK = re.compile(
    r"^.*Bates['\u2019]?\s*Pocket\s+Guide\s+to\s+Physical\s+Examination.*$",
    re.MULTILINE,
)
RE_SPACED_CHAPTER = re.compile(
    r"<header>\s*C\s+H\s+A\s+P\s+T\s+E\s+R\s+\d+\s*</header>", re.IGNORECASE
)
RE_HEADER_TAG = re.compile(r"<header>.*?</header>", re.DOTALL)
RE_PAGE_SEPARATOR = re.compile(r"^---\s*$", re.MULTILINE)


def _pass0_pre_clean(markdown: str, meta: dict[str, Any] | None = None) -> str:
    """Pre-clean markdown before page splitting.

    Fixes converter artefacts that would break page splitting:
    1. Malformed page headers: '</table>## Page N' on same line
       → inserts newline so the regex ^## Page can match.
    2. Prompt leak: the model sometimes echoes the STRICT RULES block
       from the OCR prompt instead of extracting page content.
       Detected on 9 chapter-opener pages in run 2. Strip entirely.
    3. H1 page header: '# Page 1' (heading normalization changed the
       level) → normalize to '## Page 1'.
    4. Un-renumbered pages: chapter opener pages that were stuck in
       '</table>## Page N' format weren't renumbered by the converter.
       They keep PDF page numbers while all other pages have book page
       numbers.  Detect by non-monotonic sequence and renumber.
    """
    fixes = 0

    # Fix 1: malformed page headers — ensure ## Page N starts on its own line.
    # Handles '</table>## Page N' and any other prefix stuck to the header.
    fixed, count = re.subn(r"([^\n])(#{1,2}\s+Page\s+\d+)", r"\1\n\2", markdown)
    if count:
        markdown = fixed
        fixes += count

    # Fix 2: strip STRICT RULES prompt leak blocks.
    # Pattern: starts with "STRICT RULES:" and runs through the numbered rules.
    # May end with "---" separator or run to end of page.
    markdown, count = re.subn(
        r"STRICT RULES:\n(?:.*\n)*?(?:---\s*\n|(?=\n## Page )|\Z)",
        "",
        markdown,
    )
    if count:
        fixes += count

    # Fix 3: normalize '# Page N' (H1) to '## Page N' (H2)
    markdown, count = re.subn(
        r"^# (Page\s+\d+)\s*$", r"## \1", markdown, flags=re.MULTILINE
    )
    if count:
        fixes += count

    # Fix 4: renumber un-renumbered pages.
    # After fixes 1-3, pages should be mostly monotonic (book page numbers).
    # Pages that the converter failed to renumber still have PDF page numbers.
    # Detect: a page N where N > prev+5 AND N-offset would restore order.
    page_offset = (meta or {}).get("page_offset", 0)
    if page_offset > 0:
        page_header_re = re.compile(r"^(#{1,2}\s+Page\s+)(\d+)(\s*)$", re.MULTILINE)
        matches = list(page_header_re.finditer(markdown))
        page_nums = [int(m.group(2)) for m in matches]

        # Find non-monotonic pages that are exactly offset too high
        to_fix: dict[int, int] = {}  # match_index -> corrected_page_num
        for idx in range(1, len(page_nums)):
            prev = page_nums[idx - 1]
            curr = page_nums[idx]
            corrected = curr - page_offset
            # Page is out of order AND subtracting offset restores it
            if curr > prev + 5 and corrected == prev + 1:
                to_fix[idx] = corrected

        if to_fix:
            # Apply replacements in reverse order to preserve positions
            for idx in sorted(to_fix.keys(), reverse=True):
                m = matches[idx]
                new_num = to_fix[idx]
                replacement = f"{m.group(1)}{new_num}{m.group(3)}"
                markdown = markdown[: m.start()] + replacement + markdown[m.end() :]
            fixes += len(to_fix)
            print(
                f"[pass0] Renumbered {len(to_fix)} un-renumbered page(s) "
                f"(offset={page_offset})"
            )

    if fixes:
        print(f"[pass0] Pre-cleaned {fixes} total artefact(s)")

    return markdown


def pass1_split_pages(
    markdown: str,
    meta: dict[str, Any],
    toc_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Split markdown at ## Page N boundaries, extract book page numbers,
    strip boilerplate headers. Returns list of RawPageBlock dicts."""

    page_offset = meta.get("page_offset", 0)
    # For sample PDFs, all pages are content; for full book, filter by range
    content_start = meta.get("content_pdf_page_start")
    content_end = meta.get("content_pdf_page_end")

    # Find all ## Page N positions
    splits = list(RE_PAGE_HEADER.finditer(markdown))
    if not splits:
        print("[pass1] WARNING: No ## Page N headers found")
        return []

    # Build (pdf_page, raw_content) pairs.
    #
    # Page numbering convention:
    # - Old converter: ## Page N where N = PDF page number (e.g., 18 for first content page)
    # - New converter (with --meta-file): ## Page N where N = book page number
    #   (renumbered so first content page = 1).
    #
    # We detect which convention is in use and reconvert to PDF page numbers
    # so the rest of the pipeline (content filtering, chapter matching) works
    # unchanged.
    raw_pages: list[tuple[int, str]] = []
    for i, m in enumerate(splits):
        raw_page_num = int(m.group(1))
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(markdown)
        raw_content = markdown[start:end]
        raw_pages.append((raw_page_num, raw_content))

    # Determine if page numbers are relative (sample PDF), book-page-numbered
    # (new converter), or absolute PDF page numbers (old converter).
    #
    # Detection heuristic:
    # - Sample: significantly fewer pages than total_pdf_pages
    # - Book-page-numbered: first page number is small (< page_offset) AND
    #   page_offset > 0 AND not a sample
    # - Absolute PDF: first page number >= content_pdf_page_start
    all_page_nums = [p for p, _ in raw_pages]
    total_pdf_pages = meta.get("total_pdf_pages", 0)
    is_sample = (
        len(all_page_nums) < (total_pdf_pages * 0.5) if total_pdf_pages else False
    )

    if is_sample:
        print(
            f"[pass1] Detected sample PDF ({len(all_page_nums)} pages vs "
            f"{total_pdf_pages} total). Keeping all pages."
        )

    # Detect book-page-numbered input from the new converter.
    # If the first page number is 1 (or small) and page_offset > 0 and it's
    # not a sample, the page numbers are book pages, not PDF pages.
    # Reconvert: pdf_page = book_page + page_offset.
    is_book_page_numbered = False
    if (
        not is_sample
        and page_offset > 0
        and all_page_nums
        and all_page_nums[0] < page_offset
    ):
        is_book_page_numbered = True
        print(
            f"[pass1] Detected book-page-numbered input "
            f"(first page={all_page_nums[0]}, offset={page_offset}). "
            f"Reconverting to PDF page numbers."
        )
        raw_pages = [(p + page_offset, raw) for p, raw in raw_pages]

    blocks: list[dict[str, Any]] = []
    for pdf_page, raw in raw_pages:
        # Filter to content pages for full book processing only.
        # When input is book-page-numbered, pages have already been reconverted
        # to PDF page numbers above, and front-matter was already filtered by
        # the converter, so this filter is a no-op safety net.
        if content_start and content_end and not is_sample:
            if not (content_start <= pdf_page <= content_end):
                continue

        # Extract book page numbers from <page_number> tags
        page_nums = [int(x) for x in RE_PAGE_NUMBER_TAG.findall(raw)]
        book_page_start = min(page_nums) if page_nums else None
        book_page_end = max(page_nums) if page_nums else None

        # Strip boilerplate
        cleaned = raw

        # Strip stray </table> at start of page (from converter force-close)
        cleaned = re.sub(r"^\s*</table>\s*", "", cleaned)

        # Strip <page_number> tags (we already extracted the values)
        cleaned = RE_PAGE_NUMBER_TAG.sub("", cleaned)

        # Strip running headers: "Chapter N | Title"
        cleaned = RE_RUNNING_HEADER_CHAPTER.sub("", cleaned)

        # Strip running headers: "NNN Bates' Pocket Guide..." (with or without leading page number)
        cleaned = re.sub(
            r"^\s*\d*\s*Bates['\u2019]?\s*Pocket\s+Guide\s+to\s+Physical\s+Examination.*$",
            "",
            cleaned,
            flags=re.MULTILINE,
        )

        # Strip <header>C H A P T E R N</header> lines
        cleaned = RE_SPACED_CHAPTER.sub("", cleaned)

        # Strip standalone "CHAPTER" lines (artefact from chapter opener pages)
        cleaned = re.sub(r"^\s*CHAPTER\s*$", "", cleaned, flags=re.MULTILINE)

        # Strip remaining <header>...</header> tags (running headers wrapped in header tags)
        cleaned = RE_HEADER_TAG.sub("", cleaned)

        # Strip page separator lines (---)
        cleaned = RE_PAGE_SEPARATOR.sub("", cleaned)

        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        blocks.append(
            make_raw_page_block(
                pdf_page=pdf_page,
                book_page_start=book_page_start,
                book_page_end=book_page_end,
                raw_cleaned=cleaned,
            )
        )

    # Interpolate missing book page numbers from neighbors.
    # For samples, skip the offset-based derivation (page numbers are relative).
    _interpolate_book_pages(blocks, page_offset, skip_offset=is_sample)

    print(f"[pass1] Split into {len(blocks)} page blocks")
    return blocks


def _interpolate_book_pages(
    blocks: list[dict[str, Any]],
    page_offset: int,
    skip_offset: bool = False,
) -> None:
    """Fill in missing book_page_start/end from neighbors or offset.

    When skip_offset is True (sample PDFs), the pdf_page - offset derivation
    is skipped because pdf_page numbers are relative, not real PDF pages.
    """
    for i, block in enumerate(blocks):
        if block["book_page_start"] is not None:
            continue

        # Try interpolating from previous block first (most reliable)
        if i > 0 and blocks[i - 1]["book_page_start"] is not None:
            prev_end = (
                blocks[i - 1]["book_page_end"] or blocks[i - 1]["book_page_start"]
            )
            block["book_page_start"] = prev_end + 1
            block["book_page_end"] = prev_end + 1
            continue

        # Try interpolating from next block
        if i + 1 < len(blocks) and blocks[i + 1]["book_page_start"] is not None:
            next_start = blocks[i + 1]["book_page_start"]
            block["book_page_start"] = next_start - 1
            block["book_page_end"] = next_start - 1
            continue

        # Last resort: derive from pdf_page - offset (only for full book)
        if not skip_offset:
            derived = block["pdf_page"] - page_offset
            if derived > 0:
                block["book_page_start"] = derived
                block["book_page_end"] = derived


# ---------------------------------------------------------------------------
# Pass 2 — Chapter + Section Path Assignment
# ---------------------------------------------------------------------------

RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def pass2_assign_chapters(
    blocks: list[dict[str, Any]],
    toc: list[dict[str, Any]],
    is_sample: bool = False,
) -> list[dict[str, Any]]:
    """Assign chapter_num, chapter_title, and section_path to each block."""

    def find_chapter_by_book_page(book_page: int | None) -> dict[str, Any] | None:
        if book_page is None:
            return None
        for entry in toc:
            if (
                entry["book_page_start"] is not None
                and entry["book_page_start"] <= book_page <= entry["book_page_end"]
            ):
                return entry
        return None

    def find_chapter_by_pdf_page(pdf_page: int) -> dict[str, Any] | None:
        for entry in toc:
            if entry["pdf_page_start"] <= pdf_page <= entry["pdf_page_end"]:
                return entry
        return None

    def find_chapter(pdf_page: int, book_page: int | None) -> dict[str, Any] | None:
        if is_sample:
            # For samples, pdf_page numbers are relative — use book_page first
            return find_chapter_by_book_page(book_page) or find_chapter_by_pdf_page(
                pdf_page
            )
        else:
            # For full book, pdf_page is authoritative
            return find_chapter_by_pdf_page(pdf_page) or find_chapter_by_book_page(
                book_page
            )

    # Running section state across pages
    current_sections: dict[int, str] = {}  # heading_level -> heading_text

    for block in blocks:
        # Chapter assignment
        chapter = find_chapter(block["pdf_page"], block["book_page_start"])
        if chapter:
            block["chapter_num"] = chapter["chapter_num"]
            block["chapter_title"] = chapter["title"]
        else:
            # If we can't find a chapter, inherit from previous block
            print(f"[pass2] WARNING: No chapter found for pdf_page={block['pdf_page']}")

        # Section path from headings
        headings = RE_HEADING.findall(block["raw_cleaned"])
        for level_hashes, text in headings:
            level = len(level_hashes)
            # Clear all deeper sections when a higher-level heading appears
            deeper_levels = [k for k in current_sections if k >= level]
            for k in deeper_levels:
                del current_sections[k]
            current_sections[level] = text.strip()

        # Build section_path: [chapter_title, section, subsection]
        chapter_title = block["chapter_title"] or ""
        sorted_levels = sorted(current_sections.keys())

        # Skip level-1 headings (chapter titles) in section path —
        # they're already in chapter_title
        section_levels = [lv for lv in sorted_levels if lv >= 2]

        section_path = [chapter_title]
        for lv in section_levels[:2]:  # at most section + subsection
            section_path.append(current_sections[lv])

        block["section_path"] = section_path

    print(f"[pass2] Assigned chapters and sections to {len(blocks)} blocks")
    return blocks


# ---------------------------------------------------------------------------
# Pass 3 — Exam Technique Page Detection
# ---------------------------------------------------------------------------

RE_EXAM_TECHNIQUES = re.compile(
    r"^\s*(?:#{0,6}\s*)?(?:EXAMINATION\s+TECHNIQUES?|POSSIBLE\s+FINDINGS?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def pass3_detect_exam_pages(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag pages that contain the two-column exam technique layout."""
    count = 0
    for block in blocks:
        if RE_EXAM_TECHNIQUES.search(block["raw_cleaned"]):
            block["is_exam_page"] = True
            count += 1

    print(f"[pass3] Detected {count} exam technique pages")
    return blocks


# ---------------------------------------------------------------------------
# Pass 4 — Table Continuation Merger
# ---------------------------------------------------------------------------

RE_TABLE_CONTINUES = re.compile(
    r"^\s*\(table\s+continues\s+on\s+page\s+\d+\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
RE_CONTINUED_HEADER = re.compile(
    r"^\s*\*?\*?(?:Table|Figure)\s+\d+[-–]\d+\s+.*?\(continued\)\*?\*?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
RE_STANDALONE_CONTINUED = re.compile(
    r"^\s*\((?:continued|Continued)\)\s*$", re.MULTILINE
)


def pass4_merge_tables(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge tables that span multiple pages.

    Detects continuation patterns and merges table HTML across page
    boundaries. Fixes rowspan corruption on merged tables.
    """
    consumed: set[int] = set()  # indices of blocks consumed by merges
    merge_count = 0

    i = 0
    while i < len(blocks) - 1:
        if i in consumed:
            i += 1
            continue

        block = blocks[i]
        raw = block["raw_cleaned"]

        # Check if this page has a table that continues
        has_continues_marker = bool(RE_TABLE_CONTINUES.search(raw))
        has_table = "<table" in raw.lower()

        if not (has_continues_marker and has_table):
            i += 1
            continue

        # Look at next block(s) for continuation
        j = i + 1
        while j < len(blocks) and j not in consumed:
            next_block = blocks[j]
            next_raw = next_block["raw_cleaned"]

            is_continuation = (
                bool(RE_CONTINUED_HEADER.search(next_raw))
                or next_raw.strip().startswith("<table")
                or next_raw.strip().startswith("<tbody")
            )

            if not is_continuation:
                break

            # Merge tables
            merged_raw = _merge_table_html(raw, next_raw)
            if merged_raw is not None:
                raw = merged_raw
                block["raw_cleaned"] = raw
                block["book_page_end"] = (
                    next_block["book_page_end"] or block["book_page_end"]
                )
                block["merged_from_pdf_pages"].extend(
                    next_block["merged_from_pdf_pages"]
                )
                consumed.add(j)
                merge_count += 1
                print(f"[pass4] Merged table: pages {block['merged_from_pdf_pages']}")

            j += 1

        i += 1

    # Strip continuation markers from all blocks
    for block in blocks:
        block["raw_cleaned"] = RE_TABLE_CONTINUES.sub("", block["raw_cleaned"])
        block["raw_cleaned"] = RE_STANDALONE_CONTINUED.sub("", block["raw_cleaned"])
        block["raw_cleaned"] = RE_CONTINUED_HEADER.sub("", block["raw_cleaned"])
        block["raw_cleaned"] = re.sub(r"\n{3,}", "\n\n", block["raw_cleaned"]).strip()

    # Remove consumed blocks
    result = [b for i, b in enumerate(blocks) if i not in consumed]

    print(
        f"[pass4] Merged {merge_count} table continuations, "
        f"{len(blocks)} -> {len(result)} blocks"
    )
    return result


def _merge_table_html(page_a_raw: str, page_b_raw: str) -> str | None:
    """Merge table HTML from two consecutive pages.

    Takes the last <table> from page A and the first <table> from page B,
    appends B's rows into A's table. Fixes rowspan on the merged result.
    Returns the full page A raw with merged table, or None if merge fails.
    """
    # Extract last table from page A
    table_a_match = None
    for m in re.finditer(r"<table.*?</table>", page_a_raw, re.DOTALL | re.IGNORECASE):
        table_a_match = m

    if table_a_match is None:
        return None

    # Extract first table from page B
    table_b_match = re.search(
        r"<table.*?</table>", page_b_raw, re.DOTALL | re.IGNORECASE
    )
    if table_b_match is None:
        # Page B might have rows without <table> wrapper (just <tbody> or <tr>)
        # Try to extract rows directly
        rows_match = re.search(r"(<tr>.*</tr>)", page_b_raw, re.DOTALL | re.IGNORECASE)
        if rows_match is None:
            return None
        table_b_html = f"<table><tbody>{rows_match.group(1)}</tbody></table>"
    else:
        table_b_html = table_b_match.group(0)

    table_a_html = table_a_match.group(0)

    # Parse both tables
    soup_a = BeautifulSoup(table_a_html, "lxml")
    soup_b = BeautifulSoup(table_b_html, "lxml")

    table_a = soup_a.find("table")
    table_b = soup_b.find("table")

    if not table_a or not table_b:
        return None

    # Get rows from table B (skip thead if present — it's a duplicate header)
    tbody_b = table_b.find("tbody")
    if tbody_b:
        rows_to_add = tbody_b.find_all("tr")
    else:
        # Skip thead rows, take all other tr
        rows_to_add = [
            tr
            for tr in table_b.find_all("tr")
            if tr.parent and tr.parent.name != "thead"
        ]

    if not rows_to_add:
        return None

    # Find or create tbody in table A
    tbody_a = table_a.find("tbody")
    if not tbody_a:
        tbody_a = soup_a.new_tag("tbody")
        table_a.append(tbody_a)

    # Append rows
    for row in rows_to_add:
        tbody_a.append(row)

    # Fix rowspan corruption: strip rowspan from empty cells
    _fix_rowspan(table_a)

    # Replace the table in page A's raw content
    merged_table_html = str(table_a)
    merged_raw = (
        page_a_raw[: table_a_match.start()]
        + merged_table_html
        + page_a_raw[table_a_match.end() :]
    )

    # Append non-table content from page B (text after the table)
    if table_b_match:
        after_table_b = page_b_raw[table_b_match.end() :].strip()
    else:
        after_table_b = ""

    if after_table_b:
        merged_raw = merged_raw.rstrip() + "\n\n" + after_table_b

    return merged_raw


def _fix_rowspan(table: Any) -> None:
    """Strip rowspan attributes from empty cells to fix alignment corruption."""
    for td in table.find_all(["td", "th"]):
        if td.has_attr("rowspan"):
            text = td.get_text(strip=True)
            if not text:
                del td["rowspan"]


# ---------------------------------------------------------------------------
# Pass 5 — Content Type Tagger + Block Splitter
# ---------------------------------------------------------------------------

RE_TABLE_BLOCK = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
RE_IMAGE_REF = re.compile(r"!\[([^\]]+)\]\(images/[^\)]+\)", re.DOTALL)
RE_IMAGE_REF_EMPTY = re.compile(r"!\[\s*\]\(images/[^\)]+\)")
RE_PIPE_TABLE = re.compile(r"^\|.*\|.*\|", re.MULTILINE)
RE_CLINICAL_SCALE_HEADING = re.compile(
    r"(Score\s+Index|Symptom\s+Score|BPH|AUA|CURB[-\s]65|HEART\s+Score|NEXUS|Glasgow|APGAR|WHO\s+Bone)",
    re.IGNORECASE,
)
RE_CLINICAL_CRITERIA_HEADING = re.compile(
    r"(Red\s*Flag|Signs?\s+of|Risk\s+Factor|Criteria|Mnemonic|Four\s+Signs|Important\s+Terms|Bone\s+Density)",
    re.IGNORECASE,
)
RE_NUMBERED_ITEM = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)
RE_BULLET_ITEM = re.compile(r"^\s*[\*\-\u2022\u25CB\u25A0]\s+", re.MULTILINE)
RE_BOLD_DEFINITION = re.compile(r"^\s*\*\*[^*]+\*\*[\.:]?\s+", re.MULTILINE)
RE_TABLE_FOOTNOTE = re.compile(
    r"^[A-Z]{2,}[\s,].*?;\s*[A-Z]{2,}[\s,]",  # "IBD, Inflammatory...; RA, Rheumatoid..."
)
RE_FIGURE_CAPTION = re.compile(r"^\*?\*?Figure\s+\d+[-–]\d+\*?\*?", re.MULTILINE)
RE_PAGE_NUMBER_ONLY = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def pass5_tag_content(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split each block into sub-blocks and tag by content type."""
    output_blocks: list[dict[str, Any]] = []

    for block in blocks:
        raw = block["raw_cleaned"]

        # Split raw content into segments: tables, images, and text between them
        segments = _split_into_segments(raw)

        # Track the last heading seen across segments for context
        last_heading = ""
        for seg_content, seg_type_hint in segments:
            if not seg_content.strip():
                continue

            # Track headings for context in classification
            heading_match = RE_HEADING.search(seg_content)
            if heading_match:
                last_heading = heading_match.group(2)

            content_type = _classify_segment(
                seg_content,
                seg_type_hint,
                block["is_exam_page"],
                raw,
                last_heading,
            )
            confidence = _assess_confidence(content_type, block["is_exam_page"])

            output_blocks.append(
                {
                    "pdf_page": block["pdf_page"],
                    "book_page_start": block["book_page_start"],
                    "book_page_end": block["book_page_end"],
                    "chapter_num": block["chapter_num"],
                    "chapter_title": block["chapter_title"],
                    "section_path": block["section_path"],
                    "content_type": content_type,
                    "content": seg_content.strip(),
                    "confidence": confidence,
                    "merged_from_pdf_pages": block["merged_from_pdf_pages"],
                }
            )

    print(f"[pass5] Tagged {len(output_blocks)} content blocks")
    return output_blocks


def _split_into_segments(raw: str) -> list[tuple[str, str]]:
    """Split raw content into (content, type_hint) segments.

    type_hint is one of: "table", "image", "text"
    """
    segments: list[tuple[str, str]] = []
    remaining = raw

    # Find all tables and images, split text around them
    # Build a list of (start, end, type, content) for all special blocks
    special_blocks: list[tuple[int, int, str, str]] = []

    for m in RE_TABLE_BLOCK.finditer(remaining):
        special_blocks.append((m.start(), m.end(), "table", m.group(0)))

    for m in RE_IMAGE_REF.finditer(remaining):
        special_blocks.append((m.start(), m.end(), "image", m.group(0)))

    # Sort by position
    special_blocks.sort(key=lambda x: x[0])

    # Build segments
    last_end = 0
    for start, end, stype, content in special_blocks:
        # Text before this special block — split into paragraphs
        text_before = remaining[last_end:start].strip()
        if text_before:
            _add_text_paragraphs(segments, text_before)
        segments.append((content, stype))
        last_end = end

    # Remaining text after last special block
    text_after = remaining[last_end:].strip()
    if text_after:
        _add_text_paragraphs(segments, text_after)

    return segments if segments else [(raw, "text")]


def _add_text_paragraphs(segments: list[tuple[str, str]], text: str) -> None:
    """Split text at paragraph boundaries (blank lines) and add as separate segments.

    This ensures short lines like table footnotes don't get lumped with
    subsequent prose paragraphs.
    """
    paragraphs = re.split(r"\n\n+", text)
    for para in paragraphs:
        para = para.strip()
        if para:
            segments.append((para, "text"))


def _classify_segment(
    content: str,
    type_hint: str,
    is_exam_page: bool,
    full_page_raw: str,
    heading_context: str = "",
) -> str:
    """Classify a segment into a content_type string."""

    # Priority 1: exam technique pages
    if is_exam_page:
        return "exam_technique_flattened"

    # Priority 2: explicit table
    if type_hint == "table" or content.strip().startswith("<table"):
        return "table"

    # Priority 3: pipe table
    if RE_PIPE_TABLE.search(content):
        return "table"

    # Priority 4: image description
    if type_hint == "image":
        return "image_desc"

    # Priority 5: clinical scale (numbered items with scale-related heading)
    has_numbered = len(RE_NUMBERED_ITEM.findall(content)) >= 2
    searchable = content + " " + heading_context
    if has_numbered and RE_CLINICAL_SCALE_HEADING.search(searchable):
        return "clinical_scale"

    # Priority 6: clinical criteria (bulleted items with criteria-related heading)
    has_bullets = len(RE_BULLET_ITEM.findall(content)) >= 2
    if has_bullets and RE_CLINICAL_CRITERIA_HEADING.search(searchable):
        return "clinical_criteria"

    # Priority 7: definition-heavy blocks (3+ bold term: definition patterns)
    bold_defs = RE_BOLD_DEFINITION.findall(content)
    if len(bold_defs) >= 3:
        return "clinical_criteria"

    # Priority 8: table footnote (abbreviation expansion line after a table)
    if RE_TABLE_FOOTNOTE.match(content.strip()) and len(content.strip()) < 500:
        return "table_footnote"

    # Priority 9: lists
    if has_bullets or has_numbered:
        return "list"

    # Priority 10: figure captions (standalone, not with image)
    if RE_FIGURE_CAPTION.match(content.strip()) and len(content.strip()) < 200:
        return "image_desc"

    # Default: prose
    return "prose"


def _assess_confidence(content_type: str, is_exam_page: bool) -> str:
    """Assign confidence level based on content type."""
    if content_type == "exam_technique_flattened":
        return "low"
    if content_type in ("table", "clinical_scale", "clinical_criteria"):
        return "high"
    if content_type in ("image_desc", "table_footnote"):
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Pass 6 — BM25 Noise Cleanup
# ---------------------------------------------------------------------------

RE_SURVIVING_PAGE_HEADER = re.compile(r"^#{1,2}\s+Page\s+\d+\s*$", re.MULTILINE)


def pass6_cleanup(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Final cleanup pass for BM25 retrieval quality."""
    cleaned_blocks: list[dict[str, Any]] = []

    for block in blocks:
        content = block["content"]

        # Strip surviving ## Page N headers
        content = RE_SURVIVING_PAGE_HEADER.sub("", content)

        # Strip continuation markers (should already be gone from pass 4, but safety net)
        content = RE_TABLE_CONTINUES.sub("", content)
        content = RE_STANDALONE_CONTINUED.sub("", content)

        # Strip page-number-only lines (e.g., just "293" on a line)
        content = RE_PAGE_NUMBER_ONLY.sub("", content)

        # Strip any surviving <page_number> tags
        content = RE_PAGE_NUMBER_TAG.sub("", content)

        # Normalize whitespace
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        # Skip empty blocks
        if not content:
            continue

        # Skip blocks that are only whitespace or page numbers after cleanup
        if len(content) < 5:
            continue

        block["content"] = content
        cleaned_blocks.append(block)

    print(
        f"[pass6] Cleaned {len(blocks)} -> {len(cleaned_blocks)} blocks "
        f"({len(blocks) - len(cleaned_blocks)} empty blocks removed)"
    )
    return cleaned_blocks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _detect_is_sample(markdown: str, meta: dict[str, Any]) -> bool:
    """Detect if the markdown is from a sample PDF (relative page numbers)."""
    page_count = len(RE_PAGE_HEADER.findall(markdown))
    total_pdf_pages = meta.get("total_pdf_pages", 0)
    return page_count < (total_pdf_pages * 0.5) if total_pdf_pages else False


def postprocess(
    markdown: str,
    toc: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run Pass 0 pre-clean + 6 passes and return final PageBlock list."""
    markdown = _pass0_pre_clean(markdown, meta)
    is_sample = _detect_is_sample(markdown, meta)
    blocks = pass1_split_pages(markdown, meta, toc_entries=toc)
    blocks = pass2_assign_chapters(blocks, toc, is_sample=is_sample)
    blocks = pass3_detect_exam_pages(blocks)
    blocks = pass4_merge_tables(blocks)
    blocks = pass5_tag_content(blocks)
    blocks = pass6_cleanup(blocks)
    return blocks


def main():
    parser = argparse.ArgumentParser(
        description="Post-process DocStrange markdown into PageBlock JSON"
    )
    parser.add_argument("markdown", help="Path to DocStrange markdown file")
    parser.add_argument("toc", help="Path to toc.json from extract_toc.py")
    parser.add_argument("meta", help="Path to meta.json from extract_toc.py")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: <markdown_stem>.pageblocks.json)",
    )
    args = parser.parse_args()

    md_path = Path(args.markdown)
    toc_path = Path(args.toc)
    meta_path = Path(args.meta)

    if not md_path.exists():
        print(f"Error: Markdown file not found: {md_path}")
        sys.exit(1)
    if not toc_path.exists():
        print(f"Error: TOC file not found: {toc_path}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"Error: Meta file not found: {meta_path}")
        sys.exit(1)

    markdown = md_path.read_text(encoding="utf-8")
    toc = json.loads(toc_path.read_text())
    meta = json.loads(meta_path.read_text())

    print(f"[postprocess] Input: {md_path.name} ({len(markdown):,} chars)")
    print(f"[postprocess] TOC: {len(toc)} entries")
    print(f"[postprocess] Meta: page_offset={meta.get('page_offset')}")

    blocks = postprocess(markdown, toc, meta)

    # Output
    output_path = (
        Path(args.output) if args.output else md_path.with_suffix(".pageblocks.json")
    )
    output_path.write_text(
        json.dumps(blocks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary
    type_counts: dict[str, int] = {}
    for b in blocks:
        ct = b["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    print(f"\n[postprocess] Output: {output_path}")
    print(f"[postprocess] Total blocks: {len(blocks)}")
    print(f"[postprocess] Content types:")
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
