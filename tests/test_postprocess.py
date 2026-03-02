"""Tests for pipeline/postprocess_docstrange.py — text processing passes."""

import re

from pipeline.postprocess_docstrange import (
    _assess_confidence,
    _detect_is_sample,
    _pass0_pre_clean,
    make_raw_page_block,
    pass1_split_pages,
)


# ── make_raw_page_block ───────────────────────────────────────────────────


class TestMakeRawPageBlock:
    def test_basic_creation(self):
        block = make_raw_page_block(42, 1, 1, "Some content here.")
        assert block["pdf_page"] == 42
        assert block["book_page_start"] == 1
        assert block["book_page_end"] == 1
        assert block["raw_cleaned"] == "Some content here."
        assert block["chapter_num"] is None
        assert block["chapter_title"] is None
        assert block["section_path"] == []

    def test_defaults(self):
        block = make_raw_page_block(1, 0, 0, "")
        assert block["is_exam_page"] is False
        assert block["merged_from_pdf_pages"] == [1]


# ── _pass0_pre_clean ──────────────────────────────────────────────────────


class TestPass0PreClean:
    def test_strips_prompt_leaks(self):
        md = "## Page 1\nContent\nSTRICT RULES:\n1. Extract text exactly.\n---\nMore content."
        meta = {
            "page_offset": 0,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        result = _pass0_pre_clean(md, meta)
        assert "STRICT RULES" not in result

    def test_normalizes_h1_to_h2(self):
        """# Page N should become ## Page N."""
        md = "# Page 1\nContent\n# Page 2\nMore"
        meta = {
            "page_offset": 0,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        result = _pass0_pre_clean(md, meta)
        assert "## Page 1" in result
        assert "## Page 2" in result
        # Should not have # Page (h1)
        lines = result.split("\n")
        for line in lines:
            if line.startswith("# Page "):
                assert False, f"H1 page marker not normalized: {line}"

    def test_preserves_content(self):
        md = "## Page 1\nImportant medical content about cardiac anatomy."
        meta = {
            "page_offset": 0,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        result = _pass0_pre_clean(md, meta)
        assert "cardiac anatomy" in result


# ── pass1_split_pages ─────────────────────────────────────────────────────


class TestPass1SplitPages:
    def test_splits_at_page_markers(self):
        md = "## Page 1\nFirst page.\n## Page 2\nSecond page."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        assert len(blocks) >= 2

    def test_extracts_content(self):
        md = "## Page 1\nCardiac anatomy overview.\n## Page 2\nPulmonary function."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "Cardiac anatomy" in all_content
        assert "Pulmonary function" in all_content

    def test_strips_harrison_hpim_footers(self):
        md = (
            "## Page 1\n"
            "Content here.\n"
            "HPIM21e_Part4_p481-p940.indd 653\n"
            "## Page 2\nMore content."
        )
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "harrisons21st",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "HPIM21e" not in all_content

    def test_strips_harrison_timestamps(self):
        md = "## Page 1\nContent.\n20/01/22 3:24 PM\n## Page 2\nMore."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "harrisons21st",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "20/01/22" not in all_content

    def test_strips_afkebooks_watermark(self):
        md = "## Page 1\nContent.\nAFKEBOOKS SINCE 2013\n## Page 2\nMore."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "harrisons21st",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "AFKEBOOKS" not in all_content

    def test_strips_bare_part_headers(self):
        md = "## Page 1\nContent.\nPART 6\n## Page 2\nMore."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "harrisons21st",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        # "PART 6" as a bare line should be stripped
        assert "PART 6" not in all_content

    def test_strips_bare_chapter_headers(self):
        md = "## Page 1\nContent.\nCHAPTER 202\n## Page 2\nMore."
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "harrisons21st",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "CHAPTER 202" not in all_content

    def test_strips_page_number_tags(self):
        md = "## Page 1\nContent.\n<page_number>653</page_number>\n"
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "<page_number>" not in all_content

    def test_strips_non_numeric_page_number_tags(self):
        """<page_number> tags may contain timestamps or other non-numeric content."""
        md = "## Page 1\nContent.\n<page_number>HPIM21e Part4</page_number>\n"
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        blocks = pass1_split_pages(md, meta, toc_entries=None)
        all_content = " ".join(b["raw_cleaned"] for b in blocks)
        assert "<page_number>" not in all_content


# ── _detect_is_sample ─────────────────────────────────────────────────────


class TestDetectIsSample:
    def test_full_book(self):
        md = "\n".join(f"## Page {i}\nContent" for i in range(1, 101))
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 100,
            "content_page_start": 1,
            "content_page_end": 100,
            "book": "test",
        }
        assert _detect_is_sample(md, meta) is False

    def test_sample_few_pages(self):
        md = "\n".join(f"## Page {i}\nContent" for i in range(1, 6))
        meta = {
            "page_offset": 0,
            "total_pdf_pages": 500,
            "content_page_start": 1,
            "content_page_end": 500,
            "book": "test",
        }
        assert _detect_is_sample(md, meta) is True


# ── _assess_confidence ────────────────────────────────────────────────────


class TestAssessConfidence:
    def test_prose_is_high(self):
        assert _assess_confidence("prose", False) == "high"

    def test_table_is_high(self):
        assert _assess_confidence("table", False) == "high"

    def test_exam_technique(self):
        result = _assess_confidence("exam_technique", True)
        assert result in ("high", "medium")
