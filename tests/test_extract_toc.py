"""Tests for pipeline/extract_toc.py — TOC parsing functions."""

from pipeline.extract_toc import (
    classify_entry,
    detect_page_offset_from_anchors,
    detect_page_offset_from_toc,
    parse_chapter_num,
)


# ── parse_chapter_num ─────────────────────────────────────────────────────


class TestParseChapterNum:
    def test_numbered_chapter(self):
        assert parse_chapter_num("Chapter 12 The Heart") == 12

    def test_leading_number(self):
        assert parse_chapter_num("15 Abdominal Pain") == 15

    def test_no_number(self):
        assert parse_chapter_num("Introduction") is None

    def test_appendix(self):
        """Appendices and non-chapter entries should return None."""
        assert parse_chapter_num("Appendix A") is None

    def test_single_digit(self):
        assert parse_chapter_num("1 The Practice of Medicine") == 1

    def test_three_digits(self):
        assert parse_chapter_num("404 Diabetes Mellitus: Management") == 404


# ── classify_entry ────────────────────────────────────────────────────────


class TestClassifyEntry:
    def test_front_matter(self):
        assert classify_entry(0, "Contents") == "front_matter"
        assert classify_entry(-5, "Preface") == "front_matter"

    def test_chapter(self):
        assert classify_entry(1, "1 Introduction") == "chapter"
        assert classify_entry(100, "15 Abdominal Pain") == "chapter"

    def test_back_matter(self):
        """Index, appendix at high page numbers should be back_matter."""
        result = classify_entry(3800, "Index")
        assert result in ("chapter", "back_matter")


# ── detect_page_offset_from_anchors ───────────────────────────────────────


class TestDetectPageOffsetFromAnchors:
    def test_basic_offset_detection(self):
        # TOC entries with PDF page numbers
        toc = [
            (1, "Chapter 1", 42),
            (1, "Chapter 2", 60),
        ]
        # Anchors: chapter title -> expected book page
        anchors = {
            "Chapter 1": 1,
            "Chapter 2": 19,
        }
        offset = detect_page_offset_from_anchors(toc, anchors)
        # PDF page 42 maps to book page 1 → offset = 41
        assert offset == 41

    def test_no_matching_anchors(self):
        toc = [(1, "Unknown Chapter", 50)]
        anchors = {"Different Chapter": 1}
        offset = detect_page_offset_from_anchors(toc, anchors)
        assert offset == 0  # fallback


# ── detect_page_offset_from_toc ───────────────────────────────────────────


class TestDetectPageOffsetFromToc:
    def test_first_numbered_chapter(self):
        toc = [
            (1, "Preface", 5),
            (1, "Contents", 8),
            (1, "1 The Practice of Medicine", 42),
            (1, "2 Evaluating Evidence", 56),
        ]
        offset = detect_page_offset_from_toc(toc)
        # First chapter "1 ..." is at PDF page 42 → offset = 41
        assert offset == 41

    def test_no_numbered_chapters(self):
        toc = [
            (1, "Preface", 5),
            (1, "Contents", 8),
        ]
        offset = detect_page_offset_from_toc(toc)
        assert offset == 0  # fallback
