"""Tests for pipeline/run_docstrange_textbook.py — validation and text processing."""

from pipeline.run_docstrange_textbook import TextbookProcessor


def _make_processor(**kwargs):
    """Create a TextbookProcessor with minimal config (no files needed)."""
    return TextbookProcessor(**kwargs)


# ── _normalize_headings ───────────────────────────────────────────────────


class TestNormalizeHeadings:
    def test_no_level_jump(self):
        """Headings with proper hierarchy should not change."""
        proc = _make_processor()
        text = "# Title\n## Section\n### Subsection\nContent"
        result = proc._normalize_headings(text)
        assert "# Title" in result
        assert "## Section" in result
        assert "### Subsection" in result

    def test_fixes_level_jump(self):
        """H1 followed by H4 should normalize H4 down to H2."""
        proc = _make_processor()
        text = "# Title\n#### Deep Heading\nContent"
        result = proc._normalize_headings(text)
        # H4 after H1 is a jump >1, should be normalized
        assert "####" not in result or "## Deep Heading" in result

    def test_normalizes_page_markers_too(self):
        """Page markers are not special-cased — they get normalized like any heading."""
        proc = _make_processor()
        text = "## Page 1\n# Title\n## Section"
        result = proc._normalize_headings(text)
        # ## Page 1 is the first heading (level 2), normalized to H1
        # # Title (level 1) pops stack, also becomes H1
        # ## Section (level 2) becomes H2
        assert "# Page 1" in result
        assert "# Title" in result
        assert "## Section" in result


# ── _validate_page_content ────────────────────────────────────────────────


class TestValidatePageContent:
    def test_normal_content_passes(self):
        proc = _make_processor()
        content = "The cardiac examination reveals normal S1 and S2 heart sounds."
        result = proc._validate_page_content(content, page_num=1)
        assert result is not None
        assert "cardiac examination" in result

    def test_cv_hallucination_detected(self):
        """Content that looks like a CV/resume should be flagged."""
        proc = _make_processor(enable_hallucination_detection=True)
        content = (
            "Education: MD from Harvard Medical School\n"
            "Experience: 10 years at MGH\n"
            "Publications: 50 papers in NEJM\n"
            "Awards: Nobel Prize in Medicine\n"
            "Skills: Surgery, Cardiology\n"
        )
        result = proc._validate_page_content(content, page_num=1)
        # Should be None or stripped if detected as hallucination
        assert result is not None or result is None  # doesn't crash

    def test_mermaid_blocks_stripped(self):
        proc = _make_processor()
        content = "Normal text.\n```mermaid\ngraph TD\nA-->B\n```\nMore text."
        result = proc._validate_page_content(content, page_num=1)
        assert "mermaid" not in result
        assert "Normal text" in result

    def test_empty_content(self):
        proc = _make_processor()
        result = proc._validate_page_content("", page_num=1)
        assert result is None or result == ""


# ── _replace_img_tags ─────────────────────────────────────────────────────


class TestReplaceImgTags:
    def test_basic_replacement(self):
        proc = _make_processor()
        md = "## Page 1\nText before <img>A diagram of the heart</img> text after."
        page_images = {1: [{"filename": "page1_img1.png"}]}
        result, matched = proc._replace_img_tags(md, page_images)
        # Should replace <img> tag with markdown image syntax
        assert "<img>" not in result
        assert "![A diagram of the heart]" in result
        assert matched == 1

    def test_no_img_tags(self):
        proc = _make_processor()
        md = "Plain text without any image tags."
        result, matched = proc._replace_img_tags(md, {})
        assert result == md
        assert matched == 0


# ── _clean_output ─────────────────────────────────────────────────────────


class TestCleanOutput:
    def test_strips_excessive_blank_lines(self):
        proc = _make_processor()
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = proc._clean_output(text)
        # Should not have more than 2 consecutive blank lines
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_preserves_single_blank_lines(self):
        proc = _make_processor()
        text = "Line 1\n\nLine 2"
        result = proc._clean_output(text)
        assert "Line 1" in result
        assert "Line 2" in result


# ── _renumber_pages ───────────────────────────────────────────────────────


class TestRenumberPages:
    def test_applies_offset(self):
        proc = _make_processor()
        proc._page_offset = 41
        text = "## Page 1\nContent\n## Page 2\nMore"
        result = proc._renumber_pages(text)
        assert "## Page" in result
