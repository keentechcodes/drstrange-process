"""Tests for pipeline/run_chunked_ocr.py — pure-logic functions."""

from pipeline.run_chunked_ocr import (
    add_image_prefix,
    concatenate_markdown,
    create_chunk_meta,
    renumber_chunk_pages,
)


# ── renumber_chunk_pages ──────────────────────────────────────────────────


class TestRenumberChunkPages:
    def test_no_shift_needed(self):
        """Chunk 1 at pdf_start=42, offset=41 → shift=0, no change."""
        md = "## Page 1\nContent\n## Page 2\nMore"
        result = renumber_chunk_pages(md, original_pdf_start=42, original_offset=41)
        assert "## Page 1" in result
        assert "## Page 2" in result

    def test_basic_shift(self):
        """Chunk 2 at pdf_start=542, offset=41 → shift=500."""
        md = "## Page 1\nContent\n## Page 2\nMore"
        result = renumber_chunk_pages(md, original_pdf_start=542, original_offset=41)
        assert "## Page 501" in result
        assert "## Page 502" in result

    def test_h1_page_markers(self):
        """Should renumber both # Page N and ## Page N."""
        md = "# Page 1\nContent\n## Page 2\nMore"
        result = renumber_chunk_pages(md, original_pdf_start=100, original_offset=41)
        assert "# Page 59" in result
        assert "## Page 60" in result

    def test_preserves_non_page_headings(self):
        """Should not modify headings that aren't page markers."""
        md = "## Page 1\n## Introduction\n### Overview"
        result = renumber_chunk_pages(md, original_pdf_start=100, original_offset=41)
        assert "## Introduction" in result
        assert "### Overview" in result

    def test_inline_page_not_modified(self):
        """Page references mid-line should not be renumbered."""
        md = "## Page 1\nSee Page 5 for details."
        result = renumber_chunk_pages(md, original_pdf_start=100, original_offset=41)
        assert "See Page 5 for details." in result

    def test_zero_shift(self):
        """pdf_start == offset + 1 → shift = 0."""
        md = "## Page 1\n## Page 2"
        result = renumber_chunk_pages(md, original_pdf_start=42, original_offset=41)
        assert result == md


# ── add_image_prefix ──────────────────────────────────────────────────────


class TestAddImagePrefix:
    def test_basic_prefix(self):
        md = "![alt text](images/page31_img2.png)"
        result = add_image_prefix(md, chunk_num=3)
        assert result == "![alt text](images/c003_page31_img2.png)"

    def test_multiple_images(self):
        md = "![fig1](images/page1_img1.png)\nSome text\n![fig2](images/page2_img3.jpg)"
        result = add_image_prefix(md, chunk_num=7)
        assert "images/c007_page1_img1.png" in result
        assert "images/c007_page2_img3.jpg" in result

    def test_no_images(self):
        md = "Just plain text with no images."
        result = add_image_prefix(md, chunk_num=1)
        assert result == md

    def test_empty_alt_text(self):
        md = "![](images/page5_img1.png)"
        result = add_image_prefix(md, chunk_num=12)
        assert result == "![](images/c012_page5_img1.png)"

    def test_chunk_num_padding(self):
        """Chunk numbers are zero-padded to 3 digits."""
        md = "![x](images/file.png)"
        assert "c001_" in add_image_prefix(md, 1)
        assert "c010_" in add_image_prefix(md, 10)
        assert "c100_" in add_image_prefix(md, 100)

    def test_non_image_links_unchanged(self):
        """Regular markdown links should not be affected."""
        md = "[click here](images/page1.png)"
        result = add_image_prefix(md, chunk_num=5)
        # This is a link, not an image (no !)
        assert result == md


# ── concatenate_markdown ──────────────────────────────────────────────────


class TestConcatenateMarkdown:
    def test_sorts_by_chunk_num(self):
        """Chunks should be sorted by chunk_num, not input order."""
        chunks = [
            (3, "Chunk 3 content"),
            (1, "Chunk 1 content"),
            (2, "Chunk 2 content"),
        ]
        result = concatenate_markdown(chunks)
        pos1 = result.index("Chunk 1")
        pos2 = result.index("Chunk 2")
        pos3 = result.index("Chunk 3")
        assert pos1 < pos2 < pos3

    def test_skips_none_chunks(self):
        chunks = [(1, "Content"), (2, None), (3, "More")]
        result = concatenate_markdown(chunks)
        assert "Content" in result
        assert "More" in result

    def test_empty_list(self):
        result = concatenate_markdown([])
        assert result == "\n"

    def test_single_chunk(self):
        result = concatenate_markdown([(1, "Only content")])
        assert "Only content" in result


# ── create_chunk_meta ─────────────────────────────────────────────────────


class TestCreateChunkMeta:
    def test_basic_meta(self):
        original_meta = {
            "page_offset": 41,
            "total_pdf_pages": 3897,
            "content_page_start": 42,
            "content_page_end": 3897,
            "book": "harrisons21st",
        }
        chunk = {
            "chunk_num": 2,
            "pdf_start": 542,
            "pdf_end": 1041,
            "pages": 500,
            "path": "/tmp/chunk_002.pdf",
        }
        result = create_chunk_meta(original_meta, chunk)

        assert result["page_offset"] == 0
        assert result["book"] == "harrisons21st"
        # Chunk meta uses chunk-relative pages (1-based)
        assert result["content_pdf_page_start"] == 1
        assert result["content_pdf_page_end"] == 500
        assert result["total_pdf_pages"] == 500

    def test_preserves_book_name(self):
        original_meta = {
            "page_offset": 0,
            "book": "bates",
            "total_pdf_pages": 481,
            "content_page_start": 1,
            "content_page_end": 481,
        }
        chunk = {
            "chunk_num": 1,
            "pdf_start": 1,
            "pdf_end": 481,
            "pages": 481,
            "path": "/tmp/chunk_001.pdf",
        }
        result = create_chunk_meta(original_meta, chunk)
        assert result["book"] == "bates"
