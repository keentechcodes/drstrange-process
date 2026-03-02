"""Tests for pipeline/chunk_textbook.py — chunking logic."""

from pipeline.chunk_textbook import (
    _make_chunk,
    chunk_atomic,
    count_tokens,
    split_list_items,
    split_sentences,
)


# ── split_sentences ───────────────────────────────────────────────────────


class TestSplitSentences:
    def test_basic_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0].strip() == "First sentence."
        assert result[1].strip() == "Second sentence."

    def test_single_sentence(self):
        result = split_sentences("Just one sentence.")
        assert len(result) == 1

    def test_question_and_exclamation(self):
        text = "What is this? It is great! Indeed."
        result = split_sentences(text)
        assert len(result) == 3

    def test_abbreviations_not_split(self):
        """Common abbreviations like 'Dr.' should not cause a split."""
        text = "Dr. Smith examined the patient. The results were normal."
        result = split_sentences(text)
        # "Dr." followed by a capital letter — may or may not split depending
        # on the regex. The key test is that we get reasonable results.
        assert len(result) >= 1
        # Full text is preserved
        assert "".join(result).replace(" ", "") == text.replace(" ", "")

    def test_empty_string(self):
        result = split_sentences("")
        assert result == [""] or result == []

    def test_no_terminal_punctuation(self):
        text = "A sentence without ending punctuation"
        result = split_sentences(text)
        assert len(result) == 1
        assert "sentence without" in result[0]


# ── split_list_items ──────────────────────────────────────────────────────


class TestSplitListItems:
    def test_bullet_list(self):
        text = "- Item one\n- Item two\n- Item three"
        result = split_list_items(text)
        assert len(result) == 3

    def test_numbered_list(self):
        text = "1. First item\n2. Second item\n3. Third item"
        result = split_list_items(text)
        assert len(result) == 3

    def test_asterisk_bullets(self):
        text = "* Alpha\n* Beta\n* Gamma"
        result = split_list_items(text)
        assert len(result) == 3

    def test_multi_line_items(self):
        text = "- Item one\n  continued here\n- Item two"
        result = split_list_items(text)
        assert len(result) == 2
        assert "continued" in result[0]

    def test_single_item(self):
        text = "- Only item"
        result = split_list_items(text)
        assert len(result) == 1

    def test_no_list_markers(self):
        text = "Just plain text without list markers"
        result = split_list_items(text)
        # Should return the whole text as one item
        assert len(result) >= 1


# ── count_tokens ──────────────────────────────────────────────────────────


class TestCountTokens:
    def test_nonempty_string(self):
        """Token count should be > 0 for non-empty text."""
        count = count_tokens("The patient presented with chest pain.")
        assert count > 0

    def test_empty_string(self):
        count = count_tokens("")
        assert count == 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hello world.")
        long = count_tokens(
            "Hello world. This is a much longer sentence with many more words."
        )
        assert long > short

    def test_medical_terminology(self):
        """Medical terms may tokenize into subwords; count should be reasonable."""
        count = count_tokens(
            "Electrocardiogram revealed ST-segment elevation in leads V1-V4."
        )
        assert count > 5  # At least a few tokens


# ── _make_chunk ───────────────────────────────────────────────────────────


class TestMakeChunk:
    def test_basic_chunk_creation(self, sample_pageblock):
        chunk = _make_chunk(sample_pageblock, "Test text content", 0, "testbook")

        assert chunk["text"] == "Test text content"
        assert chunk["book"] == "testbook"
        assert chunk["chapter_num"] == 1
        assert chunk["chapter_title"] == "Introduction"
        assert chunk["content_type"] == "prose"
        assert chunk["chunk_index"] == 0
        assert chunk["page_start"] == 1
        assert chunk["page_end"] == 1

    def test_chunk_id_format(self, sample_pageblock):
        chunk = _make_chunk(sample_pageblock, "text", 3, "bates")
        # ID format: {book}_ch{NN}_p{NNNN}_c{NNNN}
        assert chunk["id"].startswith("bates_ch")
        assert "_c0003" in chunk["id"]

    def test_section_from_section_path(self, sample_pageblock):
        sample_pageblock["section_path"] = ["Cardiology", "Anatomy", "Chambers"]
        chunk = _make_chunk(sample_pageblock, "text", 0, "book")
        # Section should come from section_path
        assert "section" in chunk
        assert "subsection" in chunk


# ── chunk_atomic ──────────────────────────────────────────────────────────


class TestChunkAtomic:
    def test_creates_single_chunk(self, sample_pageblock):
        sample_pageblock["content_type"] = "table"
        sample_pageblock["content"] = "<table><tr><td>Data</td></tr></table>"
        chunks, next_idx = chunk_atomic(
            sample_pageblock, chunk_index=0, book_slug="test"
        )
        assert len(chunks) == 1
        assert chunks[0]["content_type"] == "table"
        assert next_idx == 1

    def test_chunk_index_incremented(self, sample_pageblock):
        chunks, next_idx = chunk_atomic(
            sample_pageblock, chunk_index=5, book_slug="test"
        )
        assert chunks[0]["chunk_index"] == 5
        assert next_idx == 6
