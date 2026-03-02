"""Shared fixtures for drstrange-process tests."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `pipeline.*` modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_pageblock():
    """A minimal PageBlock dict for chunker tests."""
    return {
        "pdf_page": 42,
        "book_page_start": 1,
        "book_page_end": 1,
        "chapter_num": 1,
        "chapter_title": "Introduction",
        "section_path": ["Introduction", "Overview"],
        "content_type": "prose",
        "content": "This is a sample paragraph about cardiac anatomy.",
        "confidence": "high",
        "is_exam_page": False,
    }


@pytest.fixture
def sample_toc():
    """A minimal TOC list for postprocessor tests."""
    return [
        {
            "title": "1 The Practice of Medicine",
            "pdf_page_start": 42,
            "pdf_page_end": 55,
            "book_page_start": 1,
            "book_page_end": 14,
            "chapter_num": 1,
            "level": 1,
            "type": "chapter",
        },
        {
            "title": "2 Evaluating Clinical Evidence",
            "pdf_page_start": 56,
            "pdf_page_end": 70,
            "book_page_start": 15,
            "book_page_end": 29,
            "chapter_num": 2,
            "level": 1,
            "type": "chapter",
        },
    ]


@pytest.fixture
def sample_meta():
    """A minimal meta dict."""
    return {
        "page_offset": 41,
        "total_pdf_pages": 3897,
        "content_page_start": 42,
        "content_page_end": 3897,
        "book_name": "harrisons21st",
    }
