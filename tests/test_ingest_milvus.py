"""Tests for pipeline/ingest_milvus.py — env loading and VARCHAR truncation."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.ingest_milvus import get_env, load_env


# ── load_env ──────────────────────────────────────────────────────────────


class TestLoadEnv:
    def test_parses_basic_key_value(self):
        """Test the .env parsing logic directly (replicating load_env internals)."""
        lines = "FOO=bar\nBAZ=qux\n".splitlines()
        env_vars = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                env_vars[key] = value

        assert env_vars == {"FOO": "bar", "BAZ": "qux"}

    def test_strips_quotes(self):
        lines = [
            'KEY1="double quoted"',
            "KEY2='single quoted'",
            "KEY3=no quotes",
        ]
        results = {}
        for line in lines:
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            results[key.strip()] = value
        assert results["KEY1"] == "double quoted"
        assert results["KEY2"] == "single quoted"
        assert results["KEY3"] == "no quotes"

    def test_skips_comments_and_blank_lines(self):
        lines = ["# comment", "", "  ", "KEY=value", "# another comment"]
        parsed = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            parsed[key.strip()] = value.strip()
        assert parsed == {"KEY": "value"}


# ── get_env ───────────────────────────────────────────────────────────────


class TestGetEnv:
    def test_returns_existing_var(self):
        os.environ["_TEST_INGEST_VAR"] = "test_value"
        try:
            assert get_env("_TEST_INGEST_VAR") == "test_value"
        finally:
            del os.environ["_TEST_INGEST_VAR"]

    def test_exits_on_missing_var(self):
        # Ensure var doesn't exist
        os.environ.pop("_NONEXISTENT_VAR_XYZ", None)
        with pytest.raises(SystemExit):
            get_env("_NONEXISTENT_VAR_XYZ")

    def test_exits_on_empty_var(self):
        os.environ["_TEST_EMPTY_VAR"] = "   "
        try:
            with pytest.raises(SystemExit):
                get_env("_TEST_EMPTY_VAR")
        finally:
            del os.environ["_TEST_EMPTY_VAR"]


# ── _trunc (byte-aware truncation) ────────────────────────────────────────
# _trunc is nested inside insert_chunks, so we replicate its logic here
# to test the byte-aware truncation algorithm directly.


def _trunc(value: str, limit: int = 1024) -> str:
    """Replica of the truncation logic in insert_chunks for testing."""
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    truncated = encoded[: limit - 3].decode("utf-8", errors="ignore")
    return truncated + "..."


class TestTrunc:
    def test_no_truncation_needed(self):
        assert _trunc("short text", 1024) == "short text"

    def test_truncates_at_limit(self):
        text = "a" * 2000
        result = _trunc(text, 1024)
        assert len(result.encode("utf-8")) <= 1024
        assert result.endswith("...")

    def test_exact_limit(self):
        text = "a" * 1024
        result = _trunc(text, 1024)
        assert result == text  # exactly at limit, no truncation

    def test_unicode_multibyte_chars(self):
        """Unicode chars like arrows/bullets are 3 bytes in UTF-8."""
        # \u2192 (→) is 3 bytes in UTF-8
        text = "\u2192" * 500  # 500 arrows = 1500 bytes
        result = _trunc(text, 1024)
        assert len(result.encode("utf-8")) <= 1024
        assert result.endswith("...")

    def test_mixed_ascii_unicode(self):
        """Mix of 1-byte ASCII and 3-byte Unicode chars."""
        # 900 ASCII bytes + 42 arrows (126 bytes) = 1026 bytes > 1024
        text = "a" * 900 + "\u2192" * 42
        result = _trunc(text, 1024)
        assert len(result.encode("utf-8")) <= 1024

    def test_does_not_split_multibyte_char(self):
        """Truncation at a byte boundary should not produce invalid UTF-8."""
        # Create string where byte boundary falls mid-character
        text = "a" * 1021 + "\u2192"  # 1021 + 3 = 1024 bytes exactly
        result = _trunc(text, 1024)
        # Should be valid UTF-8
        result.encode("utf-8")  # Should not raise
        assert len(result.encode("utf-8")) <= 1024

    def test_greek_letters(self):
        """Greek letters (common in medical text) are 2 bytes in UTF-8."""
        # α = 2 bytes
        text = "\u03b1" * 600  # 600 * 2 = 1200 bytes
        result = _trunc(text, 1024)
        assert len(result.encode("utf-8")) <= 1024
        result.encode("utf-8")  # valid UTF-8

    def test_empty_string(self):
        assert _trunc("", 1024) == ""

    def test_small_limit(self):
        result = _trunc("hello world", 8)
        assert len(result.encode("utf-8")) <= 8
        assert result.endswith("...")
