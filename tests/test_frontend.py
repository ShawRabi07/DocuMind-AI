"""
Unit Tests for Frontend Components and Integration
===================================================
Verifies clean imports, CSS presence, session state setup, and metric displays.
"""

from pathlib import Path
import pytest

from config.settings import settings
from utils.helpers import clean_filename, deduplicate_citations, format_bytes
from utils.metrics import metrics_tracker


def test_assets_stylesheet_exists():
    """Verify that style.css exists and contains valid CSS definitions."""
    css_file = settings.base_dir / "assets" / "style.css"
    assert css_file.exists()
    content = css_file.read_text(encoding="utf-8")
    assert ".metric-card" in content
    assert ".citation-card" in content


def test_helpers_format_bytes():
    """Verify byte sizing formatter."""
    assert format_bytes(500) == "500 B"
    assert format_bytes(1024 * 500) == "500.0 KB"
    assert format_bytes(1024 * 1024 * 5) == "5.00 MB"


def test_helpers_clean_filename():
    """Verify path traversal sanitization."""
    assert clean_filename("../../../malicious.pdf") == "malicious.pdf"
    assert clean_filename("my document (v1).pdf") == "my document v1.pdf"


def test_helpers_deduplicate_citations():
    """Verify deduplication preserves order."""
    cits = ["doc1.pdf (p. 1)", "doc2.pdf (p. 3)", "doc1.pdf (p. 1)"]
    deduped = deduplicate_citations(cits)
    assert deduped == ["doc1.pdf (p. 1)", "doc2.pdf (p. 3)"]


def test_metrics_tracker_real_timing():
    """Verify real metrics accumulator calculates correct averages."""
    tracker = metrics_tracker
    tracker.clear()

    # Record 2 test queries
    tracker.record_query(
        query="Query 1",
        retrieval_ms=10.0,
        llm_ms=90.0,
        chunks_count=3,
        sources_count=1,
    )
    tracker.record_query(
        query="Query 2",
        retrieval_ms=20.0,
        llm_ms=80.0,
        chunks_count=5,
        sources_count=2,
    )

    summary = tracker.get_summary()
    assert summary["total_queries"] == 2
    assert summary["avg_retrieval_ms"] == 15.0
    assert summary["avg_llm_ms"] == 85.0
    assert summary["avg_total_ms"] == 100.0
    assert summary["avg_chunks_retrieved"] == 4.0

