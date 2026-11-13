"""
DocuMind AI – Helper Utilities Module
======================================
General utility functions for text formatting, byte sizing, and citation rendering.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def format_bytes(num_bytes: int) -> str:
    """Formats raw bytes into human-readable representation."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def clean_filename(filename: str) -> str:
    """Sanitizes uploaded filename against path traversal attacks."""
    # First extract basename only to strip any path traversal segments
    basename = Path(filename).name
    # Strip dangerous characters
    safe_name = re.sub(r"[^\w\s\.\-]", "", basename).strip()
    # Strip any leading dots to prevent hidden files
    safe_name = safe_name.lstrip(".")
    return safe_name or "document.pdf"


def deduplicate_citations(citations: List[str]) -> List[str]:
    """Preserves order while removing duplicate citation strings."""
    seen = set()
    result = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result
