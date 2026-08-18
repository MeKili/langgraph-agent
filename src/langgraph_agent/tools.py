"""Pure tools for the agent (no network calls, deterministic)."""

from __future__ import annotations


def get_length(text: str) -> int:
    """Get the length of a string."""
    return len(text)


def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())
