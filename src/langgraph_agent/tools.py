"""Pure tools for the agent (no network calls, deterministic)."""

from __future__ import annotations

from collections.abc import Callable

TOOLS_REGISTRY: dict[str, Callable[[str], int | str]] = {}


def get_length(text: str) -> int:
    """Get the length of a string."""
    return len(text)


def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def register_tools() -> None:
    """Register available tools in the global registry."""
    TOOLS_REGISTRY["get_length"] = get_length
    TOOLS_REGISTRY["uppercase"] = uppercase
    TOOLS_REGISTRY["count_words"] = count_words


def select_tool(question: str) -> str:
    """Select a tool based on the question."""
    if "length" in question.lower():
        return "get_length"
    if "upper" in question.lower():
        return "uppercase"
    return "count_words"
