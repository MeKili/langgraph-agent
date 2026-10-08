"""Pure tools for the agent (no network calls, deterministic)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDef:
    """Tool definition with metadata."""

    name: str
    description: str
    func: Callable[[str], int | str]


TOOLS_REGISTRY: dict[str, ToolDef] = {}


def get_length(text: str) -> int:
    """Get the length of a string."""
    return len(text)


def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    import re

    sentences = re.split(r"[.!?]+", text)
    return len([s.strip() for s in sentences if s.strip()])


def register_tools() -> None:
    """Register available tools in the global registry."""
    TOOLS_REGISTRY["get_length"] = ToolDef(
        name="get_length",
        description="Returns the character length of the input text",
        func=get_length,
    )
    TOOLS_REGISTRY["uppercase"] = ToolDef(
        name="uppercase",
        description="Converts the input text to uppercase",
        func=uppercase,
    )
    TOOLS_REGISTRY["lowercase"] = ToolDef(
        name="lowercase",
        description="Converts the input text to lowercase",
        func=lowercase,
    )
    TOOLS_REGISTRY["count_words"] = ToolDef(
        name="count_words",
        description="Counts the number of words in the input text",
        func=count_words,
    )
    TOOLS_REGISTRY["count_sentences"] = ToolDef(
        name="count_sentences",
        description="Counts the number of sentences in the input text",
        func=count_sentences,
    )


def select_tool(question: str) -> str:
    """Select a tool based on the question."""
    if "length" in question.lower():
        return "get_length"
    if "upper" in question.lower():
        return "uppercase"
    return "count_words"


def select_tools(question: str) -> list[str]:
    """Select multiple tools that might be needed for the question.

    Returns a list of tool names to execute in sequence.
    """
    tools: list[str] = []
    lower_q = question.lower()

    if "length" in lower_q:
        tools.append("get_length")
    if "upper" in lower_q or "uppercase" in lower_q:
        tools.append("uppercase")
    if "lower" in lower_q or "lowercase" in lower_q:
        tools.append("lowercase")
    if "sentence" in lower_q:
        tools.append("count_sentences")
    elif "word" in lower_q or "count" in lower_q:
        tools.append("count_words")

    return tools if tools else ["count_words"]
