"""Pure tools for the agent (no network calls, deterministic)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDef:
    """Immutable definition of a tool with name, description, and callable."""

    name: str
    description: str
    func: Callable[[str], int | str]


TOOLS_REGISTRY: dict[str, ToolDef] = {}  # Global registry of available tools


def get_length(text: str) -> int:
    """Return character count of text."""
    return len(text)


def uppercase(text: str) -> str:
    """Return text in uppercase."""
    return text.upper()


def lowercase(text: str) -> str:
    """Return text in lowercase."""
    return text.lower()


def count_words(text: str) -> int:
    """Return number of space-separated words."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Return number of sentences (split on .!? punctuation)."""
    import re

    sentences = re.split(r"[.!?]+", text)
    return len([s.strip() for s in sentences if s.strip()])


def register_tools() -> None:
    """Populate TOOLS_REGISTRY with all available tools."""
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
    """Return the single best tool for the question (legacy, use select_tools instead)."""
    if "length" in question.lower():
        return "get_length"
    if "upper" in question.lower():
        return "uppercase"
    return "count_words"


def get_available_tools() -> list[ToolDef]:
    """Return all registered ToolDef objects."""
    return list(TOOLS_REGISTRY.values())


def select_tools(question: str) -> list[str]:
    """Return tool names that match keywords in the question."""
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

    if not tools and len(question) > 15:
        tools.append("count_words")

    return tools


def format_tools() -> str:
    """Return a formatted string of available tools for inclusion in prompts."""
    if not TOOLS_REGISTRY:
        return "No tools available."
    lines = ["Available tools:"]
    for tool in TOOLS_REGISTRY.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
