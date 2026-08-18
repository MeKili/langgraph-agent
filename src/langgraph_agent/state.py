"""The state threaded through the agent graph."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """State passed between graph nodes."""

    question: str
    steps: list[str]
    answer: str
    tool_results: list[str]
