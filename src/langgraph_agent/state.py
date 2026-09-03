"""The state threaded through the agent graph."""

from __future__ import annotations

from typing import TypedDict


class Message(TypedDict):
    """A message in the conversation history."""

    role: str
    content: str


class AgentState(TypedDict):
    """State passed between graph nodes."""

    question: str
    steps: list[str]
    answer: str
    tool_results: list[str]
    history: list[Message]
