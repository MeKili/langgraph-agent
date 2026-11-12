"""The state threaded through the agent graph."""

from __future__ import annotations

from typing import TypedDict


class Message(TypedDict):
    """A single message with role and content."""

    role: str
    content: str


class AgentState(TypedDict):
    """State threaded through the agent graph.

    Fields:
        question: The user's input question.
        steps: Log of reasoning steps taken.
        answer: The final generated response.
        tool_results: Recorded outputs from tool execution.
        history: Previous conversation messages.
        turn_count: Total number of responses given.
        selected_tools: Tools identified for this question.
    """

    question: str
    steps: list[str]
    answer: str
    tool_results: list[str]
    history: list[Message]
    turn_count: int
    selected_tools: list[str]
