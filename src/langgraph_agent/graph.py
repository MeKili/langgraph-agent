"""A minimal LangGraph agent graph (a deterministic foundation).

Structure: plan -> act -> router -> (tool | respond). Each node returns a partial state update.
Conditional routing chooses between executing tools or responding based on question length.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from langgraph_agent.state import AgentState
from langgraph_agent.tools import count_words


def plan(state: AgentState) -> dict[str, list[str]]:
    """Record a planning step."""
    return {"steps": [*state["steps"], f"plan: understand {state['question']!r}"]}


def act(state: AgentState) -> dict[str, list[str]]:
    """Record an action step."""
    return {"steps": [*state["steps"], "act: gather what is needed"]}


def _should_use_tool(state: AgentState) -> Literal["execute_tool", "respond"]:
    """Route to tool execution or responding based on question complexity."""
    if len(state["question"]) > 10:
        return "execute_tool"
    return "respond"


def execute_tool(state: AgentState) -> dict[str, list[str]]:
    """Execute a tool (count words in the question)."""
    word_count = count_words(state["question"])
    result = f"tool: count_words returned {word_count}"
    return {
        "steps": [*state["steps"], result],
        "tool_results": [*state["tool_results"], result],
    }


def respond(state: AgentState) -> dict[str, str]:
    """Produce a final answer from the accumulated steps."""
    return {"answer": f"Handled {state['question']!r} in {len(state['steps'])} steps."}


def build_graph() -> Any:
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges(
        "act",
        _should_use_tool,
        {"execute_tool": "execute_tool", "respond": "respond"},
    )
    graph.add_edge("execute_tool", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
