"""A minimal LangGraph agent graph (a deterministic foundation).

Structure: plan -> act -> respond. Each node returns a partial state update.
LLM-backed nodes, tools and conditional routing are layered on top of this.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_agent.state import AgentState


def plan(state: AgentState) -> dict[str, list[str]]:
    """Record a planning step."""
    return {"steps": [*state["steps"], f"plan: understand {state['question']!r}"]}


def act(state: AgentState) -> dict[str, list[str]]:
    """Record an action step."""
    return {"steps": [*state["steps"], "act: gather what is needed"]}


def respond(state: AgentState) -> dict[str, str]:
    """Produce a final answer from the accumulated steps."""
    return {"answer": f"Handled {state['question']!r} in {len(state['steps'])} steps."}


def build_graph() -> Any:
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("respond", respond)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
