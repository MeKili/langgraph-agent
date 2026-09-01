"""A minimal LangGraph agent graph (a deterministic foundation).

Structure: plan -> act -> router -> (tool | respond). Each node returns a partial state update.
Conditional routing chooses between executing tools or responding based on question length.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from langgraph_agent.llm import FakeLLM, LLMBase
from langgraph_agent.state import AgentState
from langgraph_agent.tools import (
    TOOLS_REGISTRY,
    register_tools,
    select_tools,
)


def plan(state: AgentState, llm: LLMBase) -> dict[str, list[str]]:
    """Record a planning step using the LLM."""
    plan_text = llm.generate(f"Plan for: {state['question']}")
    return {"steps": [*state["steps"], f"plan: {plan_text}"]}


def act(state: AgentState) -> dict[str, list[str]]:
    """Record an action step."""
    return {"steps": [*state["steps"], "act: gather what is needed"]}


def _should_use_tool(state: AgentState) -> Literal["execute_tool", "respond"]:
    """Route to tool execution or responding based on question complexity."""
    if len(state["question"]) > 10:
        return "execute_tool"
    return "respond"


def execute_tool(state: AgentState) -> dict[str, list[str]]:
    """Execute all selected tools and accumulate results."""
    tool_names = select_tools(state["question"])
    new_steps = state["steps"].copy()
    new_tool_results = state["tool_results"].copy()

    for tool_name in tool_names:
        tool = TOOLS_REGISTRY[tool_name]
        tool_output = tool(state["question"])
        result = f"tool: {tool_name} returned {tool_output}"
        new_steps.append(result)
        new_tool_results.append(result)

    return {
        "steps": new_steps,
        "tool_results": new_tool_results,
    }


def respond(state: AgentState) -> dict[str, str]:
    """Produce a final answer from the accumulated steps."""
    return {"answer": f"Handled {state['question']!r} in {len(state['steps'])} steps."}


def build_graph(llm: LLMBase | None = None) -> Any:
    """Build and compile the agent graph.

    Args:
        llm: Language model instance. Uses FakeLLM if not provided.
    """
    if llm is None:
        llm = FakeLLM()

    register_tools()

    def plan_node(state: AgentState) -> dict[str, list[str]]:
        return plan(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
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
