"""A minimal LangGraph agent graph (a deterministic foundation).

Structure: plan -> act -> router -> (tool | respond). Each node returns a partial state update.
Conditional routing chooses between executing tools or responding based on question length.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from langgraph_agent.llm import FakeLLM, LLMBase
from langgraph_agent.state import AgentState, Message
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
    """Record selected tools and action strategy for the question."""
    selected = select_tools(state["question"])
    tools_str = ", ".join(selected) if selected else "none"
    return {"steps": [*state["steps"], f"act: identified tools [{tools_str}]"]}


def _should_use_tool(state: AgentState) -> Literal["execute_tool", "respond"]:
    """Route to tool execution if question suggests tool use (keywords or complexity)."""
    # Check for tool-related keywords
    tool_keywords = {"length", "upper", "lowercase", "word", "count", "uppercase"}
    question_lower = state["question"].lower()
    has_tool_keywords = any(kw in question_lower for kw in tool_keywords)

    if has_tool_keywords or len(state["question"]) > 15:
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


def respond(state: AgentState, llm: LLMBase) -> dict[str, str | list[Message]]:
    """Produce a final answer using the LLM, based on steps and tool results."""
    context = ""
    if state["history"]:
        context += "Conversation history:\n"
        for msg in state["history"]:
            context += f"{msg['role']}: {msg['content']}\n"
        context += "\n"
    context += f"Question: {state['question']}\n\nSteps taken:\n"
    for step in state["steps"]:
        context += f"- {step}\n"
    if state["tool_results"]:
        context += "\nTool results:\n"
        for result in state["tool_results"]:
            context += f"- {result}\n"
    answer = llm.generate(context)
    new_history = state["history"].copy()
    new_history.append({"role": "user", "content": state["question"]})
    new_history.append({"role": "assistant", "content": answer})
    return {"answer": answer, "history": new_history}


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

    def respond_node(state: AgentState) -> dict[str, str | list[Message]]:
        return respond(state, llm)

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("respond", respond_node)
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
