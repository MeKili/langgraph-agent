"""Tests for the agent graph (runs the compiled graph offline, no LLM)."""

from langgraph_agent.graph import build_graph
from langgraph_agent.llm import FakeLLM
from langgraph_agent.state import AgentState
from langgraph_agent.tools import select_tool, select_tools


def test_graph_routes_to_tool_for_long_question() -> None:
    initial: AgentState = {
        "question": "what is the meaning of life",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph().invoke(initial)

    assert result["question"] == "what is the meaning of life"
    assert "tool: count_words" in result["steps"][-1]
    assert len(result["tool_results"]) == 1
    assert result["answer"]


def test_graph_routes_to_respond_for_short_question() -> None:
    initial: AgentState = {
        "question": "hi",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph().invoke(initial)

    assert result["question"] == "hi"
    assert len(result["tool_results"]) == 0
    assert result["answer"]


def test_graph_uses_fake_llm() -> None:
    llm = FakeLLM(response="Strategic plan for the question")
    initial: AgentState = {
        "question": "what should I do",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph(llm).invoke(initial)

    assert any("Strategic plan" in step for step in result["steps"])
    assert result["answer"]


def test_tool_selection_for_length_query() -> None:
    assert select_tool("what is the length of this") == "get_length"


def test_tool_selection_for_uppercase_query() -> None:
    assert select_tool("can you make this upper") == "uppercase"


def test_tool_selection_defaults_to_count_words() -> None:
    assert select_tool("how many words here") == "count_words"


def test_graph_uses_selected_tool() -> None:
    initial: AgentState = {
        "question": "what is the length of this sentence",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph().invoke(initial)

    assert any("get_length" in step for step in result["steps"])
    assert len(result["tool_results"]) == 1


def test_select_tools_returns_multiple_tools() -> None:
    tools = select_tools("what is the length and uppercase version")
    assert "get_length" in tools
    assert "uppercase" in tools


def test_select_tools_defaults_to_count_words() -> None:
    tools = select_tools("random question")
    assert "count_words" in tools


def test_graph_executes_multiple_tools() -> None:
    initial: AgentState = {
        "question": "what is the length and can you uppercase this",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph().invoke(initial)

    assert any("get_length" in step for step in result["steps"])
    assert any("uppercase" in step for step in result["steps"])
    assert len(result["tool_results"]) >= 2
    assert result["answer"]
