"""Tests for the agent graph (runs the compiled graph offline, no LLM)."""

from langgraph_agent.graph import build_graph
from langgraph_agent.llm import FakeLLM
from langgraph_agent.state import AgentState


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
