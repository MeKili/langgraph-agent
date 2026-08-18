"""Tests for the agent graph (runs the compiled graph offline, no LLM)."""

from langgraph_agent.graph import build_graph
from langgraph_agent.state import AgentState


def test_graph_runs_plan_act_tool_respond() -> None:
    initial: AgentState = {
        "question": "hello world",
        "steps": [],
        "answer": "",
        "tool_results": [],
    }
    result = build_graph().invoke(initial)

    assert result["question"] == "hello world"
    assert len(result["steps"]) == 3
    assert len(result["tool_results"]) == 1
    assert "count_words returned 2" in result["tool_results"][0]
    assert result["answer"]
