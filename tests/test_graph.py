"""Tests for the agent graph (runs the compiled graph offline, no LLM)."""

from langgraph_agent.graph import build_graph
from langgraph_agent.state import AgentState


def test_graph_runs_plan_act_respond() -> None:
    initial: AgentState = {"question": "hello", "steps": [], "answer": ""}
    result = build_graph().invoke(initial)

    assert result["question"] == "hello"
    assert len(result["steps"]) == 2
    assert result["answer"]
