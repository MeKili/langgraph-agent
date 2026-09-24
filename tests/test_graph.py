"""Tests for the agent graph (runs the compiled graph offline, no LLM)."""

from langgraph_agent.graph import build_graph
from langgraph_agent.llm import FakeLLM
from langgraph_agent.state import AgentState
from langgraph_agent.tools import select_tool, select_tools


def test_graph_routes_to_tool_for_long_question() -> None:
    """Test that graph routes long questions to tool execution."""
    initial: AgentState = {
        "question": "what is the meaning of life",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert result["question"] == "what is the meaning of life"
    assert "tool: count_words" in result["steps"][-1]
    assert len(result["tool_results"]) == 1
    assert result["answer"]


def test_graph_routes_to_respond_for_short_question() -> None:
    """Test that graph routes short questions to respond without tools."""
    initial: AgentState = {
        "question": "hi",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert result["question"] == "hi"
    assert len(result["tool_results"]) == 0
    assert result["answer"]


def test_graph_uses_fake_llm() -> None:
    """Test that graph uses provided FakeLLM for deterministic planning."""
    llm = FakeLLM(response="Strategic plan for the question")
    initial: AgentState = {
        "question": "what should I do",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph(llm).invoke(initial)

    assert any("Strategic plan" in step for step in result["steps"])
    assert result["answer"]


def test_tool_selection_for_length_query() -> None:
    """Test that length queries select get_length tool."""
    assert select_tool("what is the length of this") == "get_length"


def test_tool_selection_for_uppercase_query() -> None:
    """Test that uppercase queries select uppercase tool."""
    assert select_tool("can you make this upper") == "uppercase"


def test_tool_selection_defaults_to_count_words() -> None:
    """Test that unknown queries default to count_words tool."""
    assert select_tool("how many words here") == "count_words"


def test_graph_uses_selected_tool() -> None:
    """Test that graph executes the tool selected for a question."""
    initial: AgentState = {
        "question": "what is the length of this sentence",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert any("get_length" in step for step in result["steps"])
    assert len(result["tool_results"]) == 1


def test_select_tools_returns_multiple_tools() -> None:
    """Test that select_tools identifies multiple tools from a single question."""
    tools = select_tools("what is the length and uppercase version")
    assert "get_length" in tools
    assert "uppercase" in tools


def test_select_tools_defaults_to_count_words() -> None:
    """Test that select_tools defaults to count_words when no keywords match."""
    tools = select_tools("random question")
    assert "count_words" in tools


def test_graph_executes_multiple_tools() -> None:
    """Test that graph executes all tools identified for a question."""
    initial: AgentState = {
        "question": "what is the length and can you uppercase this",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert any("get_length" in step for step in result["steps"])
    assert any("uppercase" in step for step in result["steps"])
    assert len(result["tool_results"]) >= 2
    assert result["answer"]


def test_graph_preserves_conversation_history() -> None:
    """Test that graph preserves and extends conversation history across invocations."""
    initial: AgentState = {
        "question": "count words",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    }
    result = build_graph().invoke(initial)

    assert len(result["history"]) == 4
    assert result["history"][0]["role"] == "user"
    assert result["history"][0]["content"] == "hello"
    assert result["history"][1]["role"] == "assistant"
    assert result["history"][1]["content"] == "Hi there!"
    assert result["history"][2]["role"] == "user"
    assert result["history"][2]["content"] == "count words"
    assert result["history"][3]["role"] == "assistant"
    assert result["answer"]


def test_graph_routes_to_tool_for_keyword_match() -> None:
    """Test that graph routes to tools when keywords are detected in question."""
    initial: AgentState = {
        "question": "uppercase",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert "uppercase" in result["steps"][-1]
    assert len(result["tool_results"]) >= 1


def test_graph_routes_to_respond_for_short_simple_question() -> None:
    """Test that very short simple questions route directly to respond."""
    initial: AgentState = {
        "question": "ok",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    assert len(result["tool_results"]) == 0
    assert result["answer"]


def test_act_node_records_selected_tools() -> None:
    """Test that act node identifies and records the selected tools."""
    initial: AgentState = {
        "question": "what is the length and uppercase version of this text",
        "steps": [],
        "answer": "",
        "tool_results": [],
        "history": [],
    }
    result = build_graph().invoke(initial)

    act_step = [s for s in result["steps"] if s.startswith("act:")][0]
    assert "get_length" in act_step
    assert "uppercase" in act_step
