"""Run the agent graph on a question from the command line."""

from __future__ import annotations

import argparse

from langgraph_agent.graph import build_graph
from langgraph_agent.state import AgentState


def run(question: str) -> AgentState:
    """Invoke the compiled graph for a single question and return the final state."""
    initial: AgentState = {"question": question, "steps": [], "answer": ""}
    result: AgentState = build_graph().invoke(initial)
    return result


def main() -> None:
    """Parse arguments, run the graph, and print the answer."""
    parser = argparse.ArgumentParser(description="Run the LangGraph agent.")
    parser.add_argument("question", help="the question to handle")
    args = parser.parse_args()
    print(run(args.question)["answer"])


if __name__ == "__main__":
    main()
