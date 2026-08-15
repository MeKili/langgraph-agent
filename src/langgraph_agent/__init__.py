"""langgraph-agent — a stateful agent built on LangGraph.

The graph threads an ``AgentState`` through a plan -> act -> respond pipeline
(see ``graph``). This deterministic foundation is where LLM-backed nodes, tools
and conditional routing are added.
"""

__version__ = "0.1.0"
