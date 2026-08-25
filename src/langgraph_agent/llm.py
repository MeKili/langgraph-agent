"""LLM interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Base interface for LLM implementations."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response to a prompt."""
        pass


class FakeLLM(LLMBase):
    """Deterministic fake LLM for testing (no network calls)."""

    def __init__(self, response: str = "This is a generated response.") -> None:
        """Initialize with a fixed response."""
        self.response = response

    def generate(self, prompt: str) -> str:
        """Return the fixed response, ignoring the prompt."""
        return self.response
