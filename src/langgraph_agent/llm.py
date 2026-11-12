"""LLM interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBase(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response text from a prompt."""
        pass


class FakeLLM(LLMBase):
    """Deterministic LLM that always returns a fixed response.

    Used for testing to ensure graph behavior is reproducible and offline.
    """

    def __init__(self, response: str = "This is a generated response.") -> None:
        """Initialize with a fixed response to return on every call."""
        self.response = response

    def generate(self, prompt: str) -> str:
        """Return the fixed response, ignoring the prompt."""
        return self.response
