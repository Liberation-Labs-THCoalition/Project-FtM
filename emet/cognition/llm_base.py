"""LLM client abstraction layer.

Defines the provider-agnostic interface that all LLM backends implement.
Emet supports four providers in cascading fallback order:

    1. Ollama       — local models via native API, default, zero cost, full privacy
    2. OpenAI-compat — any /v1/chat/completions endpoint (vLLM, llama-server, etc.)
    3. Anthropic    — Claude API, used when local models unavailable or insufficient
    4. Stub         — deterministic canned responses for testing

The active provider is selected by ``LLM_PROVIDER`` in settings and can
auto-fallback through the chain when ``LLM_FALLBACK_ENABLED`` is true.
"""

from __future__ import annotations

import abc
import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class LLMProvider(str, enum.Enum):
    """Supported LLM provider backends."""

    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI_COMPAT = "openai_compat"
    STUB = "stub"


@dataclass
class LLMResponse:
    """Response from any LLM provider."""

    text: str
    model: str
    provider: LLMProvider
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Raised when an LLM call fails irrecoverably."""


class LLMUnavailableError(LLMError):
    """Raised when a provider is unreachable — triggers fallback."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMClient(abc.ABC):
    """Provider-agnostic LLM client interface.

    Every concrete provider (Ollama, Anthropic, Stub) implements these
    four methods with identical signatures so callers never know which
    backend is active.
    """

    @property
    @abc.abstractmethod
    def provider(self) -> LLMProvider:
        """Which provider this client connects to."""

    @abc.abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
        tier: str = "balanced",
    ) -> LLMResponse:
        """Generate a completion.

        Parameters
        ----------
        prompt:
            The user message / prompt text.
        system:
            Optional system prompt.
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature (0–1).
        stop_sequences:
            Optional stop sequences.
        tier:
            Model capability tier (``fast`` / ``balanced`` / ``powerful``).
            Providers map this to a concrete model.
        """

    @abc.abstractmethod
    async def classify_intent(
        self,
        message: str,
        domains: list[str],
    ) -> tuple[str, float]:
        """Classify a message into one of the given domains.

        Returns ``(domain, confidence)`` where confidence is 0–1.
        """

    @abc.abstractmethod
    async def generate_content(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        tier: str = "balanced",
        system: str | None = None,
    ) -> str:
        """Generate content with optional context injection."""

    @abc.abstractmethod
    async def extract_entities(
        self,
        text: str,
        entity_schema: dict[str, str],
    ) -> dict[str, Any]:
        """Extract structured entities from text.

        Parameters
        ----------
        text:
            Source text to extract from.
        entity_schema:
            ``{entity_name: description}`` dict describing what to extract.

        Returns
        -------
        Dict of extracted entities (keys from *entity_schema*, values
        extracted or ``None`` if not found).
        """

    async def health_check(self) -> bool:
        """Return True if the provider is reachable and ready."""
        try:
            resp = await self.complete(
                "Respond with exactly: OK",
                max_tokens=10,
                temperature=0.0,
                tier="fast",
            )
            return bool(resp.text.strip())
        except Exception:
            return False
