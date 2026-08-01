"""LLM client factory with cascading provider fallback.

Reads ``LLM_PROVIDER`` from settings and instantiates the appropriate
client.  When ``LLM_FALLBACK_ENABLED`` is true (default), failures
cascade through the provider chain:

    Ollama → OpenAI-compat → Anthropic → Stub

This ensures Emet always works — degraded but functional — even when
no LLM backend is available.

Usage::

    from emet.cognition.llm_factory import create_llm_client

    client = await create_llm_client()     # auto-selects best available
    client = create_llm_client_sync()      # synchronous variant for startup
"""

from __future__ import annotations

import logging
from typing import Any

from emet.cognition.llm_base import (
    LLMClient,
    LLMProvider,
    LLMUnavailableError,
)
from emet.cognition.llm_anthropic import AnthropicClient
from emet.cognition.llm_ollama import OllamaClient
from emet.cognition.llm_openai_compat import OpenAICompatClient
from emet.cognition.llm_stub import StubClient
from emet.cognition.model_router import CostTracker, ModelRouter
from emet.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------


class FallbackLLMClient(LLMClient):
    """Wraps a primary client with automatic fallback to a chain of backups.

    When the primary client raises ``LLMUnavailableError``, the next
    client in the chain is tried.  The stub client at the end of the
    chain guarantees a response.

    This is transparent to callers — they interact with a single
    ``LLMClient`` and never know which backend actually served the
    request.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("At least one LLM client required")
        self._clients = clients
        self._active_index = 0

    @property
    def provider(self) -> LLMProvider:
        return self._clients[self._active_index].provider

    @property
    def active_client(self) -> LLMClient:
        """The currently active client (for inspection)."""
        return self._clients[self._active_index]

    async def _with_fallback(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Try method on each client in order until one succeeds."""
        last_error: Exception | None = None

        for i, client in enumerate(self._clients):
            try:
                result = await getattr(client, method)(*args, **kwargs)
                if i != self._active_index:
                    logger.info(
                        "LLM fallback: %s → %s",
                        self._clients[self._active_index].provider.value,
                        client.provider.value,
                    )
                    self._active_index = i
                return result
            except LLMUnavailableError as e:
                logger.warning(
                    "LLM provider %s unavailable: %s — trying next",
                    client.provider.value,
                    e,
                )
                last_error = e
                continue

        # Should never reach here if StubClient is in chain
        raise last_error or LLMUnavailableError("All LLM providers failed")

    async def complete(self, prompt: str, **kwargs: Any) -> Any:
        return await self._with_fallback("complete", prompt, **kwargs)

    async def classify_intent(self, message: str, domains: list[str]) -> tuple[str, float]:
        return await self._with_fallback("classify_intent", message, domains)

    async def generate_content(self, prompt: str, **kwargs: Any) -> str:
        return await self._with_fallback("generate_content", prompt, **kwargs)

    async def extract_entities(self, text: str, entity_schema: dict[str, str]) -> dict[str, Any]:
        return await self._with_fallback("extract_entities", text, entity_schema)

    async def health_check(self) -> bool:
        """Check health of all clients in chain."""
        for client in self._clients:
            if await client.health_check():
                return True
        return False

    async def provider_status(self) -> dict[str, bool]:
        """Return health status of each provider in the chain."""
        status = {}
        for client in self._clients:
            status[client.provider.value] = await client.health_check()
        return status


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _build_ollama_client() -> OllamaClient | None:
    """Try to build an Ollama client. Returns None if config is absent."""
    try:
        return OllamaClient(
            host=settings.OLLAMA_HOST,
            models=dict(settings.OLLAMA_MODELS),
            timeout=settings.OLLAMA_TIMEOUT,
        )
    except Exception as e:
        logger.debug("Cannot build Ollama client: %s", e)
        return None


def _build_anthropic_client(
    cost_tracker: CostTracker | None = None,
) -> AnthropicClient | None:
    """Try to build an Anthropic client. Returns None if no API key."""
    if not settings.ANTHROPIC_API_KEY:
        logger.debug("No ANTHROPIC_API_KEY — Anthropic client not available")
        return None
    try:
        return AnthropicClient(
            api_key=settings.ANTHROPIC_API_KEY,
            model_router=ModelRouter(deployment_tier=settings.DEPLOYMENT_TIER),
            cost_tracker=cost_tracker,
        )
    except Exception as e:
        logger.debug("Cannot build Anthropic client: %s", e)
        return None


def _build_openai_compat_client() -> OpenAICompatClient | None:
    """Try to build an OpenAI-compatible client. Returns None if config is absent."""
    try:
        return OpenAICompatClient(
            base_url=settings.OPENAI_COMPAT_BASE_URL,
            api_key=settings.OPENAI_COMPAT_API_KEY,
            models=dict(settings.OPENAI_COMPAT_MODELS),
            timeout=settings.OPENAI_COMPAT_TIMEOUT,
        )
    except Exception as e:
        logger.debug("Cannot build OpenAI-compat client: %s", e)
        return None


def _build_stub_client() -> StubClient:
    """Stub client is always available."""
    return StubClient()


def create_llm_client_sync(
    provider: str | None = None,
    cost_tracker: CostTracker | None = None,
) -> LLMClient:
    """Create an LLM client synchronously (for application startup).

    Builds the provider chain based on config but does NOT health-check
    providers (that would require async).  The fallback wrapper will
    handle unavailable providers at call time.

    Parameters
    ----------
    provider:
        Explicit provider name (``"ollama"``, ``"anthropic"``, ``"stub"``).
        Overrides ``settings.LLM_PROVIDER`` when given.
    cost_tracker:
        Optional cost tracker for budget enforcement.

    Returns
    -------
    A ``FallbackLLMClient`` if fallback is enabled, or a single
    provider client if not.
    """
    provider_enum = LLMProvider(provider or settings.LLM_PROVIDER)
    fallback_enabled = settings.LLM_FALLBACK_ENABLED

    # Build the requested primary client
    primary: LLMClient | None = None
    if provider_enum == LLMProvider.OLLAMA:
        primary = _build_ollama_client()
    elif provider_enum == LLMProvider.OPENAI_COMPAT:
        primary = _build_openai_compat_client()
    elif provider_enum == LLMProvider.ANTHROPIC:
        primary = _build_anthropic_client(cost_tracker)
    elif provider_enum == LLMProvider.STUB:
        primary = _build_stub_client()

    if primary is None:
        logger.warning("Requested LLM provider %s not available", provider_enum.value)
        primary = _build_stub_client()

    if not fallback_enabled:
        logger.info("LLM provider: %s (no fallback)", primary.provider.value)
        return primary

    # Build the full fallback chain
    chain: list[LLMClient] = []

    if provider_enum == LLMProvider.OLLAMA:
        chain.append(primary)
        openai_compat = _build_openai_compat_client()
        if openai_compat:
            chain.append(openai_compat)
        anthropic = _build_anthropic_client(cost_tracker)
        if anthropic:
            chain.append(anthropic)
        chain.append(_build_stub_client())

    elif provider_enum == LLMProvider.OPENAI_COMPAT:
        chain.append(primary)
        ollama = _build_ollama_client()
        if ollama:
            chain.append(ollama)
        anthropic = _build_anthropic_client(cost_tracker)
        if anthropic:
            chain.append(anthropic)
        chain.append(_build_stub_client())

    elif provider_enum == LLMProvider.ANTHROPIC:
        chain.append(primary)
        ollama = _build_ollama_client()
        if ollama:
            chain.append(ollama)
        openai_compat = _build_openai_compat_client()
        if openai_compat:
            chain.append(openai_compat)
        chain.append(_build_stub_client())

    elif provider_enum == LLMProvider.STUB:
        chain.append(primary)

    providers_str = " → ".join(c.provider.value for c in chain)
    logger.info("LLM provider chain: %s", providers_str)

    return FallbackLLMClient(chain)


async def create_llm_client(
    provider: str | None = None,
    cost_tracker: CostTracker | None = None,
) -> LLMClient:
    """Create an LLM client with async health checks.

    Probes each provider's health and logs availability before
    returning the fallback chain.  Preferred over ``create_llm_client_sync``
    when called from async context.
    """
    client = create_llm_client_sync(provider=provider, cost_tracker=cost_tracker)

    # Log provider status
    if isinstance(client, FallbackLLMClient):
        status = await client.provider_status()
        for prov, healthy in status.items():
            emoji = "✓" if healthy else "✗"
            logger.info("  %s %s", emoji, prov)

    return client


# ---------------------------------------------------------------------------
# Legacy compatibility shim
# ---------------------------------------------------------------------------


def create_llm_client_legacy(
    cost_tracker: CostTracker | None = None,
) -> AnthropicClient:
    """Legacy factory — returns AnthropicClient directly.

    .. deprecated:: 0.2.0
        Use ``create_llm_client_sync()`` or ``create_llm_client()`` instead.
    """
    client = _build_anthropic_client(cost_tracker)
    if client is None:
        raise ValueError(
            "No Anthropic API key configured. Set ANTHROPIC_API_KEY or use "
            "LLM_PROVIDER=ollama for local models."
        )
    return client
