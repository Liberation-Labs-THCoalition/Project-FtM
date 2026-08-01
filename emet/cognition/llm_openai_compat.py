"""OpenAI-compatible LLM client -- any /v1/chat/completions endpoint.

Talks to any server that exposes the OpenAI chat completions API:
  - Ollama (localhost:11434/v1)
  - vLLM (localhost:8000/v1)
  - llama-server (localhost:8080/v1)
  - LM Studio (localhost:1234/v1)
  - OpenRouter (openrouter.ai/api/v1)
  - Any OpenAI-compatible endpoint

This is the runtime-portable alternative to OllamaClient. Ollama's
native API gives richer telemetry (nanosecond timing, model load
duration), but the OpenAI-compatible API works with *any* backend,
making Emet deployable wherever a /v1 endpoint is available.

Requires: httpx (already an Emet dependency).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from emet.cognition.llm_base import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier -> model mapping (same defaults as Ollama, referenced via /v1)
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_COMPAT_MODELS: dict[str, str] = {
    "fast": "llama3.2:3b",
    "balanced": "mistral:7b",
    "powerful": "deepseek-r1:14b",
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenAICompatClient(LLMClient):
    """Async client for OpenAI-compatible inference endpoints.

    Follows the same pattern as OllamaClient but talks to
    /v1/chat/completions instead of Ollama's native /api/chat.

    Parameters
    ----------
    base_url:
        Base URL of the OpenAI-compatible endpoint (no trailing slash).
        Should include the /v1 path segment.
        Examples: ``http://localhost:11434/v1`` (Ollama),
                  ``http://localhost:8000/v1`` (vLLM).
    api_key:
        API key if required (OpenRouter, hosted endpoints).
        Not needed for local servers.
    models:
        Tier -> model name mapping. Falls back to ``DEFAULT_OPENAI_COMPAT_MODELS``.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        models: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._models = models or dict(DEFAULT_OPENAI_COMPAT_MODELS)
        self._timeout = timeout

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI_COMPAT

    def _resolve_model(self, tier: str) -> str:
        """Map tier name to concrete model tag."""
        return self._models.get(tier, self._models.get("balanced", "mistral:7b"))

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to OpenAI-compatible API with error handling."""
        url = f"{self._base_url}{endpoint}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as e:
            raise LLMUnavailableError(
                f"Cannot connect to OpenAI-compatible endpoint at {self._base_url}. "
                f"Is the server running? -- Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(
                f"OpenAI-compat request timed out after {self._timeout}s. "
                f"Model may still be loading."
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise LLMUnavailableError(
                    f"Model not found at {self._base_url}. "
                    f"Check that model '{payload.get('model', '?')}' is available."
                ) from e
            raise

    # -- Core interface ------------------------------------------------------

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
        model = self._resolve_model(tier)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        data = await self._post("/chat/completions", payload)

        # Parse OpenAI-compatible response format
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(
                text="",
                model=model,
                provider=LLMProvider.OPENAI_COMPAT,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                stop_reason="no_choices",
            )

        message = choices[0].get("message", {})
        text = message.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=text,
            model=data.get("model", model),
            provider=LLMProvider.OPENAI_COMPAT,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # Local inference is free
            stop_reason=finish_reason,
            metadata={
                "base_url": self._base_url,
            },
        )

    async def classify_intent(
        self,
        message: str,
        domains: list[str],
    ) -> tuple[str, float]:
        domain_list = ", ".join(domains)

        prompt = (
            f"Classify the following user message into exactly one of these domains: {domain_list}\n\n"
            f'User message: "{message}"\n\n'
            f"Respond with ONLY a JSON object in this exact format:\n"
            f'{{"domain": "<chosen_domain>", "confidence": <0.0-1.0>}}\n\n'
            f"Choose the single best matching domain. If unsure, use confidence below 0.5."
        )

        response = await self.complete(
            prompt,
            tier="fast",
            max_tokens=100,
            temperature=0.0,
        )

        try:
            # Strip markdown code fences if model wraps response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            domain = result.get("domain", "general")
            confidence = float(result.get("confidence", 0.5))

            if domain not in domains:
                logger.warning("OpenAI-compat returned unknown domain %r, falling back", domain)
                domain = "general"
                confidence = 0.3

            return domain, confidence
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse OpenAI-compat classification response: %s", e)
            return "general", 0.3

    async def generate_content(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        tier: str = "balanced",
        system: str | None = None,
    ) -> str:
        full_system = system or ""
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_system = f"{full_system}\n\nContext:\n{context_str}"

        response = await self.complete(
            prompt,
            tier=tier,
            system=full_system.strip() or None,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.text

    async def extract_entities(
        self,
        text: str,
        entity_schema: dict[str, str],
    ) -> dict[str, Any]:
        schema_desc = "\n".join(
            f'- "{name}": {desc}' for name, desc in entity_schema.items()
        )

        prompt = (
            f"Extract the following entities from the text:\n{schema_desc}\n\n"
            f'Text: "{text}"\n\n'
            f"Respond with ONLY a JSON object containing the extracted entities.\n"
            f"Use null for entities not found in the text."
        )

        response = await self.complete(
            prompt,
            tier="fast",
            max_tokens=500,
            temperature=0.0,
        )

        try:
            text_out = response.text.strip()
            if text_out.startswith("```"):
                text_out = text_out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text_out)
        except json.JSONDecodeError:
            logger.warning("Failed to parse OpenAI-compat entity extraction response")
            return {}

    async def health_check(self) -> bool:
        """Check if the OpenAI-compatible endpoint is reachable."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=headers,
                )
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    if models:
                        return True
                    logger.warning("OpenAI-compat endpoint running but no models available")
                    return False
                return False
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return list of available model IDs from the endpoint."""
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=headers,
                )
                resp.raise_for_status()
                return [m.get("id", "") for m in resp.json().get("data", [])]
        except Exception:
            return []
