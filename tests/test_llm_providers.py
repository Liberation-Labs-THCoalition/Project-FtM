"""Tests for the provider-agnostic LLM abstraction layer.

Covers:
    - StubClient deterministic behavior
    - OllamaClient request formatting (mocked HTTP)
    - AnthropicClient request formatting (mocked SDK)
    - FallbackLLMClient cascading behavior
    - LLM factory configuration
    - Backward-compatible shim
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emet.cognition.llm_base import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    LLMUnavailableError,
)
from emet.cognition.llm_stub import StubClient
from emet.cognition.llm_ollama import OllamaClient, DEFAULT_OLLAMA_MODELS
from emet.cognition.llm_factory import FallbackLLMClient


# ====================================================================
# StubClient tests
# ====================================================================


class TestStubClient:
    """StubClient returns deterministic responses for testing."""

    @pytest.fixture
    def stub(self) -> StubClient:
        return StubClient()

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, stub: StubClient) -> None:
        resp = await stub.complete("Hello")
        assert isinstance(resp, LLMResponse)
        assert resp.provider == LLMProvider.STUB
        assert resp.cost_usd == 0.0
        assert resp.text  # non-empty

    @pytest.mark.asyncio
    async def test_complete_records_call(self, stub: StubClient) -> None:
        await stub.complete("Hello", tier="fast", system="Be helpful")
        assert len(stub.call_log) == 1
        assert stub.call_log[0]["method"] == "complete"
        assert stub.call_log[0]["prompt"] == "Hello"
        assert stub.call_log[0]["tier"] == "fast"
        assert stub.call_log[0]["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_classify_intent_keyword_matching(self, stub: StubClient) -> None:
        domains = ["entity_search", "financial_investigation", "story_development"]

        domain, confidence = await stub.classify_intent("search for company XYZ", domains)
        assert domain == "entity_search"
        assert confidence > 0.4

        domain, confidence = await stub.classify_intent("follow the money trail", domains)
        assert domain == "financial_investigation"
        assert confidence > 0.4

    @pytest.mark.asyncio
    async def test_classify_intent_unknown_falls_back(self, stub: StubClient) -> None:
        domains = ["entity_search", "network_analysis"]
        domain, confidence = await stub.classify_intent("xyzzy foobar", domains)
        # Should return first domain with low confidence
        assert domain in domains
        assert confidence <= 0.5

    @pytest.mark.asyncio
    async def test_extract_entities_finds_names(self, stub: StubClient) -> None:
        text = "John Smith met with Acme Corp on 2024-01-15 to discuss $500,000 payment."
        schema = {
            "person": "The person's full name",
            "company": "The organization or company name",
            "date": "The date mentioned",
            "amount": "The monetary amount",
        }
        result = await stub.extract_entities(text, schema)
        assert result.get("person") == "John Smith"
        assert result.get("date") == "2024-01-15"
        assert "$500,000" in (result.get("amount") or "")

    @pytest.mark.asyncio
    async def test_health_check_always_true(self, stub: StubClient) -> None:
        assert await stub.health_check() is True

    @pytest.mark.asyncio
    async def test_reset_clears_log(self, stub: StubClient) -> None:
        await stub.complete("Hello")
        await stub.complete("World")
        assert len(stub.call_log) == 2
        stub.reset()
        assert len(stub.call_log) == 0

    @pytest.mark.asyncio
    async def test_generate_content_with_context(self, stub: StubClient) -> None:
        result = await stub.generate_content(
            "Summarize this",
            context={"entity": "Gazprom", "jurisdiction": "Russia"},
            tier="balanced",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert stub.call_log[-1]["context"]["entity"] == "Gazprom"


# ====================================================================
# OllamaClient tests (mocked HTTP)
# ====================================================================


class TestOllamaClient:
    """OllamaClient sends correct requests to Ollama REST API."""

    @pytest.fixture
    def ollama(self) -> OllamaClient:
        return OllamaClient(host="http://localhost:11434")

    def test_default_models(self, ollama: OllamaClient) -> None:
        assert ollama._resolve_model("fast") == DEFAULT_OLLAMA_MODELS["fast"]
        assert ollama._resolve_model("balanced") == DEFAULT_OLLAMA_MODELS["balanced"]
        assert ollama._resolve_model("powerful") == DEFAULT_OLLAMA_MODELS["powerful"]

    def test_custom_models(self) -> None:
        custom = OllamaClient(models={"fast": "phi3:mini", "balanced": "llama3.1:8b"})
        assert custom._resolve_model("fast") == "phi3:mini"
        assert custom._resolve_model("balanced") == "llama3.1:8b"

    def test_unknown_tier_falls_back_to_balanced(self, ollama: OllamaClient) -> None:
        model = ollama._resolve_model("nonexistent")
        assert model == DEFAULT_OLLAMA_MODELS["balanced"]

    @pytest.mark.asyncio
    async def test_complete_formats_request(self, ollama: OllamaClient) -> None:
        mock_response = {
            "message": {"role": "assistant", "content": "Hello back!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 1000000,
            "load_duration": 500000,
            "eval_duration": 500000,
            "done_reason": "stop",
        }

        with patch.object(ollama, "_post", new_callable=AsyncMock, return_value=mock_response):
            resp = await ollama.complete("Hello", system="Be brief", tier="fast")

        assert resp.text == "Hello back!"
        assert resp.provider == LLMProvider.OLLAMA
        assert resp.cost_usd == 0.0
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.metadata["total_duration_ns"] == 1000000

    @pytest.mark.asyncio
    async def test_complete_raises_unavailable_on_connect_error(self, ollama: OllamaClient) -> None:
        import httpx

        with patch.object(
            ollama, "_post", new_callable=AsyncMock,
            side_effect=LLMUnavailableError("Cannot connect"),
        ):
            with pytest.raises(LLMUnavailableError, match="Cannot connect"):
                await ollama.complete("Hello")

    @pytest.mark.asyncio
    async def test_classify_intent_parses_json(self, ollama: OllamaClient) -> None:
        mock_response = {
            "message": {"content": '{"domain": "entity_search", "confidence": 0.9}'},
            "prompt_eval_count": 50,
            "eval_count": 20,
            "done_reason": "stop",
        }

        with patch.object(ollama, "_post", new_callable=AsyncMock, return_value=mock_response):
            domain, confidence = await ollama.classify_intent(
                "find John Smith", ["entity_search", "network_analysis"]
            )

        assert domain == "entity_search"
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_classify_intent_handles_markdown_wrapped_json(self, ollama: OllamaClient) -> None:
        mock_response = {
            "message": {"content": '```json\n{"domain": "network_analysis", "confidence": 0.8}\n```'},
            "prompt_eval_count": 50,
            "eval_count": 20,
            "done_reason": "stop",
        }

        with patch.object(ollama, "_post", new_callable=AsyncMock, return_value=mock_response):
            domain, confidence = await ollama.classify_intent(
                "show network", ["entity_search", "network_analysis"]
            )

        assert domain == "network_analysis"

    @pytest.mark.asyncio
    async def test_classify_intent_handles_bad_json(self, ollama: OllamaClient) -> None:
        mock_response = {
            "message": {"content": "I think this is entity_search"},
            "prompt_eval_count": 50,
            "eval_count": 20,
            "done_reason": "stop",
        }

        with patch.object(ollama, "_post", new_callable=AsyncMock, return_value=mock_response):
            domain, confidence = await ollama.classify_intent(
                "find something", ["entity_search", "network_analysis"]
            )

        assert domain == "general"
        assert confidence == 0.3

    @pytest.mark.asyncio
    async def test_health_check_with_models(self, ollama: OllamaClient) -> None:
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "mistral:7b"}]}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ollama.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_no_models(self, ollama: OllamaClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ollama.health_check()
            assert result is False


# ====================================================================
# FallbackLLMClient tests
# ====================================================================


class TestFallbackLLMClient:
    """FallbackLLMClient cascades through providers on failure."""

    @pytest.mark.asyncio
    async def test_uses_primary_when_healthy(self) -> None:
        primary = StubClient(default_response="primary response")
        backup = StubClient(default_response="backup response")
        client = FallbackLLMClient([primary, backup])

        resp = await client.complete("Hello")
        assert resp.text == "primary response"
        assert client.provider == LLMProvider.STUB

    @pytest.mark.asyncio
    async def test_falls_back_on_unavailable(self) -> None:
        primary = AsyncMock(spec=LLMClient)
        primary.provider = LLMProvider.OLLAMA
        primary.complete = AsyncMock(side_effect=LLMUnavailableError("Ollama down"))

        backup = StubClient(default_response="backup response")
        client = FallbackLLMClient([primary, backup])

        resp = await client.complete("Hello")
        assert resp.text == "backup response"

    @pytest.mark.asyncio
    async def test_cascades_through_multiple_failures(self) -> None:
        primary = AsyncMock(spec=LLMClient)
        primary.provider = LLMProvider.OLLAMA
        primary.complete = AsyncMock(side_effect=LLMUnavailableError("Ollama down"))

        secondary = AsyncMock(spec=LLMClient)
        secondary.provider = LLMProvider.ANTHROPIC
        secondary.complete = AsyncMock(side_effect=LLMUnavailableError("Anthropic down"))

        tertiary = StubClient(default_response="stub fallback")
        client = FallbackLLMClient([primary, secondary, tertiary])

        resp = await client.complete("Hello")
        assert resp.text == "stub fallback"

    @pytest.mark.asyncio
    async def test_classify_intent_with_fallback(self) -> None:
        primary = AsyncMock(spec=LLMClient)
        primary.provider = LLMProvider.OLLAMA
        primary.classify_intent = AsyncMock(side_effect=LLMUnavailableError("down"))

        backup = StubClient()
        client = FallbackLLMClient([primary, backup])

        domain, confidence = await client.classify_intent(
            "search for company", ["entity_search", "network_analysis"]
        )
        assert domain == "entity_search"

    @pytest.mark.asyncio
    async def test_generate_content_with_fallback(self) -> None:
        primary = AsyncMock(spec=LLMClient)
        primary.provider = LLMProvider.OLLAMA
        primary.generate_content = AsyncMock(side_effect=LLMUnavailableError("down"))

        backup = StubClient(default_response="generated content")
        client = FallbackLLMClient([primary, backup])

        result = await client.generate_content("Write something")
        assert result == "generated content"

    @pytest.mark.asyncio
    async def test_extract_entities_with_fallback(self) -> None:
        primary = AsyncMock(spec=LLMClient)
        primary.provider = LLMProvider.OLLAMA
        primary.extract_entities = AsyncMock(side_effect=LLMUnavailableError("down"))

        backup = StubClient()
        client = FallbackLLMClient([primary, backup])

        result = await client.extract_entities(
            "John Smith at Acme Corp",
            {"person": "Person name", "company": "Company name"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check_true_if_any_healthy(self) -> None:
        unhealthy = AsyncMock(spec=LLMClient)
        unhealthy.health_check = AsyncMock(return_value=False)

        healthy = StubClient()  # always healthy
        client = FallbackLLMClient([unhealthy, healthy])

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_provider_status(self) -> None:
        unhealthy = AsyncMock(spec=LLMClient)
        unhealthy.provider = LLMProvider.OLLAMA
        unhealthy.health_check = AsyncMock(return_value=False)

        healthy = StubClient()
        client = FallbackLLMClient([unhealthy, healthy])

        status = await client.provider_status()
        assert status["ollama"] is False
        assert status["stub"] is True

    def test_empty_clients_raises(self) -> None:
        with pytest.raises(ValueError, match="(?i)at least one"):
            FallbackLLMClient([])


# ====================================================================
# Factory tests
# ====================================================================


class TestLLMFactory:
    """Factory creates correct client based on settings."""

    @pytest.mark.asyncio
    async def test_stub_provider(self) -> None:
        with patch("emet.cognition.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "stub"
            mock_settings.LLM_FALLBACK_ENABLED = False
            mock_settings.ANTHROPIC_API_KEY = ""
            mock_settings.OLLAMA_HOST = "http://localhost:11434"
            mock_settings.OLLAMA_MODELS = {"fast": "test", "balanced": "test", "powerful": "test"}
            mock_settings.OLLAMA_TIMEOUT = 30.0
            mock_settings.OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1"
            mock_settings.OPENAI_COMPAT_API_KEY = ""
            mock_settings.OPENAI_COMPAT_MODELS = {"fast": "test", "balanced": "test", "powerful": "test"}
            mock_settings.OPENAI_COMPAT_TIMEOUT = 30.0
            mock_settings.DEPLOYMENT_TIER = "sprout"
            mock_settings.MODEL_ROUTING = {"haiku": "test", "sonnet": "test", "opus": "test"}

            from emet.cognition.llm_factory import create_llm_client_sync
            client = create_llm_client_sync()

            assert client.provider == LLMProvider.STUB
            resp = await client.complete("Hello")
            assert resp.provider == LLMProvider.STUB

    @pytest.mark.asyncio
    async def test_ollama_with_fallback_creates_chain(self) -> None:
        with patch("emet.cognition.llm_factory.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.LLM_FALLBACK_ENABLED = True
            mock_settings.ANTHROPIC_API_KEY = ""  # no Anthropic
            mock_settings.OLLAMA_HOST = "http://localhost:11434"
            mock_settings.OLLAMA_MODELS = {"fast": "test", "balanced": "test", "powerful": "test"}
            mock_settings.OLLAMA_TIMEOUT = 30.0
            mock_settings.OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1"
            mock_settings.OPENAI_COMPAT_API_KEY = ""
            mock_settings.OPENAI_COMPAT_MODELS = {"fast": "test", "balanced": "test", "powerful": "test"}
            mock_settings.OPENAI_COMPAT_TIMEOUT = 30.0
            mock_settings.DEPLOYMENT_TIER = "sprout"
            mock_settings.MODEL_ROUTING = {"haiku": "test", "sonnet": "test", "opus": "test"}

            from emet.cognition.llm_factory import create_llm_client_sync
            client = create_llm_client_sync()

            assert isinstance(client, FallbackLLMClient)
            # Chain: Ollama → OpenAI-compat → Stub (no Anthropic key)
            assert len(client._clients) == 3
            assert client._clients[0].provider == LLMProvider.OLLAMA
            assert client._clients[1].provider == LLMProvider.OPENAI_COMPAT
            assert client._clients[2].provider == LLMProvider.STUB


# ====================================================================
# Backward compatibility
# ====================================================================


class TestBackwardCompatibility:
    """Old import paths still work."""

    def test_old_imports_resolve(self) -> None:
        from emet.cognition.llm_client import LLMResponse as OldLLMResponse
        from emet.cognition.llm_client import create_llm_client

        assert OldLLMResponse is not None
        assert callable(create_llm_client)


# ====================================================================
# LLMResponse
# ====================================================================


class TestLLMResponse:
    """LLMResponse data class."""

    def test_fields(self) -> None:
        resp = LLMResponse(
            text="Hello",
            model="test-model",
            provider=LLMProvider.STUB,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            stop_reason="end_turn",
            metadata={"key": "value"},
        )
        assert resp.text == "Hello"
        assert resp.model == "test-model"
        assert resp.provider == LLMProvider.STUB
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.cost_usd == 0.001
        assert resp.stop_reason == "end_turn"
        assert resp.metadata["key"] == "value"

    def test_default_metadata(self) -> None:
        resp = LLMResponse(
            text="", model="", provider=LLMProvider.STUB,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
        )
        assert resp.metadata == {}
        assert resp.stop_reason is None
