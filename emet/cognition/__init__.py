"""Emet cognition layer — LLM integration, model routing, and orchestration.

Provider-agnostic LLM abstraction with cascading fallback:

    Ollama (local, default) → OpenAI-compat → Anthropic (cloud) → Stub (testing)

Usage::

    from emet.cognition.llm_factory import create_llm_client_sync

    client = create_llm_client_sync()
    response = await client.complete("Analyze this entity network")
"""

from emet.cognition.llm_openai_compat import OpenAICompatClient  # noqa: F401
