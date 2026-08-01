"""Kintsugi configuration via environment / .env file."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://emet:emet@localhost:5432/emet"

    # --- Deployment ---
    DEPLOYMENT_TIER: Literal["seed", "sprout", "grove"] = "sprout"

    # --- Embeddings ---
    EMBEDDING_MODE: Literal["local", "api"] = "local"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    # --- LLM Provider ---
    LLM_PROVIDER: Literal["ollama", "anthropic", "openai_compat", "stub"] = "ollama"
    LLM_FALLBACK_ENABLED: bool = True

    # --- Ollama (local models — default) ---
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODELS: dict[str, str] = {
        "fast": "qwen3:8b",
        "balanced": "qwen3:14b",
        "powerful": "qwen3:14b",
    }
    OLLAMA_TIMEOUT: float = 120.0

    # --- OpenAI-compatible endpoint (vLLM, llama-server, LM Studio, etc.) ---
    OPENAI_COMPAT_BASE_URL: str = "http://localhost:11434/v1"
    OPENAI_COMPAT_API_KEY: str = ""
    OPENAI_COMPAT_MODELS: dict[str, str] = {
        "fast": "llama3.2:3b",
        "balanced": "mistral:7b",
        "powerful": "deepseek-r1:14b",
    }
    OPENAI_COMPAT_TIMEOUT: float = 120.0

    # --- LLM Keys (optional — used as fallback or when provider=anthropic) ---
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # --- Model routing (Anthropic tier mapping) ---
    MODEL_ROUTING: dict[str, str] = {
        "haiku": "claude-3-5-haiku-20241022",
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514",
    }

    # --- Shadow / governance ---
    KINTSUGI_SHADOW_ENABLED: bool = False

    # --- Shield budgets ---
    SHIELD_BUDGET_PER_SESSION: float = 5.0
    SHIELD_BUDGET_PER_DAY: float = 50.0

    # --- Observability ---
    OTEL_EXPORTER_ENDPOINT: str = ""

    # --- Redis / Celery ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Auth ---
    SECRET_KEY: str = "CHANGE-ME-in-production"

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- SpiderFoot (OSINT sidecar — Sprint 10b) ---
    SPIDERFOOT_HOST: str = "http://localhost:5001"
    SPIDERFOOT_USERNAME: str = "admin"
    SPIDERFOOT_PASSWORD: str = ""

    # --- MCP Server (Sprint 10a) ---
    MCP_HTTP_PORT: int = 9400

    @model_validator(mode="after")
    def _auto_shadow(self) -> "Settings":
        if self.DEPLOYMENT_TIER == "grove":
            self.KINTSUGI_SHADOW_ENABLED = True
        return self

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _fix_pg_scheme(cls, v: str) -> str:
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


settings = Settings()
