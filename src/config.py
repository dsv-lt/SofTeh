"""Конфигурация сервиса (pydantic-settings).

Все параметры читаются из переменных окружения (или файла ``.env``).
Секрет ``LLM_API_KEY`` никогда не хранится в коде и образе Docker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "fallback_models": (),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback_models": (),
    },
    "aistudio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.1-flash-lite",
        "fallback_models": ("gemini-flash-lite-latest", "gemma-4-31b-it"),
    },
}


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Внешняя LLM (стадия 2) ---
    llm_api_key: SecretStr | None = Field(
        default=None, description="Ключ Groq / OpenRouter / Google AI Studio"
    )
    llm_provider: Literal["auto", "groq", "openrouter", "aistudio"] = Field(
        default="auto",
        description=(
            "auto: определяется по префиксу ключа "
            "(gsk_ -> Groq, sk-or- -> OpenRouter, AQ./AIza -> Google AI Studio)"
        ),
    )
    llm_base_url: str | None = Field(default=None, description="Переопределение URL API")
    llm_model: str | None = Field(default=None, description="Переопределение модели")
    llm_timeout: float = Field(default=25.0, gt=0)
    llm_max_tokens: int = Field(default=1024, ge=16, le=8192)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Локальный классификатор (стадия 1) ---
    classifier_backend: Literal["lora", "baseline"] = "lora"
    models_dir: Path = PROJECT_ROOT / "models"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    chunk_chars: int = Field(default=1000, ge=100)
    max_chunks: int = Field(default=16, ge=1, description="Максимум фрагментов на запрос")

    @property
    def vectorizer_path(self) -> Path:
        return self.models_dir / "tfidf_vectorizer.pkl"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"

    @property
    def baseline_path(self) -> Path:
        return self.models_dir / "baseline_pytorch.pt"

    @property
    def lora_dir(self) -> Path:
        return self.models_dir / "lora_classifier"

    @property
    def base_model_dir(self) -> Path:
        return self.models_dir / "rubert_base"

    @property
    def resolved_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        key = self.llm_api_key.get_secret_value().strip() if self.llm_api_key else ""
        if key.startswith("sk-or-"):
            return "openrouter"
        if key.startswith(("AQ.", "AIza")):
            return "aistudio"
        return "groq"

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or str(PROVIDER_DEFAULTS[self.resolved_provider]["base_url"])

    @property
    def resolved_model(self) -> str:
        return self.llm_model or str(PROVIDER_DEFAULTS[self.resolved_provider]["model"])

    @property
    def resolved_candidate_models(self) -> tuple[str, ...]:
        if self.llm_model:
            return (self.llm_model,)
        primary = self.resolved_model
        fallbacks = tuple(PROVIDER_DEFAULTS[self.resolved_provider].get("fallback_models", ()))  # type: ignore[arg-type]
        return (primary, *(m for m in fallbacks if m != primary))


@lru_cache
def get_settings() -> Settings:
    return Settings()
