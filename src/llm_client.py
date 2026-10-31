"""Асинхронный клиент к внешней LLM (стадия 2).

Groq Cloud, OpenRouter и Google AI Studio предоставляют OpenAI-совместимый эндпоинт
``POST {base_url}/chat/completions``, поэтому клиент един для всех провайдеров.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = (
    "Ты — стилизатор русской классической литературы. Входной текст идентифицирован "
    "как стиль автора: {predicted_author}. Ответь на запрос пользователя или продолжи "
    "повествование, строго воспроизводя лексику, синтаксис, пунктуационные особенности "
    "и образы автора {predicted_author}. Выводи только сам художественный текст без "
    "вступлений и пояснений от лица ассистента."
)


def build_system_prompt(predicted_author: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(predicted_author=predicted_author)


class LLMError(Exception):
    """Ошибка обращения к внешней LLM."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class LLMClient:
    """Асинхронная обёртка над ``httpx.AsyncClient`` с ретраями и fallback-моделями."""

    RETRY_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.resolved_model
        self.candidate_models = settings.resolved_candidate_models
        self.provider = settings.resolved_provider
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value().strip()}"
        if self.provider == "openrouter":
            headers["X-Title"] = "Literary Style Assistant"
        self._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.llm_timeout, connect=10.0),
        )

    @property
    def configured(self) -> bool:
        return self.settings.llm_api_key is not None and bool(
            self.settings.llm_api_key.get_secret_value().strip()
        )

    async def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.7
    ) -> str:
        if not self.configured:
            raise LLMError("LLM_API_KEY не задан: передайте ключ через переменную окружения", 503)

        last_error = "unknown error"
        status = 502
        total_attempts = max(self.settings.llm_max_retries + 1, len(self.candidate_models))

        for attempt in range(total_attempts):
            target_model = self.candidate_models[min(attempt, len(self.candidate_models) - 1)]
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "stream": False,
            }
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                last_error = f"тайм-аут внешнего LLM API ({target_model})"
                status = 504
            except httpx.HTTPError as exc:
                last_error = f"сетевая ошибка ({target_model}): {exc.__class__.__name__}"
                status = 502
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        content = data["choices"][0]["message"]["content"]
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                        raise KeyError("empty content")
                    except (KeyError, IndexError, TypeError) as exc:
                        raise LLMError(f"Неожиданный формат ответа LLM: {data}") from exc
                status = resp.status_code
                last_error = f"HTTP {status} ({target_model}): {resp.text[:300]}"
                if status in (401, 403):
                    raise LLMError("Внешний LLM API отклонил ключ (401/403)", 502)
                if status not in self.RETRY_STATUSES:
                    break

            if attempt < total_attempts - 1:
                delay = 1.0 * (2**attempt)
                logger.warning("LLM call failed (%s), retry in %.1fs", last_error, delay)
                await asyncio.sleep(delay)

        raise LLMError(f"Ошибка внешнего LLM API: {last_error}", 504 if status == 504 else 502)

    async def aclose(self) -> None:
        await self._client.aclose()
