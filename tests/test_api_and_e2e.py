"""Интеграционные и сквозные (E2E) тесты FastAPI-сервиса (включая живой вызов AI Studio)."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import torch
from fastapi.testclient import TestClient

from src.app import app
from src.classifier import StyleClassifier
from src.config import Settings, get_settings
from src.llm_client import LLMError


class DummyClassifier(StyleClassifier):
    name = "dummy-test-classifier"

    def __init__(self) -> None:
        super().__init__(Settings(llm_api_key="gsk_dummy", _env_file=None))
        self.classes = ["Достоевский Ф", "Пушкин А", "Чехов А", "Толстой Л"]

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        return torch.tensor([[3.2, 0.5, -0.4, 0.1] for _ in chunks], dtype=torch.float32)


class MockLLM:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.generate = AsyncMock(
            return_value="Стилизованное продолжение петербургской повести над Невой."
        )
        self.aclose = AsyncMock()


class FastAPIIntegrationTests(unittest.TestCase):
    def test_health_validation_and_error_handling(self) -> None:
        dummy = DummyClassifier()
        mock_llm = MockLLM()
        with (
            patch("src.app.load_classifier", return_value=dummy),
            patch("src.app.LLMClient", return_value=mock_llm),
            TestClient(app) as client,
        ):
            # 1. GET /health
            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json(), {"status": "ok", "model_loaded": True})

            # 2. Пустой или пробельный вопрос -> 400 «Пустой вопрос»
            for endpoint in ("/ask", "/classify"):
                for q in ("", " ", "   \t\n "):
                    r_bad = client.post(endpoint, json={"question": q})
                    self.assertEqual(r_bad.status_code, 400)
                    self.assertEqual(r_bad.json()["detail"], "Пустой вопрос")

            # 3. Некорректные параметры схемы -> 422
            self.assertEqual(client.post("/ask", json={}).status_code, 422)
            self.assertEqual(
                client.post("/ask", json={"question": "Текст", "k": 0}).status_code, 422
            )
            self.assertEqual(
                client.post("/ask", json={"question": "Текст", "temperature": 3.0}).status_code,
                422,
            )

            # 4. POST /classify и POST /ask
            r_cls = client.post("/classify", json={"question": "Ночь над Невой...", "k": 3})
            self.assertEqual(r_cls.status_code, 200)
            self.assertEqual(r_cls.json()["predicted_author"], "Достоевский Ф")
            self.assertEqual(len(r_cls.json()["top_authors"]), 3)

            r_ask = client.post(
                "/ask", json={"question": "Какая прекрасная ночь над Невой...", "temperature": 0.6}
            )
            self.assertEqual(r_ask.status_code, 200)
            self.assertEqual(r_ask.json()["predicted_author"], "Достоевский Ф")
            self.assertIn("Невой", r_ask.json()["generated_text"])

            # 5. Проброс ошибки внешнего LLM API (504)
            mock_llm.generate.side_effect = LLMError("Тайм-аут провайдера", status_code=504)
            r_err = client.post("/ask", json={"question": "Какая прекрасная ночь над Невой..."})
            self.assertEqual(r_err.status_code, 504)

            # 6. Незагруженный классификатор -> 503
            app.state.classifier = None
            self.assertEqual(client.get("/health").json()["model_loaded"], False)
            self.assertEqual(client.post("/classify", json={"question": "Текст"}).status_code, 503)

    def test_live_e2e_with_real_lora_and_configured_llm_api(self) -> None:
        """Живой сквозной тест: реальный ruBERT+LoRA + реальный внешний LLM API из .env."""
        get_settings.cache_clear()
        settings = get_settings()
        if not settings.llm_api_key or not settings.llm_api_key.get_secret_value().strip():
            self.skipTest("LLM_API_KEY не задан в .env — пропуск живого E2E-вызова")

        with TestClient(app) as client:
            r = client.post(
                "/ask",
                json={"question": "Какая прекрасная ночь над Невой...", "k": 3, "temperature": 0.7},
            )
            self.assertEqual(r.status_code, 200, msg=r.text)
            body = r.json()
            self.assertEqual(body["model_name"], "rubert-base-lora")
            self.assertGreater(body["confidence"], 0.05)
            self.assertLessEqual(body["confidence"], 1.0)
            self.assertGreater(len(body["generated_text"]), 40)


if __name__ == "__main__":
    unittest.main()
