"""Модульные и интеграционные тесты микросервиса Literary Style Assistant."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import torch
from fastapi.testclient import TestClient

from src.app import app
from src.classifier import BaselineClassifier, LoraClassifier, StyleClassifier, TextClassifier
from src.config import Settings, get_settings
from src.llm_client import LLMClient, LLMError, build_system_prompt
from src.preprocessing import chunk_text, clean_text, extract_author, normalize_for_model


class DummyClassifier(StyleClassifier):
    """Легковесный тестовый классификатор для изоляции API-тестов от весов на диске."""

    name = "dummy-test-classifier"

    def __init__(self) -> None:
        settings = Settings(llm_api_key="gsk_test_dummy_key")
        super().__init__(settings)
        self.classes = ["Достоевский Ф", "Пушкин А", "Чехов А", "Толстой Л"]

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        # Детерминированные сырые логиты (без Softmax), зависящие от текста
        logits = torch.tensor([[3.2, 0.5, -0.4, 0.1] for _ in chunks], dtype=torch.float32)
        return logits


class PreprocessingTests(unittest.TestCase):
    def test_extract_author_all_templates(self) -> None:
        self.assertEqual(
            extract_author("Напиши текс в стиле автора Платонов С"),
            "Платонов С",
        )
        self.assertEqual(
            extract_author("Придумай отрывок из книги Анна Каренина автора Толстой Л"),
            "Толстой Л",
        )
        self.assertEqual(
            extract_author("Напиши текс похожий на текст автора Салтыков-Щедрин М"),
            "Салтыков-Щедрин М",
        )
        self.assertIsNone(extract_author("Просто произвольный запрос без метки"))

    def test_clean_and_chunk_text(self) -> None:
        dirty = 'Привет,\\r\\n  мир!\\n\\n\\"Цитата\\"   с пробелами.\r\n\r\nВторая строка.'
        cleaned = clean_text(dirty)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\\n", cleaned)
        self.assertIn('"Цитата"', cleaned)

        norm = normalize_for_model(dirty)
        self.assertNotIn("\n", norm)

        long_text = " ".join(["слово"] * 500)
        chunks = chunk_text(long_text, chunk_chars=300, min_chars=80)
        self.assertGreater(len(chunks), 3)
        for ch in chunks:
            self.assertGreaterEqual(len(ch), 80)


class BaselineArchitectureTests(unittest.TestCase):
    def test_raw_logits_no_softmax(self) -> None:
        model = TextClassifier(in_features=128, num_classes=64)
        # Проверяем отсутствие Softmax/LogSoftmax в архитектуре
        for module in model.modules():
            self.assertNotIsInstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax))
        x = torch.randn(4, 128)
        out = model(x)
        self.assertEqual(out.shape, (4, 64))
        # Сырые логиты не обязаны суммироваться в 1 и могут быть отрицательными
        self.assertFalse(torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-2))


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_prompt_contains_author(self) -> None:
        prompt = build_system_prompt("Гоголь Н")
        self.assertIn("Гоголь Н", prompt)
        self.assertIn("стилизатор русской классической литературы", prompt)

    async def test_missing_api_key_raises_503(self) -> None:
        client = LLMClient(Settings(llm_api_key=None, _env_file=None))
        with self.assertRaises(LLMError) as ctx:
            await client.generate("sys", "user")
        self.assertEqual(ctx.exception.status_code, 503)
        await client.aclose()

    async def test_openai_compatible_completion_mock(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Стилизованный ответ."}}]},
            )

        settings = Settings(llm_api_key="gsk_unit_test_key", _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            headers={"Authorization": "Bearer gsk_unit_test_key"},
            transport=httpx.MockTransport(handler),
        )
        text = await llm.generate("Системный промпт", "Вопрос", temperature=0.5)
        await llm.aclose()

        self.assertEqual(text, "Стилизованный ответ.")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["body"]["model"], "llama-3.1-8b-instant")
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(captured["body"]["temperature"], 0.5)


class FastAPIEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_clf = DummyClassifier()
        self.mock_llm = MagicMockLLM()

    def test_endpoints_with_mocked_dependencies(self) -> None:
        with (
            patch("src.app.load_classifier", return_value=self.dummy_clf),
            patch("src.app.LLMClient", return_value=self.mock_llm),
            TestClient(app) as client,
        ):
            # 1. GET /health
            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json(), {"status": "ok", "model_loaded": True})

            # 2. POST /ask — валидация пустого и пробельного вопроса -> 400
            for empty_q in ("", "   ", "\t\n "):
                r_bad = client.post("/ask", json={"question": empty_q})
                self.assertEqual(r_bad.status_code, 400)
                self.assertEqual(r_bad.json()["detail"], "Пустой вопрос")

            # 3. POST /classify — локальная стадия 1
            r_cls = client.post("/classify", json={"question": "Ночь над Невой...", "k": 2})
            self.assertEqual(r_cls.status_code, 200)
            cls_body = r_cls.json()
            self.assertEqual(cls_body["predicted_author"], "Достоевский Ф")
            self.assertEqual(len(cls_body["top_authors"]), 2)

            # 4. POST /ask — полный двухстадийный пайплайн
            r_ask = client.post(
                "/ask",
                json={"question": "Какая прекрасная ночь над Невой...", "k": 3, "temperature": 0.6},
            )
            self.assertEqual(r_ask.status_code, 200)
            body = r_ask.json()
            self.assertEqual(body["predicted_author"], "Достоевский Ф")
            self.assertGreater(body["confidence"], 0.5)
            self.assertEqual(body["model_name"], "dummy-test-classifier")
            self.assertIn("Невой", body["generated_text"])


class MagicMockLLM:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.generate = AsyncMock(
            side_effect=lambda system_prompt, user_message, temperature=0.7: (
                f"[Стиль подтверждён] Продолжение о том, как тиха ночь над Невой ({temperature})."
            )
        )
        self.aclose = AsyncMock()


class RealArtifactsIntegrationTests(unittest.TestCase):
    """Интеграционные тесты с реальными обученными артефактами из директории models/."""

    def test_real_baseline_and_lora_classifiers(self) -> None:
        settings_base = Settings(classifier_backend="baseline", _env_file=None)
        base_clf = BaselineClassifier(settings_base)
        self.assertEqual(len(base_clf.classes), 64)
        pred_base = base_clf.predict(
            "Какая прекрасная ночь над Невой... Белые ночи Петербурга, мечтатель шёл вдоль канала.",
            k=3,
        )
        self.assertIn(pred_base.author, base_clf.classes)
        self.assertGreater(pred_base.confidence, 0.0)
        self.assertEqual(len(pred_base.top_k), 3)
        base_clf.close()

        settings_lora = Settings(classifier_backend="lora", _env_file=None)
        lora_clf = LoraClassifier(settings_lora)
        self.assertEqual(len(lora_clf.classes), 64)
        pred_lora = lora_clf.predict("Какая прекрасная ночь над Невой...", k=3)
        self.assertIn(pred_lora.author, lora_clf.classes)
        self.assertGreater(pred_lora.confidence, 0.0)
        self.assertEqual(len(pred_lora.top_k), 3)
        lora_clf.close()

    def test_full_lifespan_with_real_lora_model(self) -> None:
        get_settings.cache_clear()
        mock_llm = MagicMockLLM()
        with patch("src.app.LLMClient", return_value=mock_llm), TestClient(app) as client:
            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json(), {"status": "ok", "model_loaded": True})

            r_ask = client.post(
                "/ask",
                json={"question": "Какая прекрасная ночь над Невой...", "k": 3},
            )
            self.assertEqual(r_ask.status_code, 200)
            data = r_ask.json()
            self.assertEqual(data["model_name"], "rubert-base-lora")
            self.assertGreater(data["confidence"], 0.1)
            self.assertTrue(len(data["generated_text"]) > 10)


if __name__ == "__main__":
    unittest.main()
