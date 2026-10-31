"""Тесты конфигурации (Settings) и асинхронного клиента внешней LLM (LLMClient)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.config import Settings
from src.llm_client import LLMClient, LLMError, build_system_prompt


class ConfigResolutionTests(unittest.TestCase):
    def test_provider_autodetection_by_key_prefix(self) -> None:
        s_groq = Settings(llm_api_key="gsk_sample123", _env_file=None)
        self.assertEqual(s_groq.resolved_provider, "groq")
        self.assertEqual(s_groq.resolved_model, "llama-3.1-8b-instant")
        self.assertIn("groq.com", s_groq.resolved_base_url)

        s_or = Settings(llm_api_key="sk-or-v1-sample", _env_file=None)
        self.assertEqual(s_or.resolved_provider, "openrouter")
        self.assertTrue(s_or.resolved_model.endswith(":free"))
        self.assertIn("openrouter.ai", s_or.resolved_base_url)

        for prefix in ("AQ.sample_aistudio_key", "AIzaSySampleKey"):
            s_ai = Settings(llm_api_key=prefix, _env_file=None)
            self.assertEqual(s_ai.resolved_provider, "aistudio")
            self.assertIn("generativelanguage.googleapis.com", s_ai.resolved_base_url)
            self.assertGreaterEqual(len(s_ai.resolved_candidate_models), 2)

    def test_explicit_overrides_take_precedence(self) -> None:
        s = Settings(
            llm_api_key="AQ.test",
            llm_provider="groq",
            llm_base_url="http://localhost:9999/v1",
            llm_model="custom-model",
            _env_file=None,
        )
        self.assertEqual(s.resolved_provider, "groq")
        self.assertEqual(s.resolved_base_url, "http://localhost:9999/v1")
        self.assertEqual(s.resolved_model, "custom-model")
        self.assertEqual(s.resolved_candidate_models, ("custom-model",))


class LLMClientUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_prompt_matches_specification(self) -> None:
        prompt = build_system_prompt("Гоголь Н")
        self.assertIn("Входной текст идентифицирован как стиль автора: Гоголь Н.", prompt)
        self.assertIn("образы автора Гоголь Н.", prompt)

    async def test_missing_api_key_raises_503(self) -> None:
        client = LLMClient(Settings(llm_api_key=None, _env_file=None))
        self.assertFalse(client.configured)
        with self.assertRaises(LLMError) as ctx:
            await client.generate("sys", "user")
        self.assertEqual(ctx.exception.status_code, 503)
        await client.aclose()

    async def test_successful_completion_and_headers(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(req.headers)
            captured["body"] = json.loads(req.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "  Стилизованный фрагмент.  "}}]},
            )

        settings = Settings(llm_api_key="sk-or-v1-test", _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            headers=llm._client.headers,
            transport=httpx.MockTransport(handler),
        )
        result = await llm.generate("Системный", "Пользовательский", temperature=0.4)
        await llm.aclose()

        self.assertEqual(result, "Стилизованный фрагмент.")
        self.assertEqual(captured["headers"].get("x-title"), "Literary Style Assistant")
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(captured["body"]["temperature"], 0.4)

    async def test_retry_with_fallback_model_on_503(self) -> None:
        models_tried: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content.decode("utf-8"))
            models_tried.append(body["model"])
            if len(models_tried) == 1:
                return httpx.Response(503, json={"error": "high demand"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Ответ резервной модели."}}]},
            )

        settings = Settings(llm_api_key="AQ.test_key", llm_max_retries=2, _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(handler),
        )
        with patch("asyncio.sleep", new=AsyncMock()):
            text = await llm.generate("sys", "msg")
        await llm.aclose()

        self.assertEqual(text, "Ответ резервной модели.")
        self.assertEqual(len(models_tried), 2)
        self.assertNotEqual(models_tried[0], models_tried[1])

    async def test_auth_401_fails_fast_without_retries(self) -> None:
        calls = 0

        def handler(_req: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, text="Unauthorized")

        settings = Settings(llm_api_key="gsk_bad", llm_max_retries=3, _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMError) as ctx:
            await llm.generate("sys", "msg")
        await llm.aclose()
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(calls, 1)

    async def test_timeout_raises_504_and_malformed_body_raises_502(self) -> None:
        def timeout_handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=req)

        settings = Settings(llm_api_key="gsk_timeout", llm_max_retries=0, _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(timeout_handler),
        )
        with self.assertRaises(LLMError) as ctx:
            await llm.generate("sys", "msg")
        self.assertEqual(ctx.exception.status_code, 504)
        await llm.aclose()

        def bad_json_handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": []})

        llm2 = LLMClient(settings)
        await llm2._client.aclose()
        llm2._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(bad_json_handler),
        )
        with self.assertRaises(LLMError) as ctx2:
            await llm2.generate("sys", "msg")
        self.assertEqual(ctx2.exception.status_code, 502)
        await llm2.aclose()

    async def test_connect_error_and_non_retryable_400(self) -> None:
        def net_err_handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=req)

        settings = Settings(llm_api_key="gsk_net", llm_max_retries=0, _env_file=None)
        llm = LLMClient(settings)
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(net_err_handler),
        )
        with self.assertRaises(LLMError) as ctx:
            await llm.generate("sys", "msg")
        self.assertEqual(ctx.exception.status_code, 502)
        await llm.aclose()

        def bad_req_handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Bad Request")

        llm2 = LLMClient(Settings(llm_api_key="gsk_400", llm_max_retries=2, _env_file=None))
        await llm2._client.aclose()
        llm2._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(bad_req_handler),
        )
        with self.assertRaises(LLMError) as ctx2:
            await llm2.generate("sys", "msg")
        self.assertEqual(ctx2.exception.status_code, 502)
        await llm2.aclose()


if __name__ == "__main__":
    unittest.main()
