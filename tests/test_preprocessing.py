"""Модульные тесты предобработки, парсинга диалогов и нарезки текста."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing import (
    chunk_text,
    clean_text,
    extract_author,
    iter_jsonl,
    normalize_for_model,
    parse_record,
)


class PreprocessingUnitTests(unittest.TestCase):
    def test_extract_author_covers_all_four_dataset_templates(self) -> None:
        cases = [
            ("Напиши текс в стиле автора Платонов С", "Платонов С"),
            ("Придумай отрывок из книги Анна Каренина автора Толстой Л", "Толстой Л"),
            ("Придумай текс похожий на текст автора Достоевский Ф", "Достоевский Ф"),
            ("Напиши текс похожий на текст автора Салтыков-Щедрин М  ", "Салтыков-Щедрин М"),
            ("Придумай отрывок из книги Белая гвардия автора Горький Максим", "Горький Максим"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_author(prompt), expected)

    def test_extract_author_returns_none_when_absent(self) -> None:
        self.assertIsNone(extract_author("Произвольный вопрос без указания писателя"))
        self.assertIsNone(extract_author(""))

    def test_clean_text_handles_broken_escapes_and_control_chars(self) -> None:
        raw = 'Строка 1\\r\\n\\r\\n\x00\x07  \\"Диалог\\"\\tс\u00a0пробелами\u200b.\r\n\r\n  Строка 2\\\' '
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, 'Строка 1\n"Диалог" с пробелами.\nСтрока 2\'')
        self.assertEqual(clean_text(""), "")

    def test_normalize_for_model_collapses_newlines(self) -> None:
        text = "Первая строка.\r\n\r\n   Вторая\tстрока."
        self.assertEqual(normalize_for_model(text), "Первая строка. Вторая строка.")

    def test_chunk_text_splits_on_words_and_merges_short_tail(self) -> None:
        long_text = " ".join(f"слово{i}" for i in range(450))
        chunks = chunk_text(long_text, chunk_chars=500, min_chars=150)
        self.assertGreater(len(chunks), 4)
        for ch in chunks:
            self.assertGreaterEqual(len(ch), 150)

    def test_chunk_text_preserves_single_short_fragment_and_handles_empty(self) -> None:
        short = "Какая прекрасная ночь над Невой..."
        self.assertEqual(chunk_text(short, chunk_chars=1000, min_chars=200), [short])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_iter_jsonl_and_parse_record(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "Ты помощник писателя."},
                {"role": "user", "content": "Напиши текс в стиле автора Чехов А"},
                {"role": "assistant", "content": "В саду пахло яблоками."},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.jsonl"
            p.write_text("\n" + json.dumps(record, ensure_ascii=False) + "\n\n", encoding="utf-8")
            loaded = list(iter_jsonl(p))
        self.assertEqual(len(loaded), 1)
        author, text = parse_record(loaded[0])
        self.assertEqual(author, "Чехов А")
        self.assertEqual(text, "В саду пахло яблоками.")


if __name__ == "__main__":
    unittest.main()
