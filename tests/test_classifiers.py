"""Тесты нейросетевых классификаторов и целостности обученных артефактов в models/."""

from __future__ import annotations

import json
import unittest

import joblib
import torch
from torch import nn

from src.classifier import (
    BaselineClassifier,
    LoraClassifier,
    TextClassifier,
    load_classifier,
    resolve_device,
)
from src.config import Settings


class ClassifierAndArtifactTests(unittest.TestCase):
    def test_text_classifier_architecture_and_gradients(self) -> None:
        model = TextClassifier(in_features=256, num_classes=64)
        for mod in model.modules():
            self.assertNotIsInstance(mod, (nn.Softmax, nn.LogSoftmax))
        layers = list(model.net.children())
        self.assertEqual(len(layers), 3)
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertIsInstance(layers[1], nn.ReLU)
        self.assertIsInstance(layers[2], nn.Linear)

        x = torch.randn(8, 256)
        y = torch.randint(0, 64, (8,))
        logits = model(x)
        self.assertEqual(logits.shape, (8, 64))
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        self.assertIsNotNone(layers[0].weight.grad)

    def test_resolve_device(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertIn(resolve_device("auto").type, ("cpu", "cuda"))

    def test_saved_artifacts_integrity(self) -> None:
        settings = Settings(_env_file=None)
        vec = joblib.load(settings.vectorizer_path)
        le = joblib.load(settings.label_encoder_path)
        self.assertEqual(vec.analyzer, "char")
        self.assertEqual(tuple(vec.ngram_range), (2, 4))
        self.assertEqual(vec.max_features, 50000)
        self.assertTrue(vec.sublinear_tf)
        self.assertEqual(len(vec.vocabulary_), 50000)
        self.assertEqual(len(le.classes_), 64)

        ckpt = torch.load(settings.baseline_path, map_location="cpu", weights_only=True)
        self.assertIn("model_state_dict", ckpt)
        self.assertEqual(ckpt["epoch"], 20)
        self.assertEqual(ckpt["in_features"], 50000)
        self.assertEqual(ckpt["num_classes"], 64)

        adapter_cfg = json.loads(
            (settings.lora_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(adapter_cfg["r"], 16)
        self.assertEqual(adapter_cfg["lora_alpha"], 32)
        self.assertEqual(adapter_cfg["task_type"], "SEQ_CLS")
        self.assertTrue((settings.lora_dir / "adapter_model.safetensors").exists())

    def test_baseline_and_lora_inference_single_and_multichunk(self) -> None:
        sentence = (
            "В начале июля, в чрезвычайно жаркое время, под вечер, молодой человек вышел из каморки "
            "в переулке и медленно отправился к мосту через канал в Петербурге."
        )
        long_fragment = " ".join([sentence] * 20)
        for backend in ("baseline", "lora"):
            with self.subTest(backend=backend):
                settings = Settings(classifier_backend=backend, _env_file=None)
                clf = (
                    BaselineClassifier(settings)
                    if backend == "baseline"
                    else LoraClassifier(settings)
                )
                pred = clf.predict(long_fragment, k=5)
                self.assertIn(pred.author, clf.classes)
                self.assertGreater(pred.confidence, 0.0)
                self.assertLessEqual(pred.confidence, 1.0)
                self.assertEqual(len(pred.top_k), 5)
                probs = [p for _, p in pred.top_k]
                self.assertEqual(probs, sorted(probs, reverse=True))
                with self.assertRaises(ValueError):
                    clf.predict("   ")
                clf.close()

        # Проверка load_classifier в режиме baseline
        clf_fb = load_classifier(Settings(classifier_backend="baseline", _env_file=None))
        self.assertIsInstance(clf_fb, BaselineClassifier)
        clf_fb.close()


if __name__ == "__main__":
    unittest.main()
