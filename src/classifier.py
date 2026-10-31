"""Инференс локального классификатора стиля (стадия 1).

Поддерживаются два бэкенда:

* ``baseline`` — посимвольный TF-IDF + PyTorch MLP (``models/baseline_pytorch.pt``);
* ``lora`` — ruBERT-base + LoRA-адаптер (``models/lora_classifier``).

Длинный текст режется на фрагменты ~1000 символов (как при обучении), итоговое
распределение по авторам — среднее softmax-вероятностей по фрагментам.
"""

from __future__ import annotations

import gc
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from torch import nn

from src.config import Settings
from src.preprocessing import chunk_text

logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """MLP-классификатор поверх TF-IDF. На выходе — сырые логиты (без Softmax)."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Prediction:
    author: str
    confidence: float
    top_k: list[tuple[str, float]]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class StyleClassifier(ABC):
    """Общий интерфейс классификаторов."""

    name: str = "unknown"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self.classes: list[str] = []
        self._lock = threading.Lock()

    @abstractmethod
    def _logits(self, chunks: list[str]) -> torch.Tensor:
        """Сырые логиты [n_chunks, n_classes]."""

    def predict(self, text: str, k: int = 3) -> Prediction:
        chunks = chunk_text(text, chunk_chars=self.settings.chunk_chars)
        if not chunks:
            raise ValueError("Пустой текст")
        chunks = chunks[: self.settings.max_chunks]
        with self._lock, torch.inference_mode():
            logits = self._logits(chunks).float()
            probs = torch.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        top = np.argsort(-probs)[: max(1, k)]
        top_k = [(self.classes[i], float(probs[i])) for i in top]
        return Prediction(author=top_k[0][0], confidence=top_k[0][1], top_k=top_k)

    def close(self) -> None:
        """Освобождает память модели (RAM/GPU)."""
        for attr in ("model", "vectorizer", "tokenizer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class BaselineClassifier(StyleClassifier):
    name = "pytorch-mlp-char-tfidf"

    def __init__(self, settings: Settings, vectorizer: object | None = None):
        super().__init__(settings)
        self.vectorizer = (
            vectorizer if vectorizer is not None else joblib.load(settings.vectorizer_path)
        )
        ckpt = torch.load(settings.baseline_path, map_location="cpu", weights_only=True)
        self.model = TextClassifier(ckpt["in_features"], ckpt["num_classes"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.classes = list(ckpt["classes"])
        logger.info(
            "Baseline loaded: %d features, %d classes", ckpt["in_features"], len(self.classes)
        )

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        x = torch.from_numpy(self.vectorizer.transform(chunks).toarray()).to(self.device)
        return self.model(x)


class LoraClassifier(StyleClassifier):
    name = "rubert-base-lora"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((settings.lora_dir / "labels.json").read_text(encoding="utf-8"))
        self.classes = list(meta["classes"])
        self.max_length = int(meta.get("max_length", 256))
        base_path = (
            settings.base_model_dir if settings.base_model_dir.exists() else meta["base_model"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.lora_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=len(self.classes),
            id2label=dict(enumerate(self.classes)),
            label2id={c: i for i, c in enumerate(self.classes)},
        )
        # Слияние LoRA-весов с базовыми: нет накладных расходов адаптера на инференсе
        self.model = PeftModel.from_pretrained(base, settings.lora_dir).merge_and_unload()
        self.model.to(self.device).eval()
        logger.info(
            "LoRA classifier loaded from %s (base: %s) on %s",
            settings.lora_dir,
            base_path,
            self.device,
        )

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            chunks, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(self.device)
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**enc).logits
        return self.model(**enc).logits


def load_classifier(settings: Settings, vectorizer: object | None = None) -> StyleClassifier:
    if (
        settings.classifier_backend == "lora"
        and (settings.lora_dir / "adapter_config.json").exists()
    ):
        return LoraClassifier(settings)
    if settings.classifier_backend == "lora":
        logger.warning(
            "LoRA adapter not found at %s, falling back to PyTorch baseline", settings.lora_dir
        )
    return BaselineClassifier(settings, vectorizer=vectorizer)
