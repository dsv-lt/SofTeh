FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Шаг 1: Установка зависимостей с кэшированием слоя
# Сначала ставится легковесная CPU-сборка PyTorch (экономия ~2.5 ГБ относительно CUDA-колеса)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('ai-forever/ruBert-base'); AutoModel.from_pretrained('ai-forever/ruBert-base')"

# Шаг 2: Копирование исходного кода и обученных артефактов
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Шаг 3: Точка входа
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
