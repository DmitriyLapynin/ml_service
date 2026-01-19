# ---------- Builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# system deps (добавляй по мере необходимости: gcc, g++, poppler-utils, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Poetry
ENV POETRY_VERSION=2.1.3
RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main --no-root


# Копируем код
COPY . /app

# (Опционально) тесты — пока выключены
# RUN poetry install --no-interaction --no-ansi
# RUN pytest -q


# ---------- Runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_DOMAIN_DATA_DIR=/data \
    AI_DOMAIN_MODELS_DIR=/models \
    PYTHONPATH=/app/src

WORKDIR /app

# Минимальные системные зависимости runtime (если нужно читать pdf/excel и т.п.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# создаём пользователя
RUN useradd -m -u 10001 appuser

# Копируем установленные пакеты и приложение из builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Готовим директории volume
RUN mkdir -p /data /models && chown -R appuser:appuser /data /models /app

USER appuser

EXPOSE 8080

# Healthcheck (желательно иметь /health)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Старт (лучше через uvicorn, либо gunicorn+uvicorn workers)
CMD ["python", "-m", "uvicorn", "ai_domain.main:app", "--host", "0.0.0.0", "--port", "8080"]