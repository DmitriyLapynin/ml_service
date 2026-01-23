SHELL := /bin/bash

PROJECT := ai-domain
DATA_VOL := ai_domain_data
MODELS_VOL := ai_domain_models

HOST_MODELS_DIR := /Users/dmitrij/ai-domain/embeddings_models
HOST_DATA_DIR := /Users/dmitrij/ai-domain/data
HOST_SECRETS_DIR := /Users/dmitrij/secrets

.PHONY: help build up down logs ps restart shell init-models init-data fix-perms test lint

help:
	@echo "Targets:"
	@echo "  build        Build docker images"
	@echo "  up           Start сервис"
	@echo "  down         Stop сервис"
	@echo "  logs         Tail logs"
	@echo "  shell        Shell inside container"
	@echo "  init-models  Copy embedding model into models volume (idempotent)"
	@echo "  init-data    Copy data into data volume (idempotent-ish)"
	@echo "  fix-perms    Fix volume permissions for appuser (uid 10001)"
	@echo "  test         Run pytest locally"
	@echo "  lint         (optional) run ruff/mypy if you add them"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

restart: down up

shell:
	docker compose exec $(PROJECT) sh

init-models:
	docker pull python:3.11-slim
	docker run --rm \
	  -v $(MODELS_VOL):/models \
	  -v $(HOST_MODELS_DIR):/host_models:ro \
	  python:3.11-slim \
	  sh -lc 'if [ ! -d /models/rubert-mini-frida ]; then cp -R /host_models/rubert-mini-frida /models/; else echo "Model already exists, skipping"; fi'

init-data:
	docker pull python:3.11-slim
	docker run --rm \
	  -v $(DATA_VOL):/data \
	  -v $(HOST_DATA_DIR):/host_data:ro \
	  python:3.11-slim \
	  sh -lc 'cp -R /host_data/* /data/ || true'

fix-perms:
	docker pull python:3.11-slim
	docker run --rm \
	  -v $(DATA_VOL):/data \
	  python:3.11-slim \
	  sh -lc "chown -R 10001:10001 /data && chmod -R u+rwX /data"

test:
	poetry run pytest -q

bootstrap: build init-models init-data fix-perms up
	@echo "Service started. Use 'make logs' to view logs."