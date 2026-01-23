## ai-domain

Полноценный сервис для домена AI/LLM с API, оркестратором, LangGraph-графами, единым LLM-слоем, RAG (Supabase + FAISS) и трассировкой (включая LangSmith).

### Архитектура и потоки

**API → Orchestrator → Graph → Nodes → LLM/Tools/RAG**
1) **FastAPI** принимает запрос (`/v1/chat`, `/v1/system-analysis`, `/v1/kb/upload`, `/v1/kb/{funnel_id}/{kb_id}`, `/health`).
2) **Middleware** ставит `trace_id`, читает заголовки, пишет structured-логи, прокидывает `request.state`.
3) **Orchestrator** собирает `GraphState` и запускает граф через `graph.ainvoke`.
4) **Nodes** читают промпты, вызывают LLM/Tools/RAG, пишут runtime/trace.
5) **Ответ** возвращается с `answer`, `status`, `trace_id`, `meta`.

### Основные компоненты

**Оркестратор**
- `src/ai_domain/orchestrator/service.py` — `Orchestrator.run(request)`:
  - idempotency
  - version/policy resolve
  - сбор `GraphState`
  - запуск графа
  - сбор ответа (`status/answer/stage/trace_id/versions/meta`)
- `src/ai_domain/orchestrator/context_builder.py` — нормализация входа и сбор `GraphState`.
- `src/ai_domain/graphs/state.py` — структура состояния.

**Графы**
- `src/ai_domain/graphs/main_graph.py` и flow’ы (`chat/email/voice`).
- `src/ai_domain/agent/graph.py` — агентский граф с safety → agent → retrieve → generate → safety_out.

**Nodes**
- `src/ai_domain/agent/nodes/*` — safety, tools loop, tool executor, генерация ответа.
- `src/ai_domain/nodes/*` — stage analysis, rag retrieve, final answer.
- `src/ai_domain/utils/memory.py` — memory selection (buffer/summary) + structured logs `memory_trim`.

**LLM слой**
- `src/ai_domain/llm/client.py` — единый вход:
  - `invoke_text`, `invoke_structured`, `invoke_tool_calls`, `invoke_tool_response`, `generate`
  - ретраи, лимитеры, circuit breaker
  - sanitization параметров по возможностям моделей
- `src/ai_domain/llm/model_caps.py` — матрица поддерживаемых параметров моделей.
- `src/ai_domain/llm/openai_provider.py` — маппинг параметров, включая `max_tokens`/`max_completion_tokens`.

**Tools**
- `src/ai_domain/tools/registry.py` — регистрация/валидация/исполнение инструментов, глобальный limiter, structured-логи tool_call.
- `knowledge_search` уважает `rag_enabled`.

**RAG (Supabase + FAISS)**
- Supabase — источник истины: `kb_files`, `kb_chunks`, `kb_embeddings`.
- FAISS — локальный кэш/ускоритель: `data/funnels/{funnel_id}/kb/{kb_id}/`.
- `src/ai_domain/rag/kb_client.py` — сбор чанков, эмбеддингов, запись в Supabase, build FAISS.
- `src/ai_domain/rag/supabase_store.py` — `upsert_kb_file`, `find_kb_file_by_hash`, hash/тип источника.
- `scripts/build_faiss_index.py` — билд индекса с Supabase-записью.

**Ingest/удаление KB**
- `/v1/kb/upload` — multipart upload с `funnel_id`, сразу запускает ingest (chunks + embeddings + FAISS).
- `/v1/kb/{funnel_id}/{kb_id}` — удаление:
  - manifest → локальные файлы → Supabase
  - идемпотентный ответ `deleted/not_found/conflict`.
- `src/ai_domain/rag/delete_service.py` — централизованный delete (lock, manifest, локальные файлы, Supabase).

**System Analysis**
- `/v1/system-analysis` — отдельный анализ диалога, structured output (`FastAnalytics`).
- `src/ai_domain/system_analysis/service.py` — memory trim + prompt + `invoke_structured`.
- `src/ai_domain/system_analysis/prompts.py` — генерация системного промпта.

**Промпты (Supabase)**
- `src/ai_domain/registry/prompts.py` — `PromptRepository` (кэш, шаблоны, совместимость).
- `src/ai_domain/registry/static_prompt_repo.py` — fallback для dev.

**Telemetry и трассировка**
- Structured JSON логи: `api_request`, `memory_trim`, `node_start/node_end`, `llm_call_end`, `tool_call_end`, `api_response`.
- `src/ai_domain/llm/metrics.py` — `StateMetricsWriter` + `LangSmithWriter` (без credentials).
- Trace meta возвращается в ответе.

### API эндпоинты

- `GET /health`
- `POST /v1/chat`
- `POST /v1/system-analysis`
- `POST /v1/kb/upload` (multipart/form-data: `file`, `funnel_id`)
- `DELETE /v1/kb/{funnel_id}/{kb_id}`

### Хранилища и данные

**Локальные файлы**
- `AI_DOMAIN_DATA_DIR` (по умолчанию `data/`)
- структура: `data/funnels/{funnel_id}/kb/{kb_id}/...`

**Модели**
- `AI_DOMAIN_MODELS_DIR` (по умолчанию `embeddings_models/`)

**Секреты**
- `AI_DOMAIN_SECRETS_DIR` (по умолчанию `../secrets`)
- env `.env` поддерживается.

### Логи и наблюдаемость

Все логи — JSON. В debug режиме (`AI_DOMAIN_DEBUG_LOGGING=true`) дополнительно логируются входные/выходные payloadы, но без секретов.

### Тесты

Структура тестов:
- `tests/unit/agent`
- `tests/unit/memory`
- `tests/unit/prompts`
- `tests/unit/tools`
- `tests/unit/rag`
- `tests/unit/llm`
- `tests/unit/routing`
- `tests/unit/orchestrator`
- `tests/unit/versioning`
- `tests/integration/graph`
- `tests/integration/orchestrator`

Запуск:
- `poetry run pytest -q`

### Запуск приложения

**FastAPI**
- `uvicorn ai_domain.main:app --reload --log-level info`

**Pipeline (скрипты)**
- `poetry run python scripts/run_pipeline.py`

### Переменные окружения (минимум)

- `OPENAI_API_KEY`
- `SUPABASE_URL` / `SUPABASE_KEY`
- `AI_DOMAIN_DATA_DIR` (опционально)
- `AI_DOMAIN_MODELS_DIR` (опционально)
- `AI_DOMAIN_SECRETS_DIR` (опционально)
- `LANGSMITH_API_KEY` (для LangSmith)
- `LANGSMITH_PROJECT` (опционально)