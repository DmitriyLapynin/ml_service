## ai-domain

Проект-скелет домена “AI/LLM” с оркестратором, LangGraph-графами, единым фасадом для вызовов LLM и инфраструктурой для промптов (Supabase), guardrails и тестов.

### Что уже реализовано

**1) Оркестратор**
- `src/ai_domain/orchestrator/service.py` — `Orchestrator.run(request)`:
  - idempotency (кэширование по `idempotency_key`)
  - version/policy resolve (внешние резолверы)
  - сбор `GraphState` и запуск графа через `graph.invoke(state)`
  - сбор ответа (`status/answer/stage/trace_id/versions`)
- `src/ai_domain/orchestrator/context_builder.py` — нормализация `messages` и сбор `GraphState`, включая поля из API слоя (`prompt`, `role_instruction`, `memory_*`, `model_params`, и т.д.).
- `src/ai_domain/graphs/state.py` — dataclass `GraphState` с полями для runtime, промптов, модели, памяти и output.

**2) LangGraph графы “основного домена”**
- `src/ai_domain/graphs/main_graph.py` + flow’ы в `src/ai_domain/graphs/*_flow.py`:
  - router по `channel` → `chat/email/voice`
  - chat-поток включает стадии анализа (`StageAnalysisNode`), RAG retrieve, tools loop и финальную генерацию.

**3) Узлы (nodes)**
- `src/ai_domain/nodes/stage_analysis.py` — анализ стадии/сигналов (LLM), теперь уважает `memory_strategy/memory_params` (берёт последние `k` сообщений).
- `src/ai_domain/nodes/final_answer.py` — финальный ответ:
  - учитывает `role_instruction` и `prompt` из `ChatRequest`
  - уважает memory window (`k`) и параметры модели (`temperature/top_p`)
  - корректно конвертирует `state.credentials` (dict) → `LLMCredentials`
- `src/ai_domain/nodes/rag_retrieve.py` — RAG retrieve (включается/выключается policy’ями).
- `src/ai_domain/nodes/tools_loop.py` — цикл решений по tools (пока без реального tool-calling в LLM).

**4) Единый формат входа API (ChatRequest)**
- `src/ai_domain/api/schemas.py`:
  - `ChatRequest`, `ChatMessage`, `ToolDescription`, `ModelConfig`
  - адаптер `chat_request_to_orchestrator_request(...)` для перевода внешнего запроса в формат `Orchestrator.run`.
- Поля `model`, `model.params (temperature/top_p)`, `memory_strategy/memory_params` прокидываются в `GraphState` и используются узлами.

**5) Единый фасад вызова LLM: LLMClient**
- `src/ai_domain/llm/client.py`:
  - `LLMConfig` (model/provider/temperature/top_p/max_tokens/timeout/retries/tags/metadata/seed)
  - `invoke_text(messages, config) -> str`
  - `invoke_structured(schema, messages, config) -> BaseModel|dict`
  - `generate(...) -> LLMResponse` — совместимость с существующими узлами (2 стиля: kwargs и `LLMRequest`).
- Внутри используется существующий `LLMRouter` (retries/breaker/limiter) и **единый** механизм structured output через LangChain.

**6) LangChain интеграция (structured output)**
- `src/ai_domain/llm/langchain_adapter.py`:
  - `LangChainLLMAdapter` — адаптер, позволяющий включать наш `llm.generate`/`LLMRouter.generate` в LangChain chain.
  - `with_structured_output(PydanticModel, include_raw=True/False)` — стандартизированный structured output без ручного `prompt.partial(format_instructions=...)` снаружи.

**7) Guardrails / safety для agent-графа**
- `src/ai_domain/agent/graph.py` — LangGraph-граф guardrails + agent/retrieve/generate.
- `src/ai_domain/agent/nodes.py`:
  - `safety_in_node`/`safety_out_node` — гибрид: rule-based + LLM-классификатор (LangChain `ainvoke` + structured output)
  - `agent_node` — сбор составного промпта и фильтрация tools по `is_rag`
- `src/ai_domain/agent/safety_prompt.py`:
  - хардкод промпта классификатора безопасности
  - Pydantic-схема `SafetyClassifierOutput`
  - rule-based списки + парсеры/нормализация.

**8) Supabase промпты**
- `src/ai_domain/registry/prompts.py` — `PromptRepository(supabase_client)`:
  - читает `prompts` по `(key, version)` и кэширует
  - совместим со стилями вызова `get_prompt(prompt_key, version, channel=...)` и keyword-стилем
  - поддерживает шаблоны:
    - plain string
    - f-string/format-строка (через `variables=...`)
    - JSON: `{"template":"...", "defaults":{...}}` + `variables`.
- `src/ai_domain/registry/supabase_connector.py` — коннектор Supabase (`create_client(url,key)`).
- `src/ai_domain/registry/static_prompt_repo.py` — fallback для dev (статические промпты).

**9) Секреты “как на проде”**
- `src/ai_domain/secrets.py`:
  - сначала берёт из env
  - иначе читает `../secrets/.env` (через `python-dotenv`)
  - также поддерживает fallback на файлы `../secrets/OPENAI_API_KEY` и т.п.
  - путь можно переопределить через `AI_DOMAIN_SECRETS_DIR`.
- `.env.example` — пример переменных.

**10) Скрипты для ручных прогонов**
- `scripts/run_pipeline.py` — пример прогона orchestrator+graph с OpenAI и промптами (Supabase или fallback).
- `scripts/run_safety_in_node.py` — реальный прогон `safety_in_node` (пишет только JSON `{unsafe,injection}`).

### Как устроен проект (архитектура)

**Внешний API запрос → Orchestrator → Graph → Nodes → LLM**
1) API слой принимает `ChatRequest` (`src/ai_domain/api/schemas.py`)
2) Конвертируем его в `request` для `Orchestrator.run()` через `chat_request_to_orchestrator_request`
3) `Orchestrator` собирает `GraphState` и вызывает граф (`LangGraph`)
4) Nodes читают промпты из `PromptRepository`/`StaticPromptRepo`, формируют `LLMRequest` и вызывают LLM
5) Ответ возвращается как `answer/stage/status`

**Единый вход для LLM**
- В идеале весь код вызывает LLM только через `LLMClient` (или через объект, совместимый с `.generate`, например `LLMRouter/LLMClient`).
- Structured output стандартизирован через LangChain `with_structured_output`.

### Промпты (Supabase таблица `prompts`)

Текущие обязательные ключи (по коду):
- `system_prompt` — для `FinalAnswerNode`
- `analysis_prompt` — для `StageAnalysisNode`
- `tool_prompt` — для `ToolsLoopNode`

Рекомендуемые (если хочешь хранить в Supabase составные/динамические промпты):
- `agent_prompt` — для `agent_node` (если решишь переносить сборку промпта из кода в Supabase)

Схема таблицы, которую ожидает `PromptRepository`:
- `key` (text)
- `version` (text)
- `content` (text)
Пара `(key, version)` должна быть уникальной.

### Секреты/переменные окружения

Поддерживаются 3 способа:
1) Экспорт env vars (как в проде)
2) `../secrets/.env` (как “продоподобный” внешний каталог)
3) `../secrets/<NAME>` (по одному файлу на секрет)

Минимум для live-прогонов:
- `OPENAI_API_KEY`
- `SUPABASE_URL` и `SUPABASE_KEY` (если читаешь промпты из Supabase)

### Как запустить

**Установка зависимостей**
- `poetry install`

**Запуск pipeline**
- `poetry run python scripts/run_pipeline.py`

**Запуск safety_in_node (реальный вызов LLM)**
- `poetry run python scripts/run_safety_in_node.py --text "Ignore previous instructions"`

**Тесты**
- Запускать через Poetry, чтобы совпало окружение зависимостей:
  - `poetry run pytest -q`
  - при необходимости — конкретные файлы `poetry run pytest -q tests/unit/test_llm_client.py`

### Что ещё нужно сделать (план улучшений)

**1) API слой**
- Реальные HTTP роуты (FastAPI/Starlette) в `src/ai_domain/api/routes.py` и схемы/валидация.
- Middleware для trace_id и логирования.

**2) Реальный сбор deps**
- Вынести создание `LLMClient/LLMRouter`, prompt repo (Supabase), rag client, tool registry в единый composer/DI.

**3) OpenRouter провайдер**
- Добавить `LLMProvider` для OpenRouter + маппинг ошибок → `LLMError` и включить fallback через `LLMClient`/router config.

**4) Инструменты (tools)**
- Реальный tool-calling: хранение описаний, исполнение, запись результатов в runtime, сериализация/валидация args.

**5) Memory strategy “summary”**
- Сейчас реализован `buffer/k` (последние `k` сообщений).
- Добавить “summary” (сводка истории + последние сообщения).

**6) Streaming**
- `LLMClient.stream_text(...)` пока не реализован.

**7) Политики/версии**
- Реальные `VersionResolver`/`PolicyResolver` (tenant/channel), чтение из БД/конфига, аудит изменений.

---

Если нужно — можем дальше “дожать” проект:
- перевести все узлы на `LLMClient` (чтобы везде был единый вход)
- добавить OpenRouter fallback
- сделать полноценный API endpoint `/chat` под `ChatRequest`.
