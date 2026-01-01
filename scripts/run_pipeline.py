# src/ai_domain/fixtures/dentistry_one_dialog.py

DENTISTRY_DIALOG = [
    {"role": "user", "content": "Здравствуйте! Хочу записаться на чистку зубов. Сколько стоит и есть ли окна на этой неделе после 17:00?"},
    {"role": "assistant", "content": "Здравствуйте! Да, подскажите, пожалуйста: вам нужна профессиональная гигиена (чистка + полировка) и вы готовы к визиту в будни после 17:00?"},
    {"role": "user", "content": "Да, профессиональная. Лучше во вторник или четверг вечером."},
]

PROMPTS = {
    # Общий системный стиль
    "system_prompt": (
        "Ты администратор стоматологической клиники. "
        "Отвечай вежливо, коротко и по делу. "
        "Твоя цель: уточнить данные и помочь записаться."
    ),
    # Анализ стадии/намерения: ВАЖНО — строго JSON
    "analysis_prompt": (
        "Верни СТРОГО JSON без пояснений.\n"
        "Определи:\n"
        "- stage: один из [lead, booking, confirmation, post_visit]\n"
        "- intent: краткий ключ (например booking_cleaning)\n"
        "- missing: список недостающих полей для записи (name, phone, date, time, doctor_preference)\n"
        "- rag_suggested: true/false\n"
        "Формат:\n"
        '{"stage":"...","intent":"...","missing":["..."],"rag_suggested":false}'
    ),
    # Инструменты: в этом прогоне инструменты не используем (должно вернуть null)
    "tool_prompt": (
        "Если нужен инструмент — верни JSON вида "
        '{"tool":"name","args":{...}}. '
        "Если не нужен — верни null. СТРОГО без пояснений."
    ),
    # Финальная генерация ответа клиенту
    "final_prompt": (
        "Составь ответ клиенту от лица администратора стоматологии.\n"
        "1) Дай ориентир по стоимости (если нет точных цен — укажи, что зависит от объёма и предложи диапазон).\n"
        "2) Предложи 2-3 варианта времени на этой неделе после 17:00.\n"
        "3) Спроси 2-3 уточнения для записи (имя, телефон, предпочтение врача/языка).\n"
        "Ответ короткий, дружелюбный, без лишних деталей."
    ),
}

# scripts/run_dentistry_pipeline.py
import os
import asyncio

from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.llm.retry import RetryPolicy
from ai_domain.llm.routing import LLMRouter, ModelRoute
from ai_domain.llm.types import LLMMessage, LLMRequest

from ai_domain.orchestrator.context_builder import normalize_messages
from ai_domain.orchestrator.service import Orchestrator
from ai_domain.registry.static_prompt_repo import StaticPromptRepo
from ai_domain.registry.prompts import PromptRepository
from ai_domain.registry.supabase_connector import create_supabase_client_from_env
from ai_domain.secrets import get_secret

from ai_domain.telemetry.noop import NoOpTelemetry
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Deps:
    llm: Any
    prompt_repo: Any
    rag_client: Optional[Any] = None
    telemetry: Optional[Any] = None
    tool_executor: Optional[Any] = None


# --- такие же inline зависимости, как в твоём live-тесте ---

class InlineIdempotency:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def mark_in_progress(self, key):
        self.store[key] = None

    async def save(self, key, value):
        self.store[key] = value

    async def clear(self, key):
        self.store.pop(key, None)


class InlineVersionResolver:
    async def resolve(self, tenant_id: str, channel: str):
        return {
            "system_prompt": "v1",
            "analysis_prompt": "v1",
            "tool_prompt": "v1",
            "final_prompt": "v1",
            "rag_config_id": "rc1",
            "model": "gpt-4.1-mini",
        }


class InlinePolicyResolver:
    def resolve(self, channel: str):
        return {
            "rag_enabled": False,
            "rag_top_k": 2,
            "max_output_tokens": 180,
            "voice_ssml": False,
        }


class InlineTelemetry:
    def error(self, trace_id, exc):
        pass


class RouterLLMAdapter:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def generate(self, *args, **kwargs):
        if args and len(args) == 1 and not kwargs:
            req = args[0]
        else:
            req = LLMRequest(
                messages=[LLMMessage(**m) for m in kwargs["messages"]],
                model=kwargs["model"],
                max_output_tokens=kwargs.get("max_output_tokens", 256),
                temperature=kwargs.get("temperature", 0.2),
            )
        return await self.router.generate(req)

    async def decide_tool(self, _):
        return None


class GraphWrapper:
    def __init__(self, graph):
        self.graph = graph

    async def invoke(self, state):
        res = await self.graph.ainvoke(state)
        if isinstance(res, dict):
            class Wrapper:
                def __init__(self, data):
                    self.__dict__.update(data)

            return Wrapper(res)
        return res


def make_deps(llm):
    # Если заданы SUPABASE_URL/SUPABASE_KEY — используем Supabase как источник промптов.
    # Иначе — fallback на статические PROMPTS из этого файла.
    # if get_secret("SUPABASE_URL") and get_secret("SUPABASE_KEY"):
    #     sb = create_supabase_client_from_env()
    #     prompt_repo = PromptRepository(sb)
    if False:
        pass
    else:
        # Важно: ключи промптов должны совпадать с тем, что ожидают узлы.
        # Если у тебя узлы читают только system/analysis/tool — оставь минимум.
        prompt_repo = StaticPromptRepo(
            {
                "system_prompt": PROMPTS["system_prompt"],
                "analysis_prompt": PROMPTS["analysis_prompt"],
                "tool_prompt": PROMPTS["tool_prompt"],
                "final_prompt": PROMPTS["final_prompt"],
            }
        )

    return Deps(
        llm=llm,
        prompt_repo=prompt_repo,
        rag_client=None,
        telemetry=NoOpTelemetry(),
        tool_executor=None,
    )


def build_live_router_llm():
    provider = OpenAIProvider(platform_api_key=get_secret("OPENAI_API_KEY", required=True))
    router = LLMRouter(
        providers={"openai": provider},
        route=ModelRoute(
            primary_provider="openai",
            primary_model="gpt-4.1-mini",
            retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.3, max_delay_s=1.0),
        ),
        limiter=ConcurrencyLimiter(max_inflight=1),
        breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=5),
    )
    return RouterLLMAdapter(router)


async def main():
    # Проверка вынесена на get_secret(required=True) в build_live_router_llm().

    from ai_domain.graphs.main_graph import build_graph

    llm = build_live_router_llm()
    deps = make_deps(llm)
    graph = build_graph(deps=deps)

    orchestrator = Orchestrator(
        graph=GraphWrapper(graph),
        idempotency=InlineIdempotency(),
        version_resolver=InlineVersionResolver(),
        policy_resolver=InlinePolicyResolver(),
        telemetry=InlineTelemetry(),
    )

    request = {
        "tenant_id": "tenant",
        "conversation_id": "dent-live-001",
        "channel": "chat",
        "messages": normalize_messages(DENTISTRY_DIALOG),
        "idempotency_key": "dent-live-001",
        # если у тебя orchestrator/state поддерживает meta — можно добавить:
        "meta": {"domain": "dentistry", "clinic_name": "Стоматология Улыбка+"},
    }

    resp = await orchestrator.run(request)

    print("STATUS:", resp.get("status"))
    print("STAGE:", resp.get("stage"))
    print("ANSWER:", (resp.get("answer") or {}).get("text"))


if __name__ == "__main__":
    asyncio.run(main())
