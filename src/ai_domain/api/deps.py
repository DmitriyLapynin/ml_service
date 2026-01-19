from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Any

from ai_domain.api.config import APISettings
from ai_domain.fakes.fake_idempotency import FakeIdempotency
from ai_domain.fakes.fake_policy_resolver import FakePolicyResolver
from ai_domain.fakes.fake_telemetry import FakeTelemetry
from ai_domain.fakes.fake_version_resolver import FakeVersionResolver
from ai_domain.agent.graph import build_agent_graph
from ai_domain.agent.nodes import AgentNodes
from ai_domain.llm.circuit_breaker import CircuitBreaker
from ai_domain.llm.client_cache import TTLRUClientCache
from ai_domain.llm.client import LLMClient
from ai_domain.llm.metrics import CompositeMetricsWriter, LangSmithWriter, StateMetricsWriter
from ai_domain.llm.openai_provider import OpenAIProvider
from ai_domain.llm.rate_limit import ConcurrencyLimiter
from ai_domain.orchestrator.service import Orchestrator
from ai_domain.orchestrator.context_builder import normalize_messages
import os

from ai_domain.rag.embedder import LocalEmbedder
from ai_domain.rag.funnel_store import FunnelKBResolver, FunnelKBStore
from ai_domain.registry.prompts import PromptRepository, PromptNotFound
from ai_domain.registry.static_prompt_repo import StaticPromptRepo
from ai_domain.registry.supabase_connector import create_supabase_client_from_env, SupabaseConfigError
from ai_domain.secrets import get_secret
from ai_domain.tools.registry import ToolRegistry, default_registry


@dataclass
class AppDeps:
    llm: LLMClient
    prompt_repo: Any
    tool_executor: Any
    telemetry: Any
    rag_client: Any
    tool_registry: ToolRegistry
    kb_resolver: FunnelKBResolver
    orchestrator: Orchestrator


class ToolExecutorAdapter:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(self, *, tool_name: str, args: dict, state):
        return await self._registry.execute(
            tool_name,
            args,
            state=state.__dict__ if hasattr(state, "__dict__") else dict(state),
            trace_id=getattr(state, "trace_id", None),
        )


class FallbackPromptRepo:
    def __init__(self, primary, fallback):
        self._primary = primary
        self._fallback = fallback

    def get_prompt(self, *args, **kwargs) -> str:
        try:
            return self._primary.get_prompt(*args, **kwargs)
        except PromptNotFound:
            return self._fallback.get_prompt(*args, **kwargs)


class ResolverRagClient:
    def __init__(self, resolver: FunnelKBResolver):
        self._resolver = resolver

    async def search(self, *, query: str, top_k: int = 5, rag_config_id: str | None = None):
        if not rag_config_id:
            return []
        return await self._resolver.search(
            funnel_id=str(rag_config_id),
            query=query,
            top_k=top_k,
            top_k_per_doc=top_k,
        )


@lru_cache
def get_settings() -> APISettings:
    return APISettings.from_env()


@lru_cache
def get_llm_client() -> LLMClient:
    provider = OpenAIProvider(platform_api_key=get_secret("OPENAI_API_KEY", required=True))
    provider.byok_cache = TTLRUClientCache(ttl_seconds=600, max_size=64)
    return LLMClient(
        providers={"openai": provider},
        default_provider="openai",
        default_model="gpt-4.1-mini",
        limiter=ConcurrencyLimiter(max_inflight=5),
        breaker=CircuitBreaker(failure_threshold=5, reset_timeout_s=5),
    )


@lru_cache
def get_prompt_repo():
    try:
        sb = create_supabase_client_from_env()
        primary = PromptRepository(sb)
        fallback = StaticPromptRepo(
            prompts={
                "analysis_prompt": "Analyze the request and decide next step.",
                "system_prompt": "You are a helpful assistant.",
                "tool_prompt": "Decide whether to call a tool.",
                "memory_summary_prompt": "Summarize the conversation.",
            }
        )
        return FallbackPromptRepo(primary, fallback)
    except SupabaseConfigError:
        return StaticPromptRepo(
            prompts={
                "analysis_prompt": "Analyze the request and decide next step.",
                "system_prompt": "You are a helpful assistant.",
                "tool_prompt": "Decide whether to call a tool.",
                "memory_summary_prompt": "Summarize the conversation.",
            }
        )


@lru_cache
def get_kb_resolver() -> FunnelKBResolver:
    settings = get_settings()
    models_dir = os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models")
    embedder = LocalEmbedder(model_path=str(Path(models_dir) / "rubert-mini-frida"))
    return FunnelKBResolver(base_dir=settings.rag_base_dir, embedder=embedder)


@lru_cache
def get_kb_store() -> FunnelKBStore:
    settings = get_settings()
    models_dir = os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models")
    embedder = LocalEmbedder(model_path=str(Path(models_dir) / "rubert-mini-frida"))
    return FunnelKBStore(base_dir=settings.rag_base_dir, embedder=embedder)


@lru_cache
def get_supabase_client() -> Any | None:
    try:
        return create_supabase_client_from_env()
    except SupabaseConfigError:
        return None


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return default_registry(max_concurrency_global=20)


@lru_cache
def get_orchestrator() -> Orchestrator:
    llm = get_llm_client()
    telemetry = FakeTelemetry()
    tool_registry = get_tool_registry()
    kb_resolver = get_kb_resolver()
    rag_client = ResolverRagClient(kb_resolver)
    graph = build_agent_graph(AgentNodes())

    def _build_agent_state(
        *,
        request: dict,
        versions: dict,
        policies: dict,
        credentials: dict | None,
        task_configs: dict,
        trace_id: str,
        idempotency_key: str | None,
        graph_name: str,
    ) -> dict:
        _ = versions, task_configs
        model_name = request.get("model") or "gpt-4.1-mini"
        settings = get_settings()
        trace = {
            "steps": [],
            "llm_calls": [],
            "tool_calls": [],
            "rag": None,
            "errors": [],
        }
        state = {
            "trace_id": trace_id,
            "tenant_id": request["tenant_id"],
            "channel": request["channel"],
            "request_id": idempotency_key,
            "graph": graph_name,
            "messages": normalize_messages(request["messages"]),
            "policies": policies,
            "prompt": request.get("prompt"),
            "role_instruction": request.get("role_instruction") or "",
            "user_instruction": request.get("prompt"),
            "is_rag": bool(request.get("is_rag", False)),
            "tools": request.get("tools") or None,
            "funnel_id": request.get("funnel_id"),
            "model": model_name,
            "model_params": request.get("model_params") or {},
            "memory_strategy": request.get("memory_strategy"),
            "memory_params": request.get("memory_params") or {},
            "credentials": credentials or {},
            "llm": llm,
            "safety_llm": llm,
            "safety_model": "gpt-4.1-nano",
            "tool_registry": tool_registry,
            "kb_resolver": kb_resolver,
            "rag_client": rag_client,
            "trace": trace,
            "runtime": {"degraded": False, "errors": []},
        }
        langsmith_api_key = get_secret("LANGSMITH_API_KEY") or get_secret("LANGCHAIN_API_KEY")
        langsmith_writer = LangSmithWriter(
            trace_id=trace_id,
            project_name=settings.langsmith_project,
            enabled=True,
            api_key=langsmith_api_key,
            endpoint=settings.langsmith_endpoint,
            inputs={
                "trace_id": trace_id,
                "channel": request.get("channel"),
            },
        )
        state["metrics_writer"] = CompositeMetricsWriter(
            [StateMetricsWriter(state), langsmith_writer]
        )
        return state

    return Orchestrator(
        graph=graph,
        idempotency=FakeIdempotency(),
        version_resolver=FakeVersionResolver(),
        policy_resolver=FakePolicyResolver(),
        telemetry=telemetry,
        state_builder=_build_agent_state,
        graph_name="agent_graph",
    )


def get_app_deps() -> AppDeps:
    llm = get_llm_client()
    prompt_repo = get_prompt_repo()
    telemetry = FakeTelemetry()
    tool_registry = get_tool_registry()
    tool_executor = ToolExecutorAdapter(tool_registry)
    kb_resolver = get_kb_resolver()
    rag_client = ResolverRagClient(kb_resolver)
    orchestrator = get_orchestrator()
    return AppDeps(
        llm=llm,
        prompt_repo=prompt_repo,
        tool_executor=tool_executor,
        telemetry=telemetry,
        rag_client=rag_client,
        tool_registry=tool_registry,
        kb_resolver=kb_resolver,
        orchestrator=orchestrator,
    )
