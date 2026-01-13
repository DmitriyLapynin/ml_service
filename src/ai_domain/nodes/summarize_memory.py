from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ai_domain.llm.observability import build_llm_metadata
from ai_domain.llm.types import LLMCallContext, LLMMessage, LLMRequest
from ai_domain.orchestrator.tasks import get_task_config
from ai_domain.utils.hashing import hash_message_contents
from ai_domain.utils.memory import estimate_message_tokens


@dataclass
class SummarizeMemoryNode:
    llm: Any
    prompt_repo: Any
    telemetry: Any | None = None
    prompt_key: str = "memory_summary_prompt"
    task_name: str = "memory_summary"

    async def __call__(self, state) -> Any:
        runtime = getattr(state, "runtime", None) or {}
        runtime.setdefault("executed", [])
        runtime.setdefault("errors", [])
        runtime.setdefault("degraded", False)

        runtime["executed"].append("memory_summary")

        if getattr(state, "memory_strategy", None) != "summary":
            setattr(state, "runtime", runtime)
            return state

        messages: List[Dict[str, str]] = getattr(state, "messages", []) or []
        if not messages:
            setattr(state, "runtime", runtime)
            return state

        params = getattr(state, "memory_params", {}) or {}
        trigger_tokens = int(params.get("summary_trigger_tokens", 1200))
        token_estimate = estimate_message_tokens(messages)
        if token_estimate < trigger_tokens:
            setattr(state, "runtime", runtime)
            return state

        task_config = get_task_config(state, self.task_name)
        prompt_version = task_config.prompt_versions.get(self.prompt_key) or "active"

        try:
            system_prompt = self.prompt_repo.get_prompt(self.prompt_key, prompt_version, channel=state.channel)
        except Exception as e:
            runtime["degraded"] = True
            runtime["errors"].append({"node": "memory_summary", "type": "prompt_load_error", "msg": str(e)})
            setattr(state, "runtime", runtime)
            return state

        message_hash = hash_message_contents([m.get("content") or "" for m in messages])
        existing_meta = getattr(state.memory, "summary_meta", {}) or {}
        if existing_meta.get("source_message_hash") == message_hash:
            setattr(state, "runtime", runtime)
            return state

        llm_cfg = task_config.llm
        llm_messages = [{"role": "system", "content": system_prompt}, *messages]
        metadata = {**(llm_cfg.metadata or {}), **build_llm_metadata(
            state=state,
            node_name="memory_summary",
            task=self.task_name,
            prompt_key=self.prompt_key,
            prompt_version=prompt_version,
        )}
        context = LLMCallContext(
            trace_id=getattr(state, "trace_id", None),
            graph=getattr(state, "graph_name", None),
            node=self.__class__.__name__,
            task=self.task_name,
            channel=getattr(state, "channel", None),
            tenant_id=getattr(state, "tenant_id", None),
            request_id=getattr(state, "request_id", None),
        )
        req = LLMRequest(
            messages=[LLMMessage(role=m["role"], content=m["content"]) for m in llm_messages],
            model=llm_cfg.model,
            max_output_tokens=int(llm_cfg.max_tokens),
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            seed=llm_cfg.seed,
            metadata=metadata,
        )

        try:
            resp = await self.llm.generate(req, context=context)
        except Exception as e:
            runtime["degraded"] = True
            runtime["errors"].append({"node": "memory_summary", "type": "llm_error", "msg": str(e)})
            setattr(state, "runtime", runtime)
            return state

        summary = (getattr(resp, "content", None) or "").strip()
        if summary:
            state.memory.summary = summary
            state.memory.summary_meta = {
                "source_message_hash": message_hash,
                "source_message_count": len(messages),
                "prompt_key": self.prompt_key,
                "prompt_version": prompt_version,
            }

        setattr(state, "runtime", runtime)
        return state
