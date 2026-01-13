from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ai_domain.llm.observability import build_llm_metadata
from ai_domain.llm.types import LLMCallContext, LLMMessage, LLMRequest
from ai_domain.orchestrator.tasks import get_task_config
from ai_domain.utils.memory import select_memory_messages


class StageAnalysisError(Exception):
    pass


@dataclass
class StageAnalysisNode:
    llm: Any
    prompt_repo: Any
    telemetry: Any | None = None
    prompt_key: str = "analysis_prompt"  # в Supabase prompt_versions
    task_name: str = "stage_analysis"

    async def __call__(self, state) -> Any:
        runtime = getattr(state, "runtime", None) or {}
        runtime.setdefault("executed", [])
        runtime.setdefault("errors", [])
        runtime.setdefault("degraded", False)
        runtime.setdefault("prompts_used", [])

        runtime["executed"].append("stage_analysis")

        channel = getattr(state, "channel", "any")
        task_config = get_task_config(state, self.task_name)
        prompt_version = task_config.prompt_versions.get(self.prompt_key) or "active"
        try:
            system_prompt = self.prompt_repo.get_prompt(self.prompt_key, prompt_version, channel=channel)
        except Exception as e:
            # промпт не нашли -> деградация, но не падаем
            runtime["degraded"] = True
            runtime["errors"].append({"node": "stage_analysis", "type": "prompt_load_error", "msg": str(e)})
            setattr(state, "runtime", runtime)
            return state

        runtime["prompts_used"].append(
            {"prompt_key": self.prompt_key, "prompt_version": prompt_version, "channel": channel}
        )

        # messages у тебя список dict(role, content)
        memory_messages = select_memory_messages(state)
        llm_messages = [{"role": "system", "content": system_prompt}, *memory_messages]

        llm_cfg = task_config.llm
        metadata = {**(llm_cfg.metadata or {}), **build_llm_metadata(
            state=state,
            node_name="stage_analysis",
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
            runtime["errors"].append({"node": "stage_analysis", "type": "llm_error", "msg": str(e)})
            setattr(state, "runtime", runtime)
            return state

        text = (getattr(resp, "content", None) or "").strip()

        # Пытаемся распарсить JSON (если промпт так настроен)
        parsed: Optional[Dict[str, Any]] = None
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        # Кладём результаты в state
        if parsed and isinstance(parsed, dict):
            # ожидаем формат вроде:
            # { "stage": "...", "confidences": {...}, "signals": {...}, "rag_suggested": true/false, "rationale": "..." }
            setattr(state, "stage", parsed.get("stage") or getattr(state, "stage", None))
            runtime["analysis"] = parsed
            # можно мягко подсказать rag policy
            rag_suggested = parsed.get("rag_suggested")
            if rag_suggested is not None:
                policies = getattr(state, "policies", {}) or {}
                policies.setdefault("rag_enabled", bool(rag_suggested))
                setattr(state, "policies", policies)
        else:
            runtime["analysis_text"] = text

        setattr(state, "runtime", runtime)

        # telemetry (опционально)
        if self.telemetry:
            try:
                self.telemetry.log_step(
                    trace_id=getattr(state, "trace_id", None),
                    node="stage_analysis",
                    meta={"prompt_key": self.prompt_key, "prompt_version": prompt_version},
                )
            except Exception:
                # telemetry никогда не должна ломать выполнение
                pass

        return state
