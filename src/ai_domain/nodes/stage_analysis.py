from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ai_domain.utils.memory import select_memory_messages


class StageAnalysisError(Exception):
    pass


@dataclass
class StageAnalysisNode:
    llm: Any
    prompt_repo: Any
    telemetry: Any | None = None
    prompt_key: str = "analysis_prompt"  # в Supabase prompt_versions
    version_key: str = "analysis_prompt"  # ключ в state.versions

    async def __call__(self, state) -> Any:
        runtime = getattr(state, "runtime", None) or {}
        runtime.setdefault("executed", [])
        runtime.setdefault("errors", [])
        runtime.setdefault("degraded", False)
        runtime.setdefault("prompts_used", [])

        runtime["executed"].append("stage_analysis")

        channel = getattr(state, "channel", "any")
        versions: Dict[str, str] = getattr(state, "versions", {}) or {}

        prompt_version = versions.get(self.version_key) or "active"
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

        policies = getattr(state, "policies", {}) or {}
        llm_kwargs = {
            "messages": llm_messages,
            "model": versions.get("model") or None,
            "max_output_tokens": policies.get("max_output_tokens", 256),
            "temperature": policies.get("temperature", 0.2),
        }
        top_p = policies.get("top_p")
        if top_p is not None:
            llm_kwargs["top_p"] = top_p

        try:
            resp = await self.llm.generate(**llm_kwargs)
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
