from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ai_domain.llm.client import ResolvedLLMConfig
from ai_domain.orchestrator.tasks import ResolvedTaskConfig, TaskName


TASK_PROMPT_KEYS: Mapping[TaskName, list[str]] = {
    "stage_analysis": ["analysis_prompt"],
    "final_answer": ["system_prompt"],
    "tools_loop": ["tool_prompt"],
    "memory_summary": ["memory_summary_prompt"],
    "safety_classifier": [],
}


@dataclass(frozen=True)
class PolicyBundle:
    policies: Dict[str, Any]
    task_configs: Dict[str, ResolvedTaskConfig]


def build_task_configs(
    *,
    versions: Mapping[str, str],
    policies: Mapping[str, Any],
    model_override: str | None,
    model_params: Mapping[str, Any],
    default_provider: str = "openai",
    default_model: str = "gpt-4.1-mini",
) -> Dict[str, ResolvedTaskConfig]:
    provider = str(model_params.get("provider") or policies.get("provider") or default_provider)
    model = str(model_override or versions.get("model") or default_model)

    base_temperature = float(model_params.get("temperature") or policies.get("temperature") or 0.2)
    base_top_p = model_params.get("top_p")
    if base_top_p is None:
        base_top_p = policies.get("top_p")
    max_tokens = model_params.get("max_output_tokens") or policies.get("max_output_tokens") or 2048

    base = ResolvedLLMConfig(
        provider=provider,
        model=model,
        temperature=base_temperature,
        top_p=base_top_p,
        max_tokens=int(max_tokens),
        retries=int(model_params.get("retries") or policies.get("retries") or 2),
        seed=model_params.get("seed"),
        metadata=dict(model_params.get("metadata") or {}),
    )

    task_overrides = policies.get("task_overrides") or {}
    task_configs: Dict[str, ResolvedTaskConfig] = {}
    for task, prompt_keys in TASK_PROMPT_KEYS.items():
        overrides = task_overrides.get(task, {})
        llm = ResolvedLLMConfig(
            provider=str(overrides.get("provider") or base.provider),
            model=str(overrides.get("model") or base.model),
            temperature=float(overrides.get("temperature") or base.temperature),
            top_p=overrides.get("top_p", base.top_p),
            max_tokens=int(overrides.get("max_tokens") or base.max_tokens),
            retries=int(overrides.get("retries") or base.retries),
            seed=overrides.get("seed", base.seed),
            tags=list(overrides.get("tags") or base.tags),
            metadata={**base.metadata, **(overrides.get("metadata") or {})},
        )
        prompt_versions = {k: (versions.get(k) or "active") for k in prompt_keys}
        task_configs[task] = ResolvedTaskConfig(
            task=task,
            llm=llm,
            prompt_versions=prompt_versions,
            metadata={"channel": policies.get("channel")},
        )

    return task_configs
