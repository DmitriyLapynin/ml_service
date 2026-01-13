from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping

from ai_domain.llm.client import ResolvedLLMConfig


TaskName = Literal[
    "stage_analysis",
    "final_answer",
    "tools_loop",
    "memory_summary",
    "safety_classifier",
]


@dataclass(frozen=True)
class TaskSpec:
    task: TaskName


@dataclass(frozen=True)
class ResolvedTaskConfig:
    task: TaskName
    llm: ResolvedLLMConfig
    prompt_versions: Mapping[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_task_config(state: Any, task: TaskName) -> ResolvedTaskConfig:
    configs = getattr(state, "task_configs", None) or {}
    if task not in configs:
        raise KeyError(f"Task config missing: {task}")
    return configs[task]
