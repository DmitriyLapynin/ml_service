from __future__ import annotations

from typing import Any, Dict, Mapping


def build_llm_metadata(
    *,
    state: Any,
    node_name: str,
    task: str,
    prompt_key: str | None = None,
    prompt_version: str | None = None,
    prompt_vars: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "trace_id": getattr(state, "trace_id", None),
        "graph_name": getattr(state, "graph_name", None),
        "node_name": node_name,
        "task": task,
    }
    if prompt_key:
        metadata["prompt_key"] = prompt_key
    if prompt_version:
        metadata["prompt_version"] = prompt_version
    if prompt_vars:
        metadata["prompt_vars"] = dict(prompt_vars)
    return metadata
