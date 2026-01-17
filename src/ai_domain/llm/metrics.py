from typing import Any, Dict, Iterable


def _scrub_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, val in value.items():
            key_lower = str(key).lower()
            if any(
                token in key_lower
                for token in (
                    "api_key",
                    "apikey",
                    "secret",
                    "token",
                    "password",
                    "credentials",
                    "authorization",
                    "crypted_api_key",
                    "key",
                )
            ):
                cleaned[key] = "***"
            else:
                cleaned[key] = _scrub_sensitive(val)
        return cleaned
    if isinstance(value, list):
        return [_scrub_sensitive(item) for item in value]
    return value


def _allowlist_dict(payload: Dict[str, Any] | None, allowed: Iterable[str]) -> Dict[str, Any]:
    if not payload:
        return {}
    out: Dict[str, Any] = {}
    for key in allowed:
        if key in payload:
            out[key] = payload.get(key)
    return out


class MetricsWriter:
    def begin_trace(self, payload: Dict[str, Any]) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_step(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def end_step(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_llm_call(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_tool_call(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_rag_call(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def add_error(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self, payload: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class StateMetricsWriter(MetricsWriter):
    def __init__(self, state: Dict[str, Any]) -> None:
        self._state = state

    def _trace(self) -> Dict[str, Any]:
        return self._state.setdefault("trace", {})

    def add_llm_call(self, payload: Dict[str, Any]) -> None:
        self._trace().setdefault("llm_calls", []).append(payload)

    def add_error(self, payload: Dict[str, Any]) -> None:
        self._trace().setdefault("errors", []).append(payload)

    def add_tool_call(self, payload: Dict[str, Any]) -> None:
        _ = payload

    def add_rag_call(self, payload: Dict[str, Any]) -> None:
        _ = payload

    def begin_trace(self, payload: Dict[str, Any]) -> str | None:
        _ = payload
        return None

    def begin_step(self, payload: Dict[str, Any]) -> None:
        _ = payload

    def end_step(self, payload: Dict[str, Any]) -> None:
        _ = payload

    def finalize(self, payload: Dict[str, Any]) -> None:
        _ = payload


class CompositeMetricsWriter(MetricsWriter):
    def __init__(self, writers: Iterable[MetricsWriter]) -> None:
        self._writers = [w for w in writers if w is not None]

    def add_llm_call(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.add_llm_call(payload)

    def add_error(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.add_error(payload)

    def add_tool_call(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.add_tool_call(payload)

    def add_rag_call(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.add_rag_call(payload)

    def begin_trace(self, payload: Dict[str, Any]) -> str | None:
        run_id = None
        for writer in self._writers:
            candidate = writer.begin_trace(payload)
            if candidate:
                run_id = candidate
        return run_id

    def begin_step(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.begin_step(payload)

    def end_step(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.end_step(payload)

    def finalize(self, payload: Dict[str, Any]) -> None:
        for writer in self._writers:
            writer.finalize(payload)


class LangSmithWriter(MetricsWriter):
    def __init__(
        self,
        *,
        trace_id: str,
        project_name: str | None,
        enabled: bool,
        api_key: str | None,
        endpoint: str | None,
        inputs: Dict[str, Any] | None = None,
    ) -> None:
        self._enabled = enabled and bool(api_key)
        self._run = None
        self._inputs = inputs or {}
        self._project_name = project_name
        self._step_runs: Dict[int, Any] = {}
        self._node_runs: Dict[str, list[Any]] = {}
        self._child_runs: list[Any] = []
        if not self._enabled:
            return
        try:
            import os
            from langsmith.run_trees import RunTree  # type: ignore

            os.environ["LANGSMITH_API_KEY"] = api_key or ""
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_API_KEY"] = api_key or ""
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            if endpoint:
                os.environ["LANGSMITH_ENDPOINT"] = endpoint
                os.environ["LANGCHAIN_ENDPOINT"] = endpoint
            if project_name:
                os.environ["LANGSMITH_PROJECT"] = project_name

        except Exception:
            self._enabled = False
            self._run = None

    def begin_trace(self, payload: Dict[str, Any]) -> str | None:
        if not self._enabled:
            return None
        if self._run is not None:
            return getattr(self._run, "id", None)
        try:
            from langsmith.run_trees import RunTree  # type: ignore

            inputs = _allowlist_dict(
                payload.get("inputs") or self._inputs,
                ("input_fingerprint", "messages_count", "total_chars"),
            )
            metadata = _allowlist_dict(
                payload.get("metadata"),
                (
                    "trace_id",
                    "tenant_id",
                    "channel",
                    "graph_name",
                    "graph_version",
                    "versions",
                    "route",
                ),
            )
            self._run = RunTree(
                name=payload.get("name") or "ai_request",
                run_type="chain",
                inputs=_scrub_sensitive(inputs),
                project_name=self._project_name,
                metadata=_scrub_sensitive(metadata),
            )
            self._run.post()
            return getattr(self._run, "id", None)
        except Exception:
            self._run = None
            return None

    def begin_step(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            node = payload.get("node") or "step"
            index = int(payload.get("index") or 0)
            child = self._run.create_child(
                name=node,
                run_type="chain",
                inputs=_scrub_sensitive({"node": node}),
                metadata=_scrub_sensitive({"started_at": payload.get("started_at")}),
            )
            child.post()
            self._step_runs[index] = child
            self._node_runs.setdefault(node, []).append(child)
            self._child_runs.append(child)
        except Exception:
            return

    def end_step(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            index = int(payload.get("index") or 0)
            node_run = self._step_runs.get(index)
            if node_run is None:
                return
            node_run.end(
                outputs={
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                }
            )
            node_run.patch()
        except Exception:
            return

    def add_llm_call(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            node = payload.get("node") or "llm_call"
            parent = self._run
            runs = self._node_runs.get(node)
            if runs:
                parent = runs[-1]
            child = parent.create_child(
                name="llm_call",
                run_type="llm",
                inputs=_scrub_sensitive(
                    {
                        "messages": payload.get("messages"),
                        "model": payload.get("model"),
                        "provider": payload.get("provider"),
                    }
                ),
                metadata=_scrub_sensitive(
                    {
                        "structured": payload.get("structured"),
                        "task": payload.get("task"),
                        "response_format": payload.get("response_format"),
                    }
                ),
            )
            child.post()
            child.end(
                outputs=_scrub_sensitive(
                    {
                        "latency_ms": payload.get("latency_ms"),
                        "usage": payload.get("usage"),
                        "finish_reason": payload.get("finish_reason"),
                        "output_fingerprint": payload.get("output_fingerprint"),
                        "completion_chars": payload.get("output_chars"),
                    }
                )
            )
            child.patch()
            self._child_runs.append(child)
        except Exception:
            return

    def add_tool_call(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            node = payload.get("node") or "tool"
            parent = self._run
            runs = self._node_runs.get(node)
            if runs:
                parent = runs[-1]
            child = parent.create_child(
                name=f"tool:{payload.get('tool_name') or 'tool'}",
                run_type="tool",
                inputs=_scrub_sensitive(
                    {
                        "args_fingerprint": payload.get("args_fingerprint"),
                        "args_preview": payload.get("args_preview"),
                    }
                ),
                metadata=_scrub_sensitive(
                    {
                        "tool_name": payload.get("tool_name"),
                        "ok": payload.get("ok"),
                        "latency_ms": payload.get("latency_ms"),
                        "call_id": payload.get("call_id"),
                    }
                ),
            )
            child.post()
            child.end(
                outputs=_scrub_sensitive(
                    {
                        "result_fingerprint": payload.get("result_fingerprint"),
                        "result_preview": payload.get("result_preview"),
                    }
                )
            )
            child.patch()
            self._child_runs.append(child)
        except Exception:
            return

    def add_rag_call(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            node = payload.get("node") or "retrieve"
            parent = self._run
            runs = self._node_runs.get(node)
            if runs:
                parent = runs[-1]
            child = parent.create_child(
                name="rag_retrieve",
                run_type="retriever",
                inputs=_scrub_sensitive(
                    {
                        "query_fingerprint": payload.get("query_fingerprint"),
                        "query_preview": payload.get("query_preview"),
                    }
                ),
                metadata=_scrub_sensitive(payload.get("metadata") or {}),
            )
            child.post()
            child.end(
                outputs=_scrub_sensitive(
                    {
                        "docs_fingerprint": payload.get("docs_fingerprint"),
                        "docs_count": payload.get("docs_count"),
                    }
                )
            )
            child.patch()
            self._child_runs.append(child)
        except Exception:
            return

    def add_error(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            child = self._run.create_child(
                name=payload.get("node") or "error",
                run_type="tool",
                inputs=_scrub_sensitive({"error": payload.get("error")}),
            )
            child.post()
            child.end(error=str(payload.get("error") or "error"))
            child.patch()
            self._child_runs.append(child)
        except Exception:
            return

    def finalize(self, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            outputs = _allowlist_dict(
                payload,
                ("trace_id", "status", "route", "degraded", "total_latency_ms"),
            )
            self._run.end(outputs=_scrub_sensitive(outputs))
            self._run.patch()
            for child in list(self._child_runs):
                try:
                    child.wait()
                except Exception:
                    continue
            try:
                self._run.wait()
            except Exception:
                pass
        except Exception:
            return
