from __future__ import annotations


class NoOpTelemetry:
    def log_step(self, *, trace_id: str | None, node: str, meta: dict):  # noqa: ARG002
        return None

    def event(self, name: str, payload: dict):  # noqa: ARG002
        return None

    def error(self, trace_id: str, exc: Exception):  # noqa: ARG002
        return None

