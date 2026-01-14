# src/ai_domain/orchestrator/service.py
from typing import Any, Dict
from uuid import uuid4

from ai_domain.orchestrator.context_builder import build_graph_state
from ai_domain.orchestrator.policy_resolver import PolicyBundle, build_task_configs


class OrchestratorError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        code: str = "orchestrator_error",
        trace_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.trace_id = trace_id


class Orchestrator:
    def __init__(
        self,
        *,
        graph,
        idempotency,
        version_resolver,
        policy_resolver,
        telemetry,
    ):
        self.graph = graph
        self.idempotency = idempotency
        self.version_resolver = version_resolver
        self.policy_resolver = policy_resolver
        self.telemetry = telemetry

    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = request.get("trace_id") or str(uuid4())
        idempotency_key = request.get("idempotency_key")
        marked_in_progress = False

        for key in ("tenant_id", "conversation_id", "channel", "messages"):
            if key not in request or request[key] is None:
                raise OrchestratorError(
                    f"Missing required field: {key}",
                    status_code=400,
                    code="bad_request",
                    trace_id=trace_id,
                )

        # 1️⃣ Idempotency
        if idempotency_key:
            cached = await self.idempotency.get(idempotency_key)
            if cached and isinstance(cached, dict) and cached.get("status") == "in_progress":
                raise OrchestratorError(
                    "Idempotency key is already in progress",
                    status_code=409,
                    code="idempotency_in_progress",
                    trace_id=trace_id,
                )
            if cached:
                return cached

            await self.idempotency.mark_in_progress(idempotency_key, ttl_seconds=120)
            marked_in_progress = True

        try:
            # 2️⃣ Resolve versions
            versions = await self.version_resolver.resolve(
                tenant_id=request["tenant_id"],
                channel=request["channel"],
            )
            versions = dict(versions or {})

            # 3️⃣ Resolve policies
            policy_result = self.policy_resolver.resolve(
                channel=request["channel"],
            )
            if isinstance(policy_result, PolicyBundle):
                policies = dict(policy_result.policies)
                task_configs = dict(policy_result.task_configs)
            else:
                policies = dict(policy_result or {})
                task_configs = {}

            credentials = request.get("credentials")

            model_params = request.get("model_params") or {}

            # request overrides (API layer)
            if request.get("is_rag") is not None:
                policies["rag_enabled"] = bool(request["is_rag"])

            model_override = request.get("model")
            if model_override:
                versions["model"] = str(model_override)

            policies_with_model = dict(policies)
            policies_with_model["channel"] = request["channel"]
            policies_with_model.setdefault("temperature", 0.2)
            if "temperature" in model_params:
                policies_with_model["temperature"] = float(model_params["temperature"])
            if "top_p" in model_params:
                policies_with_model["top_p"] = float(model_params["top_p"])
            else:
                policies_with_model.pop("top_p", None)
            if "max_output_tokens" in model_params:
                policies_with_model["max_output_tokens"] = model_params["max_output_tokens"]
            funnel_id = request.get("funnel_id")

            if not task_configs:
                task_configs = build_task_configs(
                    versions=versions,
                    policies=policies_with_model,
                    model_override=model_override,
                    model_params=model_params,
                )

            # 4️⃣ Build graph state
            state = build_graph_state(
                tenant_id=request["tenant_id"],
                conversation_id=request["conversation_id"],
                channel=request["channel"],
                messages=request["messages"],
                versions=versions,
                policies=policies_with_model,
                credentials=credentials,
                task_configs=task_configs,
                prompt=request.get("prompt"),
                role_instruction=request.get("role_instruction"),
                is_rag=request.get("is_rag"),
                tools=request.get("tools"),
                funnel_id=funnel_id,
                request_id=idempotency_key,
                memory_strategy=request.get("memory_strategy"),
                memory_params=request.get("memory_params"),
                model_params=model_params,
                trace_id=trace_id,
                graph_name="main_graph",
            )

            # 5️⃣ Run graph
            result_state = await self.graph.ainvoke(state)

            def _get(attr, default=None):
                if hasattr(result_state, attr):
                    return getattr(result_state, attr)
                if isinstance(result_state, dict):
                    return result_state.get(attr, default)
                return default

            runtime = _get("runtime", {}) or {}
            degraded = bool(runtime.get("degraded"))

            # 6️⃣ Build response
            response = {
                "status": "ok" if not degraded else "degraded",
                "answer": _get("answer"),
                "stage": _get("stage"),
                "trace_id": trace_id,
                "versions": _get("versions"),
            }

            if idempotency_key:
                await self.idempotency.save(idempotency_key, response)

            return response

        except OrchestratorError:
            if idempotency_key and marked_in_progress:
                await self.idempotency.clear(idempotency_key)
            raise
        except Exception as e:
            self.telemetry.error(trace_id, e)
            if idempotency_key and marked_in_progress:
                await self.idempotency.clear(idempotency_key)
            raise OrchestratorError(str(e), code="internal_error", trace_id=trace_id)
        
