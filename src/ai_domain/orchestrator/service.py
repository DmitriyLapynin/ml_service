# src/ai_domain/orchestrator/service.py
from typing import Any, Dict
from uuid import uuid4

from ai_domain.orchestrator.context_builder import build_graph_state


class OrchestratorError(Exception):
    pass


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
        trace_id = str(uuid4())
        idempotency_key = request.get("idempotency_key")

        # 1️⃣ Idempotency
        if idempotency_key:
            cached = await self.idempotency.get(idempotency_key)
            if cached:
                return cached

            await self.idempotency.mark_in_progress(idempotency_key)

        try:
            # 2️⃣ Resolve versions
            versions = await self.version_resolver.resolve(
                tenant_id=request["tenant_id"],
                channel=request["channel"],
            )

            # 3️⃣ Resolve policies
            policies = self.policy_resolver.resolve(
                channel=request["channel"],
            )

            credentials = self._decrypt_credentials(request.get("credentials"))

            model_params = request.get("model_params") or {}

            # request overrides (API layer)
            if request.get("is_rag") is not None:
                policies["rag_enabled"] = bool(request["is_rag"])

            model_override = request.get("model")
            if model_override:
                versions["model"] = str(model_override)

            policies_with_model = dict(policies)
            policies_with_model.setdefault("temperature", 0.2)
            if "temperature" in model_params:
                policies_with_model["temperature"] = float(model_params["temperature"])
            if "top_p" in model_params:
                policies_with_model["top_p"] = float(model_params["top_p"])
            else:
                policies_with_model.pop("top_p", None)
            if "max_output_tokens" in model_params:
                policies_with_model["max_output_tokens"] = model_params["max_output_tokens"]

            # 4️⃣ Build graph state
            state = build_graph_state(
                tenant_id=request["tenant_id"],
                conversation_id=request["conversation_id"],
                channel=request["channel"],
                messages=request["messages"],
                versions=versions,
                policies=policies_with_model,
                credentials=credentials,
                prompt=request.get("prompt"),
                role_instruction=request.get("role_instruction"),
                is_rag=request.get("is_rag"),
                tools=request.get("tools"),
                funnel_id=request.get("funnel_id"),
                memory_strategy=request.get("memory_strategy"),
                memory_params=request.get("memory_params"),
                model_params=model_params,
                trace_id=trace_id,
            )

            # 5️⃣ Run graph
            result_state = await self.graph.invoke(state)

            # 6️⃣ Build response
            response = {
                "status": "ok" if not result_state.runtime["degraded"] else "degraded",
                "answer": result_state.answer,
                "stage": result_state.stage,
                "trace_id": trace_id,
                "versions": result_state.versions,
            }

            if idempotency_key:
                await self.idempotency.save(idempotency_key, response)

            return response

        except Exception as e:
            self.telemetry.error(trace_id, e)
            if idempotency_key:
                await self.idempotency.clear(idempotency_key)
            raise OrchestratorError(str(e))
        
    def _decrypt_credentials(self, encrypted: dict | None) -> dict | None:
        if not encrypted:
            return None

        # пример, замени на свой KMS / vault
        return {
            "openai_api_key": self.crypto.decrypt(encrypted["openai"]),
            "source": encrypted.get("source", "campaign"),
        }
