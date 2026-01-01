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


            # 4️⃣ Build graph state
            state = build_graph_state(
                tenant_id=request["tenant_id"],
                conversation_id=request["conversation_id"],
                channel=request["channel"],
                messages=request["messages"],
                versions=versions,
                policies=policies,
                credentials=credentials,
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
