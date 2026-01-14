from ai_domain.orchestrator.tasks import get_task_config


class ToolsLoopNode:
    def __init__(self, llm, prompt_repo, tool_executor, telemetry, *, max_iters=3):
        self.llm = llm
        self.prompts = prompt_repo
        self.tools = tool_executor
        self.telemetry = telemetry
        self.max_iters = max_iters

    async def __call__(self, state):
        prompt_key = "tool_prompt"
        task_config = get_task_config(state, "tools_loop")
        version = task_config.prompt_versions.get(prompt_key) or "active"

        prompt = self.prompts.get_prompt(prompt_key=prompt_key, version=version)

        def _get(attr, default=None):
            if hasattr(state, attr):
                return getattr(state, attr)
            if isinstance(state, dict):
                return state.get(attr, default)
            return default

        def _set(attr, value):
            if hasattr(state, attr):
                setattr(state, attr, value)
            elif isinstance(state, dict):
                state[attr] = value

        runtime = _get("runtime", {}) or {}
        messages = _get("messages", []) or []

        for _ in range(self.max_iters):
            req = {
                "messages": [
                    {"role": "system", "content": prompt},
                    *messages,
                ],
                "analysis": runtime.get("analysis"),
            }

            decision = await self.llm.decide_tool(req)

            if not decision:
                break

            result = await self.tools.execute(
                tool_name=decision.name,
                args=decision.args,
                state=state,
            )

            runtime.setdefault("tool_results", []).append(
                {
                    "tool": decision.name,
                    "result": result,
                }
            )

        _set("runtime", runtime)
        return state
