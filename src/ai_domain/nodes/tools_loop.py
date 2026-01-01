class ToolsLoopNode:
    def __init__(self, llm, prompt_repo, tool_executor, telemetry, *, max_iters=3):
        self.llm = llm
        self.prompts = prompt_repo
        self.tools = tool_executor
        self.telemetry = telemetry
        self.max_iters = max_iters

    async def __call__(self, state):
        prompt_key = "tool_prompt"
        version = state.versions[prompt_key]

        prompt = self.prompts.get_prompt(prompt_key=prompt_key, version=version)

        for _ in range(self.max_iters):
            req = {
                "messages": [
                    {"role": "system", "content": prompt},
                    *state.messages,
                ],
                "analysis": state.runtime.get("analysis"),
            }

            decision = await self.llm.decide_tool(req)

            if not decision:
                break

            result = await self.tools.execute(
                tool_name=decision.name,
                args=decision.args,
                state=state,
            )

            state.runtime.setdefault("tool_results", []).append(
                {
                    "tool": decision.name,
                    "result": result,
                }
            )

        return state