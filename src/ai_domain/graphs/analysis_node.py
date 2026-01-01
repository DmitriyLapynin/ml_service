from ai_domain.llm.types import LLMMessage, LLMRequest

class AnalysisNode:
    def __init__(self, llm, prompt_repo, telemetry):
        self.llm = llm
        self.prompts = prompt_repo
        self.telemetry = telemetry

    async def __call__(self, state):
        prompt_key = "analysis_prompt"
        version = state.versions[prompt_key]

        prompt = self.prompts.get_prompt(
            prompt_key=prompt_key,
            version=version,
        )

        self.telemetry.event(
            "prompt_used",
            {
                "trace_id": state.trace_id,
                "prompt_key": prompt_key,
                "prompt_version": version,
            },
        )

        req = LLMRequest(
            messages=[
                LLMMessage(role="system", content=prompt),
                *[LLMMessage(**m) for m in state.messages],
            ],
            model=state.versions["model"],
            max_output_tokens=256,
            metadata={"trace_id": state.trace_id},
        )

        resp = await self.llm.generate(req)

        state.runtime["analysis"] = resp.content
        return state