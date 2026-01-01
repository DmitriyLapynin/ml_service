from ai_domain.llm.types import LLMMessage, LLMRequest

class FinalAnswerNode:
    def __init__(self, llm, prompt_repo, telemetry):
        self.llm = llm
        self.prompts = prompt_repo
        self.telemetry = telemetry

    async def __call__(self, state):
        prompt_key = "system_prompt"
        version = state.versions[prompt_key]

        system_prompt = self.prompts.get_prompt(
            prompt_key=prompt_key,
            version=version,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            *[LLMMessage(**m) for m in state.messages],
        ]

        req = LLMRequest(
            messages=messages,
            model=state.versions["model"],
            max_output_tokens=state.policies["max_output_tokens"],
            credentials=state.credentials,
            metadata={"trace_id": state.trace_id},
        )

        resp = await self.llm.generate(req)

        state.answer = {"text": resp.content}
        state.stage = {"current": "final"}
        return state