from ai_domain.llm.types import LLMCredentials, LLMMessage, LLMRequest

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

        credentials = getattr(state, "credentials", None)
        llm_credentials = None
        if isinstance(credentials, dict):
            llm_credentials = LLMCredentials(openai_api_key=credentials.get("openai_api_key"))
        else:
            llm_credentials = credentials

        req = LLMRequest(
            messages=messages,
            model=state.versions["model"],
            max_output_tokens=state.policies["max_output_tokens"],
            credentials=llm_credentials,
            metadata={"trace_id": state.trace_id},
        )

        resp = await self.llm.generate(req)

        state.answer = {"text": resp.content}
        state.stage = {"current": "final"}
        return state
