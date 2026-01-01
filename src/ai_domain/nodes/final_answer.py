from ai_domain.llm.types import LLMCredentials, LLMMessage, LLMRequest
from ai_domain.utils.memory import select_memory_messages

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

        role_instruction = getattr(state, "role_instruction", None)
        prompt = getattr(state, "prompt", None)
        if role_instruction:
            system_prompt = f"{system_prompt}\n\nРоль:\n{role_instruction}"
        if prompt:
            system_prompt = f"{system_prompt}\n\nИнструкция:\n{prompt}"

        messages = [
            LLMMessage(role="system", content=system_prompt),
            *[LLMMessage(**m) for m in select_memory_messages(state)],
        ]

        credentials = getattr(state, "credentials", None)
        llm_credentials = None
        if isinstance(credentials, dict):
            llm_credentials = LLMCredentials(openai_api_key=credentials.get("openai_api_key"))
        else:
            llm_credentials = credentials

        policies = getattr(state, "policies", {}) or {}
        req = LLMRequest(
            messages=messages,
            model=state.versions["model"],
            max_output_tokens=policies.get("max_output_tokens"),
            temperature=policies.get("temperature", 0.2),
            top_p=policies.get("top_p"),
            credentials=llm_credentials,
            metadata={"trace_id": state.trace_id},
        )

        resp = await self.llm.generate(req)

        state.answer = {"text": resp.content}
        state.stage = {"current": "final"}
        return state
