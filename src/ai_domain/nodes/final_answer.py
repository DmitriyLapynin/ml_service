from ai_domain.llm.observability import build_llm_metadata
from ai_domain.llm.types import LLMCredentials, LLMCallContext, LLMMessage, LLMRequest
from ai_domain.orchestrator.tasks import get_task_config
from ai_domain.utils.memory import select_memory_messages

class FinalAnswerNode:
    def __init__(self, llm, prompt_repo, telemetry):
        self.llm = llm
        self.prompts = prompt_repo
        self.telemetry = telemetry
        self.task_name = "final_answer"

    async def __call__(self, state):
        prompt_key = "system_prompt"
        task_config = get_task_config(state, self.task_name)
        version = task_config.prompt_versions.get(prompt_key) or "active"

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

        llm_cfg = task_config.llm
        prompt_vars = {}
        if role_instruction:
            prompt_vars["role_instruction"] = role_instruction
        if prompt:
            prompt_vars["prompt"] = prompt
        metadata = {**(llm_cfg.metadata or {}), **build_llm_metadata(
            state=state,
            node_name="final_answer",
            task=self.task_name,
            prompt_key=prompt_key,
            prompt_version=version,
            prompt_vars=prompt_vars or None,
        )}
        context = LLMCallContext(
            trace_id=getattr(state, "trace_id", None),
            graph=getattr(state, "graph_name", None),
            node=self.__class__.__name__,
            task=self.task_name,
            channel=getattr(state, "channel", None),
            tenant_id=getattr(state, "tenant_id", None),
            request_id=getattr(state, "request_id", None),
        )
        req = LLMRequest(
            messages=messages,
            model=llm_cfg.model,
            max_output_tokens=int(llm_cfg.max_tokens),
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            seed=llm_cfg.seed,
            credentials=llm_credentials,
            metadata=metadata,
        )

        resp = await self.llm.generate(req, context=context)

        state.answer = {"text": resp.content}
        state.stage = {"current": "final"}
        return state
