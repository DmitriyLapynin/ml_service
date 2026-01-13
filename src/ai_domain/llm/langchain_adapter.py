from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class LangChainLLMAdapter:
    """
    Adapter that allows using our LLM objects inside LangChain pipelines.

    It is compatible with two call styles:
    1) node-style: `await llm.generate(messages=[...], model=..., ...)`
    2) router/provider-style: `await llm.generate(LLMRequest(...))`
    """

    llm: Any
    model: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    metadata: dict = field(default_factory=dict)

    def with_structured_output(self, pydantic_model: Any, *, include_raw: bool = False):
        """
        LangChain-like structured output helper.

        Returns a Runnable-like object supporting `.ainvoke(...)` that:
        - appends parser format instructions to the first system message (if present)
        - calls this adapter
        - parses JSON into the given Pydantic model

        If `include_raw=True`, returns:
          {"raw": AIMessage, "parsed": <PydanticModel|None>, "parsing_error": <str|None>}
        Otherwise returns the Pydantic model instance.
        """
        try:
            from langchain_core.output_parsers import JsonOutputParser  # type: ignore
            from langchain_core.messages import AIMessage, BaseMessage  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("langchain-core is required for with_structured_output") from e

        from langchain_core.runnables import RunnableLambda  # type: ignore

        parser = JsonOutputParser(pydantic_object=pydantic_model)
        format_instructions = parser.get_format_instructions()

        adapter = self

        async def _run(input: Any) -> Any:
            lc_messages: list[Any]
            if hasattr(input, "to_messages"):
                lc_messages = input.to_messages()
            elif isinstance(input, list):
                lc_messages = input
            else:
                lc_messages = [input]

            injected: list[Any] = []
            injected_done = False
            for m in lc_messages:
                if (
                    not injected_done
                    and isinstance(m, BaseMessage)
                    and getattr(m, "type", None) == "system"
                ):
                    content = str(getattr(m, "content", ""))
                    injected.append(type(m)(content=content + "\n\n" + format_instructions))
                    injected_done = True
                else:
                    injected.append(m)

            raw_msg = await adapter.ainvoke(injected)
            raw_text = str(getattr(raw_msg, "content", ""))

            parsed_obj = None
            parsing_error = None
            try:
                parsed_dict = parser.parse(raw_text)
                parsed_obj = pydantic_model(**parsed_dict)
            except Exception as e:  # pragma: no cover
                parsing_error = str(e)

            if include_raw:
                return {"raw": raw_msg, "parsed": parsed_obj, "parsing_error": parsing_error}

            if parsing_error:
                raise ValueError(parsing_error)
            return parsed_obj

        return RunnableLambda(_run)

    async def __call__(self, input: Any, **kwargs: Any):  # noqa: ARG002
        # Callable instances are coerced by LangChain into Runnables.
        return await self.ainvoke(input)

    async def ainvoke(self, input: Any, config: Any | None = None, **kwargs: Any):  # noqa: ARG002
        try:
            from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # type: ignore
            from langchain_core.prompt_values import PromptValue  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("langchain-core is required for LangChainLLMAdapter") from e

        messages: list[dict[str, str]] = []

        # ChatPromptTemplate produces PromptValue with `.to_messages()`.
        if isinstance(input, PromptValue):
            lc_messages = input.to_messages()
        elif isinstance(input, list) and all(hasattr(m, "content") for m in input):
            lc_messages = input
        else:
            lc_messages = [HumanMessage(content=str(input))]  # type: ignore[list-item]

        for m in lc_messages:
            if isinstance(m, BaseMessage):
                role = getattr(m, "type", None) or "user"
                if role == "human":
                    role = "user"
                if role == "ai":
                    role = "assistant"
                messages.append({"role": str(role), "content": str(getattr(m, "content", ""))})
            else:
                messages.append({"role": "user", "content": str(m)})

        # Try node-style generate(**kwargs) first.
        try:
            resp = await self.llm.generate(
                messages=messages,
                model=self.model,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                metadata=self.metadata,
            )
        except TypeError:
            # Fallback to router/provider style: generate(LLMRequest)
            from ai_domain.llm.types import LLMMessage, LLMRequest

            req = LLMRequest(
                messages=[LLMMessage(role=m["role"], content=m["content"]) for m in messages],
                model=self.model or "",
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=self.top_p,
                metadata=self.metadata,
            )
            resp = await self.llm.generate(req)

        return AIMessage(content=str(getattr(resp, "content", "")))
