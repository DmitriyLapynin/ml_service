import logging
import time

from ai_domain.llm.client import LLMConfig
from ai_domain.llm.types import LLMCallContext

from .prompts import (
    GENERAL_WITH_INSTRUCTION_PROMPT,
    GENERAL_WITH_INSTRUCTION_SUFFIX,
    RAG_WITH_INSTRUCTION_PROMPT,
    RAG_WITH_INSTRUCTION_SUFFIX,
)
from .utils import ensure_lists, log_node, step_begin, step_end


async def generate_node(state: dict) -> dict:
    ensure_lists(state)
    log_node(state, "generate")
    step_index = step_begin(state, "generate")
    start = time.perf_counter()
    state["executed"].append("generate")
    messages = state.get("messages") or []
    tool_messages = state.get("tool_messages") or []
    llm = state.get("llm")
    is_rag = bool(state.get("is_rag", False))
    sub_query_results = state.get("sub_query_results") or []
    user_instruction = state.get("user_instruction")
    role_instruction = state.get("role_instruction") or ""

    logging.info(
        "node_generate_start",
        extra={
            "trace_id": state.get("trace_id"),
            "has_tools": bool(tool_messages),
            "is_rag": is_rag,
        },
    )

    if not user_instruction or not str(user_instruction).strip():
        user_instruction = "Сформулируй полезный и связный ответ на основе предоставленной информации."

    found_documents = ""
    if sub_query_results:
        chunks = []
        for result in sub_query_results:
            sub_q = result.get("sub_question") or ""
            docs = result.get("documents") or []
            docs_str = "\n".join([f"- {doc}" for doc in docs])
            if sub_q:
                chunks.append(f"Информация по под-вопросу '{sub_q}':\n{docs_str}")
            else:
                chunks.append(docs_str)
        found_documents = "\n\n".join([c for c in chunks if c])
    elif state.get("context"):
        found_documents = str(state.get("context") or "")

    if sub_query_results and is_rag:
        system_prompt = RAG_WITH_INSTRUCTION_PROMPT.format(role_instruction=role_instruction)
        system_suffix = RAG_WITH_INSTRUCTION_SUFFIX.format(
            found_documents=found_documents,
            user_instruction=user_instruction,
        )
    else:
        system_prompt = GENERAL_WITH_INSTRUCTION_PROMPT.format(role_instruction=role_instruction)
        system_suffix = GENERAL_WITH_INSTRUCTION_SUFFIX.format(user_instruction=user_instruction)

    llm_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
        {"role": "system", "content": system_suffix},
    ]
    if tool_messages and not hasattr(llm, "invoke_tool_response"):
        tool_outputs = []
        for msg in tool_messages:
            content = msg.get("content") or ""
            tool_outputs.append(f"- {content}")
        tool_outputs_text = "\n".join(tool_outputs)
        if tool_outputs_text:
            llm_messages[-1]["content"] = (
                f"{system_suffix}\n\nРЕЗУЛЬТАТЫ ИНСТРУМЕНТОВ:\n{tool_outputs_text}"
            )

    model_name = state.get("model") or state.get("model_name") or "gpt-4.1-mini"
    model_params = state.get("model_params") or {}
    top_p = model_params.get("top_p")
    if top_p is not None:
        top_p = float(top_p)
    config = LLMConfig(
        model=model_name,
        max_tokens=int(model_params.get("max_tokens") or model_params.get("max_output_tokens") or 2048),
        temperature=float(model_params.get("temperature") or 0.2),
        top_p=top_p,
        metadata={"required_capabilities": {"supports_structured": False}},
    )

    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph") or state.get("graph_name"),
            node="generate",
            task="agent_generate",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
            metrics=state.get("llm_metrics"),
        )
        if trace_id
        else None
    )
    if tool_messages and hasattr(llm, "invoke_tool_response"):
        response = await llm.invoke_tool_response(
            [
                {"role": "system", "content": system_prompt},
                *messages,
                *tool_messages,
                {"role": "system", "content": system_suffix},
            ],
            config=config,
            context=context,
        )
        text = response
        if not isinstance(text, str):
            text = getattr(response, "content", response)
        logging.info(
            "node_generate_raw_response",
            extra={"trace_id": state.get("trace_id"), "content": text},
        )
        state["answer"] = {"text": text, "format": "plain"}
        latency_ms = int((time.perf_counter() - start) * 1000)
        step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
        return state

    if not hasattr(llm, "invoke_text"):
        user_text = (messages[-1].get("content") if messages else "") or ""
        ctx = f"\n{found_documents}" if found_documents else ""
        state["answer"] = {"text": f"Ответ: {user_text}{ctx}", "format": "plain"}
        return state

    if sub_query_results and is_rag:
        logging.info(
            "node_generate_prompt_path",
            extra={"trace_id": state.get("trace_id"), "prompt_path": "rag"},
        )
    else:
        logging.info(
            "node_generate_prompt_path",
            extra={"trace_id": state.get("trace_id"), "prompt_path": "no_rag"},
        )

    final_answer = await llm.invoke_text(
        llm_messages,
        config=config,
        context=context,
    )
    logging.info(
        "node_generate_raw_response",
        extra={"trace_id": state.get("trace_id"), "content": final_answer},
    )
    state["answer"] = {"text": final_answer, "format": "plain"}
    latency_ms = int((time.perf_counter() - start) * 1000)
    step_end(state, index=step_index, latency_ms=latency_ms, status="ok")
    return state
