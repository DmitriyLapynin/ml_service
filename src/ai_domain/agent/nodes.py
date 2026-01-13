from dataclasses import dataclass
import logging
from typing import Callable, Any, Dict, List, Optional

from ai_domain.agent.safety_prompt import (
    SAFETY_CLASSIFIER_PROMPT,
    SafetyClassifierOutput,
    normalize_classifier_output,
    rule_based_flags,
)
from ai_domain.llm.client import LLMConfig
from ai_domain.llm.types import LLMCallContext
from ai_domain.tools.registry import ToolSpec, default_registry


def _ensure_lists(state: dict):
    state.setdefault("messages", [])
    state.setdefault("executed", [])


def _extract_last_user_content(state: dict) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


async def _check_unsafe_and_injection(
    text: str,
    *,
    llm: Any | None = None,
    model: str | None = None,
    call_context: LLMCallContext | None = None,
) -> Dict[str, bool]:
    """
    Комбинированный чек:
    1) rule-based (ключевые слова),
    2) LLM-классификатор, который возвращает unsafe / injection (strict JSON).

    LLM передаётся извне (через state['safety_llm']) и должен поддерживать:
      await llm.invoke_structured(schema, messages, config, include_raw=...)
    """
    if not text:
        return {"unsafe": False, "injection_suspected": False}

    rules = rule_based_flags(text)
    llm_unsafe = False
    llm_injection = False

    if llm is not None:
        try:
            messages = [
                {"role": "system", "content": SAFETY_CLASSIFIER_PROMPT},
                {"role": "user", "content": text},
            ]
            config = LLMConfig(
                model=model,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                metadata={"required_capabilities": {"supports_structured": True}},
            )
            res = await llm.invoke_structured(
                SafetyClassifierOutput,
                messages,
                config=config,
                include_raw=True,
                context=call_context,
            )
            payload = res.parsed
            if payload is not None:
                parsed = normalize_classifier_output(payload.model_dump())
                llm_unsafe = bool(parsed["unsafe"])
                llm_injection = bool(parsed["injection_suspected"])
        except Exception as e:
            logging.error(f"Ошибка при LLM-классификации безопасности: {e}")

    unsafe = rules["unsafe"] or llm_unsafe
    injection_suspected = rules["injection_suspected"] or llm_injection
    return {"unsafe": unsafe, "injection_suspected": injection_suspected}


def _format_tools(tools: List[dict], is_rag: bool) -> List[dict]:
    filtered = [
        tool for tool in tools
        if is_rag or tool.get("name") != "knowledge_search"
    ]
    return filtered


def _normalize_tools(tools: List[object]) -> List[dict]:
    normalized: List[dict] = []
    for tool in tools:
        if isinstance(tool, ToolSpec):
            normalized.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.schema,
                }
            )
        else:
            normalized.append(tool)
    return normalized


def create_agent_prompt(tools: List[dict], is_rag: bool = True) -> str:
    filtered_tools = _format_tools(tools, is_rag)
    if filtered_tools:
        tool_lines = [
            f"- `{tool['name']}`: {tool.get('description', 'без описания')}."
            for tool in filtered_tools
        ]
        tool_description_string = "\n".join(tool_lines)
        tools_section = (
            "**ДОСТУПНЫЕ ИНСТРУМЕНТЫ:**\n---\n"
            f"{tool_description_string}\n---"
        )
    else:
        tools_section = (
            "**У тебя нет доступных инструментов.** "
            "Отвечай на основе истории диалога и своих знаний."
        )

    base_instruction = (
        "Ты — умный ассистент-исследователь. "
        "Твоя задача — проанализировать вопрос пользователя и дать полезный ответ."
    )

    if is_rag and filtered_tools:
        rag_instructions = """
**КЛЮЧЕВОЕ ПРАВИЛО: Если вопрос пользователя содержит несколько независимых тем (например, цена И адрес, или две разные услуги), ты ДОЛЖЕН вызывать инструменты для каждой темы ОТДЕЛЬНО и ПАРАЛЛЕЛЬНО в одном ответе.**

Пример:
- Вопрос пользователя: "Сколько стоят брекеты и где вы находитесь?"
- Твой правильный ответ (вызов инструментов):
  - `knowledge_search(query="стоимость брекетов")`
  - `knowledge_search(query="адрес клиники")`

1. **ИСПОЛЬЗУЙ `knowledge_search`**, если вопрос касается:
   - Цен на услуги (например, "сколько стоят брекеты?")
   - Описания услуг (например, "что такое имплантация?")
   - Адреса, времени работы, контактов клиники
   - Любых других фактических данных, которые могут быть в базе знаний.

2. **НЕ ИСПОЛЬЗУЙ `knowledge_search`**, если вопрос является:
   - Приветствием или прощанием ("привет", "до свидания")
   - Благодарностью ("спасибо")
   - Общим разговором, не требующим фактов ("как дела?", "ты бот?")
"""
    else:
        rag_instructions = """
**ПРАВИЛА ПРИНЯТИЯ РЕШЕНИЙ:**
1. **Разбивай сложные запросы:** Если вопрос пользователя содержит несколько независимых тем, ты ДОЛЖЕН вызывать инструменты для каждой темы ОТДЕЛЬНО и ПАРАЛЛЕЛЬНО.
2. **Выбирай лучший инструмент:** Внимательно прочитай описания инструментов и выбери тот, который лучше всего подходит для под-запроса.
3. **Не используй инструменты** для простого разговора ("привет", "спасибо") или если ни один из инструментов не подходит для ответа.

Проанализируй последний вопрос и вызови один или несколько инструментов, если это необходимо.
"""

    return f"{base_instruction}\n\n{tools_section}\n\n{rag_instructions}"


async def safety_in_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_in")
    last_user = _extract_last_user_content(state)
    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="safety_in",
            task="safety_classifier",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
        )
        if trace_id
        else None
    )
    flags = await _check_unsafe_and_injection(
        last_user,
        llm=state.get("safety_llm"),
        model=state.get("safety_model"),
        call_context=context,
    )
    state["unsafe"] = flags["unsafe"]
    state["injection_suspected"] = flags["injection_suspected"]
    state["blocked"] = flags["unsafe"] or flags["injection_suspected"]
    if state["blocked"]:
        state["safety_reason"] = "unsafe_request"
    return state


def safety_router_condition(state: dict) -> str:
    if state.get("unsafe") or state.get("injection_suspected"):
        return "block"
    return "ok"


def safety_block_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_block")
    text = "Я не могу ответить на этот запрос."
    if state.get("unsafe"):
        text = (
            "Запрос нарушает политику безопасности. "
            "Уточните вопрос про услуги или запись."
        )
    elif state.get("injection_suspected"):
        text = (
            "Нельзя обходить правила. "
            "Я могу ответить только в рамках политики безопасности."
        )
    state["answer"] = {"text": text, "format": "plain"}
    return state


async def agent_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("agent")
    tools = state.get("tools")
    if tools is None:
        tools = default_registry().list()
    tools = _normalize_tools(list(tools))
    is_rag = bool(state.get("is_rag", True))
    logging.info(f"Режим RAG: {is_rag}")
    state["agent_prompt"] = create_agent_prompt(tools, is_rag=is_rag)
    state["filtered_tools"] = _format_tools(tools, is_rag)
    state["wants_retrieve"] = bool(state["filtered_tools"]) and is_rag

    llm = state.get("llm")
    if not hasattr(llm, "invoke_text"):
        return state

    model_name = state.get("model") or state.get("model_name") or "gpt-4.1-mini"
    model_params = state.get("model_params") or {}
    if model_name in {"gpt-5-nano", "gpt-5-mini"}:
        model_params = {}

    config = LLMConfig(
        model=model_name,
        max_tokens=int(model_params.get("max_tokens") or model_params.get("max_output_tokens") or 128),
        temperature=float(model_params.get("temperature") or 0.2),
        top_p=model_params.get("top_p"),
        metadata={"required_capabilities": {"supports_structured": False}},
    )

    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="agent_node",
            task="agent_decision",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
        )
        if trace_id
        else None
    )

    llm_messages = [
        {"role": "system", "content": state["agent_prompt"]},
        *state.get("messages", []),
    ]
    response = await llm.invoke_text(llm_messages, config=config, context=context)
    state["messages"] = [*state.get("messages", []), {"role": "assistant", "content": response}]
    return state


def tool_router_condition(state: dict) -> str:
    return "retrieve" if state.get("wants_retrieve") else "skip"


def tool_executor_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("retrieve")
    tool_name = "knowledge_search"
    logging.info(f"Tool call: {tool_name}")
    state["context"] = "Контекст: найден факт."
    return state


async def generate_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("generate")
    logging.info("--- УЗЕЛ: УСЛОВНАЯ ГЕНЕРАЦИЯ ОТВЕТА ---")

    messages = state.get("messages") or []
    is_rag = bool(state.get("is_rag", False))
    sub_query_results = state.get("sub_query_results") or []
    user_instruction = state.get("user_instruction")
    role_instruction = state.get("role_instruction") or ""

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

    model_name = state.get("model") or state.get("model_name") or "gpt-4.1-mini"
    model_params = state.get("model_params") or {}
    if model_name in {"gpt-5-nano", "gpt-5-mini"}:
        model_params = {}

    config = LLMConfig(
        model=model_name,
        max_tokens=int(model_params.get("max_tokens") or model_params.get("max_output_tokens") or 2048),
        temperature=float(model_params.get("temperature") or 0.2),
        top_p=model_params.get("top_p"),
        metadata={"required_capabilities": {"supports_structured": False}},
    )

    llm = state.get("llm")
    if not hasattr(llm, "invoke_text"):
        user_text = (messages[-1].get("content") if messages else "") or ""
        ctx = f"\n{found_documents}" if found_documents else ""
        state["answer"] = {"text": f"Ответ: {user_text}{ctx}", "format": "plain"}
        return state

    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="generate",
            task="agent_generate",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
        )
        if trace_id
        else None
    )

    if sub_query_results and is_rag:
        logging.info("Обнаружены результаты RAG. Используется промпт с RAG и инструкцией.")
    else:
        logging.info("Результаты RAG отсутствуют. Используется промпт с историей и инструкцией.")

    final_answer = await llm.invoke_text(
        llm_messages,
        config=config,
        context=context,
    )
    state["answer"] = {"text": final_answer, "format": "plain"}
    return state


async def safety_out_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_out")
    answer = state.get("answer") or {}
    text = (answer.get("text") or "").strip()
    trace_id = state.get("trace_id")
    context = (
        LLMCallContext(
            trace_id=trace_id,
            graph=state.get("graph"),
            node="safety_out",
            task="safety_classifier",
            channel=state.get("channel"),
            tenant_id=state.get("tenant_id"),
            request_id=state.get("request_id"),
        )
        if trace_id
        else None
    )
    flags = await _check_unsafe_and_injection(
        text,
        llm=state.get("safety_llm"),
        model=state.get("safety_model"),
        call_context=context,
    )
    if flags["unsafe"]:
        state["answer"] = {"text": "Ответ скрыт политикой безопасности.", "format": "plain"}
    return state


@dataclass
class AgentNodes:
    safety_in_node: Callable[[dict], dict] = safety_in_node
    safety_block_node: Callable[[dict], dict] = safety_block_node
    safety_out_node: Callable[[dict], dict] = safety_out_node

    agent_node: Callable[[dict], dict] = agent_node
    tool_executor_node: Callable[[dict], dict] = tool_executor_node
    generate_node: Callable[[dict], dict] = generate_node

    safety_router_condition: Callable[[dict], str] = safety_router_condition
    tool_router_condition: Callable[[dict], str] = tool_router_condition
RAG_WITH_INSTRUCTION_PROMPT = """{role_instruction}

Ты должен выполнить ИНСТРУКЦИЮ ПОЛЬЗОВАТЕЛЯ.
Ты видишь список сообщений диалога ЕСТЕСТВЕННОГО формата (Human, AI, Tool).
Сообщения с ролью 'tool' являются результатами вызовов инструментов (например, календаря).
Считай их источником истины и не противоречь им.

Если данных недостаточно — честно скажи об этом.
"""

RAG_WITH_INSTRUCTION_SUFFIX = """
НАЙДЕННЫЕ ДОКУМЕНТЫ:
{found_documents}

Сформируй финальный ответ строго на основании диалога и найденных документов.
ИНСТРУКЦИЯ ПОЛЬЗОВАТЕЛЯ:
{user_instruction}
"""

GENERAL_WITH_INSTRUCTION_PROMPT = """{role_instruction}

Перед тобой история диалога в естественном формате (Human, AI, Tool).
Сообщения c ролью 'tool' — это результаты вызовов инструментов.
Считай их правдивыми фактами и используй при необходимости.
"""

GENERAL_WITH_INSTRUCTION_SUFFIX = """
Ты должен выполнить ИНСТРУКЦИЮ ПОЛЬЗОВАТЕЛЯ.
Используй только историю диалога.
ИНСТРУКЦИЯ ПОЛЬЗОВАТЕЛЯ:
{user_instruction}
"""
