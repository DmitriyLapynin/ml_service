from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _ensure_lists(state: dict):
    state.setdefault("messages", [])
    state.setdefault("executed", [])


def safety_in_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_in")
    text = (state["messages"][-1].get("content") if state["messages"] else "") or ""
    lowered = text.lower()
    banned = ["пароль", "взорвать", "ignore previous", "prompt injection"]
    if any(b in lowered for b in banned):
        state["blocked"] = True
        state["safety_reason"] = "unsafe_request"
    else:
        state["blocked"] = state.get("blocked", False)
    return state


def safety_router_condition(state: dict) -> str:
    return "block" if state.get("blocked") else "ok"


def safety_block_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_block")
    state["answer"] = {"text": "Запрос отклонён политикой безопасности.", "format": "plain"}
    return state


def agent_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("agent")
    text = (state["messages"][-1].get("content") if state["messages"] else "") or ""
    state["plan"] = "answer"
    state["wants_retrieve"] = "факт" in text.lower()
    return state


def tool_router_condition(state: dict) -> str:
    return "retrieve" if state.get("wants_retrieve") else "skip"


def tool_executor_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("retrieve")
    state["context"] = "Контекст: найден факт про продукт."
    return state


def generate_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("generate")
    user_text = (state["messages"][-1].get("content") if state["messages"] else "") or ""
    ctx = f"\n{state.get('context')}" if state.get("context") else ""
    state["answer"] = {"text": f"Ответ: {user_text}{ctx}", "format": "plain"}
    return state


def safety_out_node(state: dict) -> dict:
    _ensure_lists(state)
    state["executed"].append("safety_out")
    answer_text = (state.get("answer") or {}).get("text", "").lower()
    if any(b in answer_text for b in ["пароль", "взорвать"]):
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
