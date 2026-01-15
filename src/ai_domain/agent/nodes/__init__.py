from dataclasses import dataclass
from typing import Awaitable, Callable

from .final_answer import generate_node
from .prompts import create_agent_prompt, create_agent_prompt_short
from .router import safety_router_condition, tool_router_condition
from .safety_in import safety_block_node, safety_in_node
from .safety_out import safety_out_node
from .tool_executor import tool_executor_node
from .tools_loop import agent_node

AsyncNode = Callable[[dict], Awaitable[dict]]
Condition = Callable[[dict], str]


@dataclass
class AgentNodes:
    safety_in_node: AsyncNode = safety_in_node
    safety_block_node: AsyncNode = safety_block_node
    safety_out_node: AsyncNode = safety_out_node

    agent_node: AsyncNode = agent_node
    tool_executor_node: AsyncNode = tool_executor_node
    generate_node: AsyncNode = generate_node

    safety_router_condition: Condition = safety_router_condition
    tool_router_condition: Condition = tool_router_condition


__all__ = [
    "AgentNodes",
    "agent_node",
    "create_agent_prompt",
    "create_agent_prompt_short",
    "generate_node",
    "safety_in_node",
    "safety_block_node",
    "safety_out_node",
    "tool_executor_node",
    "safety_router_condition",
    "tool_router_condition",
]
