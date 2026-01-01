from dataclasses import dataclass
from typing import Any


@dataclass
class TestDeps:
    """
    Контейнер зависимостей для build_graph(deps).

    Используется ТОЛЬКО в тестах.
    В проде deps собирается в orchestrator.
    """
    llm: Any
    prompt_repo: Any
    rag_client: Any
    telemetry: Any