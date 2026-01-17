from __future__ import annotations

import json
from typing import Dict, List


DEFAULT_STAGE_DESCRIPTIONS = {
    "1": "Первое сообщение пользователя в диалоге, содержит только приветствие.",
    "2": "Сбор информации о запросе клиента, без презентации продукта.",
    "3": "Презентация продукта на основе запроса клиента.",
    "4": "Работа с возражениями и запись на встречу.",
}


SYSTEM_ANALYSIS_PROMPT = """Ты аналитик диалогов продаж.
Верни ТОЛЬКО валидный JSON, соответствующий Pydantic-схеме FastAnalytics. Никакого текста вокруг. Без markdown. Ответ должен начинаться с '{' и заканчиваться '}'.

Как анализировать:
- Учитывай всю историю.
- stage определяй по последнему сообщению клиента (role=user). Если оно короткое подтверждение ("да", "ок", "хорошо"), опирайся на ближайшее предыдущее содержательное сообщение клиента.

Правила заполнения:
- Верни все поля схемы FastAnalytics.
- client_info.name/phone: только если явно есть в сообщениях, иначе "".
- stage: числом (stage_number).
- stage_confidences: для каждого этапа из stages_info верни объект {"stage_number":N,"confidence":0..1}.
- current_stage_number — предыдущий этап, можно выбрать другой.
- use_rag=true только если нужны точные цены/условия/детали из базы знаний.
- Не выдумывай факты: если данных нет — ""/null/[] по типам схемы."""


def _format_current_stage_info(
    stages_info: List[Dict],
    current_stage_number: int,
) -> str:
    current_stage = next(
        (stage for stage in stages_info if stage.get("stage_number") == current_stage_number),
        None,
    )
    if not current_stage:
        return "Не определен."
    stage_num_str = str(current_stage.get("stage_number"))
    description = current_stage.get("description") or DEFAULT_STAGE_DESCRIPTIONS.get(stage_num_str, "Нет описания.")
    return (
        f"- Название: {current_stage.get('name', 'N/A')}\n"
        f"- Описание: {description}"
    )


def _format_all_stages_rules(stages_info: List[Dict]) -> str:
    rules = []
    for stage in stages_info:
        stage_num_str = str(stage.get("stage_number"))
        description = stage.get("description") or DEFAULT_STAGE_DESCRIPTIONS.get(stage_num_str, "Нет описания.")
        rules.append(
            "  - Этап \"{name}\":\n"
            "    - Описание: {description}\n".format(
                name=stage.get("name", "N/A"),
                description=description,
            )
        )
    return "\n".join(rules)


def format_analysis_user_prompt(
    *,
    stages_info: List[Dict],
    current_stage_number: int,
) -> str:
    normalized_stages = []
    for stage in stages_info:
        normalized_stages.append(
            {
                "stage_number": stage.get("stage_number"),
                "stage_name": stage.get("stage_name") or stage.get("name"),
                "description": stage.get("description"),
            }
        )
    payload = {
        "current_stage_number": current_stage_number,
        "stages_info": normalized_stages,
    }
    return json.dumps(payload, ensure_ascii=False)
