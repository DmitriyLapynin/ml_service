def safety_router_condition(state: dict) -> str:
    if state.get("unsafe") or state.get("injection_suspected"):
        return "block"
    return "ok"


def tool_router_condition(state: dict) -> str:
    return "retrieve" if state.get("wants_retrieve") else "skip"
