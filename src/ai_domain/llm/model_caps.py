from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_seed: bool = True


_MODEL_CAPS_BY_PREFIX = {
    "gpt-5-mini": ModelCapabilities(supports_top_p=False, supports_temperature=False),
    "gpt-5-nano": ModelCapabilities(supports_top_p=False, supports_temperature=False),
}


def get_model_capabilities(model: str | None) -> ModelCapabilities:
    if not model:
        return ModelCapabilities()
    for prefix, caps in _MODEL_CAPS_BY_PREFIX.items():
        if model.startswith(prefix):
            return caps
    return ModelCapabilities()
