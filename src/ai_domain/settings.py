OPENAI_PLATFORM_API_KEY: str | None = None

# BYOK client cache
OPENAI_BYOK_CACHE_TTL_SECONDS: int = 3600      # 1 час
OPENAI_BYOK_CACHE_MAX_SIZE: int = 500          # сколько разных ключей держим в памяти