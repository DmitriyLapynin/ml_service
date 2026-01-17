from ai_domain.api.routers.chat import router as chat_router
from ai_domain.api.routers.health import router as health_router
from ai_domain.api.routers.system_analysis import router as system_analysis_router

__all__ = ["health_router", "chat_router", "system_analysis_router"]
