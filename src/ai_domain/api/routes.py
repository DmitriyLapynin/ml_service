from fastapi import APIRouter

from ai_domain.api.routers import chat_router, health_router, system_analysis_router

router = APIRouter(prefix="/v1")
router.include_router(health_router)
router.include_router(chat_router)
router.include_router(system_analysis_router)
