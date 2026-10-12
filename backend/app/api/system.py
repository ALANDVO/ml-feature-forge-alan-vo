"""System diagnostics and health monitoring API endpoints."""
from typing import Any, Dict
from fastapi import APIRouter
from backend.app.core.config import get_settings

router = APIRouter(prefix="", tags=["System"])


@router.get("/health")
def healthcheck() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/system/info")
def system_info() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "name": settings.app_name,
        "author": "Alan Vo",
        "email": "alanvo@gmail.com",
        "version": settings.app_version,
        "environment": settings.environment,
        "demo_mode": settings.demo_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_configured": bool(settings.llm_api_key),
        "oidc_issuer": settings.oidc_issuer_url,
    }
