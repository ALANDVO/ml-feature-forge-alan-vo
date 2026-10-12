"""Main FastAPI application factory and entrypoint for ML Feature Forge."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.audit import router as audit_router
from backend.app.api.auth import router as auth_router
from backend.app.api.datasets import router as datasets_router
from backend.app.api.evaluations import router as evaluations_router
from backend.app.api.features import router as features_router
from backend.app.api.pipelines import router as pipelines_router
from backend.app.api.system import router as system_router
from backend.app.core.config import get_settings
from backend.app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Initialize SQLite database schema
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="ML feature engineering toolkit that profiles data, generates candidate transforms, evaluates importance, and exports production pipelines.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    application.include_router(system_router, prefix="/api")
    application.include_router(auth_router, prefix="/api")
    application.include_router(datasets_router, prefix="/api")
    application.include_router(features_router, prefix="/api")
    application.include_router(evaluations_router, prefix="/api")
    application.include_router(pipelines_router, prefix="/api")
    application.include_router(audit_router, prefix="/api")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
