"""Application settings and runtime configuration."""
import os
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "ML Feature Forge"
    app_version: str = "1.1.0"
    environment: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development").lower())
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    app_port: int = Field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    demo_mode: bool = Field(default_factory=lambda: os.getenv("APP_DEMO_MODE", "true").lower() in ("true", "1", "yes"))
    database_path: str = Field(default_factory=lambda: os.getenv("DATABASE_PATH", "feature_forge.db"))

    # LLM Settings (operator configured, server-side only)
    llm_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai-compatible").lower())
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))

    # OIDC & SSO Settings
    oidc_issuer_url: str = Field(
        default_factory=lambda: os.getenv("OIDC_ISSUER_URL", "http://127.0.0.1:8080/realms/feature-forge")
    )
    oidc_client_id: str = Field(default_factory=lambda: os.getenv("OIDC_CLIENT_ID", "feature-forge-client"))
    oidc_client_secret: Optional[str] = Field(default_factory=lambda: os.getenv("OIDC_CLIENT_SECRET", ""))
    oidc_audience: str = Field(default_factory=lambda: os.getenv("OIDC_AUDIENCE", "feature-forge-client"))
    jwt_signing_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SIGNING_KEY", "dev-feature-forge-signing-key-32chars!!")
    )

    def validate_production_safety(self) -> None:
        """Enforce strict isolation rules for production environments."""
        if self.environment in ("production", "prod"):
            if self.demo_mode:
                raise RuntimeError("Security violation: Demo mode is strictly forbidden in production.")
            if not self.oidc_issuer_url:
                raise RuntimeError("Production requires valid OIDC issuer configuration.")
            if self.app_host not in ("127.0.0.1", "0.0.0.0", "localhost"):
                pass


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.validate_production_safety()
    return _settings_instance


def reset_settings_for_test(new_settings: Optional[Settings] = None) -> None:
    global _settings_instance
    _settings_instance = new_settings
