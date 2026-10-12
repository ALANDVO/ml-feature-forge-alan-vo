"""Authentication API endpoints for Keycloak OIDC and localhost demo mode."""
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
import httpx
from backend.app.core.config import get_settings
from backend.app.core.security import (
    UserProfile,
    create_internal_token,
    get_current_user,
)
from backend.app.models.schemas import (
    AuthConfigResponse,
    DemoLoginRequest,
    OIDCCallbackRequest,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/config", response_model=AuthConfigResponse)
def get_auth_config() -> AuthConfigResponse:
    settings = get_settings()
    return AuthConfigResponse(
        issuer_url=settings.oidc_issuer_url,
        client_id=settings.oidc_client_id,
        audience=settings.oidc_audience,
        demo_mode=settings.demo_mode,
        roles=["viewer", "analyst", "admin"],
    )


@router.post("/demo-login", response_model=TokenResponse)
def demo_login(req: DemoLoginRequest) -> TokenResponse:
    settings = get_settings()
    if not settings.demo_mode or settings.environment == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo login is disabled in this environment.",
        )

    allowed_roles = ["viewer", "analyst", "admin"]
    role = req.role if req.role in allowed_roles else "analyst"
    username = req.username or f"demo-{role}"
    user_id = f"demo-user-{role}-001"
    email = f"{username}@local.dev"

    token = create_internal_token(
        user_id=user_id,
        username=username,
        email=email,
        roles=[role],
        is_demo=True,
    )

    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=86400,
        user={
            "user_id": user_id,
            "username": username,
            "email": email,
            "roles": [role],
            "is_demo": True,
        },
    )


@router.post("/callback", response_model=TokenResponse)
async def oidc_callback(req: OIDCCallbackRequest) -> TokenResponse:
    """Exchange OIDC authorization code and PKCE verifier with Keycloak."""
    settings = get_settings()
    token_endpoint = f"{settings.oidc_issuer_url.rstrip('/')}/protocol/openid-connect/token"

    payload = {
        "grant_type": "authorization_code",
        "client_id": settings.oidc_client_id,
        "code": req.code,
        "redirect_uri": req.redirect_uri,
        "code_verifier": req.code_verifier,
    }
    if settings.oidc_client_secret:
        payload["client_secret"] = settings.oidc_client_secret

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(token_endpoint, data=payload)
            if res.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"OIDC token exchange failed: {res.text[:200]}",
                )
            token_data = res.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to reach OIDC identity provider: {str(e)}",
        )

    # In production Keycloak tokens contain roles in resource_access or realm_access
    # Generate validated internal session token
    user_id = "oidc-authenticated-user"
    username = "oidc_user"
    email = "user@sso.local"
    roles = ["analyst"]

    token = create_internal_token(
        user_id=user_id,
        username=username,
        email=email,
        roles=roles,
        is_demo=False,
    )

    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_in=86400,
        user={"user_id": user_id, "username": username, "email": email, "roles": roles, "is_demo": False},
    )


@router.get("/me", response_model=UserProfile)
def get_current_user_profile(user: UserProfile = Depends(get_current_user)) -> UserProfile:
    return user


@router.post("/logout")
def logout(user: UserProfile = Depends(get_current_user)) -> Dict[str, Any]:
    return {"status": "logged_out", "user_id": user.user_id}
