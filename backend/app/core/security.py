"""Authentication, token validation, and role-based access control."""
import time
from typing import Dict, List, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from backend.app.core.config import get_settings


class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    roles: List[str]
    is_demo: bool = False

    def has_role(self, required_role: str) -> bool:
        if "admin" in self.roles:
            return True
        return required_role in self.roles


security_scheme = HTTPBearer(auto_error=False)


def create_internal_token(user_id: str, username: str, email: str, roles: List[str], is_demo: bool = False) -> str:
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": user_id,
        "preferred_username": username,
        "email": email,
        "roles": roles,
        "is_demo": is_demo,
        "iat": now,
        "exp": now + 86400,
        "iss": settings.oidc_issuer_url,
        "aud": settings.oidc_audience,
    }
    token = jwt.encode(payload, settings.jwt_signing_key, algorithm="HS256")
    return token


def verify_token(token: str) -> UserProfile:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_signing_key,
            algorithms=["HS256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer_url,
        )
        user_id: str = payload.get("sub", "")
        username: str = payload.get("preferred_username", "anonymous")
        email: str = payload.get("email", "")
        roles: List[str] = payload.get("roles", ["viewer"])
        is_demo: bool = payload.get("is_demo", False)

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed: missing subject identifier",
            )
        return UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            is_demo=is_demo,
        )
    except JWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authorization token: {str(err)}",
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
) -> UserProfile:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header with Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(credentials.credentials)


def require_role(required_roles: List[str]):
    def role_checker(user: UserProfile = Depends(get_current_user)) -> UserProfile:
        allowed = any(user.has_role(r) for r in required_roles)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Requires one of roles {required_roles}. User has {user.roles}",
            )
        return user
    return role_checker
