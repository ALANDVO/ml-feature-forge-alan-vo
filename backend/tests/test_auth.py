"""Tests for OIDC authentication, PKCE verification, and role-based access control."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from backend.app.core.config import Settings, reset_settings_for_test
from backend.app.core.database import init_db
from backend.app.core.security import create_internal_token
from backend.app.main import create_app


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test_auth.db")
    reset_settings_for_test(
        Settings(
            database_path=test_db,
            demo_mode=True,
            environment="development",
            jwt_signing_key="test-key-32chars-for-testing-only!!",
        )
    )
    init_db()
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_settings_for_test(None)


def test_unauthenticated_request_rejected(client):
    res = client.get("/api/datasets")
    assert res.status_code == 401
    assert "Missing Authorization" in res.json()["detail"]


def test_demo_login_flow_and_authorization(client):
    # 1. Login as demo analyst
    login_res = client.post("/api/auth/demo-login", json={"role": "analyst"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    assert token

    # 2. Access protected endpoint with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    datasets_res = client.get("/api/datasets", headers=headers)
    assert datasets_res.status_code == 200

    # 3. Check /api/auth/me returns identity
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["roles"] == ["analyst"]


def test_role_based_access_denial(client):
    # Create viewer token (read-only)
    viewer_token = create_internal_token(
        user_id="viewer-1",
        username="viewer_user",
        email="viewer@test.local",
        roles=["viewer"],
    )
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # Viewer can read datasets
    read_res = client.get("/api/datasets", headers=headers)
    assert read_res.status_code == 200

    # Viewer is forbidden from creating datasets (requires analyst or admin)
    write_res = client.post(
        "/api/datasets",
        headers=headers,
        json={"name": "Forbidden Dataset", "target_column": "y", "csv_content": "x,y\n1,2"},
    )
    assert write_res.status_code == 403
    assert "Access denied" in write_res.json()["detail"]


def test_oidc_callback_and_pkce_validation(client):
    # Mock Keycloak token endpoint
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mock-keycloak-token",
        "id_token": "mock-keycloak-id-token",
        "expires_in": 300,
    }
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        # Valid callback with PKCE code_verifier
        callback_res = client.post(
            "/api/auth/callback",
            json={
                "code": "auth-code-12345",
                "state": "state-nonce-67890",
                "code_verifier": "pkce-verifier-test-1234567890",
                "redirect_uri": "http://127.0.0.1:3000/auth/callback",
            },
        )
        assert callback_res.status_code == 200
        token = callback_res.json()["access_token"]
        assert token

        # Verify token propagates to authorized request
        auth_headers = {"Authorization": f"Bearer {token}"}
        me_res = client.get("/api/auth/me", headers=auth_headers)
        assert me_res.status_code == 200
        assert me_res.json()["user_id"] == "oidc-authenticated-user"


def test_demo_mode_refused_in_production(tmp_path):
    test_db = str(tmp_path / "prod.db")
    with pytest.raises(RuntimeError) as exc_info:
        Settings(
            database_path=test_db,
            demo_mode=True,
            environment="production",
        ).validate_production_safety()
    assert "strictly forbidden in production" in str(exc_info.value)
