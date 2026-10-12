"""End-to-end integration tests crossing the HTTP boundary for core workflows."""
import pytest
from fastapi.testclient import TestClient
from backend.app.core.config import Settings, reset_settings_for_test
from backend.app.core.database import init_db
from backend.app.core.security import create_internal_token
from backend.app.main import create_app


@pytest.fixture
def test_setup(tmp_path):
    test_db = str(tmp_path / "test_workflows.db")
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
    analyst_token = create_internal_token(
        user_id="analyst-1",
        username="lead_analyst",
        email="analyst@test.local",
        roles=["analyst"],
    )
    admin_token = create_internal_token(
        user_id="admin-1",
        username="sys_admin",
        email="admin@test.local",
        roles=["admin"],
    )
    with TestClient(app) as client:
        yield client, analyst_token, admin_token
    reset_settings_for_test(None)


def test_full_feature_engineering_workflow(test_setup):
    client, analyst_token, admin_token = test_setup
    headers = {"Authorization": f"Bearer {analyst_token}"}

    # Step 1: Ingest sample dataset via HTTP boundary
    load_res = client.post("/api/datasets/demo-load", headers=headers)
    assert load_res.status_code == 200
    dataset = load_res.json()
    dataset_id = dataset["id"]
    assert dataset["target_column"] == "approved"
    assert dataset["row_count"] == 20

    # Step 2: Propose candidate feature transforms
    propose_res = client.post(
        "/api/features/propose",
        headers=headers,
        json={"dataset_id": dataset_id, "enable_llm": False, "max_candidates": 10},
    )
    assert propose_res.status_code == 200
    candidates = propose_res.json()
    assert len(candidates) >= 5
    cand_names = [c["name"] for c in candidates]
    assert any("ratio" in n or "log1p" in n for n in cand_names)

    # Step 3: Run statistical evaluation (importance, leakage, collinearity)
    eval_res = client.post(
        "/api/evaluations",
        headers=headers,
        json={"dataset_id": dataset_id, "collinearity_threshold": 0.85},
    )
    assert eval_res.status_code == 200
    evals = eval_res.json()
    assert len(evals) == len(candidates)
    assert all(e["verdict"] in ("keep", "drop") for e in evals)

    # Step 4: Compile production pipeline
    pipeline_res = client.post(
        "/api/pipelines",
        headers=headers,
        json={"dataset_id": dataset_id, "name": "Credit Assessment Pipeline"},
    )
    assert pipeline_res.status_code == 200
    pipeline = pipeline_res.json()
    assert "def build_features(df: pd.DataFrame)" in pipeline["python_code"]
    assert len(pipeline["selected_features"]) > 0

    # Step 5: Verify persistent audit trail
    audit_res = client.get("/api/audit", headers=headers)
    assert audit_res.status_code == 200
    audit_logs = audit_res.json()
    actions = [log["action"] for log in audit_logs]
    assert "create_dataset" in actions
    assert "propose_candidates" in actions
    assert "evaluate_features" in actions
    assert "create_pipeline" in actions


def test_system_health_and_info(test_setup):
    client, _, _ = test_setup
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    info_res = client.get("/api/system/info")
    assert info_res.status_code == 200
    assert info_res.json()["author"] == "Alan Vo"
    assert info_res.json()["email"] == "alanvo@gmail.com"


def test_not_found_handling(test_setup):
    client, analyst_token, _ = test_setup
    headers = {"Authorization": f"Bearer {analyst_token}"}
    res = client.get("/api/datasets/nonexistent-id", headers=headers)
    assert res.status_code == 404
