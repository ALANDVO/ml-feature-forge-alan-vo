"""Pydantic schemas for request, response, and domain models."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    row_count: int
    col_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    missing_counts: Dict[str, int]
    missing_rates: Dict[str, float]
    stats: Dict[str, Dict[str, Any]]
    target_column: str
    problem_type: str


class DatasetCreateRequest(BaseModel):
    name: str
    target_column: str
    problem_type: Optional[str] = "classification"
    csv_content: Optional[str] = None
    data_rows: Optional[List[Dict[str, Any]]] = None


class DatasetResponse(BaseModel):
    id: str
    name: str
    filename: str
    row_count: int
    col_count: int
    target_column: str
    problem_type: str
    profile: DatasetProfile
    sample_rows: List[Dict[str, Any]]
    created_by: str
    created_at: str


class FeatureCandidate(BaseModel):
    id: str
    dataset_id: str
    name: str
    transform_type: str
    expression: str
    input_columns: List[str]
    hypothesis: str
    source: str  # deterministic | llm_advisory
    expected_signal: str = "moderate"
    risk: str = "none"
    created_at: str


class ProposeFeaturesRequest(BaseModel):
    dataset_id: str
    enable_llm: bool = False
    max_candidates: int = 15
    include_interactions: bool = True
    include_encodings: bool = True
    include_nonlinear: bool = True
    include_datetime: bool = True


class FeatureEvaluation(BaseModel):
    id: str
    dataset_id: str
    feature_name: str
    importance_score: float
    correlation_target: float
    leakage_risk: str
    collinearity_with: List[str]
    verdict: str  # keep | drop
    reason: str
    evaluated_at: str


class EvaluateFeaturesRequest(BaseModel):
    dataset_id: str
    collinearity_threshold: float = 0.85
    min_importance_threshold: float = 0.02


class PipelineCompileRequest(BaseModel):
    dataset_id: str
    name: Optional[str] = "Production Feature Pipeline"
    selected_feature_ids: Optional[List[str]] = None


class PipelineResponse(BaseModel):
    id: str
    dataset_id: str
    name: str
    selected_features: List[str]
    python_code: str
    json_spec: Dict[str, Any]
    created_by: str
    created_at: str


class AuditLogEntry(BaseModel):
    id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    timestamp: str


class AuthConfigResponse(BaseModel):
    issuer_url: str
    client_id: str
    audience: str
    demo_mode: bool
    roles: List[str]


class DemoLoginRequest(BaseModel):
    role: str = "analyst"  # viewer | analyst | admin
    username: Optional[str] = "demo-analyst"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 86400
    user: Dict[str, Any]


class OIDCCallbackRequest(BaseModel):
    code: str
    state: str
    code_verifier: str
    redirect_uri: str
