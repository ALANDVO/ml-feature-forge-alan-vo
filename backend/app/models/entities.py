"""Database entity mapping and conversion helpers."""
import json
import sqlite3
from typing import Any, Dict, List
from backend.app.models.schemas import (
    AuditLogEntry,
    DatasetProfile,
    DatasetResponse,
    FeatureCandidate,
    FeatureEvaluation,
    PipelineResponse,
)


def row_to_dataset(row: sqlite3.Row) -> DatasetResponse:
    profile_dict = json.loads(row["profile_json"])
    sample_rows = json.loads(row["sample_data_json"])
    profile = DatasetProfile(**profile_dict)
    return DatasetResponse(
        id=row["id"],
        name=row["name"],
        filename=row["filename"],
        row_count=row["row_count"],
        col_count=row["col_count"],
        target_column=row["target_column"],
        problem_type=row["problem_type"],
        profile=profile,
        sample_rows=sample_rows,
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


def row_to_candidate(row: sqlite3.Row) -> FeatureCandidate:
    input_cols = json.loads(row["input_columns_json"])
    return FeatureCandidate(
        id=row["id"],
        dataset_id=row["dataset_id"],
        name=row["name"],
        transform_type=row["transform_type"],
        expression=row["expression"],
        input_columns=input_cols,
        hypothesis=row["hypothesis"],
        source=row["source"],
        created_at=row["created_at"],
    )


def row_to_evaluation(row: sqlite3.Row) -> FeatureEvaluation:
    collinear = json.loads(row["collinearity_with_json"])
    return FeatureEvaluation(
        id=row["id"],
        dataset_id=row["dataset_id"],
        feature_name=row["feature_name"],
        importance_score=row["importance_score"],
        correlation_target=row["correlation_target"],
        leakage_risk=row["leakage_risk"],
        collinearity_with=collinear,
        verdict=row["verdict"],
        reason=row["reason"],
        evaluated_at=row["evaluated_at"],
    )


def row_to_pipeline(row: sqlite3.Row) -> PipelineResponse:
    selected = json.loads(row["selected_features_json"])
    spec = json.loads(row["json_spec"])
    return PipelineResponse(
        id=row["id"],
        dataset_id=row["dataset_id"],
        name=row["name"],
        selected_features=selected,
        python_code=row["python_code"],
        json_spec=spec,
        created_by=row["created_by"],
        created_at=row["created_at"],
    )


def row_to_audit(row: sqlite3.Row) -> AuditLogEntry:
    details = json.loads(row["details_json"])
    return AuditLogEntry(
        id=row["id"],
        user_id=row["user_id"],
        action=row["action"],
        resource_type=row["resource_type"],
        resource_id=row["resource_id"],
        details=details,
        ip_address=row["ip_address"],
        timestamp=row["timestamp"],
    )
