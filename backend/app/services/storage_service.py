"""Database operations and repository functions for all entity types."""
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.database import get_db_connection, now_utc_iso, record_audit
from backend.app.models.entities import (
    row_to_audit,
    row_to_candidate,
    row_to_dataset,
    row_to_evaluation,
    row_to_pipeline,
)
from backend.app.models.schemas import (
    AuditLogEntry,
    DatasetProfile,
    DatasetResponse,
    FeatureCandidate,
    FeatureEvaluation,
    PipelineResponse,
)


class StorageService:
    @staticmethod
    def create_dataset(
        name: str,
        filename: str,
        row_count: int,
        col_count: int,
        target_column: str,
        problem_type: str,
        profile: DatasetProfile,
        sample_rows: List[Dict[str, Any]],
        created_by: str,
    ) -> DatasetResponse:
        dataset_id = str(uuid.uuid4())
        created_at = now_utc_iso()
        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO datasets (id, name, filename, row_count, col_count, target_column, problem_type, profile_json, sample_data_json, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        name,
                        filename,
                        row_count,
                        col_count,
                        target_column,
                        problem_type,
                        profile.model_dump_json(),
                        json.dumps(sample_rows),
                        created_by,
                        created_at,
                    ),
                )
            record_audit(
                user_id=created_by,
                action="create_dataset",
                resource_type="dataset",
                resource_id=dataset_id,
                details={"name": name, "rows": row_count, "cols": col_count, "target": target_column},
            )
            return StorageService.get_dataset(dataset_id)  # type: ignore
        finally:
            conn.close()

    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[DatasetResponse]:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cur.fetchone()
            return row_to_dataset(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def list_datasets(limit: int = 50, offset: int = 0) -> List[DatasetResponse]:
        conn = get_db_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [row_to_dataset(r) for r in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def delete_dataset(dataset_id: str, deleted_by: str) -> bool:
        conn = get_db_connection()
        try:
            with conn:
                cur = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
                deleted = cur.rowcount > 0
            if deleted:
                record_audit(
                    user_id=deleted_by,
                    action="delete_dataset",
                    resource_type="dataset",
                    resource_id=dataset_id,
                    details={"deleted": True},
                )
            return deleted
        finally:
            conn.close()

    @staticmethod
    def save_candidates(
        dataset_id: str,
        candidates: List[Dict[str, Any]],
        user_id: str,
    ) -> List[FeatureCandidate]:
        created_at = now_utc_iso()
        conn = get_db_connection()
        try:
            with conn:
                for c in candidates:
                    cid = c.get("id") or str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO feature_candidates
                        (id, dataset_id, name, transform_type, expression, input_columns_json, hypothesis, source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            cid,
                            dataset_id,
                            c["name"],
                            c["transform_type"],
                            c["expression"],
                            json.dumps(c.get("input_columns", [])),
                            c.get("hypothesis", ""),
                            c.get("source", "deterministic"),
                            created_at,
                        ),
                    )
            record_audit(
                user_id=user_id,
                action="propose_candidates",
                resource_type="candidates",
                resource_id=dataset_id,
                details={"candidate_count": len(candidates)},
            )
            return StorageService.list_candidates(dataset_id)
        finally:
            conn.close()

    @staticmethod
    def list_candidates(dataset_id: str) -> List[FeatureCandidate]:
        conn = get_db_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM feature_candidates WHERE dataset_id = ? ORDER BY created_at ASC",
                (dataset_id,),
            )
            return [row_to_candidate(r) for r in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def save_evaluations(
        dataset_id: str,
        evals: List[Dict[str, Any]],
        user_id: str,
    ) -> List[FeatureEvaluation]:
        evaluated_at = now_utc_iso()
        conn = get_db_connection()
        try:
            with conn:
                # Remove previous evaluations for this dataset
                conn.execute("DELETE FROM feature_evaluations WHERE dataset_id = ?", (dataset_id,))
                for e in evals:
                    eid = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO feature_evaluations
                        (id, dataset_id, feature_name, importance_score, correlation_target, leakage_risk, collinearity_with_json, verdict, reason, evaluated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            eid,
                            dataset_id,
                            e["feature_name"],
                            e["importance_score"],
                            e["correlation_target"],
                            e["leakage_risk"],
                            json.dumps(e.get("collinearity_with", [])),
                            e["verdict"],
                            e["reason"],
                            evaluated_at,
                        ),
                    )
            record_audit(
                user_id=user_id,
                action="evaluate_features",
                resource_type="evaluations",
                resource_id=dataset_id,
                details={"evaluations_count": len(evals)},
            )
            return StorageService.list_evaluations(dataset_id)
        finally:
            conn.close()

    @staticmethod
    def list_evaluations(dataset_id: str) -> List[FeatureEvaluation]:
        conn = get_db_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM feature_evaluations WHERE dataset_id = ? ORDER BY importance_score DESC",
                (dataset_id,),
            )
            return [row_to_evaluation(r) for r in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def create_pipeline(
        dataset_id: str,
        name: str,
        selected_features: List[str],
        python_code: str,
        json_spec: Dict[str, Any],
        created_by: str,
    ) -> PipelineResponse:
        pipe_id = str(uuid.uuid4())
        created_at = now_utc_iso()
        conn = get_db_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO feature_pipelines (id, dataset_id, name, selected_features_json, python_code, json_spec, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pipe_id,
                        dataset_id,
                        name,
                        json.dumps(selected_features),
                        python_code,
                        json.dumps(json_spec),
                        created_by,
                        created_at,
                    ),
                )
            record_audit(
                user_id=created_by,
                action="create_pipeline",
                resource_type="pipeline",
                resource_id=pipe_id,
                details={"name": name, "features": selected_features},
            )
            return StorageService.get_pipeline(pipe_id)  # type: ignore
        finally:
            conn.close()

    @staticmethod
    def get_pipeline(pipeline_id: str) -> Optional[PipelineResponse]:
        conn = get_db_connection()
        try:
            cur = conn.execute("SELECT * FROM feature_pipelines WHERE id = ?", (pipeline_id,))
            row = cur.fetchone()
            return row_to_pipeline(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def list_pipelines(dataset_id: Optional[str] = None) -> List[PipelineResponse]:
        conn = get_db_connection()
        try:
            if dataset_id:
                cur = conn.execute(
                    "SELECT * FROM feature_pipelines WHERE dataset_id = ? ORDER BY created_at DESC",
                    (dataset_id,),
                )
            else:
                cur = conn.execute("SELECT * FROM feature_pipelines ORDER BY created_at DESC")
            return [row_to_pipeline(r) for r in cur.fetchall()]
        finally:
            conn.close()

    @staticmethod
    def list_audit_logs(limit: int = 100, offset: int = 0) -> List[AuditLogEntry]:
        conn = get_db_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [row_to_audit(r) for r in cur.fetchall()]
        finally:
            conn.close()
