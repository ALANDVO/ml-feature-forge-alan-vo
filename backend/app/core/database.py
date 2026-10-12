"""SQLite persistent storage layer with parameterized SQL and transactions."""
import sqlite3
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.config import get_settings


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db_connection() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                col_count INTEGER NOT NULL,
                target_column TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                sample_data_json TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feature_candidates (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                transform_type TEXT NOT NULL,
                expression TEXT NOT NULL,
                input_columns_json TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feature_evaluations (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                correlation_target REAL NOT NULL,
                leakage_risk TEXT NOT NULL,
                collinearity_with_json TEXT NOT NULL,
                verdict TEXT NOT NULL,
                reason TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feature_pipelines (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                selected_features_json TEXT NOT NULL,
                python_code TEXT NOT NULL,
                json_spec TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                details_json TEXT NOT NULL,
                ip_address TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_dataset ON feature_candidates(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON feature_evaluations(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_pipelines_dataset ON feature_pipelines(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp);
            """)
    finally:
        conn.close()


def record_audit(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: Dict[str, Any],
    ip_address: Optional[str] = None
) -> None:
    import uuid
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO audit_logs (id, user_id, action, resource_type, resource_id, details_json, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(details),
                    ip_address or "127.0.0.1",
                    now_utc_iso()
                )
            )
    finally:
        conn.close()
