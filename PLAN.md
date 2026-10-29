# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

streaming-data-ingestion

## Roadmap

### core-feature-forge-engine — Full-Stack Feature Engineering Baseline with Statistical Evaluation and OIDC Auth

Status: done

Acceptance:
- FastAPI backend and React Vite frontend baseline
- Deterministic statistical feature generation and profiling
- Feature quality evaluation with leakage and multicollinearity pruning
- Production Python pipeline export
- Keycloak OIDC PKCE authentication and localhost demo mode

Implementation: `backend/app/main.py`, `backend/app/services/feature_engine.py`, `backend/app/services/importance.py`, `backend/app/services/pipeline_generator.py`, `backend/app/services/llm_service.py`, `backend/app/services/storage_service.py`, `backend/app/core/security.py`, `frontend/src/App.tsx`

Tests: `backend/tests/test_feature_engine.py`, `backend/tests/test_api_workflows.py`, `backend/tests/test_auth.py`, `backend/tests/test_llm_service.py`, `backend/tests/test_pipeline_generation.py`, `frontend/src/App.test.tsx`

### target-encode-pipeline-repair — Fix non-executable target-encoded pipeline code generation

Status: done

Acceptance:
- Compiled pipeline code for target_encode features runs without NameError
- Generated target-encoded values match the offline compute_feature_values engine

Implementation: `backend/app/services/feature_engine.py`, `backend/tests/test_pipeline_generation.py`

Tests: `backend/tests/test_pipeline_generation.py::test_target_encode_pipeline_expression_is_self_contained`, `backend/tests/test_pipeline_generation.py::test_all_transform_types_produce_compilable_pipeline`

### streaming-data-ingestion — Streaming Tabular Data Ingestion with Apache Arrow / Parquet

Status: in_progress

Acceptance:
- Support streaming large parquet files without loading entire dataset into memory
- Chunked profiling and running statistical variance updates

Implementation: `backend/app/services/streaming_ingestion.py`, `backend/app/api/datasets.py`

Tests: `backend/tests/test_streaming_ingestion.py`

### automated-hyperparameter-tuning — Automated Downstream Model Cross-Validation and Feature Ablation

Status: planned

Acceptance:
- Train baseline LightGBM / XGBoost models on engineered features
- Measure SHAP values and cross-validated ROC-AUC / RMSE gain

