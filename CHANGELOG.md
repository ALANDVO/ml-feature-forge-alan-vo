# Changelog

## 1.3.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.3.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

### Added
- Streaming Parquet ingestion service (`backend/app/services/streaming_ingestion.py`) with Apache Arrow for large dataset processing without full memory loading.
- Chunked profiling with Welford's online algorithm for numerically stable running variance computation.
- Reservoir sampling for incremental correlation computation with target column.
- API endpoints for Parquet schema inspection (`POST /api/datasets/parquet/schema`), streaming profile (`POST /api/datasets/parquet/profile`), and dataset ingestion (`POST /api/datasets/parquet/ingest`).
- ColumnProfile class supporting both numeric and categorical data types with null handling.

## [1.2.0] - 2026-10-12

### Added
- Complete modular FastAPI backend architecture with `backend/app/main.py`, `backend/app/api/`, `backend/app/services/`, `backend/app/models/`, and `backend/app/core/`.
- Full-stack React + TypeScript + Vite web dashboard with dataset management, candidate hypothesis forge, statistical evaluation dashboard, and pipeline code export.
- Keycloak OIDC authorization-code SSO with PKCE, state/nonce validation, role-based access control (`viewer`, `analyst`, `admin`), and localhost demo mode.
- Keycloak realm configuration export with corporate SAML identity provider brokering documentation.
- Provider-agnostic LLM service supporting OpenAI-compatible, Anthropic messages, Google Gemini, and Ollama with error redaction and deterministic offline core fallback.
- SQLite transactional database layer with audit logging for all mutations.
- Multi-stage Dockerfiles for backend and frontend with non-root execution.
- Docker Compose specification (`compose.yaml`) with Keycloak, persistent volumes, and healthchecks bound to 127.0.0.1.
- End-to-end integration and unit test suite with 21 backend pytest tests and 3 frontend Vitest tests.
- GitHub Actions CI workflow for backend pytest, frontend build/test, and Docker build verification.
- Deterministic offline feature engine (log, scale, quantile bin, ratio, product, frequency, target encoding, cyclical, weekend) with Pearson-based importance ranking, leakage detection, and multicollinearity pruning.

### Fixed
- Generated production pipeline code for target-encoded features (`target_encode`) now emits a self-contained `groupby().transform()` expression computing the Bayesian-smoothed target mean. Previously the emitted expression referenced an undefined `category_mean_map` symbol, raising `NameError` at runtime whenever a target-encoded feature survived pruning and was compiled into a pipeline. The offline evaluation engine (`compute_feature_values`) and the generated pipeline now produce identical results.

## [1.0.0] - 2026-10-07

### Added
- Initial proof-of-concept CLI interface for candidate feature exploration.

