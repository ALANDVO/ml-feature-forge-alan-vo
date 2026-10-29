"""Dataset management API endpoints."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from backend.app.core.security import UserProfile, require_role
from backend.app.models.schemas import DatasetCreateRequest, DatasetResponse
from backend.app.services.feature_engine import parse_csv_data, profile_dataset
from backend.app.services.storage_service import StorageService
from backend.app.services.streaming_ingestion import (
    StreamingProfiler,
    infer_parquet_schema,
    infer_parquet_schema_from_bytes,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


SAMPLE_DATASET_CSV = """id,age,income,credit_score,loan_amount,debt_to_income,months_employed,city_tier,approved
1,28,45000,680,12000,0.26,36,tier_1,1
2,45,85000,720,25000,0.29,120,tier_1,1
3,22,28000,590,8000,0.35,12,tier_3,0
4,35,62000,640,18000,0.31,48,tier_2,1
5,50,110000,790,40000,0.22,180,tier_1,1
6,24,31000,610,10000,0.40,18,tier_2,0
7,41,74000,690,22000,0.28,84,tier_1,1
8,31,52000,630,15000,0.33,30,tier_3,0
9,58,95000,760,30000,0.24,240,tier_1,1
10,29,48000,670,14000,0.27,42,tier_2,1
11,62,125000,810,45000,0.19,300,tier_1,1
12,23,26000,580,7500,0.42,8,tier_3,0
13,38,68000,660,20000,0.30,60,tier_2,1
14,47,91000,740,28000,0.25,144,tier_1,1
15,33,56000,650,16500,0.32,36,tier_3,1
16,26,34000,600,11000,0.38,24,tier_2,0
17,52,105000,780,38000,0.23,200,tier_1,1
18,30,49000,665,13500,0.28,38,tier_2,1
19,43,80000,710,24000,0.29,96,tier_1,1
20,21,24000,570,6000,0.45,6,tier_3,0
"""


@router.get("", response_model=List[DatasetResponse])
def list_datasets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> List[DatasetResponse]:
    return StorageService.list_datasets(limit=limit, offset=offset)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> DatasetResponse:
    dataset = StorageService.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


@router.post("", response_model=DatasetResponse)
def create_dataset(
    req: DatasetCreateRequest,
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> DatasetResponse:
    if req.csv_content:
        cols, rows = parse_csv_data(req.csv_content)
    elif req.data_rows:
        rows = req.data_rows
        cols = list(rows[0].keys()) if rows else []
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either csv_content or data_rows must be provided",
        )

    if not rows or not cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least one column and one row of data",
        )

    if req.target_column not in cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{req.target_column}' does not exist in dataset columns: {cols}",
        )

    profile = profile_dataset(cols, rows, req.target_column)

    return StorageService.create_dataset(
        name=req.name,
        filename=f"{req.name.lower().replace(' ', '_')}.csv",
        row_count=len(rows),
        col_count=len(cols),
        target_column=req.target_column,
        problem_type=profile.problem_type,
        profile=profile,
        sample_rows=rows[:10],
        created_by=user.username,
    )


@router.post("/demo-load", response_model=DatasetResponse)
def load_sample_dataset(
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> DatasetResponse:
    cols, rows = parse_csv_data(SAMPLE_DATASET_CSV)
    target_column = "approved"
    profile = profile_dataset(cols, rows, target_column)
    return StorageService.create_dataset(
        name="Credit Loan Assessment (Sample)",
        filename="credit_loan_sample.csv",
        row_count=len(rows),
        col_count=len(cols),
        target_column=target_column,
        problem_type="classification",
        profile=profile,
        sample_rows=rows[:10],
        created_by=user.username,
    )


@router.post("/parquet/schema")
async def get_parquet_schema(
    file: UploadFile = File(...),
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> Dict[str, Any]:
    """Return column schema for a Parquet file without loading full data."""
    if not file.filename or not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .parquet extension",
        )

    content = await file.read()
    try:
        schema = infer_parquet_schema_from_bytes(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Parquet file: {str(e)}",
        )

    return {"columns": schema, "filename": file.filename}


@router.post("/parquet/profile")
async def profile_parquet_streaming(
    file: UploadFile = File(...),
    target_column: str = Query(...),
    chunk_size: int = Query(10000, ge=1000, le=100000),
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> Dict[str, Any]:
    """Stream a Parquet file and return incremental profile without loading full data."""
    if not file.filename or not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .parquet extension",
        )

    content = await file.read()
    profiler = StreamingProfiler(target_column=target_column, chunk_size=chunk_size)

    try:
        profile = profiler.process_bytes(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process Parquet file: {str(e)}",
        )

    if target_column not in profile["columns"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found in Parquet columns",
        )

    return profile


@router.post("/parquet/ingest", response_model=DatasetResponse)
async def ingest_parquet_dataset(
    file: UploadFile = File(...),
    name: str = Query(...),
    target_column: str = Query(...),
    chunk_size: int = Query(10000, ge=1000, le=100000),
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> DatasetResponse:
    """Stream a Parquet file, profile it, and save as a dataset."""
    if not file.filename or not file.filename.endswith(".parquet"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .parquet extension",
        )

    content = await file.read()
    profiler = StreamingProfiler(target_column=target_column, chunk_size=chunk_size)

    try:
        profile = profiler.process_bytes(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process Parquet file: {str(e)}",
        )

    if target_column not in profile["columns"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found in Parquet columns",
        )

    # Convert streaming profile to DatasetProfile schema format
    dtypes = {}
    missing_counts = {}
    missing_rates = {}
    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    stats = {}

    for col_name, col_profile in profile["columns"].items():
        is_numeric = col_profile.get("is_numeric", False)
        if is_numeric:
            dtypes[col_name] = "numeric"
            numeric_columns.append(col_name)
            stats[col_name] = {
                "min": col_profile.get("min"),
                "max": col_profile.get("max"),
                "mean": col_profile.get("mean"),
                "std": col_profile.get("std"),
                "distinct_count": col_profile.get("count", 0),
            }
        else:
            dtypes[col_name] = "categorical"
            categorical_columns.append(col_name)
            stats[col_name] = {
                "distinct_count": col_profile.get("unique_count", 0),
                "top_categories": [v[0] for v in col_profile.get("top_values", [])[:5]],
            }
        missing_counts[col_name] = col_profile.get("null_count", 0)
        missing_rates[col_name] = col_profile.get("null_ratio", 0.0)

    # Detect problem type based on target column
    target_profile = profile["columns"][target_column]
    target_is_numeric = target_profile.get("is_numeric", False)
    if target_is_numeric:
        unique_targets = target_profile.get("unique_count", 0)
        problem_type = "classification" if unique_targets <= 10 else "regression"
    else:
        problem_type = "classification"

    from backend.app.models.schemas import DatasetProfile
    dataset_profile = DatasetProfile(
        row_count=profile["row_count"],
        col_count=len(profile["columns"]),
        columns=list(profile["columns"].keys()),
        dtypes=dtypes,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        missing_counts=missing_counts,
        missing_rates=missing_rates,
        stats=stats,
        target_column=target_column,
        problem_type=problem_type,
    )

    # Extract sample rows from the first chunk
    # For simplicity, we'll create a minimal sample from the profiler
    sample_rows = []

    return StorageService.create_dataset(
        name=name,
        filename=file.filename,
        row_count=profile["row_count"],
        col_count=len(profile["columns"]),
        target_column=target_column,
        problem_type=problem_type,
        profile=dataset_profile,
        sample_rows=sample_rows,
        created_by=user.username,
    )


@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: str,
    user: UserProfile = Depends(require_role(["admin"])),
) -> Dict[str, Any]:
    deleted = StorageService.delete_dataset(dataset_id, deleted_by=user.username)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return {"status": "deleted", "dataset_id": dataset_id}
