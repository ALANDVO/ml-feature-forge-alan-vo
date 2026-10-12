"""Pipeline compilation and export API endpoints."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.app.core.security import UserProfile, require_role
from backend.app.models.schemas import PipelineCompileRequest, PipelineResponse
from backend.app.services.pipeline_generator import (
    generate_pipeline_json_spec,
    generate_python_pipeline_code,
)
from backend.app.services.storage_service import StorageService

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])


@router.get("", response_model=List[PipelineResponse])
def list_pipelines(
    dataset_id: Optional[str] = Query(None),
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> List[PipelineResponse]:
    return StorageService.list_pipelines(dataset_id=dataset_id)


@router.get("/{pipeline_id}", response_model=PipelineResponse)
def get_pipeline(
    pipeline_id: str,
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> PipelineResponse:
    pipe = StorageService.get_pipeline(pipeline_id)
    if not pipe:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline not found")
    return pipe


@router.post("", response_model=PipelineResponse)
def compile_pipeline(
    req: PipelineCompileRequest,
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> PipelineResponse:
    dataset = StorageService.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    evaluations = StorageService.list_evaluations(req.dataset_id)
    candidates = StorageService.list_candidates(req.dataset_id)

    # Determine which features to include
    if req.selected_feature_ids:
        chosen_candidates = [c for c in candidates if c.id in req.selected_feature_ids]
    else:
        # Default: include features where evaluation verdict == "keep"
        keep_names = {e.feature_name for e in evaluations if e.verdict == "keep"}
        if keep_names:
            chosen_candidates = [c for c in candidates if c.name in keep_names]
        else:
            chosen_candidates = candidates[:6]

    if not chosen_candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No candidate features selected for pipeline compilation.",
        )

    cand_dicts = [
        {
            "name": c.name,
            "transform_type": c.transform_type,
            "expression": c.expression,
            "input_columns": c.input_columns,
            "hypothesis": c.hypothesis,
        }
        for c in chosen_candidates
    ]

    python_code = generate_python_pipeline_code(
        selected_features=cand_dicts,
        target_column=dataset.target_column,
        dataset_name=dataset.name,
    )

    json_spec = generate_pipeline_json_spec(
        selected_features=cand_dicts,
        target_column=dataset.target_column,
        dataset_name=dataset.name,
    )

    saved_pipeline = StorageService.create_pipeline(
        dataset_id=req.dataset_id,
        name=req.name or f"{dataset.name} Pipeline",
        selected_features=[c["name"] for c in cand_dicts],
        python_code=python_code,
        json_spec=json_spec,
        created_by=user.username,
    )

    return saved_pipeline
