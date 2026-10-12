"""Feature evaluation and statistical quality API endpoints."""
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.core.security import UserProfile, require_role
from backend.app.models.schemas import EvaluateFeaturesRequest, FeatureEvaluation
from backend.app.services.feature_engine import evaluate_candidates
from backend.app.services.storage_service import StorageService

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get("/{dataset_id}", response_model=List[FeatureEvaluation])
def list_evaluations(
    dataset_id: str,
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> List[FeatureEvaluation]:
    return StorageService.list_evaluations(dataset_id)


@router.post("", response_model=List[FeatureEvaluation])
def run_evaluation(
    req: EvaluateFeaturesRequest,
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> List[FeatureEvaluation]:
    dataset = StorageService.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    candidates = StorageService.list_candidates(req.dataset_id)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No candidate features found for this dataset. Please propose features first.",
        )

    cand_dicts = [
        {
            "name": c.name,
            "transform_type": c.transform_type,
            "expression": c.expression,
            "input_columns": c.input_columns,
            "hypothesis": c.hypothesis,
            "source": c.source,
        }
        for c in candidates
    ]

    eval_results = evaluate_candidates(
        candidates=cand_dicts,
        rows=dataset.sample_rows,
        target_column=dataset.target_column,
        collinearity_threshold=req.collinearity_threshold,
        min_importance_threshold=req.min_importance_threshold,
    )

    saved = StorageService.save_evaluations(
        dataset_id=req.dataset_id,
        evals=eval_results,
        user_id=user.username,
    )
    return saved
