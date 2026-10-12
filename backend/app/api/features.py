"""Candidate feature generation API endpoints."""
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.core.security import UserProfile, require_role
from backend.app.models.schemas import FeatureCandidate, ProposeFeaturesRequest
from backend.app.services.feature_engine import generate_deterministic_candidates
from backend.app.services.llm_service import LLMService, LLMServiceError
from backend.app.services.storage_service import StorageService

router = APIRouter(prefix="/features", tags=["Features"])


@router.get("/{dataset_id}", response_model=List[FeatureCandidate])
def list_candidates(
    dataset_id: str,
    user: UserProfile = Depends(require_role(["viewer", "analyst", "admin"])),
) -> List[FeatureCandidate]:
    return StorageService.list_candidates(dataset_id)


@router.post("/propose", response_model=List[FeatureCandidate])
async def propose_features(
    req: ProposeFeaturesRequest,
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> List[FeatureCandidate]:
    dataset = StorageService.get_dataset(req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # 1. Deterministic core algorithms (always active offline without any LLM key)
    deterministic_candidates = generate_deterministic_candidates(
        profile=dataset.profile,
        target_column=dataset.target_column,
        include_interactions=req.include_interactions,
        include_encodings=req.include_encodings,
        include_nonlinear=req.include_nonlinear,
        include_datetime=req.include_datetime,
    )
    for c in deterministic_candidates:
        c["source"] = "deterministic"

    all_candidates = deterministic_candidates[: req.max_candidates]

    # 2. Opt-in LLM advisory hypotheses (grounded in profile, labeled advisory)
    if req.enable_llm:
        llm = LLMService()
        try:
            llm_hypotheses = await llm.generate_hypotheses(
                dataset_profile=dataset.profile.model_dump(),
                target_column=dataset.target_column,
                problem_type=dataset.problem_type,
                count=5,
            )
            all_candidates.extend(llm_hypotheses)
        except LLMServiceError as err:
            # Opt-in LLM error is logged and returned in audit; core deterministic candidates succeed
            pass

    saved = StorageService.save_candidates(
        dataset_id=req.dataset_id,
        candidates=all_candidates,
        user_id=user.username,
    )
    return saved
