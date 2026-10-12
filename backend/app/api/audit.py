"""Audit log inspection API endpoints."""
from typing import List
from fastapi import APIRouter, Depends, Query
from backend.app.core.security import UserProfile, require_role
from backend.app.models.schemas import AuditLogEntry
from backend.app.services.storage_service import StorageService

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("", response_model=List[AuditLogEntry])
def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: UserProfile = Depends(require_role(["analyst", "admin"])),
) -> List[AuditLogEntry]:
    return StorageService.list_audit_logs(limit=limit, offset=offset)
