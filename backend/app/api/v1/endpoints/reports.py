"""
Community Incident Reporting Endpoints.
"""

from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.repositories.report_repository import ReportRepository, get_report_repository
from app.schemas.report import IncidentReportCreate, IncidentReportRead, VerificationStatus
from app.utils.datetime_utils import utc_now

router = APIRouter(prefix="/reports", tags=["Community Incident Reports"])


@router.get("", response_model=List[IncidentReportRead], summary="List community environmental reports")
async def list_reports(
    location_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    repo: ReportRepository = Depends(get_report_repository),
) -> List[IncidentReportRead]:
    if location_id:
        return await repo.list_by_location(location_id=location_id, limit=limit)
    return await repo.list_all(limit=limit, skip=skip)


@router.post("", response_model=IncidentReportRead, status_code=status.HTTP_201_CREATED, summary="Submit community report")
async def create_report(
    payload: IncidentReportCreate,
    repo: ReportRepository = Depends(get_report_repository),
) -> IncidentReportRead:
    report = IncidentReportRead(
        id=f"rep_{uuid4().hex[:10]}",
        location_id=payload.location_id,
        coordinates=payload.coordinates,
        incident_type=payload.incident_type,
        severity=payload.severity,
        description=payload.description,
        reporter_id=payload.reporter_id,
        metadata=payload.metadata,
        reported_at=utc_now(),
        verification_status=VerificationStatus.UNVERIFIED,
        upvote_count=0,
    )
    return await repo.save(report)


@router.post("/{report_id}/upvote", response_model=IncidentReportRead, summary="Upvote incident report")
async def upvote_report(
    report_id: str,
    repo: ReportRepository = Depends(get_report_repository),
) -> IncidentReportRead:
    updated = await repo.upvote(report_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found.",
        )
    return updated
