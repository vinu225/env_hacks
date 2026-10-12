"""
Community Incident Report Repository.

Handles crowdsourced reports and community upvotes.
"""

from abc import abstractmethod
from typing import Dict, List, Optional
from app.repositories.base import BaseRepository
from app.schemas.report import IncidentReportRead, VerificationStatus


class ReportRepository(BaseRepository[IncidentReportRead, str]):
    """Abstract interface for incident report storage."""

    @abstractmethod
    async def list_by_location(
        self, location_id: str, limit: int = 50
    ) -> List[IncidentReportRead]:
        pass

    @abstractmethod
    async def upvote(self, report_id: str) -> Optional[IncidentReportRead]:
        pass


class InMemoryReportRepository(ReportRepository):
    """In-memory incident report storage adapter."""

    def __init__(self):
        self._reports: Dict[str, IncidentReportRead] = {}

    async def get_by_id(self, entity_id: str) -> Optional[IncidentReportRead]:
        return self._reports.get(entity_id)

    async def list_all(self, limit: int = 100, skip: int = 0) -> List[IncidentReportRead]:
        return list(self._reports.values())[skip : skip + limit]

    async def list_by_location(
        self, location_id: str, limit: int = 50
    ) -> List[IncidentReportRead]:
        results = [r for r in self._reports.values() if r.location_id == location_id]
        results.sort(key=lambda x: x.reported_at, reverse=True)
        return results[:limit]

    async def save(self, entity: IncidentReportRead) -> IncidentReportRead:
        self._reports[entity.id] = entity
        return entity

    async def delete(self, entity_id: str) -> bool:
        if entity_id in self._reports:
            del self._reports[entity_id]
            return True
        return False

    async def upvote(self, report_id: str) -> Optional[IncidentReportRead]:
        report = self._reports.get(report_id)
        if not report:
            return None
        updated = report.model_copy(
            update={
                "upvote_count": report.upvote_count + 1,
                "verification_status": (
                    VerificationStatus.COMMUNITY_UPVOTED
                    if report.upvote_count + 1 >= 3
                    else report.verification_status
                ),
            }
        )
        self._reports[report_id] = updated
        return updated


_report_repo_instance: Optional[ReportRepository] = None


def get_report_repository() -> ReportRepository:
    global _report_repo_instance
    if _report_repo_instance is None:
        _report_repo_instance = InMemoryReportRepository()
    return _report_repo_instance
