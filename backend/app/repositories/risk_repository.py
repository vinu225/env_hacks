"""
Risk Prediction Repository.

Persists traceable risk predictions with timestamps and query capabilities by location.
Ready for DynamoDB Single-Table Design (PK=LOCATION#<id>, SK=RISK#<timestamp>).
"""

from abc import abstractmethod
from typing import Dict, List, Optional
from app.repositories.base import BaseRepository
from app.schemas.risk import RiskCategory, RiskPrediction


class RiskRepository(BaseRepository[RiskPrediction, str]):
    """Abstract interface for risk prediction persistence."""

    @abstractmethod
    async def get_latest_for_location(
        self, location_id: str, category: Optional[RiskCategory] = None
    ) -> Optional[RiskPrediction]:
        """Fetch the most recent risk prediction for a location."""
        pass

    @abstractmethod
    async def list_history_for_location(
        self, location_id: str, limit: int = 50
    ) -> List[RiskPrediction]:
        """Fetch chronological prediction history for auditing and trends."""
        pass


class InMemoryRiskRepository(RiskRepository):
    """In-memory storage adapter for risk predictions."""

    def __init__(self):
        # Key: composite key f"{location_id}:{category}:{timestamp}"
        self._predictions: Dict[str, RiskPrediction] = {}

    async def get_by_id(self, entity_id: str) -> Optional[RiskPrediction]:
        return self._predictions.get(entity_id)

    async def list_all(self, limit: int = 100, skip: int = 0) -> List[RiskPrediction]:
        return list(self._predictions.values())[skip : skip + limit]

    async def save(self, entity: RiskPrediction) -> RiskPrediction:
        key = f"{entity.location_id}:{entity.category.value}:{entity.generated_at.isoformat()}"
        self._predictions[key] = entity
        return entity

    async def delete(self, entity_id: str) -> bool:
        if entity_id in self._predictions:
            del self._predictions[entity_id]
            return True
        return False

    async def get_latest_for_location(
        self, location_id: str, category: Optional[RiskCategory] = None
    ) -> Optional[RiskPrediction]:
        matching = [
            p
            for p in self._predictions.values()
            if p.location_id == location_id and (category is None or p.category == category)
        ]
        if not matching:
            return None
        matching.sort(key=lambda x: x.generated_at, reverse=True)
        return matching[0]

    async def list_history_for_location(
        self, location_id: str, limit: int = 50
    ) -> List[RiskPrediction]:
        matching = [p for p in self._predictions.values() if p.location_id == location_id]
        matching.sort(key=lambda x: x.generated_at, reverse=True)
        return matching[:limit]


_risk_repo_instance: Optional[RiskRepository] = None


def get_risk_repository() -> RiskRepository:
    global _risk_repo_instance
    if _risk_repo_instance is None:
        _risk_repo_instance = InMemoryRiskRepository()
    return _risk_repo_instance
