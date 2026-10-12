"""
Location Repository Interface and In-Memory Adapter.

Designed to be swapped with a DynamoDB implementation (Boto3 / Aiobotocore)
without altering service or API layers.
"""

from abc import abstractmethod
from typing import Dict, List, Optional
from app.repositories.base import BaseRepository
from app.schemas.location import LocationRead


class LocationRepository(BaseRepository[LocationRead, str]):
    """Abstract interface for location persistence."""

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[LocationRead]:
        """Lookup location by name or ward."""
        pass


class InMemoryLocationRepository(LocationRepository):
    """
    In-memory location store for local execution, unit tests, and fast development.
    Pre-seeded with key Indian urban centers.
    """

    def __init__(self):
        self._store: Dict[str, LocationRead] = {}
        self._seed_default_locations()

    def _seed_default_locations(self) -> None:
        defaults = [
            LocationRead(
                id="loc_delhi_central",
                name="Central Delhi (Connaught Place / NDMC)",
                state="Delhi",
                country="India",
                coordinates={"latitude": 28.6315, "longitude": 77.2167},
                elevation_meters=216.0,
                population_density_per_sq_km=11297.0,
                ward_or_zone="Zone 1",
                metadata={"hotspot_type": "Dense Urban Heat Island"},
            ),
            LocationRead(
                id="loc_mumbai_kurla",
                name="Kurla (Mithi River Basin)",
                state="Maharashtra",
                country="India",
                coordinates={"latitude": 19.0726, "longitude": 72.8845},
                elevation_meters=8.0,
                population_density_per_sq_km=25000.0,
                ward_or_zone="Ward L",
                metadata={"hotspot_type": "High Vulnerability Flood & Waterlogging Basin"},
            ),
            LocationRead(
                id="loc_chennai_velachery",
                name="Velachery (South Chennai Basin)",
                state="Tamil Nadu",
                country="India",
                coordinates={"latitude": 12.9815, "longitude": 80.2180},
                elevation_meters=6.0,
                population_density_per_sq_km=18000.0,
                ward_or_zone="Zone 13",
                metadata={"hotspot_type": "Inundation & Groundwater Stress"},
            ),
            LocationRead(
                id="loc_bengaluru_bellandur",
                name="Bellandur / Outer Ring Road",
                state="Karnataka",
                country="India",
                coordinates={"latitude": 12.9345, "longitude": 77.6775},
                elevation_meters=880.0,
                population_density_per_sq_km=14000.0,
                ward_or_zone="Ward 150",
                metadata={"hotspot_type": "Urban Runoff & Lake Catchment Stress"},
            ),
        ]
        for loc in defaults:
            self._store[loc.id] = loc

    async def get_by_id(self, entity_id: str) -> Optional[LocationRead]:
        return self._store.get(entity_id)

    async def get_by_name(self, name: str) -> Optional[LocationRead]:
        for loc in self._store.values():
            if loc.name.lower() == name.lower():
                return loc
        return None

    async def list_all(self, limit: int = 100, skip: int = 0) -> List[LocationRead]:
        return list(self._store.values())[skip : skip + limit]

    async def save(self, entity: LocationRead) -> LocationRead:
        self._store[entity.id] = entity
        return entity

    async def delete(self, entity_id: str) -> bool:
        if entity_id in self._store:
            del self._store[entity_id]
            return True
        return False


# Singleton instance for repository dependency injection
_location_repo_instance: Optional[LocationRepository] = None


def get_location_repository() -> LocationRepository:
    global _location_repo_instance
    if _location_repo_instance is None:
        _location_repo_instance = InMemoryLocationRepository()
    return _location_repo_instance
