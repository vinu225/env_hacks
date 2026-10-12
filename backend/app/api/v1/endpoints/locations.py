"""
Location API Endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.repositories.location_repository import LocationRepository, get_location_repository
from app.schemas.location import LocationCreate, LocationRead

router = APIRouter(prefix="/locations", tags=["Locations"])


@router.get("", response_model=List[LocationRead], summary="List monitored geographic locations")
async def list_locations(
    limit: int = Query(default=50, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    repo: LocationRepository = Depends(get_location_repository),
) -> List[LocationRead]:
    return await repo.list_all(limit=limit, skip=skip)


@router.get("/{location_id}", response_model=LocationRead, summary="Get location details by ID")
async def get_location(
    location_id: str,
    repo: LocationRepository = Depends(get_location_repository),
) -> LocationRead:
    location = await repo.get_by_id(location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{location_id}' not found.",
        )
    return location


@router.post("", response_model=LocationRead, status_code=status.HTTP_201_CREATED, summary="Register new location")
async def create_location(
    payload: LocationCreate,
    repo: LocationRepository = Depends(get_location_repository),
) -> LocationRead:
    loc_id = payload.id or f"loc_{payload.name.lower().replace(' ', '_')}"
    new_loc = LocationRead(
        id=loc_id,
        name=payload.name,
        state=payload.state,
        country=payload.country,
        coordinates=payload.coordinates,
        elevation_meters=payload.elevation_meters,
        population_density_per_sq_km=payload.population_density_per_sq_km,
        ward_or_zone=payload.ward_or_zone,
        metadata=payload.metadata,
    )
    return await repo.save(new_loc)
