"""
Location Schemas.

Defines schemas for geographic entities (cities, wards, districts, coordinates).
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Coordinates(BaseModel):
    """Geographic coordinate representation."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")


class LocationBase(BaseModel):
    """Base attributes for a location."""
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable location or ward name")
    state: str = Field(..., min_length=1, max_length=100, description="State or province (e.g. Maharashtra, Delhi)")
    country: str = Field(default="India", max_length=100)
    coordinates: Coordinates
    elevation_meters: Optional[float] = Field(default=None, description="Elevation above sea level in meters")
    population_density_per_sq_km: Optional[float] = Field(default=None, ge=0.0)
    ward_or_zone: Optional[str] = Field(default=None, description="Municipal ward, zone, or sub-district ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for the location")


class LocationCreate(LocationBase):
    """Schema for registering a new location."""
    id: Optional[str] = Field(default=None, description="Optional custom ID; auto-generated if omitted")


class LocationRead(LocationBase):
    """Schema for returning location information."""
    id: str = Field(..., description="Unique location identifier")

    model_config = {"from_attributes": True}
