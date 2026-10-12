"""
Domain Entities for Locations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LocationEntity:
    """Core domain representation of a monitored geographic area."""
    id: str
    name: str
    state: str
    country: str
    latitude: float
    longitude: float
    elevation_meters: Optional[float] = None
    population_density: Optional[float] = None
    ward_or_zone: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
