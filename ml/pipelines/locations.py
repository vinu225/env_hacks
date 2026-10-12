"""
Location Registry and Geographic Coverage Configurations for Environmental Data Pipelines.

Defines canonical monitoring and historical training locations across India,
representing diverse climatic regimes (arid, humid subtropical, tropical wet/dry, coastal monsoon).
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LocationConfig(BaseModel):
    """Configuration schema for a geographic observation or reanalysis station."""
    location_id: str = Field(..., description="Unique alphanumeric location identifier")
    name: str = Field(..., description="Common city or station name")
    state: str = Field(..., description="State or Union Territory")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    climate_zone: str = Field(..., description="Köppen-Geiger / IMD climate classification")
    elevation_m: Optional[float] = Field(default=None, description="Elevation above mean sea level in meters")


# 10 Canonical Indian Cities for Heat Risk Modeling spanning diverse climate zones
DEFAULT_LOCATIONS: List[LocationConfig] = [
    LocationConfig(
        location_id="loc_delhi",
        name="Delhi",
        state="Delhi",
        latitude=28.6139,
        longitude=77.2090,
        climate_zone="Semi-Arid (Hot Subtropical)",
        elevation_m=216.0,
    ),
    LocationConfig(
        location_id="loc_bhopal",
        name="Bhopal",
        state="Madhya Pradesh",
        latitude=23.2599,
        longitude=77.4126,
        climate_zone="Humid Subtropical",
        elevation_m=527.0,
    ),
    LocationConfig(
        location_id="loc_jaipur",
        name="Jaipur",
        state="Rajasthan",
        latitude=26.9124,
        longitude=75.7873,
        climate_zone="Hot Semi-Arid",
        elevation_m=431.0,
    ),
    LocationConfig(
        location_id="loc_nagpur",
        name="Nagpur",
        state="Maharashtra",
        latitude=21.1458,
        longitude=79.0882,
        climate_zone="Tropical Wet and Dry",
        elevation_m=310.0,
    ),
    LocationConfig(
        location_id="loc_lucknow",
        name="Lucknow",
        state="Uttar Pradesh",
        latitude=26.8467,
        longitude=80.9462,
        climate_zone="Humid Subtropical (Gangetic Plain)",
        elevation_m=123.0,
    ),
    LocationConfig(
        location_id="loc_ahmedabad",
        name="Ahmedabad",
        state="Gujarat",
        latitude=23.0225,
        longitude=72.5714,
        climate_zone="Hot Semi-Arid",
        elevation_m=53.0,
    ),
    LocationConfig(
        location_id="loc_hyderabad",
        name="Hyderabad",
        state="Telangana",
        latitude=17.3850,
        longitude=78.4867,
        climate_zone="Tropical Wet and Dry",
        elevation_m=542.0,
    ),
    LocationConfig(
        location_id="loc_chennai",
        name="Chennai",
        state="Tamil Nadu",
        latitude=13.0827,
        longitude=80.2707,
        climate_zone="Tropical Wet and Dry (Coastal)",
        elevation_m=6.0,
    ),
    LocationConfig(
        location_id="loc_mumbai",
        name="Mumbai",
        state="Maharashtra",
        latitude=19.0760,
        longitude=72.8777,
        climate_zone="Tropical Monsoon (Coastal)",
        elevation_m=14.0,
    ),
    LocationConfig(
        location_id="loc_bengaluru",
        name="Bengaluru",
        state="Karnataka",
        latitude=12.9716,
        longitude=77.5946,
        climate_zone="Tropical Savanna (Plateau)",
        elevation_m=920.0,
    ),
]


def get_location_registry() -> Dict[str, LocationConfig]:
    """Returns mapping of location_id -> LocationConfig."""
    return {loc.location_id: loc for loc in DEFAULT_LOCATIONS}


def get_location_by_id(location_id: str) -> Optional[LocationConfig]:
    """Lookup a location configuration by its canonical ID."""
    return get_location_registry().get(location_id)
