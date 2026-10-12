"""
Domain Entities for Rainfall, Inundation, and Flood Dynamics.
"""

from dataclasses import dataclass


@dataclass
class PrecipitationMetrics:
    """Core domain metrics for rainfall intensity and drainage load."""
    intensity_hourly_mm: float
    accumulated_24h_mm: float
    soil_saturation_pct: float
    drainage_capacity_rating: float
    is_low_lying: bool = False
