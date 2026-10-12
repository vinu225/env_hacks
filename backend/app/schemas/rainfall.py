"""
Rainfall and Flood / Waterlogging Risk Schemas.

Domain-specific schemas for hyperlocal precipitation extremes, drainage bottlenecks,
and waterlogging vulnerabilities.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.risk import RiskPrediction


class RainfallFeatureInput(BaseModel):
    """Input features for rainfall risk, flash-flood, and urban waterlogging assessment."""
    location_id: str
    precipitation_current_hour_mm: float = Field(..., ge=0.0)
    precipitation_last_24h_mm: float = Field(..., ge=0.0)
    precipitation_forecast_next_6h_mm: float = Field(default=0.0, ge=0.0)
    soil_saturation_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    drainage_capacity_rating: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Drainage efficiency rating (0.0=severely blocked/poor, 1.0=optimal storm drains)",
    )
    is_low_lying_basin: bool = Field(default=False, description="Topographical depression flag")


class RainfallRiskAssessment(BaseModel):
    """Rainfall risk assessment payload with traceable prediction and flood metrics."""
    prediction: RiskPrediction
    waterlogging_risk: str = Field(default="LOW", description="LOW, MODERATE, HIGH, SEVERE")
    flash_flood_potential: bool = Field(default=False)
    runoff_coefficient: float = Field(..., ge=0.0, le=1.0)
    drainage_overflow_expected: bool = Field(default=False)
    flood_warning_summary: str = Field(..., description="Actionable drainage and movement advisory")
