"""
Heat Risk Schemas.

Domain-specific schemas for heatwaves, apparent temperature, wet-bulb metrics,
and urban thermal stress indicators.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.risk import RiskPrediction


class HeatFeatureInput(BaseModel):
    """Input features required for heat risk calculation (deterministic or ML)."""
    location_id: str
    temperature_c: float = Field(..., description="Current dry-bulb temperature in °C")
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0)
    wind_speed_kmh: float = Field(default=5.0, ge=0.0)
    max_temperature_forecast_24h_c: Optional[float] = Field(default=None)
    min_temperature_night_c: Optional[float] = Field(default=None, description="Nighttime recovery temperature")
    consecutive_hot_days: int = Field(default=0, ge=0, description="Consecutive days exceeding regional threshold")
    urban_heat_island_factor: float = Field(default=1.0, ge=1.0, le=2.0, description="Localized UHI amplification multiplier")


class HeatRiskAssessment(BaseModel):
    """Heat-specific assessment payload containing traceable prediction and domain metrics."""
    prediction: RiskPrediction
    heat_index_c: float = Field(..., description="Calculated heat index in °C")
    wet_bulb_temp_c: Optional[float] = Field(default=None, description="Calculated wet-bulb temperature in °C")
    is_heatwave_condition: bool = Field(default=False, description="Flag indicating if regional heatwave criteria are met")
    nighttime_recovery_adequate: bool = Field(default=True, description="False if night temperatures remain dangerously high")
    advisory_recommendation: str = Field(..., description="Deterministic public safety guidance")
