"""
Water Stress and Supply Risk Schemas.

Domain-specific schemas for groundwater stress, drought indices, reservoir storage,
and municipal water-supply disruptions.
"""

from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.risk import RiskPrediction


class WaterFeatureInput(BaseModel):
    """Input features required for water stress and supply disruption assessment."""
    location_id: str
    soil_moisture_index: float = Field(..., ge=0.0, le=1.0, description="Normalized soil moisture index (0=bone dry, 1=saturated)")
    consecutive_dry_days: int = Field(default=0, ge=0, description="Consecutive days without significant rainfall (>2.5mm)")
    groundwater_table_depth_m: Optional[float] = Field(default=None, description="Depth to water table in meters")
    reservoir_capacity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Regional reservoir storage %")
    supply_cut_hours_last_week: int = Field(default=0, ge=0, description="Reported hours of municipal water outage")
    water_demand_index: float = Field(default=1.0, ge=0.5, le=3.0, description="Demographic / seasonal water demand multiplier")


class WaterRiskAssessment(BaseModel):
    """Water risk assessment payload containing traceable prediction and domain indicators."""
    prediction: RiskPrediction
    water_stress_index: float = Field(..., ge=0.0, le=1.0, description="Composite water scarcity index")
    drought_stage: str = Field(default="NONE", description="NONE, WATCH, WARNING, EMERGENCY")
    supply_disruption_risk: str = Field(default="LOW", description="LOW, MODERATE, HIGH")
    groundwater_status: str = Field(default="ADEQUATE", description="ADEQUATE, SEMI_CRITICAL, CRITICAL, OVER_EXPLOITED")
    conservation_advisory: str = Field(..., description="Actionable water management recommendation")
