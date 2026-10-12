"""
Risk Prediction Schemas.

Implements the mandatory data model principles for traceable predictions:
- location_id
- risk_score (0.0 to 1.0 or 0 to 100)
- risk_level (LOW, MODERATE, HIGH, SEVERE, CRITICAL)
- confidence (0.0 to 1.0)
- generated_at
- valid_until
- model_name
- model_version
- drivers (list of key contributing factors)
- evidence (supporting metric values and flags)
- data_version
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.utils.datetime_utils import utc_now


class RiskLevel(str, Enum):
    """Categorical risk severity levels."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


class RiskCategory(str, Enum):
    """Environmental risk categories."""
    HEAT = "HEAT"
    WATER_STRESS = "WATER_STRESS"
    RAINFALL = "RAINFALL"
    FLOOD = "FLOOD"
    DROUGHT = "DROUGHT"
    COMPOSITE = "COMPOSITE"


class RiskDriver(BaseModel):
    """Traceable driver explaining why a risk score was elevated."""
    parameter: str = Field(..., description="Name of the driving feature (e.g. wet_bulb_temperature, dry_spell_days)")
    impact_level: str = Field(..., description="HIGH, MODERATE, or LOW contribution")
    observed_value: Any = Field(..., description="The observed or computed value")
    threshold_reference: Optional[str] = Field(default=None, description="Explanation or benchmark value")
    description: str = Field(..., description="Human/system readable explanation of the risk driver")


class RiskPrediction(BaseModel):
    """
    Standardized, fully traceable risk prediction contract.
    Shared by Rule-Based baselines, ML Models, and Risk Engines.
    """
    location_id: str = Field(..., description="Target geographic entity ID")
    category: RiskCategory = Field(..., description="Risk category (HEAT, WATER_STRESS, RAINFALL, etc.)")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Normalized risk score from 0 (safest) to 100 (most severe)")
    risk_level: RiskLevel = Field(..., description="Standardized severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction (0.0 - 1.0)")
    generated_at: datetime = Field(default_factory=utc_now, description="When prediction was computed")
    valid_until: datetime = Field(..., description="Prediction expiration/validity horizon")
    model_name: str = Field(..., description="Identifier of the model that produced this prediction")
    model_version: str = Field(..., description="Exact version tag of the model artifact or rule baseline")
    drivers: List[RiskDriver] = Field(default_factory=list, description="Key environmental factors driving the risk")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Raw feature inputs and calculated indicators")
    data_version: str = Field(default="v1.0", description="Schema/data pipeline version tag")


class CompositeRiskSummary(BaseModel):
    """Aggregated multi-hazard risk synthesis for a location."""
    location_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    overall_risk_score: float = Field(..., ge=0.0, le=100.0)
    overall_risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    sub_risks: Dict[str, RiskPrediction] = Field(..., description="Breakdown by individual risk categories")
    primary_hazard: RiskCategory = Field(..., description="The dominant risk factor")
    summary_text: str = Field(..., description="Deterministic synthesis statement")
