"""
Feature Schema and Traceability Definitions.

Defines a uniform tabular feature contract shared identically by:
- Online live inference (FastAPI serving)
- Offline ML training and backtesting pipelines

Enforces data_version and feature_version for auditability and drift tracking.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now

DATA_VERSION: str = "v1.0"
FEATURE_VERSION: str = "weather-features-v1"


class FeatureValue(BaseModel):
    """Individual feature measurement with unit, provenance, and availability."""
    name: str = Field(..., description="Feature identifier (e.g. heat_index_c, rolling_24h_rainfall_mm)")
    value: Optional[float] = Field(default=None, description="Computed numeric feature value, or None if unavailable")
    unit: Optional[str] = Field(default=None, description="Physical unit of measurement (°C, mm, %, km/h)")
    is_available: bool = Field(default=True, description="True if value was computed from real observations, False if missing/unobserved")
    category: str = Field(default="general", description="Feature category: weather, heat, water, rainfall")
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Inherited quality score from underlying observations")
    description: Optional[str] = Field(default=None, description="Human-readable description of the feature calculation")


class EnvironmentalFeatureVector(BaseModel):
    """
    Standardized tabular feature vector produced by feature engineering.
    Contains both the detailed typed feature objects and helper methods to output
    flat numerical dictionaries for ML models (XGBoost, Scikit-learn, etc.).
    """
    location_id: Optional[str] = Field(default=None)
    coordinates: Coordinates
    generated_at: datetime = Field(default_factory=utc_now)
    data_version: str = Field(default=DATA_VERSION)
    feature_version: str = Field(default=FEATURE_VERSION)
    features: Dict[str, FeatureValue] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_feature(
        self,
        name: str,
        value: Optional[float],
        unit: Optional[str] = None,
        category: str = "general",
        is_available: bool = True,
        data_quality: float = 1.0,
        description: Optional[str] = None,
    ) -> None:
        """Helper to append or update a feature."""
        self.features[name] = FeatureValue(
            name=name,
            value=value if is_available and value is not None else None,
            unit=unit,
            is_available=is_available and (value is not None),
            category=category,
            data_quality=data_quality,
            description=description,
        )

    def to_numeric_dict(self) -> Dict[str, Optional[float]]:
        """
        Produce flat numeric dictionary for ML inference or training dataframe construction.
        Unavailable features yield None (or np.nan in tabular contexts).
        """
        return {name: feat.value if feat.is_available else None for name, feat in self.features.items()}

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Produce traceability metadata payload."""
        return {
            "location_id": self.location_id,
            "coordinates": self.coordinates.model_dump(),
            "generated_at": self.generated_at.isoformat(),
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "feature_count": len(self.features),
            "available_feature_count": sum(1 for f in self.features.values() if f.is_available),
        }
