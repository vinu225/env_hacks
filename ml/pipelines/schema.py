"""
Schema and Provenance Definitions for Offline ML Environmental Data Pipelines.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now


class HistoricalObservation(BaseModel):
    """
    Normalized historical meteorological observation representation.
    Completely decoupled from specific provider JSON structures (Open-Meteo, IMD, ERA5).
    """
    location_id: str = Field(..., description="Unique location identifier")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    timestamp: datetime = Field(..., description="Observation timestamp (strictly timezone-aware UTC)")
    temperature_c: Optional[float] = Field(default=None, description="Air temperature at 2m in Celsius")
    relative_humidity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Relative humidity %")
    apparent_temperature_c: Optional[float] = Field(default=None, description="Apparent / perceived temperature in Celsius")
    precipitation_mm: Optional[float] = Field(default=None, ge=0.0, description="Precipitation accumulation in mm")
    wind_speed_kmh: Optional[float] = Field(default=None, ge=0.0, description="Wind speed at 10m in km/h")
    surface_pressure_hpa: Optional[float] = Field(default=None, description="Surface barometric pressure in hPa")
    precipitation_probability_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Probability of precipitation %")
    source: str = Field(..., description="Origin data provider or archive (e.g. open_meteo_archive, imd_gridded, era5)")
    origin_type: str = Field(default="OBSERVED", description="Distinguishes 'OBSERVED' station/reanalysis data from 'FORECAST' data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provenance attributes (elevation, archive_id, etc.)")

    def get_coordinates(self) -> Coordinates:
        return Coordinates(latitude=self.latitude, longitude=self.longitude)


class DatasetMetadata(BaseModel):
    """
    Comprehensive traceability manifest recorded alongside training dataset artifacts.
    Enables future trained models to be audited against the exact data & feature versions.
    """
    dataset_name: str = Field(default="heat_risk_training_dataset")
    dataset_version: str = Field(default="v1.0", description="Version of the generated dataset artifact")
    data_version: str = Field(default="v1.0", description="Version of the raw underlying observations")
    feature_version: str = Field(default="weather-features-v1", description="Version of feature engineering logic used")
    target_version: str = Field(default="heat-target-v1", description="Version of target definition and thresholds used")
    source: str = Field(..., description="Data source provenance (e.g. open-meteo-era5, synthetic_generator)")
    data_type: str = Field(default="reanalysis", description="Data provenance tier: observation, reanalysis, or forecast")
    location_coverage: List[str] = Field(default_factory=list, description="List of location IDs included in dataset")
    locations: List[Dict[str, Any]] = Field(default_factory=list, description="Structured location configurations")
    time_range: Dict[str, str] = Field(default_factory=dict, description="Temporal range {'start': ISO, 'end': ISO}")
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    row_counts: Dict[str, int] = Field(default_factory=dict, description="Counts per split: {'total': N, 'train': N, 'val': N, 'test': N}")
    features_list: List[str] = Field(default_factory=list, description="Ordered list of feature column names")
    target_column: str = Field(default="target_heat_risk_category", description="Name of the target column")
    target_type: str = Field(default="classification", description="'classification' (ordinal categories) or 'regression' (0-100 score)")
    forecast_horizon_hours: int = Field(default=6, description="Forward forecast window horizon H in hours")
    schema_version: str = Field(default="v1.1", description="Contract schema version")
    is_synthetic: bool = Field(default=False, description="Flag explicitly indicating if data is synthetic/sample for testing")
    provider_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters for the historical provider")
    target_generator_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters for the target generator")
    provenance_note: Optional[str] = Field(default=None, description="Human-readable provenance or disclaimer")
    ready_for_model_training: bool = Field(default=False, description="Audit readiness status")


class DatasetQualityReport(BaseModel):
    """
    Quality and health summary computed prior to dataset export or model training.
    """
    dataset_version: str = Field(default="heat-risk-dataset-v1")
    source: str = Field(default="open-meteo-era5")
    data_type: str = Field(default="reanalysis")
    total_rows: int
    location_count: int
    temporal_frequency: str = Field(default="hourly")
    observation_reanalysis_distinction: str = Field(
        default="ECMWF ERA5 / ERA5-Land reanalysis via Open-Meteo Archive API (assimilated numerical models + observations, not raw station gauge)"
    )
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    missing_percentage_by_column: Dict[str, float] = Field(default_factory=dict)
    feature_availability: Dict[str, bool] = Field(default_factory=dict)
    target_distribution: Dict[str, Any] = Field(default_factory=dict)
    target_percentages: Dict[str, float] = Field(default_factory=dict)
    location_distribution: Dict[str, int] = Field(default_factory=dict)
    observations_per_location: Dict[str, int] = Field(default_factory=dict)
    dropped_rows_summary: Dict[str, int] = Field(default_factory=dict)
    temporal_split_timestamps: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    yearly_distribution: Dict[str, int] = Field(default_factory=dict)
    monthly_distribution: Dict[str, int] = Field(default_factory=dict)
    seasonal_distribution: Dict[str, int] = Field(default_factory=dict)
    heat_events_summary: Dict[str, Any] = Field(default_factory=dict)
    persistence_baseline_summary: Dict[str, Any] = Field(default_factory=dict)
    target_predictability_check: Dict[str, Any] = Field(default_factory=dict)
    duplicate_timestamp_count: int = 0
    invalid_value_count: int = 0
    validation_passed: bool = True
    issues: List[str] = Field(default_factory=list)
