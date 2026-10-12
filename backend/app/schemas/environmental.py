"""
Environmental Observation Schemas.

Enforces strict traceability for every environmental observation:
- source (e.g. open_meteo, mock_weather_provider)
- location (coordinates or location_id)
- observed_at vs ingested_at vs forecast_valid_at (strictly timezone-aware UTC)
- value and unit mappings
- freshness (age in seconds, is_fresh boolean)
- data_quality_score, is_usable, validation_issues
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.environmental.freshness import FreshnessEvaluator
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now


class HourlyForecastObservation(BaseModel):
    """Forecast observation at an explicit hourly validity horizon."""
    forecast_valid_at: datetime = Field(..., description="UTC time for which this forecast is valid")
    temperature_c: float
    apparent_temperature_c: Optional[float] = None
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0)
    precipitation_mm: float = Field(..., ge=0.0)
    precipitation_probability_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    wind_speed_kmh: float = Field(..., ge=0.0)


class DailyForecastObservation(BaseModel):
    """Aggregated daily forecast observation."""
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    temperature_max_c: float
    temperature_min_c: float
    precipitation_sum_mm: float = Field(..., ge=0.0)
    precipitation_probability_max_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class WeatherSnapshot(BaseModel):
    """
    Standardized aggregated weather snapshot for a location at a point in time.
    Supplies foundational observations to feature engineering services.
    """
    location_id: str
    source: str
    observed_at: datetime
    ingested_at: datetime = Field(default_factory=utc_now)
    temperature_c: Optional[float] = Field(default=None, description="Ambient air temperature in Celsius")
    apparent_temperature_c: Optional[float] = Field(default=None, description="Apparent / heat index temperature")
    relative_humidity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Relative humidity percentage")
    precipitation_mm: Optional[float] = Field(default=None, ge=0.0, description="Precipitation over observation window in mm")
    precipitation_probability_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    wind_speed_kmh: Optional[float] = Field(default=None, ge=0.0, description="Wind speed in km/h")
    wind_direction_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    surface_pressure_hpa: Optional[float] = Field(default=None)
    soil_moisture_m3_m3: Optional[float] = Field(default=None, ge=0.0, description="Volumetric soil water content")
    freshness_seconds: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    forecast_horizon_hours: int = Field(default=0, description="0 for current observation, >0 for forecast")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedWeatherObservation(BaseModel):
    """
    Normalized internal representation of current meteorological conditions,
    completely independent of external provider payload formats.
    """
    source: str = Field(..., description="Provider source identifier (e.g. open_meteo)")
    location_id: Optional[str] = Field(default=None, description="Associated location identifier if resolved")
    coordinates: Coordinates = Field(..., description="Exact coordinates of the observation")
    observed_at: datetime = Field(..., description="Time of observation (UTC)")
    ingested_at: datetime = Field(default_factory=utc_now, description="Time ingested into system (UTC)")
    temperature_c: Optional[float] = Field(default=None, description="Ambient temperature in °C")
    apparent_temperature_c: Optional[float] = Field(default=None, description="Apparent temperature in °C")
    relative_humidity_pct: Optional[float] = Field(default=None, description="Relative humidity %")
    precipitation_mm: Optional[float] = Field(default=None, description="Recent precipitation in mm")
    rain_mm: Optional[float] = Field(default=None, description="Liquid rain in mm")
    wind_speed_kmh: Optional[float] = Field(default=None, description="Wind speed in km/h")
    precipitation_probability_pct: Optional[float] = Field(default=None, description="Precipitation probability %")
    units: Dict[str, str] = Field(
        default_factory=lambda: {
            "temperature": "°C",
            "apparent_temperature": "°C",
            "relative_humidity": "%",
            "precipitation": "mm",
            "rain": "mm",
            "wind_speed": "km/h",
            "precipitation_probability": "%",
        }
    )
    freshness_seconds: float = Field(default=0.0, ge=0.0)
    is_fresh: bool = Field(default=True)
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    is_usable: bool = Field(default=True)
    validation_issues: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Convenience alias properties for standard naming
    @property
    def temperature(self) -> Optional[float]:
        return self.temperature_c

    @property
    def humidity(self) -> Optional[float]:
        return self.relative_humidity_pct

    @property
    def precipitation(self) -> Optional[float]:
        return self.precipitation_mm

    @property
    def rain(self) -> Optional[float]:
        return self.rain_mm

    @property
    def wind_speed(self) -> Optional[float]:
        return self.wind_speed_kmh

    @property
    def apparent_temperature(self) -> Optional[float]:
        return self.apparent_temperature_c

    @property
    def precipitation_probability(self) -> Optional[float]:
        return self.precipitation_probability_pct

    @property
    def data_quality(self) -> float:
        return self.data_quality_score

    @property
    def freshness(self) -> float:
        return self.freshness_seconds

    @property
    def age_seconds(self) -> float:
        return FreshnessEvaluator.calculate_age_seconds(self.observed_at, self.ingested_at)

    def is_fresh_enough(self, max_age_seconds: Optional[float] = None) -> bool:
        """Evaluate if observation is fresh enough for a specific SLA threshold."""
        return FreshnessEvaluator.is_observation_fresh(self.observed_at, max_age_seconds=max_age_seconds)

    def to_weather_snapshot(self, location_id: Optional[str] = None) -> WeatherSnapshot:
        """Converts to standard WeatherSnapshot for full compatibility with risk models."""
        loc_id = location_id or self.location_id or f"coord_{round(self.coordinates.latitude, 3)}_{round(self.coordinates.longitude, 3)}"
        return WeatherSnapshot(
            location_id=loc_id,
            source=self.source,
            observed_at=self.observed_at,
            ingested_at=self.ingested_at,
            temperature_c=self.temperature_c,
            apparent_temperature_c=self.apparent_temperature_c,
            relative_humidity_pct=self.relative_humidity_pct,
            precipitation_mm=self.precipitation_mm,
            precipitation_probability_pct=self.precipitation_probability_pct,
            wind_speed_kmh=self.wind_speed_kmh,
            freshness_seconds=self.freshness_seconds,
            data_quality_score=self.data_quality_score,
            forecast_horizon_hours=0,
            metadata=self.metadata,
        )


# EnvironmentalObservation is a unified alias of NormalizedWeatherObservation
# providing the exact internal representation required by the pipeline.
EnvironmentalObservation = NormalizedWeatherObservation


class NormalizedWeatherData(BaseModel):
    """
    Complete bundle of current conditions, hourly forecasts, and daily outlooks.
    """
    current: NormalizedWeatherObservation
    hourly_forecast: List[HourlyForecastObservation] = Field(default_factory=list)
    daily_forecast: List[DailyForecastObservation] = Field(default_factory=list)
