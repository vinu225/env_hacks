"""
Rainfall and Inundation Feature Extraction.

Prepares structured metrics for flash-flood and urban waterlogging assessment.
"""

from typing import Any, Dict, Optional
from app.schemas.environmental import WeatherSnapshot
from app.schemas.rainfall import RainfallFeatureInput


def is_heavy_rain_condition(
    hourly_precipitation_mm: float,
    accumulated_24h_mm: float = 0.0,
) -> bool:
    """
    True if precipitation meets IMD criteria for heavy downpour
    (>= 15 mm/hr or >= 64.5 mm over 24 hours).
    """
    return hourly_precipitation_mm >= 15.0 or accumulated_24h_mm >= 64.5


def extract_rainfall_features(
    snapshot: WeatherSnapshot,
    accumulated_24h_mm: float = 0.0,
    forecast_6h_mm: float = 0.0,
    drainage_capacity_rating: float = 0.5,
    is_low_lying: bool = False,
) -> RainfallFeatureInput:
    """Transform weather snapshot and topography metadata into structured rainfall features."""
    return RainfallFeatureInput(
        location_id=snapshot.location_id,
        precipitation_current_hour_mm=snapshot.precipitation_mm,
        precipitation_last_24h_mm=max(snapshot.precipitation_mm, accumulated_24h_mm),
        precipitation_forecast_next_6h_mm=forecast_6h_mm,
        soil_saturation_pct=70.0 if (snapshot.soil_moisture_m3_m3 or 0.3) > 0.35 else 40.0,
        drainage_capacity_rating=drainage_capacity_rating,
        is_low_lying_basin=is_low_lying,
    )


def compute_rainfall_feature_vector(
    input_data: RainfallFeatureInput,
    precipitation_probability_pct: Optional[float] = None,
    accumulated_24h_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate normalized numeric features for rainfall / flood risk engines."""
    # Drainage impedance factor (higher means worse drainage)
    drainage_impedance = 1.0 - input_data.drainage_capacity_rating
    topography_multiplier = 1.4 if input_data.is_low_lying_basin else 1.0

    # Runoff coefficient estimate (rational method proxy)
    base_c = 0.3 + (0.5 * (input_data.soil_saturation_pct / 100.0))
    runoff_c = min(0.95, base_c * (1.0 + 0.3 * drainage_impedance))

    rain_24h = accumulated_24h_mm if accumulated_24h_mm is not None else input_data.precipitation_last_24h_mm
    heavy_rain_flag = is_heavy_rain_condition(input_data.precipitation_current_hour_mm, rain_24h)

    return {
        "precipitation_current_hour_mm": float(input_data.precipitation_current_hour_mm),
        "precipitation_last_24h_mm": float(rain_24h),
        "precipitation_forecast_next_6h_mm": float(input_data.precipitation_forecast_next_6h_mm),
        "precipitation_probability_pct": precipitation_probability_pct,
        "heavy_rain_flag": 1.0 if heavy_rain_flag else 0.0,
        "soil_saturation_pct": float(input_data.soil_saturation_pct),
        "drainage_impedance": float(round(drainage_impedance, 3)),
        "topography_risk_multiplier": float(topography_multiplier),
        "estimated_runoff_coefficient": float(round(runoff_c, 3)),
    }
