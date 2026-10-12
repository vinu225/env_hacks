"""
Heat Feature Extraction and Transformation.

Prepares structured feature matrices for Rule-Based Baselines and ML Heat Models.
"""

from typing import Any, Dict, Optional, Sequence
from app.features.rolling import calculate_consecutive_hot_days, calculate_temperature_anomaly
from app.features.weather_features import calculate_heat_index, calculate_wet_bulb_temperature
from app.schemas.environmental import WeatherSnapshot
from app.schemas.heat import HeatFeatureInput


def is_high_temperature_condition(temperature_c: float, threshold_c: float = 40.0) -> bool:
    """True if temperature exceeds regional heatwave alert mark."""
    return temperature_c >= threshold_c


def is_humidity_stress_condition(temperature_c: float, relative_humidity_pct: float) -> bool:
    """
    True if concurrent elevated temperature and high humidity creates dangerous sweat-inhibition stress
    (e.g., Temperature >= 34°C with Humidity >= 60%).
    """
    return temperature_c >= 34.0 and relative_humidity_pct >= 60.0


def extract_heat_features(
    snapshot: WeatherSnapshot,
    consecutive_hot_days: int = 0,
    urban_heat_island_factor: float = 1.0,
    min_night_temp_c: Optional[float] = None,
) -> HeatFeatureInput:
    """Transform raw weather observations into structured heat features."""
    return HeatFeatureInput(
        location_id=snapshot.location_id,
        temperature_c=snapshot.temperature_c,
        relative_humidity_pct=snapshot.relative_humidity_pct,
        wind_speed_kmh=snapshot.wind_speed_kmh,
        max_temperature_forecast_24h_c=snapshot.temperature_c,
        min_temperature_night_c=min_night_temp_c,
        consecutive_hot_days=consecutive_hot_days,
        urban_heat_island_factor=urban_heat_island_factor,
    )


def compute_heat_feature_vector(
    input_data: HeatFeatureInput,
    climatological_normal_temp_c: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Produce normalized numeric vector ready for ML or Rule-Based inference.
    Traceable feature keys ensure consistent tabular alignment.
    """
    hi = calculate_heat_index(input_data.temperature_c, input_data.relative_humidity_pct)
    wb = calculate_wet_bulb_temperature(input_data.temperature_c, input_data.relative_humidity_pct)
    adjusted_temp = input_data.temperature_c * input_data.urban_heat_island_factor
    high_temp_flag = is_high_temperature_condition(input_data.temperature_c)
    humidity_stress_flag = is_humidity_stress_condition(input_data.temperature_c, input_data.relative_humidity_pct)
    temp_anomaly = calculate_temperature_anomaly(input_data.temperature_c, climatological_normal_temp_c)

    return {
        "temperature_c": float(input_data.temperature_c),
        "apparent_temperature_c": float(hi),
        "wet_bulb_temperature_c": float(wb),
        "uhi_adjusted_temp_c": float(round(adjusted_temp, 2)),
        "relative_humidity_pct": float(input_data.relative_humidity_pct),
        "wind_speed_kmh": float(input_data.wind_speed_kmh),
        "consecutive_hot_days": float(input_data.consecutive_hot_days),
        "high_temperature_flag": 1.0 if high_temp_flag else 0.0,
        "humidity_stress_flag": 1.0 if humidity_stress_flag else 0.0,
        "temperature_anomaly_c": temp_anomaly,
        "night_temperature_recovery_gap": float(
            (input_data.temperature_c - input_data.min_temperature_night_c)
            if input_data.min_temperature_night_c is not None
            else 5.0
        ),
    }
