"""
Feature Engineering Package.
"""

from app.features.builder import FeatureVectorBuilder
from app.features.heat_features import (
    compute_heat_feature_vector,
    extract_heat_features,
    is_high_temperature_condition,
    is_humidity_stress_condition,
)
from app.features.rainfall_features import (
    compute_rainfall_feature_vector,
    extract_rainfall_features,
    is_heavy_rain_condition,
)
from app.features.rolling import (
    calculate_consecutive_dry_days,
    calculate_consecutive_hot_days,
    calculate_rainfall_deficit_pct,
    calculate_rolling_rainfall,
    calculate_temperature_anomaly,
)
from app.features.water_features import (
    compute_water_feature_vector,
    extract_water_features,
)
from app.features.weather_features import (
    calculate_dew_point,
    calculate_heat_index,
    calculate_wet_bulb_temperature,
)

__all__ = [
    "calculate_dew_point",
    "calculate_heat_index",
    "calculate_wet_bulb_temperature",
    "compute_heat_feature_vector",
    "extract_heat_features",
    "is_high_temperature_condition",
    "is_humidity_stress_condition",
    "compute_water_feature_vector",
    "extract_water_features",
    "compute_rainfall_feature_vector",
    "extract_rainfall_features",
    "is_heavy_rain_condition",
    "calculate_rolling_rainfall",
    "calculate_consecutive_dry_days",
    "calculate_consecutive_hot_days",
    "calculate_temperature_anomaly",
    "calculate_rainfall_deficit_pct",
    "FeatureVectorBuilder",
]
