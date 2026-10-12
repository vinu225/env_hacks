"""
Water Feature Extraction and Transformation.

Prepares structured indicators for soil moisture deficit, drought progression,
and municipal water-supply vulnerability.
"""

from typing import Any, Dict, Optional, Sequence
from app.features.rolling import (
    calculate_consecutive_dry_days,
    calculate_rainfall_deficit_pct,
    calculate_rolling_rainfall,
)
from app.schemas.environmental import WeatherSnapshot
from app.schemas.water import WaterFeatureInput


def extract_water_features(
    snapshot: WeatherSnapshot,
    consecutive_dry_days: int = 0,
    groundwater_table_depth_m: Optional[float] = None,
    reservoir_capacity_pct: Optional[float] = None,
    supply_cut_hours: int = 0,
    water_demand_index: float = 1.0,
) -> WaterFeatureInput:
    """Transform weather snapshot and municipal metrics into structured water features."""
    # Convert volumetric soil moisture (m3/m3 typically 0.0 - 0.5) to normalized index 0-1 ONLY IF present
    soil_index = 0.5
    if snapshot.soil_moisture_m3_m3 is not None:
        soil_index = max(0.0, min(1.0, snapshot.soil_moisture_m3_m3 / 0.45))

    return WaterFeatureInput(
        location_id=snapshot.location_id,
        soil_moisture_index=round(soil_index, 3),
        consecutive_dry_days=consecutive_dry_days,
        groundwater_table_depth_m=groundwater_table_depth_m,
        reservoir_capacity_pct=reservoir_capacity_pct,
        supply_cut_hours_last_week=supply_cut_hours,
        water_demand_index=water_demand_index,
    )


def compute_water_feature_vector(
    input_data: WaterFeatureInput,
    recent_rain_1h_mm: float = 0.0,
    accumulated_rain_7d_mm: Optional[float] = None,
    accumulated_rain_30d_mm: Optional[float] = None,
    normal_rain_30d_mm: Optional[float] = None,
    precipitation_probability_pct: Optional[float] = None,
    has_actual_soil_moisture: bool = False,
) -> Dict[str, Any]:
    """
    Compute aligned feature dictionary for water risk prediction.
    Strictly preserves availability and does NOT invent unobserved groundwater or soil data.
    """
    moisture_deficit = 1.0 - input_data.soil_moisture_index
    reservoir_shortage = (
        (100.0 - input_data.reservoir_capacity_pct) / 100.0
        if input_data.reservoir_capacity_pct is not None
        else None
    )

    deficit_pct = None
    if accumulated_rain_30d_mm is not None and normal_rain_30d_mm is not None:
        deficit_pct = calculate_rainfall_deficit_pct(accumulated_rain_30d_mm, normal_rain_30d_mm)

    features: Dict[str, Any] = {
        "recent_rain_1h_mm": float(recent_rain_1h_mm),
        "rainfall_accumulation_7d_mm": accumulated_rain_7d_mm,
        "rainfall_accumulation_30d_mm": accumulated_rain_30d_mm,
        "rainfall_deficit_30d_pct": deficit_pct,
        "consecutive_dry_days": float(input_data.consecutive_dry_days),
        "precipitation_probability_pct": precipitation_probability_pct,
        "soil_moisture_index": float(input_data.soil_moisture_index) if input_data.soil_moisture_index is not None else 0.5,
        "soil_moisture_deficit": float(round(moisture_deficit, 3)),
        "reservoir_shortage_ratio": reservoir_shortage,
        "groundwater_depth_m": input_data.groundwater_table_depth_m,
        "supply_outage_hours": float(input_data.supply_cut_hours_last_week) if input_data.supply_cut_hours_last_week > 0 else None,
        "demand_stress_factor": float(input_data.water_demand_index),
        # Legacy backward-compatible keys used by rule baseline model
        "_legacy_soil_moisture_index": float(input_data.soil_moisture_index),
        "_legacy_groundwater_depth_m": float(input_data.groundwater_table_depth_m or 15.0),
        "_legacy_reservoir_shortage_ratio": float(reservoir_shortage if reservoir_shortage is not None else 0.4),
        "_legacy_supply_outage_hours": float(input_data.supply_cut_hours_last_week),
    }

    return features
