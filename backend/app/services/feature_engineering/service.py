"""
Feature Engineering Service.

Coordinates extraction and formatting of tabular features across weather,
hydrology, and topography data streams.
"""

from typing import Any, Dict
from app.features.heat_features import compute_heat_feature_vector, extract_heat_features
from app.features.rainfall_features import (
    compute_rainfall_feature_vector,
    extract_rainfall_features,
)
from app.features.water_features import (
    compute_water_feature_vector,
    extract_water_features,
)
from app.schemas.environmental import WeatherSnapshot
from app.schemas.location import LocationRead


class FeatureEngineeringService:
    """Service transforming raw ingested domain data into model-ready feature vectors."""

    def prepare_heat_features(
        self, snapshot: WeatherSnapshot, location: LocationRead
    ) -> Dict[str, Any]:
        uhi_multiplier = float(location.metadata.get("uhi_factor", 1.1))
        consecutive_hot_days = int(location.metadata.get("consecutive_hot_days", 1))
        feature_input = extract_heat_features(
            snapshot=snapshot,
            consecutive_hot_days=consecutive_hot_days,
            urban_heat_island_factor=uhi_multiplier,
        )
        return compute_heat_feature_vector(feature_input)

    def prepare_water_features(
        self, snapshot: WeatherSnapshot, location: LocationRead
    ) -> Dict[str, Any]:
        dry_days = int(location.metadata.get("consecutive_dry_days", 5))
        gw_depth = location.metadata.get("groundwater_depth_m", 18.0)
        res_cap = location.metadata.get("reservoir_capacity_pct", 45.0)
        outage = int(location.metadata.get("supply_outage_hours", 2))

        feature_input = extract_water_features(
            snapshot=snapshot,
            consecutive_dry_days=dry_days,
            groundwater_table_depth_m=gw_depth,
            reservoir_capacity_pct=res_cap,
            supply_cut_hours=outage,
        )
        return compute_water_feature_vector(feature_input)

    def prepare_rainfall_features(
        self, snapshot: WeatherSnapshot, location: LocationRead
    ) -> Dict[str, Any]:
        drainage_rating = float(location.metadata.get("drainage_capacity_rating", 0.5))
        is_basin = "basin" in location.name.lower() or "flood" in str(location.metadata).lower()

        feature_input = extract_rainfall_features(
            snapshot=snapshot,
            accumulated_24h_mm=snapshot.precipitation_mm * 2.0,
            forecast_6h_mm=snapshot.precipitation_mm * 1.5,
            drainage_capacity_rating=drainage_rating,
            is_low_lying=is_basin,
        )
        return compute_rainfall_feature_vector(feature_input)
