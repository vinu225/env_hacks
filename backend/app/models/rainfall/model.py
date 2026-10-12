"""
Rule-Based Rainfall and Flood Risk Baseline Model.

Implements the RiskModel contract using precipitation intensity, accumulated rainfall,
drainage impedance, and topography depression factors.
"""

from typing import Any, Dict, List
from app.models.base import RiskModel
from app.schemas.risk import RiskCategory, RiskDriver, RiskPrediction
from app.utils.exceptions import FeatureComputationError


class RuleBasedRainfallModel(RiskModel):
    """
    Deterministic rule-based baseline for rainfall, waterlogging, and flash-flood risk.
    Can be replaced transparently by MLRainfallModel without modifying API contracts.
    """

    @property
    def model_name(self) -> str:
        return "rule-baseline-rainfall"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> RiskCategory:
        return RiskCategory.RAINFALL

    @property
    def is_ml_model(self) -> bool:
        return False

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "required_features": [
                "precipitation_current_hour_mm",
                "precipitation_last_24h_mm",
            ],
            "optional_features": [
                "precipitation_forecast_next_6h_mm",
                "soil_saturation_pct",
                "drainage_impedance",
                "topography_risk_multiplier",
                "estimated_runoff_coefficient",
            ],
            "thresholds": {
                "heavy_hourly_rain_mm": 20.0,
                "torrential_hourly_rain_mm": 50.0,
                "heavy_24h_rain_mm": 64.5,  # IMD standard heavy rainfall mark
            },
        }

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 12,
    ) -> RiskPrediction:
        for req in ["precipitation_current_hour_mm", "precipitation_last_24h_mm"]:
            if req not in features or features[req] is None:
                raise FeatureComputationError(req, "Required rainfall feature is missing")

        curr_rain = float(features["precipitation_current_hour_mm"])
        rain_24h = float(features["precipitation_last_24h_mm"])
        forecast_6h = float(features.get("precipitation_forecast_next_6h_mm", 0.0))
        impedance = float(features.get("drainage_impedance", 0.5))
        topo_mult = float(features.get("topography_risk_multiplier", 1.0))

        score = 5.0
        drivers: List[RiskDriver] = []

        # Current rain intensity (up to 40 pts)
        if curr_rain >= 50.0:
            score += 40.0
            drivers.append(
                RiskDriver(
                    parameter="precipitation_current_hour_mm",
                    impact_level="CRITICAL",
                    observed_value=curr_rain,
                    threshold_reference=">= 50 mm/hr (Torrential / Cloudburst rate)",
                    description="Extreme short-duration cloudburst intensity overwhelms urban storm drains.",
                )
            )
        elif curr_rain >= 20.0:
            score += 25.0
            drivers.append(
                RiskDriver(
                    parameter="precipitation_current_hour_mm",
                    impact_level="HIGH",
                    observed_value=curr_rain,
                    threshold_reference=">= 20 mm/hr (Heavy downpour)",
                    description="Heavy rain rate exceeds regular storm gutter absorption.",
                )
            )
        elif curr_rain >= 7.5:
            score += 12.0
            drivers.append(
                RiskDriver(
                    parameter="precipitation_current_hour_mm",
                    impact_level="MODERATE",
                    observed_value=curr_rain,
                    threshold_reference=">= 7.5 mm/hr (Moderate rain)",
                    description="Steady rainfall accumulating in low drainage zones.",
                )
            )

        # 24h accumulated rain (up to 25 pts)
        if rain_24h >= 115.5:
            score += 25.0
            drivers.append(
                RiskDriver(
                    parameter="precipitation_last_24h_mm",
                    impact_level="CRITICAL",
                    observed_value=rain_24h,
                    threshold_reference=">= 115.5 mm (IMD Very Heavy Rainfall)",
                    description="Prolonged cumulative saturation causes widespread waterlogging.",
                )
            )
        elif rain_24h >= 64.5:
            score += 16.0
            drivers.append(
                RiskDriver(
                    parameter="precipitation_last_24h_mm",
                    impact_level="HIGH",
                    observed_value=rain_24h,
                    threshold_reference=">= 64.5 mm (IMD Heavy Rainfall)",
                    description="Substantial 24-hour accumulation saturates urban soils.",
                )
            )

        # Drainage blockage / impedance factor (up to 20 pts)
        if impedance >= 0.7:
            drainage_pt = 20.0 * impedance
            score += drainage_pt
            drivers.append(
                RiskDriver(
                    parameter="drainage_impedance",
                    impact_level="HIGH",
                    observed_value=impedance,
                    threshold_reference=">= 0.7 (Severe drainage bottleneck)",
                    description="Local drainage network impaired or silted, causing rapid road ponding.",
                )
            )

        # Topographical depression multiplier
        if topo_mult > 1.0:
            score *= topo_mult
            drivers.append(
                RiskDriver(
                    parameter="topography_risk_multiplier",
                    impact_level="MODERATE",
                    observed_value=topo_mult,
                    threshold_reference="Low-lying basin depression",
                    description="Depression topography collects runoff from surrounding wards.",
                )
            )

        confidence = 0.88

        evidence = {
            "precipitation_current_hour_mm": curr_rain,
            "precipitation_last_24h_mm": rain_24h,
            "precipitation_forecast_next_6h_mm": forecast_6h,
            "drainage_impedance": impedance,
            "topography_risk_multiplier": topo_mult,
            "rule_set_version": "IMD_URBAN_FLOOD_v1",
        }

        return self.create_traceable_prediction(
            location_id=location_id,
            risk_score=score,
            confidence=confidence,
            drivers=drivers,
            evidence=evidence,
            validity_hours=validity_hours,
        )
