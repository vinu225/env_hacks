"""
Rule-Based Water Stress Baseline Model.

Implements the RiskModel contract using soil moisture deficit, dry spell duration,
reservoir levels, and groundwater decline thresholds.
"""

from typing import Any, Dict, List
from app.models.base import RiskModel
from app.schemas.risk import RiskCategory, RiskDriver, RiskPrediction
from app.utils.exceptions import FeatureComputationError


class RuleBasedWaterModel(RiskModel):
    """
    Deterministic rule-based baseline for water stress and drought risk assessment.
    Can be replaced transparently by MLWaterModel without altering consumers.
    """

    @property
    def model_name(self) -> str:
        return "rule-baseline-water"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> RiskCategory:
        return RiskCategory.WATER_STRESS

    @property
    def is_ml_model(self) -> bool:
        return False

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "required_features": [
                "soil_moisture_index",
                "consecutive_dry_days",
            ],
            "optional_features": [
                "groundwater_depth_m",
                "reservoir_shortage_ratio",
                "supply_outage_hours",
                "demand_stress_factor",
            ],
            "thresholds": {
                "critical_dry_days": 21,
                "critical_soil_moisture_index": 0.15,
                "critical_groundwater_depth_m": 35.0,
            },
        }

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 48,
    ) -> RiskPrediction:
        for req in ["soil_moisture_index", "consecutive_dry_days"]:
            if req not in features or features[req] is None:
                raise FeatureComputationError(req, "Required water feature is missing")

        smi = float(features["soil_moisture_index"])
        dry_days = float(features["consecutive_dry_days"])
        gw_depth = float(features.get("groundwater_depth_m", 15.0))
        res_shortage = float(features.get("reservoir_shortage_ratio", 0.3))
        outage_hours = float(features.get("supply_outage_hours", 0.0))

        score = 15.0
        drivers: List[RiskDriver] = []

        # Soil moisture deficit component (up to 35 pts)
        if smi <= 0.15:
            score += 35.0
            drivers.append(
                RiskDriver(
                    parameter="soil_moisture_index",
                    impact_level="CRITICAL",
                    observed_value=smi,
                    threshold_reference="<= 0.15 (Severe drought / wilting point)",
                    description="Soil moisture is critically depleted, threatening urban trees and agriculture.",
                )
            )
        elif smi <= 0.30:
            score += 20.0
            drivers.append(
                RiskDriver(
                    parameter="soil_moisture_index",
                    impact_level="HIGH",
                    observed_value=smi,
                    threshold_reference="<= 0.30",
                    description="Significant soil moisture deficit indicating prolonged dry conditions.",
                )
            )

        # Consecutive dry spell component (up to 25 pts)
        if dry_days >= 30:
            score += 25.0
            drivers.append(
                RiskDriver(
                    parameter="consecutive_dry_days",
                    impact_level="CRITICAL",
                    observed_value=dry_days,
                    threshold_reference=">= 30 days without rain",
                    description="Prolonged meteorological drought conditions.",
                )
            )
        elif dry_days >= 14:
            score += 15.0
            drivers.append(
                RiskDriver(
                    parameter="consecutive_dry_days",
                    impact_level="MODERATE",
                    observed_value=dry_days,
                    threshold_reference=">= 14 days without rain",
                    description="Two weeks without significant precipitation.",
                )
            )

        # Municipal supply cuts / Outages (up to 20 pts) - Only if explicitly observed
        if outage_hours is not None and outage_hours > 0:
            if outage_hours >= 24:
                score += 20.0
                drivers.append(
                    RiskDriver(
                        parameter="supply_outage_hours",
                        impact_level="HIGH",
                        observed_value=outage_hours,
                        threshold_reference=">= 24 hrs outage in last week",
                        description="Severe disruption in piped municipal potable water supply.",
                    )
                )
            elif outage_hours >= 6:
                score += 10.0
                drivers.append(
                    RiskDriver(
                        parameter="supply_outage_hours",
                        impact_level="MODERATE",
                        observed_value=outage_hours,
                        threshold_reference=">= 6 hrs outage",
                        description="Intermittent piped water supply cuts reported.",
                    )
                )

        # Groundwater depth indicator (up to 15 pts) - Only if telemetry actually observed
        if gw_depth is not None and gw_depth >= 35.0:
            score += 15.0
            drivers.append(
                RiskDriver(
                    parameter="groundwater_depth_m",
                    impact_level="HIGH",
                    observed_value=gw_depth,
                    threshold_reference=">= 35m below surface",
                    description="Deep water table indicates over-extracted aquifer.",
                )
            )

        confidence = 0.85

        evidence = {
            "assessment_scope": "weather-derived water stress risk",
            "soil_moisture_index": smi,
            "consecutive_dry_days": dry_days,
            "groundwater_depth_m": gw_depth,
            "reservoir_shortage_ratio": res_shortage,
            "supply_outage_hours": outage_hours,
            "rule_set_version": "WEATHER_DERIVED_WATER_STRESS_v1",
        }

        return self.create_traceable_prediction(
            location_id=location_id,
            risk_score=score,
            confidence=confidence,
            drivers=drivers,
            evidence=evidence,
            validity_hours=validity_hours,
        )
