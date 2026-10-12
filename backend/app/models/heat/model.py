"""
Rule-Based Heat Risk Baseline Model.

Implements the RiskModel contract using established meteorological criteria
and public health heat thresholds (IMD / NOAA heat index and wet-bulb indicators).
"""

from typing import Any, Dict, List
from app.models.base import RiskModel
from app.schemas.risk import RiskCategory, RiskDriver, RiskPrediction
from app.utils.exceptions import FeatureComputationError


class RuleBasedHeatModel(RiskModel):
    """
    Deterministic rule-based baseline for heat risk assessment.
    Can be replaced transparently by MLHeatModel without changing API contracts.
    """

    @property
    def model_name(self) -> str:
        return "rule-baseline-heat"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> RiskCategory:
        return RiskCategory.HEAT

    @property
    def is_ml_model(self) -> bool:
        return False

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "required_features": [
                "temperature_c",
                "apparent_temperature_c",
                "wet_bulb_temperature_c",
                "consecutive_hot_days",
            ],
            "optional_features": [
                "uhi_adjusted_temp_c",
                "relative_humidity_pct",
                "wind_speed_kmh",
            ],
            "thresholds": {
                "moderate_heat_index_c": 32.0,
                "high_heat_index_c": 41.0,
                "extreme_heat_index_c": 54.0,
                "critical_wet_bulb_c": 31.0,
            },
        }

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 24,
    ) -> RiskPrediction:
        # Validate essential inputs
        for req in ["temperature_c", "apparent_temperature_c", "wet_bulb_temperature_c"]:
            if req not in features or features[req] is None:
                raise FeatureComputationError(req, "Required heat feature is missing")

        temp = float(features["temperature_c"])
        hi = float(features["apparent_temperature_c"])
        wb = float(features["wet_bulb_temperature_c"])
        hot_days = float(features.get("consecutive_hot_days", 0.0))
        uhi_temp = float(features.get("uhi_adjusted_temp_c", temp))

        score = 10.0
        drivers: List[RiskDriver] = []

        # Apparent Temperature / Heat Index component (up to 45 pts)
        if hi >= 50.0:
            score += 45.0
            drivers.append(
                RiskDriver(
                    parameter="apparent_temperature_c",
                    impact_level="CRITICAL",
                    observed_value=hi,
                    threshold_reference=">= 50°C (Dangerous heat stroke territory)",
                    description="Extreme heat index creates immediate physiological danger.",
                )
            )
        elif hi >= 41.0:
            score += 32.0
            drivers.append(
                RiskDriver(
                    parameter="apparent_temperature_c",
                    impact_level="HIGH",
                    observed_value=hi,
                    threshold_reference=">= 41°C (Heat exhaustion likely)",
                    description="Severe heat index poses substantial heat stress risk.",
                )
            )
        elif hi >= 35.0:
            score += 18.0
            drivers.append(
                RiskDriver(
                    parameter="apparent_temperature_c",
                    impact_level="MODERATE",
                    observed_value=hi,
                    threshold_reference=">= 35°C (Caution threshold)",
                    description="Elevated heat index requires hydration precautions.",
                )
            )

        # Wet-bulb temperature risk component (up to 30 pts)
        if wb >= 31.0:
            score += 30.0
            drivers.append(
                RiskDriver(
                    parameter="wet_bulb_temperature_c",
                    impact_level="CRITICAL",
                    observed_value=wb,
                    threshold_reference=">= 31°C (Dangerous survivability limit)",
                    description="Wet-bulb temperature approaches human thermal cooling limits.",
                )
            )
        elif wb >= 28.0:
            score += 18.0
            drivers.append(
                RiskDriver(
                    parameter="wet_bulb_temperature_c",
                    impact_level="HIGH",
                    observed_value=wb,
                    threshold_reference=">= 28°C",
                    description="High humidity severely restricts sweat evaporation.",
                )
            )

        # Consecutive hot days (accumulated heatwave burden, up to 15 pts)
        if hot_days >= 3:
            pts = min(15.0, hot_days * 3.0)
            score += pts
            drivers.append(
                RiskDriver(
                    parameter="consecutive_hot_days",
                    impact_level="HIGH" if hot_days >= 5 else "MODERATE",
                    observed_value=hot_days,
                    threshold_reference=">= 3 days",
                    description=f"Prolonged heatwave accumulation over {int(hot_days)} consecutive days.",
                )
            )

        # Urban Heat Island amplification (up to 10 pts)
        if uhi_temp > temp + 1.5:
            score += 8.0
            drivers.append(
                RiskDriver(
                    parameter="uhi_adjusted_temp_c",
                    impact_level="MODERATE",
                    observed_value=uhi_temp,
                    threshold_reference=f"+{round(uhi_temp - temp, 1)}°C UHI delta",
                    description="Dense built environment retains nighttime and ambient thermal load.",
                )
            )

        # Deterministic confidence based on feature completeness
        confidence = 0.90 if wb and hi and hot_days is not None else 0.75

        evidence = {
            "temperature_c": temp,
            "apparent_temperature_c": hi,
            "wet_bulb_temperature_c": wb,
            "consecutive_hot_days": hot_days,
            "uhi_adjusted_temp_c": uhi_temp,
            "rule_set_version": "IMD_NOAA_v1",
        }

        return self.create_traceable_prediction(
            location_id=location_id,
            risk_score=score,
            confidence=confidence,
            drivers=drivers,
            evidence=evidence,
            validity_hours=validity_hours,
        )
