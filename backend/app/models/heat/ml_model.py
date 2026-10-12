"""
Production-Grade ML Heat Risk Model.

Wraps the trained LightGBM multi-class classification artifact under the RiskModel contract.
Preserves strict precedence:
    Safety override -> Validated ML prediction -> Rule-based fallback -> Safe error response.

Key Guarantees:
1. Loads model.joblib and feature_schema.json once during initialization.
2. Extracts canonical features in exact declared order from EnvironmentalFeatureVector or feature dict.
3. Deterministic safety overrides (HI >= 54°C or Tw >= 31°C) cannot be suppressed by lower ML output.
4. Robust fallback to RuleBasedHeatModel if ML inference fails.
5. Never fabricates predictions on missing or non-finite core inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.features.weather_features import calculate_dew_point, calculate_heat_index, calculate_wet_bulb_temperature
from app.models.base import RiskModel, score_to_risk_level
from app.models.heat.model import RuleBasedHeatModel
from app.schemas.risk import RiskCategory, RiskDriver, RiskLevel, RiskPrediction
from app.utils.exceptions import FeatureComputationError

logger = get_logger(__name__)


def resolve_artifact_dir(custom_path: Optional[str] = None) -> Path:
    """
    Resolve the artifact directory reliably from the repository structure,
    independent of the current working directory.
    If a custom_path is explicitly provided, validate that path directly.
    """
    if custom_path and custom_path.strip():
        p = Path(custom_path.strip()).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Specified artifact directory does not exist: {p}")
        if not (p / "model.joblib").exists():
            raise FileNotFoundError(f"Model artifact 'model.joblib' missing from: {p}")
        return p

    # Walk up parent directories to find ml/artifacts/heat_model_v1
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / "ml" / "artifacts" / "heat_model_v1"
        if candidate.is_dir() and (candidate / "model.joblib").exists():
            return candidate.resolve()

    # Direct fallback relative to repository root
    fallback = Path(__file__).resolve().parents[4] / "ml" / "artifacts" / "heat_model_v1"
    return fallback.resolve()


class MLHeatModel(RiskModel):
    """
    Production-grade ML Heat Model powered by trained LightGBM Classifier.
    """

    _cached_model: Any = None
    _cached_schema: Optional[Dict[str, Any]] = None
    _cached_dir: Optional[Path] = None

    def __init__(self, artifact_dir: Optional[str] = None):
        self._artifact_dir = resolve_artifact_dir(artifact_dir)
        self._fallback_model = RuleBasedHeatModel()

        # Load and validate artifacts (cached after first initialization)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load and validate serialized model and schema JSON."""
        if MLHeatModel._cached_model is not None and MLHeatModel._cached_dir == self._artifact_dir:
            self._model = MLHeatModel._cached_model
            self._schema = MLHeatModel._cached_schema
            return

        model_path = self._artifact_dir / "model.joblib"
        schema_path = self._artifact_dir / "feature_schema.json"

        if not model_path.exists():
            raise FileNotFoundError(f"ML heat model artifact not found at: {model_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at: {schema_path}")

        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            model = joblib.load(model_path)

            # Validate loaded model attributes
            if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
                raise ValueError("Loaded artifact does not implement predict and predict_proba")
            if not hasattr(model, "classes_"):
                raise ValueError("Loaded classifier missing classes_ attribute")

            MLHeatModel._cached_model = model
            MLHeatModel._cached_schema = schema
            MLHeatModel._cached_dir = self._artifact_dir

            self._model = model
            self._schema = schema
            logger.info(
                f"MLHeatModel successfully loaded from {self._artifact_dir} "
                f"(classes={list(model.classes_)}, features={len(schema.get('ordered_features', []))})"
            )
        except Exception as e:
            logger.error(f"Failed to load MLHeatModel artifact from {self._artifact_dir}: {e}")
            raise

    @property
    def model_name(self) -> str:
        return "lightgbm-heat-v1"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def category(self) -> RiskCategory:
        return RiskCategory.HEAT

    @property
    def is_ml_model(self) -> bool:
        return True

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "required_features": [
                "temperature_c",
                "apparent_temperature_c",
                "relative_humidity_pct",
                "dew_point_c",
                "heat_index_c",
                "wet_bulb_temperature_c",
                "wind_speed_kmh",
                "consecutive_hot_days",
                "is_high_temperature",
                "is_humidity_stress",
                "precipitation_mm",
                "rainfall_accumulation_24h_mm",
                "rainfall_accumulation_7d_mm",
                "consecutive_dry_days",
            ],
            "safety_thresholds": {
                "critical_heat_index_c": 54.0,
                "critical_wet_bulb_c": 31.0,
            },
            "empirical_classes": list(self._model.classes_),
            "artifact_dir": str(self._artifact_dir),
        }

    def _extract_canonical_features(self, features: Dict[str, Any]) -> Tuple[List[float], Dict[str, float]]:
        """
        Extract the 14 canonical features in the exact order declared by feature_schema.json.
        Gracefully handles both EnvironmentalFeatureVector numeric dicts and prepare_heat_features dicts.
        Raises FeatureComputationError if essential inputs are missing or non-finite.
        """
        # 1. Validate fundamental temperature input
        temp_val = features.get("temperature_c")
        if temp_val is None or (isinstance(temp_val, float) and (math.isnan(temp_val) or math.isinf(temp_val))):
            raise FeatureComputationError("temperature_c", "Essential temperature observation is missing or non-finite")
        temp_c = float(temp_val)

        # 2. Validate relative humidity or compute/fallback
        rh_val = features.get("relative_humidity_pct")
        if rh_val is None or (isinstance(rh_val, float) and (math.isnan(rh_val) or math.isinf(rh_val))):
            rh_pct = 50.0  # Documented neutral humidity fallback if absent
        else:
            rh_pct = float(rh_val)

        # 3. Derive or extract meteorological indicators
        app_temp = features.get("apparent_temperature_c")
        if app_temp is None or (isinstance(app_temp, float) and math.isnan(app_temp)):
            hi = calculate_heat_index(temp_c, rh_pct) or temp_c
            app_temp = hi
        else:
            app_temp = float(app_temp)

        heat_idx = features.get("heat_index_c")
        if heat_idx is None or (isinstance(heat_idx, float) and math.isnan(heat_idx)):
            heat_idx = float(calculate_heat_index(temp_c, rh_pct) or app_temp)
        else:
            heat_idx = float(heat_idx)

        wb_temp = features.get("wet_bulb_temperature_c")
        if wb_temp is None or (isinstance(wb_temp, float) and math.isnan(wb_temp)):
            wb_temp = float(calculate_wet_bulb_temperature(temp_c, rh_pct) or temp_c)
        else:
            wb_temp = float(wb_temp)

        dew_p = features.get("dew_point_c")
        if dew_p is None or (isinstance(dew_p, float) and math.isnan(dew_p)):
            dew_p = float(calculate_dew_point(temp_c, rh_pct) or (temp_c - 5.0))
        else:
            dew_p = float(dew_p)

        wind = float(features.get("wind_speed_kmh") or 10.0)
        hot_days = float(features.get("consecutive_hot_days") or 0.0)

        # Binary condition flags
        high_temp_flag = features.get("is_high_temperature")
        if high_temp_flag is None:
            high_temp_flag = features.get("high_temperature_flag", 1.0 if temp_c >= 40.0 else 0.0)
        high_temp = float(high_temp_flag)

        hum_stress_flag = features.get("is_humidity_stress")
        if hum_stress_flag is None:
            hum_stress_flag = features.get("humidity_stress_flag", 1.0 if (temp_c >= 34.0 and rh_pct >= 60.0) else 0.0)
        hum_stress = float(hum_stress_flag)

        precip = float(features.get("precipitation_mm") or 0.0)
        rain_24h = float(features.get("rainfall_accumulation_24h_mm") or 0.0)
        rain_7d = float(features.get("rainfall_accumulation_7d_mm") or 0.0)
        dry_days = float(features.get("consecutive_dry_days") or 0.0)

        values_dict = {
            "temperature_c": temp_c,
            "apparent_temperature_c": app_temp,
            "relative_humidity_pct": rh_pct,
            "dew_point_c": dew_p,
            "heat_index_c": heat_idx,
            "wet_bulb_temperature_c": wb_temp,
            "wind_speed_kmh": wind,
            "consecutive_hot_days": hot_days,
            "is_high_temperature": high_temp,
            "is_humidity_stress": hum_stress,
            "precipitation_mm": precip,
            "rainfall_accumulation_24h_mm": rain_24h,
            "rainfall_accumulation_7d_mm": rain_7d,
            "consecutive_dry_days": dry_days,
        }

        # Verify all 14 values are finite
        ordered_list = []
        ordered_names = self._schema.get("ordered_features", list(values_dict.keys()))
        for name in ordered_names:
            val = values_dict[name]
            if math.isnan(val) or math.isinf(val):
                raise FeatureComputationError(name, f"Feature '{name}' has non-finite value: {val}")
            ordered_list.append(val)

        return ordered_list, values_dict

    def _check_critical_safety_override(
        self, features_dict: Dict[str, float]
    ) -> Optional[Tuple[float, List[RiskDriver]]]:
        """
        Evaluate deterministic physiological emergency benchmarks.
        Returns (override_score, override_drivers) if safety conditions are met, else None.
        Precedence: Safety override can NEVER be suppressed by a lower ML prediction.
        """
        hi = features_dict.get("heat_index_c", 0.0)
        wb = features_dict.get("wet_bulb_temperature_c", 0.0)

        is_hi_critical = hi >= 54.0
        is_wb_critical = wb >= 31.0

        if is_hi_critical or is_wb_critical:
            drivers = []
            score = 85.0

            if is_hi_critical:
                score += min(10.0, (hi - 54.0) * 2.0)
                drivers.append(
                    RiskDriver(
                        parameter="heat_index_c",
                        impact_level="CRITICAL",
                        observed_value=round(hi, 1),
                        threshold_reference=">= 54.0°C (NOAA NWS Extreme Danger benchmark)",
                        description="Dangerous heat stroke territory; physiological thermal regulation overwhelmed.",
                    )
                )

            if is_wb_critical:
                score += min(10.0, (wb - 31.0) * 3.0)
                drivers.append(
                    RiskDriver(
                        parameter="wet_bulb_temperature_c",
                        impact_level="CRITICAL",
                        observed_value=round(wb, 1),
                        threshold_reference=">= 31.0°C (Vecellio 2022 uncompensable wet-bulb limit)",
                        description="Sustained physical labor causes continuous heat accumulation regardless of hydration.",
                    )
                )

            return min(100.0, round(score, 2)), drivers

        return None

    def _map_ml_prediction_to_score(
        self,
        pred_class: str,
        probs: Dict[str, float],
    ) -> float:
        """
        Map predicted categorical class and probability vector into a standardized 0-100 risk score.
        Guarantees monotonicity: score_to_risk_level(score) strictly equals pred_class.
        """
        p_low = probs.get("LOW", 0.0)
        p_mod = probs.get("MODERATE", 0.0)
        p_high = probs.get("HIGH", 0.0)
        p_sev = probs.get("SEVERE", 0.0)

        if pred_class == "LOW":
            # [0.0, 19.99]
            score = 10.0 + 8.0 * (p_mod + p_high + p_sev - p_low)
            return round(max(2.0, min(19.5, score)), 2)
        elif pred_class == "MODERATE":
            # [20.0, 39.99]
            score = 25.0 + 12.0 * (p_high + p_sev - p_low)
            return round(max(20.0, min(39.5, score)), 2)
        elif pred_class == "HIGH":
            # [40.0, 59.99]
            score = 45.0 + 12.0 * (p_sev - p_low)
            return round(max(40.0, min(59.5, score)), 2)
        elif pred_class == "SEVERE":
            # [60.0, 79.99]
            score = 65.0 + 12.0 * p_sev
            return round(max(60.0, min(79.5, score)), 2)
        else:
            return 10.0

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 24,
    ) -> RiskPrediction:
        """
        Execute risk prediction with the strict precedence:
            1. Deterministic safety override check
            2. Validated ML prediction via LightGBM
            3. Rule-based fallback upon ML failure
            4. Established safe error response
        """
        # Step 1: Feature Extraction & Validation
        try:
            canonical_values, values_dict = self._extract_canonical_features(features)
        except FeatureComputationError:
            # Re-raise directly: never fabricate a prediction on fundamentally invalid input
            raise
        except Exception as e:
            logger.warning(f"Feature extraction unexpected error ({e}); attempting rule-based fallback")
            return self._fallback_model.predict(features, location_id, validity_hours)

        # Step 2: Deterministic Safety Override Check
        safety_result = self._check_critical_safety_override(values_dict)
        if safety_result is not None:
            override_score, override_drivers = safety_result
            logger.warning(
                f"SAFETY OVERRIDE ACTIVATED for location='{location_id}': "
                f"Heat Index={values_dict.get('heat_index_c')}°C, Wet Bulb={values_dict.get('wet_bulb_temperature_c')}°C"
            )
            return self.create_traceable_prediction(
                location_id=location_id,
                risk_score=override_score,
                confidence=0.95,
                drivers=override_drivers,
                evidence={
                    **values_dict,
                    "prediction_source": "deterministic_safety_override",
                    "override_reason": "NOAA / Wet-Bulb physiological survivability benchmark exceeded",
                },
                validity_hours=validity_hours,
            )

        # Step 3: Validated ML Prediction via LightGBM
        try:
            ordered_names = self._schema.get("ordered_features", list(values_dict.keys()))
            X_input = pd.DataFrame([canonical_values], columns=ordered_names)

            # Raw prediction & probabilities
            raw_pred = self._model.predict(X_input)[0]
            raw_probs = self._model.predict_proba(X_input)[0]

            # Validate probability outputs
            if len(raw_probs) != len(self._model.classes_):
                raise ValueError("Probability vector size does not match model classes")
            if any(math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0 for p in raw_probs):
                raise ValueError(f"Invalid non-finite or out-of-bounds probability vector: {raw_probs}")

            # Map actual model.classes_ strictly to probability dict
            class_probs = {cls_name: float(raw_probs[i]) for i, cls_name in enumerate(self._model.classes_)}
            predicted_class = str(raw_pred)

            if predicted_class not in class_probs:
                raise ValueError(f"Predicted class '{predicted_class}' not in model classes {list(self._model.classes_)}")

            risk_score = self._map_ml_prediction_to_score(predicted_class, class_probs)
            confidence = round(float(class_probs[predicted_class]), 2)

            # Generate traceable risk drivers
            drivers: List[RiskDriver] = []
            hi = values_dict["heat_index_c"]
            wb = values_dict["wet_bulb_temperature_c"]
            hot_days = values_dict["consecutive_hot_days"]

            if wb >= 28.0:
                drivers.append(
                    RiskDriver(
                        parameter="wet_bulb_temperature_c",
                        impact_level="HIGH" if wb >= 29.0 else "MODERATE",
                        observed_value=round(wb, 1),
                        threshold_reference=">= 28.0°C (Elevated sweat inhibition)",
                        description="High humidity severely restricts atmospheric evaporative cooling.",
                    )
                )

            if hi >= 40.0:
                drivers.append(
                    RiskDriver(
                        parameter="heat_index_c",
                        impact_level="HIGH" if hi >= 46.0 else "MODERATE",
                        observed_value=round(hi, 1),
                        threshold_reference=">= 40.0°C (Caution/Danger zone)",
                        description="Elevated apparent temperature poses significant thermal strain.",
                    )
                )

            if hot_days >= 3:
                drivers.append(
                    RiskDriver(
                        parameter="consecutive_hot_days",
                        impact_level="HIGH" if hot_days >= 5 else "MODERATE",
                        observed_value=hot_days,
                        threshold_reference=">= 3 days",
                        description=f"Cumulative multi-day heatwave burden over {int(hot_days)} days.",
                    )
                )

            # Add primary ML classification driver
            drivers.append(
                RiskDriver(
                    parameter="ml_classification",
                    impact_level=predicted_class,
                    observed_value=round(confidence, 2),
                    threshold_reference=f"LightGBM 6-Hour Forecaster (P={confidence:.0%})",
                    description=f"Ensemble predicts forward 6-hour thermal severity as {predicted_class}.",
                )
            )

            evidence = {
                **values_dict,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "prediction_source": "lightgbm_ensemble",
                "class_probabilities": class_probs,
                "safety_override_checked": True,
            }

            return self.create_traceable_prediction(
                location_id=location_id,
                risk_score=risk_score,
                confidence=confidence,
                drivers=drivers,
                evidence=evidence,
                validity_hours=validity_hours,
            )

        except Exception as e:
            # Step 4: Rule-based fallback upon ML failure
            logger.warning(
                f"ML inference failed ({e}); safely activating RuleBasedHeatModel fallback "
                f"for location='{location_id}'"
            )
            enriched_features = {**features, **values_dict}
            fallback_prediction = self._fallback_model.predict(enriched_features, location_id, validity_hours)
            fallback_prediction.evidence["fallback_active"] = True
            fallback_prediction.evidence["fallback_reason"] = str(e)
            fallback_prediction.evidence["prediction_source"] = "rule_based_fallback"
            return fallback_prediction
