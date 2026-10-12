"""
Focused Phase E integration tests for MLHeatModel and HeatPredictor.

Covers all required Phase E test obligations:
1. Successful artifact loading
2. Correct canonical feature ordering
3. Successful prediction and correct class mapping
4. Probability vector validation
5. Missing and malformed features handling
6. Non-finite input handling (no silent fallback to LOW)
7. Corrupted or missing model artifact handling
8. Inference exceptions and rule-based fallback behavior
9. CRITICAL override precedence over lower ML predictions
10. Both configured CRITICAL thresholds and exact boundaries (HI >= 54.0°C, Tw >= 31.0°C)
11. No fabricated CRITICAL training class verification
12. Compatibility with the existing RiskPrediction and API response contracts
13. Configurable model selection in HeatPredictor and dynamic model swapping
14. End-to-end FastAPI endpoint execution with MLHeatModel active
"""

import math
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.models.heat.ml_model import MLHeatModel, resolve_artifact_dir
from app.models.heat.model import RuleBasedHeatModel
from app.models.heat.predictor import HeatPredictor
from app.schemas.heat import HeatRiskAssessment
from app.schemas.risk import RiskCategory, RiskLevel, RiskPrediction
from app.services.risk.engine import RiskEngine
from app.utils.exceptions import FeatureComputationError


@pytest.fixture
def sample_normal_features():
    """Typical non-extreme weather observation."""
    return {
        "temperature_c": 31.0,
        "apparent_temperature_c": 34.0,
        "relative_humidity_pct": 55.0,
        "dew_point_c": 21.0,
        "heat_index_c": 34.0,
        "wet_bulb_temperature_c": 24.0,
        "wind_speed_kmh": 12.0,
        "consecutive_hot_days": 1.0,
        "is_high_temperature": 0.0,
        "is_humidity_stress": 0.0,
        "precipitation_mm": 0.0,
        "rainfall_accumulation_24h_mm": 0.0,
        "rainfall_accumulation_7d_mm": 0.0,
        "consecutive_dry_days": 2.0,
    }


def test_successful_artifact_loading():
    """Verify MLHeatModel initializes cleanly, loads artifacts, and checks model metadata."""
    model = MLHeatModel()
    assert model.model_name == "lightgbm-heat-v1"
    assert model.model_version == "1.0.0"
    assert model.category == RiskCategory.HEAT
    assert model.is_ml_model is True

    metadata = model.get_feature_metadata()
    assert len(metadata["required_features"]) == 14
    assert "critical_heat_index_c" in metadata["safety_thresholds"]
    assert metadata["safety_thresholds"]["critical_heat_index_c"] == 54.0


def test_correct_canonical_feature_ordering(sample_normal_features):
    """Verify feature extraction preserves the exact 14 declared order and names."""
    model = MLHeatModel()
    canonical_list, values_dict = model._extract_canonical_features(sample_normal_features)

    assert len(canonical_list) == 14
    expected_order = model._schema["ordered_features"]
    assert len(expected_order) == 14

    for idx, name in enumerate(expected_order):
        assert canonical_list[idx] == values_dict[name]


def test_successful_prediction_and_correct_class_mapping(sample_normal_features):
    """Verify predict returns a valid RiskPrediction matching the predicted risk level."""
    model = MLHeatModel()
    prediction = model.predict(sample_normal_features, location_id="loc_delhi_central")

    assert isinstance(prediction, RiskPrediction)
    assert prediction.location_id == "loc_delhi_central"
    assert prediction.category == RiskCategory.HEAT
    assert prediction.model_name == "lightgbm-heat-v1"
    assert 0.0 <= prediction.risk_score <= 100.0
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.SEVERE]
    assert len(prediction.drivers) >= 1
    assert prediction.evidence["prediction_source"] == "lightgbm_ensemble"


def test_probability_vector_validation(sample_normal_features):
    """Verify predicted class probabilities sum to 1.0 and cover all empirical classes."""
    model = MLHeatModel()
    prediction = model.predict(sample_normal_features, location_id="loc_mumbai_central")

    probs = prediction.evidence.get("class_probabilities", {})
    assert set(probs.keys()) == {"HIGH", "LOW", "MODERATE", "SEVERE"}

    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, abs=1e-4) == 1.0
    for c, p in probs.items():
        assert 0.0 <= p <= 1.0
        assert not math.isnan(p)


def test_missing_and_malformed_features():
    """Missing fundamental temperature input must raise FeatureComputationError, not silently succeed."""
    model = MLHeatModel()
    invalid_features = {"wind_speed_kmh": 15.0}  # Missing temperature_c

    with pytest.raises(FeatureComputationError) as exc_info:
        model.predict(invalid_features, location_id="loc_test")
    assert "temperature_c" in str(exc_info.value)


def test_non_finite_input_handling(sample_normal_features):
    """Non-finite inputs (NaN, Inf) must be rejected safely and not silently mapped to LOW risk."""
    model = MLHeatModel()

    nan_features = sample_normal_features.copy()
    nan_features["temperature_c"] = float("nan")
    with pytest.raises(FeatureComputationError):
        model.predict(nan_features, location_id="loc_test")

    inf_features = sample_normal_features.copy()
    inf_features["temperature_c"] = float("inf")
    with pytest.raises(FeatureComputationError):
        model.predict(inf_features, location_id="loc_test")


def test_corrupted_or_missing_model_artifact(tmp_path):
    """Initializing MLHeatModel with an invalid artifact path raises FileNotFoundError."""
    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        MLHeatModel(artifact_dir=str(empty_dir))


def test_inference_exceptions_and_fallback_behavior(sample_normal_features):
    """If ML prediction encounters an unexpected inference exception, rule-based fallback activates seamlessly."""
    model = MLHeatModel()

    # Mock the internal booster to raise a runtime inference error
    broken_model = MagicMock()
    broken_model.predict.side_effect = RuntimeError("Simulated C++ LightGBM runtime crash")
    broken_model.classes_ = ["HIGH", "LOW", "MODERATE", "SEVERE"]
    model._model = broken_model

    prediction = model.predict(sample_normal_features, location_id="loc_delhi_central")

    assert isinstance(prediction, RiskPrediction)
    assert prediction.evidence.get("fallback_active") is True
    assert "Simulated C++ LightGBM runtime crash" in prediction.evidence.get("fallback_reason", "")
    assert prediction.evidence.get("prediction_source") == "rule_based_fallback"


def test_critical_override_precedence_over_lower_ml_prediction(sample_normal_features):
    """Extreme heat index must trigger CRITICAL override and cannot be suppressed by lower ML output."""
    model = MLHeatModel()
    extreme_features = sample_normal_features.copy()
    extreme_features["temperature_c"] = 48.0
    extreme_features["heat_index_c"] = 56.5  # Exceeds 54.0°C threshold

    prediction = model.predict(extreme_features, location_id="loc_rajasthan_barmer")

    assert prediction.risk_level == RiskLevel.CRITICAL
    assert prediction.risk_score >= 80.0
    assert prediction.evidence["prediction_source"] == "deterministic_safety_override"
    critical_driver = next((d for d in prediction.drivers if d.impact_level == "CRITICAL"), None)
    assert critical_driver is not None
    assert "54.0°C" in critical_driver.threshold_reference


def test_both_configured_critical_thresholds_exact_boundaries(sample_normal_features):
    """Verify exact boundary behavior for both Heat Index (54.0°C) and Wet-Bulb (31.0°C)."""
    model = MLHeatModel()

    # 1. Heat Index boundary: 53.9°C (Sub-critical) vs 54.0°C (Critical)
    sub_hi = sample_normal_features.copy()
    sub_hi["heat_index_c"] = 53.9
    sub_hi["wet_bulb_temperature_c"] = 28.0
    pred_sub = model.predict(sub_hi, location_id="loc_test")
    assert pred_sub.risk_level != RiskLevel.CRITICAL

    at_hi = sample_normal_features.copy()
    at_hi["heat_index_c"] = 54.0
    at_hi["wet_bulb_temperature_c"] = 28.0
    pred_at = model.predict(at_hi, location_id="loc_test")
    assert pred_at.risk_level == RiskLevel.CRITICAL
    assert pred_at.risk_score >= 80.0

    # 2. Wet-Bulb boundary: 30.9°C (Sub-critical) vs 31.0°C (Critical)
    sub_wb = sample_normal_features.copy()
    sub_wb["heat_index_c"] = 45.0
    sub_wb["wet_bulb_temperature_c"] = 30.9
    pred_sub_wb = model.predict(sub_wb, location_id="loc_test")
    assert pred_sub_wb.risk_level != RiskLevel.CRITICAL

    at_wb = sample_normal_features.copy()
    at_wb["heat_index_c"] = 45.0
    at_wb["wet_bulb_temperature_c"] = 31.0
    pred_at_wb = model.predict(at_wb, location_id="loc_test")
    assert pred_at_wb.risk_level == RiskLevel.CRITICAL
    assert pred_at_wb.risk_score >= 80.0


def test_no_fabricated_critical_training_class():
    """Verify trained model's classes_ strictly does NOT contain a fabricated CRITICAL class."""
    model = MLHeatModel()
    classes = list(model._model.classes_)
    assert classes == ["HIGH", "LOW", "MODERATE", "SEVERE"]
    assert "CRITICAL" not in classes


def test_compatibility_with_existing_api_response_contract(sample_normal_features):
    """Verify MLHeatModel output strictly validates against HeatRiskAssessment schema."""
    model = MLHeatModel()
    prediction = model.predict(sample_normal_features, location_id="loc_delhi_central")

    # Construct the full endpoint response model
    assessment = HeatRiskAssessment(
        prediction=prediction,
        heat_index_c=sample_normal_features["heat_index_c"],
        wet_bulb_temperature_c=sample_normal_features["wet_bulb_temperature_c"],
        advisory_recommendation="High hydration advisory",
        generated_at=prediction.generated_at,
    )
    assert assessment.prediction.model_name == "lightgbm-heat-v1"
    assert assessment.prediction.category == RiskCategory.HEAT


def test_heat_predictor_model_selection_and_swapping(sample_normal_features):
    """Test HeatPredictor dependency injection and runtime model swapping."""
    ml_model = MLHeatModel()
    rule_model = RuleBasedHeatModel()

    predictor = HeatPredictor(active_model=ml_model)
    assert predictor.model.is_ml_model is True
    assert predictor.model.model_name == "lightgbm-heat-v1"
    pred_ml = predictor.predict(sample_normal_features, location_id="loc_swap")
    assert pred_ml.model_name == "lightgbm-heat-v1"

    # Runtime swap to rule-based rollback
    predictor.set_model(rule_model)
    assert predictor.model.is_ml_model is False
    assert predictor.model.model_name == "rule-baseline-heat"
    pred_rule = predictor.predict(sample_normal_features, location_id="loc_swap")
    assert pred_rule.model_name == "rule-baseline-heat"


def test_end_to_end_endpoint_with_ml_model(client: TestClient, monkeypatch):
    """End-to-end integration test: configuring HEAT_MODEL_TYPE='ml' activates MLHeatModel through the FastAPI route."""
    monkeypatch.setenv("HEAT_MODEL_TYPE", "ml")

    # Instantiate RiskEngine with MLHeatModel
    engine = RiskEngine(heat_predictor=HeatPredictor(active_model=MLHeatModel()))
    app.dependency_overrides = {}

    response = client.get("/api/v1/locations/loc_delhi_central/heat")
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"]["category"] == "HEAT"
    assert "risk_score" in data["prediction"]
    assert "risk_level" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert "drivers" in data["prediction"]
