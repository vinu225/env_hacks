"""
Unit Tests for RiskModel Abstraction and Prediction Contract.
"""

from typing import Any, Dict
import pytest
from app.models.base import RiskModel, score_to_risk_level
from app.schemas.risk import RiskCategory, RiskLevel, RiskPrediction


class DummyRiskModel(RiskModel):
    """Concrete test implementation of RiskModel to test the abstraction."""

    @property
    def model_name(self) -> str:
        return "test-dummy-model"

    @property
    def model_version(self) -> str:
        return "0.0.1"

    @property
    def category(self) -> RiskCategory:
        return RiskCategory.HEAT

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {"required_features": ["temperature_c"]}

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 24,
    ) -> RiskPrediction:
        temp = float(features.get("temperature_c", 25.0))
        score = min(100.0, temp * 2.0)
        return self.create_traceable_prediction(
            location_id=location_id,
            risk_score=score,
            confidence=0.85,
            drivers=[],
            evidence={"temperature_c": temp},
            validity_hours=validity_hours,
        )


def test_score_to_risk_level_mapping():
    """Verify monotonic severity scale mapping."""
    assert score_to_risk_level(15.0) == RiskLevel.LOW
    assert score_to_risk_level(25.0) == RiskLevel.MODERATE
    assert score_to_risk_level(45.0) == RiskLevel.HIGH
    assert score_to_risk_level(65.0) == RiskLevel.SEVERE
    assert score_to_risk_level(85.0) == RiskLevel.CRITICAL


def test_risk_model_prediction_traceability():
    """Verify that predictions adhere to the complete traceability contract."""
    model = DummyRiskModel()
    prediction = model.predict(features={"temperature_c": 35.0}, location_id="loc_delhi_01")

    assert prediction.location_id == "loc_delhi_01"
    assert prediction.category == RiskCategory.HEAT
    assert prediction.risk_score == 70.0
    assert prediction.risk_level == RiskLevel.SEVERE
    assert prediction.confidence == 0.85
    assert prediction.model_name == "test-dummy-model"
    assert prediction.model_version == "0.0.1"
    assert prediction.generated_at is not None
    assert prediction.valid_until > prediction.generated_at
    assert prediction.data_version == "v1.0"
    assert "temperature_c" in prediction.evidence
