"""
Base Model Abstraction and Prediction Contract.

This is the architectural boundary ensuring the platform depends strictly on
the RiskModel contract rather than specific ML libraries (XGBoost, PyTorch, Scikit-learn).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from app.schemas.risk import RiskCategory, RiskDriver, RiskLevel, RiskPrediction
from app.utils.datetime_utils import utc_now


def score_to_risk_level(score: float) -> RiskLevel:
    """Standardized monotonic mapping of 0-100 score to risk levels."""
    if score >= 80.0:
        return RiskLevel.CRITICAL
    if score >= 60.0:
        return RiskLevel.SEVERE
    if score >= 40.0:
        return RiskLevel.HIGH
    if score >= 20.0:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


class RiskModel(ABC):
    """
    Abstract Base Class for all Environmental Risk Models.
    Unifies Rule-Based baselines and future ML Models under an identical contract.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier of the model (e.g. 'rule-baseline-heat', 'xgboost-heat-v1')."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version tag of the model artifact or rule specification."""
        pass

    @property
    @abstractmethod
    def category(self) -> RiskCategory:
        """Primary risk category assessed by this model."""
        pass

    @property
    def is_ml_model(self) -> bool:
        """Boolean flag declaring whether this model is ML-trained or rule-based."""
        return False

    @abstractmethod
    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 24,
    ) -> RiskPrediction:
        """
        Execute prediction pipeline:
        1. Validates required input features
        2. Computes risk score (0-100) and confidence (0-1)
        3. Identifies key contributing drivers and supporting evidence
        4. Returns standardized, traceable RiskPrediction
        """
        pass

    @abstractmethod
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return metadata regarding expected feature names, data types, and thresholds."""
        pass

    def create_traceable_prediction(
        self,
        location_id: str,
        risk_score: float,
        confidence: float,
        drivers: List[RiskDriver],
        evidence: Dict[str, Any],
        validity_hours: int = 24,
        data_version: str = "v1.0",
    ) -> RiskPrediction:
        """Helper to construct fully traceable, immutable prediction objects."""
        now = utc_now()
        bounded_score = max(0.0, min(100.0, float(risk_score)))
        return RiskPrediction(
            location_id=location_id,
            category=self.category,
            risk_score=round(bounded_score, 2),
            risk_level=score_to_risk_level(bounded_score),
            confidence=round(max(0.0, min(1.0, float(confidence))), 2),
            generated_at=now,
            valid_until=now + timedelta(hours=validity_hours),
            model_name=self.model_name,
            model_version=self.model_version,
            drivers=drivers,
            evidence=evidence,
            data_version=data_version,
        )
