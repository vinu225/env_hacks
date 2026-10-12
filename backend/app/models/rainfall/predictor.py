"""
Rainfall Predictor Orchestrator.

Loads and executes the active rainfall/flood risk model (Rule-Based baseline or ML Rainfall Model).
"""

from typing import Any, Dict, Optional
from app.core.logging import get_logger
from app.models.base import RiskModel
from app.models.rainfall.model import RuleBasedRainfallModel
from app.schemas.risk import RiskPrediction

logger = get_logger(__name__)


class RainfallPredictor:
    """Predictor orchestrating rainfall risk inference across baseline or ML models."""

    def __init__(self, active_model: Optional[RiskModel] = None):
        self._model: RiskModel = active_model or RuleBasedRainfallModel()
        logger.info(
            f"RainfallPredictor initialized with model='{self._model.model_name}' "
            f"version='{self._model.model_version}' is_ml={self._model.is_ml_model}"
        )

    @property
    def model(self) -> RiskModel:
        return self._model

    def set_model(self, model: RiskModel) -> None:
        """Dynamic runtime model swap."""
        self._model = model
        logger.info(f"RainfallPredictor switched to model='{model.model_name}:{model.model_version}'")

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 12,
    ) -> RiskPrediction:
        return self._model.predict(
            features=features,
            location_id=location_id,
            validity_hours=validity_hours,
        )
