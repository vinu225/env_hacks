"""
Water Predictor Orchestrator.

Loads and executes the active water model (Rule-Based baseline or ML Water Model).
"""

from typing import Any, Dict, Optional
from app.core.logging import get_logger
from app.models.base import RiskModel
from app.models.water.model import RuleBasedWaterModel
from app.schemas.risk import RiskPrediction

logger = get_logger(__name__)


class WaterPredictor:
    """Predictor orchestrating water risk inference across baseline or ML models."""

    def __init__(self, active_model: Optional[RiskModel] = None):
        self._model: RiskModel = active_model or RuleBasedWaterModel()
        logger.info(
            f"WaterPredictor initialized with model='{self._model.model_name}' "
            f"version='{self._model.model_version}' is_ml={self._model.is_ml_model}"
        )

    @property
    def model(self) -> RiskModel:
        return self._model

    def set_model(self, model: RiskModel) -> None:
        """Dynamic runtime model swap."""
        self._model = model
        logger.info(f"WaterPredictor switched to model='{model.model_name}:{model.model_version}'")

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 48,
    ) -> RiskPrediction:
        return self._model.predict(
            features=features,
            location_id=location_id,
            validity_hours=validity_hours,
        )
