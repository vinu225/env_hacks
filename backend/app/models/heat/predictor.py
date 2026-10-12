"""
Heat Predictor Orchestrator.

Loads and executes the active heat model (Rule-Based baseline or ML Heat Model).
Ensures zero disruption to upstream callers when switching model implementations.
"""

from typing import Any, Dict, Optional
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.base import RiskModel
from app.models.heat.ml_model import MLHeatModel
from app.models.heat.model import RuleBasedHeatModel
from app.schemas.risk import RiskPrediction

logger = get_logger(__name__)


class HeatPredictor:
    """Predictor orchestrating heat risk inference across baseline or ML models."""

    def __init__(self, active_model: Optional[RiskModel] = None):
        if active_model is not None:
            self._model: RiskModel = active_model
        else:
            settings = get_settings()
            if settings.HEAT_MODEL_TYPE == "ml":
                try:
                    self._model = MLHeatModel(artifact_dir=settings.HEAT_MODEL_ARTIFACT_DIR or None)
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize MLHeatModel ({e}); falling back to RuleBasedHeatModel"
                    )
                    self._model = RuleBasedHeatModel()
            else:
                self._model = RuleBasedHeatModel()

        logger.info(
            f"HeatPredictor initialized with model='{self._model.model_name}' "
            f"version='{self._model.model_version}' is_ml={self._model.is_ml_model}"
        )

    @property
    def model(self) -> RiskModel:
        return self._model

    def set_model(self, model: RiskModel) -> None:
        """Dynamic runtime model swap (e.g. for A/B testing or ML rollout)."""
        self._model = model
        logger.info(f"HeatPredictor switched to model='{model.model_name}:{model.model_version}'")

    def predict(
        self,
        features: Dict[str, Any],
        location_id: str,
        validity_hours: int = 24,
    ) -> RiskPrediction:
        """Execute risk prediction with the configured model."""
        return self._model.predict(
            features=features,
            location_id=location_id,
            validity_hours=validity_hours,
        )
