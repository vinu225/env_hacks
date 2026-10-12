"""
Risk Engine Service.

Coordinates evaluation across domain predictors (Heat, Water, Rainfall).
Decoupled from model algorithms via the RiskModel abstraction contract.
"""

from typing import Any, Dict, Optional
from app.core.logging import get_logger
from app.models.heat.predictor import HeatPredictor
from app.models.rainfall.predictor import RainfallPredictor
from app.models.water.predictor import WaterPredictor
from app.repositories.risk_repository import RiskRepository, get_risk_repository
from app.schemas.risk import RiskCategory, RiskPrediction

logger = get_logger(__name__)


class RiskEngine:
    """Core computational engine driving risk model predictions."""

    def __init__(
        self,
        heat_predictor: Optional[HeatPredictor] = None,
        water_predictor: Optional[WaterPredictor] = None,
        rainfall_predictor: Optional[RainfallPredictor] = None,
        risk_repository: Optional[RiskRepository] = None,
    ):
        self.heat_predictor = heat_predictor or HeatPredictor()
        self.water_predictor = water_predictor or WaterPredictor()
        self.rainfall_predictor = rainfall_predictor or RainfallPredictor()
        self.risk_repository = risk_repository or get_risk_repository()

    async def evaluate_heat_risk(
        self, location_id: str, features: Dict[str, Any]
    ) -> RiskPrediction:
        prediction = self.heat_predictor.predict(features=features, location_id=location_id)
        await self.risk_repository.save(prediction)
        return prediction

    async def evaluate_water_risk(
        self, location_id: str, features: Dict[str, Any]
    ) -> RiskPrediction:
        prediction = self.water_predictor.predict(features=features, location_id=location_id)
        await self.risk_repository.save(prediction)
        return prediction

    async def evaluate_rainfall_risk(
        self, location_id: str, features: Dict[str, Any]
    ) -> RiskPrediction:
        prediction = self.rainfall_predictor.predict(features=features, location_id=location_id)
        await self.risk_repository.save(prediction)
        return prediction
