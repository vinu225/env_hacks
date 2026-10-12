"""
Risk Fusion Service.

Synthesizes multi-hazard risks (Heat, Water, Rainfall) into a coherent,
composite index. Identifies primary hazards and computes deterministic summaries.
"""

from typing import Dict
from app.models.base import score_to_risk_level
from app.schemas.risk import CompositeRiskSummary, RiskCategory, RiskPrediction
from app.utils.datetime_utils import utc_now


class RiskFusionService:
    """Combines individual hazard predictions into an integrated situational picture."""

    def fuse_risks(
        self,
        location_id: str,
        heat_risk: RiskPrediction,
        water_risk: RiskPrediction,
        rainfall_risk: RiskPrediction,
    ) -> CompositeRiskSummary:
        sub_risks: Dict[str, RiskPrediction] = {
            "heat": heat_risk,
            "water": water_risk,
            "rainfall": rainfall_risk,
        }

        # Weighted multi-hazard score (prioritizes active extremes)
        # Dominant maximum principle + weighted background risk
        max_sub = max(heat_risk.risk_score, water_risk.risk_score, rainfall_risk.risk_score)
        avg_sub = (heat_risk.risk_score + water_risk.risk_score + rainfall_risk.risk_score) / 3.0
        composite_score = round((0.7 * max_sub) + (0.3 * avg_sub), 2)
        composite_level = score_to_risk_level(composite_score)

        # Average confidence weighted across active inputs
        overall_confidence = round(
            (heat_risk.confidence + water_risk.confidence + rainfall_risk.confidence) / 3.0, 2
        )

        # Identify dominant hazard
        scores = [
            (heat_risk.risk_score, RiskCategory.HEAT),
            (water_risk.risk_score, RiskCategory.WATER_STRESS),
            (rainfall_risk.risk_score, RiskCategory.RAINFALL),
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        primary_hazard = scores[0][1]

        summary_text = (
            f"Overall environmental risk is classified as {composite_level.value} "
            f"(Score: {composite_score}/100). Dominant driver is {primary_hazard.value} "
            f"(Score: {scores[0][0]}/100)."
        )

        return CompositeRiskSummary(
            location_id=location_id,
            generated_at=utc_now(),
            overall_risk_score=composite_score,
            overall_risk_level=composite_level,
            confidence=overall_confidence,
            sub_risks=sub_risks,
            primary_hazard=primary_hazard,
            summary_text=summary_text,
        )
