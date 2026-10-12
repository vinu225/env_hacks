"""
Decision Service (Jev Routing Boundary).

Orchestrates operational workflows, early warning dispatches, and emergency routing
via the DecisionProvider abstraction without coupling to specific rule engine code.
"""

from typing import Any, Dict, List, Optional
from app.integrations.jev_provider import (
    DecisionOutcome,
    DecisionProvider,
    get_decision_provider,
)
from app.schemas.risk import CompositeRiskSummary, RiskPrediction


class DecisionService:
    """Coordinates policy execution and operational routing."""

    def __init__(self, provider: Optional[DecisionProvider] = None):
        self._provider = provider or get_decision_provider()

    async def evaluate_risk_decision(
        self, risk_summary: CompositeRiskSummary
    ) -> List[DecisionOutcome]:
        """Trigger policy evaluation based on multi-hazard summary."""
        context = {
            "location_id": risk_summary.location_id,
            "overall_score": risk_summary.overall_risk_score,
            "risk_level": risk_summary.overall_risk_level.value,
            "primary_hazard": risk_summary.primary_hazard.value,
        }
        return await self._provider.evaluate_decision(
            context=context, policy_group="municipal_disaster_action"
        )
