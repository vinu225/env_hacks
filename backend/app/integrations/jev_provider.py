"""
Decision Provider Abstraction (Jev Extension Point).

Jev is reserved for bounded decision-making, policy routing, and workflow orchestration.
It is strictly NOT responsible for raw numerical environmental risk prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionOutcome(BaseModel):
    """Result of a bounded decision-making or policy routing evaluation."""
    decision_id: str
    action_type: str = Field(..., description="Action to trigger (e.g. DISPATCH_WATER_TANKER, ISSUE_HEAT_ADVISORY)")
    target_recipients: List[str] = Field(default_factory=list)
    priority: str = Field(default="MEDIUM", description="LOW, MEDIUM, HIGH, EMERGENCY")
    routing_destination: Optional[str] = Field(default=None, description="Department, queue, or automated handler")
    rule_name: str = Field(..., description="Jev decision rule or policy applied")
    is_automated_execution_allowed: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class DecisionProvider(ABC):
    """Abstract interface for bounded decision-making and routing engines."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of the decision engine."""
        pass

    @abstractmethod
    async def evaluate_decision(
        self,
        context: Dict[str, Any],
        policy_group: str = "environmental_response",
    ) -> List[DecisionOutcome]:
        """Evaluate contextual risk data against business/municipal policies."""
        pass


class MockDecisionProvider(DecisionProvider):
    """
    Mock implementation of DecisionProvider.
    Acts as a placeholder prior to full Jev engine integration.
    """

    @property
    def provider_name(self) -> str:
        return "mock_jev_decision_provider"

    async def evaluate_decision(
        self,
        context: Dict[str, Any],
        policy_group: str = "environmental_response",
    ) -> List[DecisionOutcome]:
        # Placeholder skeleton returning bounded routing decision
        risk_level = context.get("risk_level", "LOW")
        actions: List[DecisionOutcome] = []

        if risk_level in ["SEVERE", "CRITICAL"]:
            actions.append(
                DecisionOutcome(
                    decision_id="dec_mock_001",
                    action_type="NOTIFY_MUNICIPAL_DISASTER_DESK",
                    target_recipients=["ward_officer@gov.in"],
                    priority="HIGH",
                    routing_destination="municipal_alert_queue",
                    rule_name="MOCK_CRITICAL_RISK_POLICY",
                    is_automated_execution_allowed=False,
                    details={"evaluated_risk_level": risk_level},
                )
            )

        return actions


def get_decision_provider() -> DecisionProvider:
    """Factory retrieving configured decision provider."""
    return MockDecisionProvider()
