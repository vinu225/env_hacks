"""
AI & Natural Language Service.

Coordinates human-readable explanations, multilingual advisories, and citizen dialogue.
Relies on LLMProvider. Never computes numerical risks.
"""

from typing import Any, Dict, List, Optional
from app.integrations.llm_provider import (
    LLMExplanation,
    LLMProvider,
    get_llm_provider,
)
from app.schemas.risk import RiskPrediction


class AIService:
    """Service mediating natural language explanation and report summarization."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider or get_llm_provider()

    async def explain_risk(
        self,
        prediction: RiskPrediction,
        audience: str = "general_public",
        language: str = "en",
    ) -> LLMExplanation:
        """Produce clear, accessible narrative explanation for a risk prediction."""
        risk_data = {
            "category": prediction.category.value,
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level.value,
            "confidence": prediction.confidence,
            "model_name": prediction.model_name,
            "drivers": [d.model_dump() for d in prediction.drivers],
        }
        return await self._provider.generate_explanation(
            risk_data=risk_data, audience=audience, language=language
        )

    async def summarize_community_reports(
        self, reports: List[Dict[str, Any]], location_name: str
    ) -> str:
        """Synthesize community reports into an executive briefing."""
        return await self._provider.summarize_reports(reports=reports, location_name=location_name)
