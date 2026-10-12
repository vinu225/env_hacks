"""
LLM Provider Abstraction.

LLMs are strictly used for natural-language explanations, multilingual advisories,
and citizen interaction. They MUST NOT compute numerical risk scores.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMExplanation(BaseModel):
    """Structured natural language advisory produced by LLM services."""
    summary: str = Field(..., description="High-level synthesis in plain language")
    key_bullet_points: List[str] = Field(default_factory=list)
    actionable_guidance: str = Field(...)
    language: str = Field(default="en")
    model_id: str = Field(...)


class LLMProvider(ABC):
    """Abstract interface for LLM natural language generation and synthesis."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate_explanation(
        self,
        risk_data: Dict[str, Any],
        audience: str = "general_public",
        language: str = "en",
    ) -> LLMExplanation:
        """Translate deterministic and ML risk outputs into clear human advisory."""
        pass

    @abstractmethod
    async def summarize_reports(
        self,
        reports: List[Dict[str, Any]],
        location_name: str,
    ) -> str:
        """Synthesize multiple community reports into a concise situation briefing."""
        pass


class MockLLMProvider(LLMProvider):
    """
    Mock LLM Provider for local development.
    Uses deterministic templates without calling external APIs (e.g. AWS Bedrock / OpenAI).
    """

    @property
    def provider_name(self) -> str:
        return "mock_llm_provider"

    async def generate_explanation(
        self,
        risk_data: Dict[str, Any],
        audience: str = "general_public",
        language: str = "en",
    ) -> LLMExplanation:
        category = risk_data.get("category", "ENVIRONMENTAL")
        score = risk_data.get("risk_score", 0.0)
        level = risk_data.get("risk_level", "LOW")

        summary = (
            f"[MOCK ADVISORY] Current {category} risk level is evaluated as {level} "
            f"(Index: {score}/100). Stay hydrated and monitor regional updates."
        )

        bullets = [
            f"Assessed category: {category}",
            f"Severity classification: {level}",
            "Deterministic baseline active; advisory produced by mock LLM skeleton.",
        ]

        return LLMExplanation(
            summary=summary,
            key_bullet_points=bullets,
            actionable_guidance="Follow official municipal guidelines and avoid peak sun/waterlogging areas.",
            language=language,
            model_id=self.provider_name,
        )

    async def summarize_reports(
        self,
        reports: List[Dict[str, Any]],
        location_name: str,
    ) -> str:
        return (
            f"[MOCK SUMMARY] {len(reports)} community reports received for {location_name}. "
            "Primary incidents relate to drainage or water disruptions."
        )


def get_llm_provider() -> LLMProvider:
    """Factory retrieving configured LLM provider."""
    return MockLLMProvider()
