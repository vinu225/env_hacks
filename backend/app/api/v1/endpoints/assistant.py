"""
AI Assistant and Natural Language Explanation Endpoints.

Extension point for LLM narrative synthesis.
Reminder: Never make LLM responsible for numerical risk scoring.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.integrations.llm_provider import LLMExplanation
from app.repositories.location_repository import LocationRepository, get_location_repository
from app.services.ai.service import AIService
from app.services.feature_engineering.service import FeatureEngineeringService
from app.services.risk.engine import RiskEngine
from app.services.weather.service import WeatherService

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])


class AssistantExplanationRequest(BaseModel):
    location_id: str
    audience: str = Field(default="general_public", description="general_public, municipal_officer, volunteer")
    language: str = Field(default="en", description="en, hi, ta, mr")


@router.post(
    "/explain",
    response_model=LLMExplanation,
    summary="Generate Plain-Language Environmental Risk Explanation",
)
async def explain_hyperlocal_risk(
    payload: AssistantExplanationRequest,
    location_repo: LocationRepository = Depends(get_location_repository),
) -> LLMExplanation:
    """
    Placeholder endpoint for AI natural language explanation.
    Uses AIService backed by MockLLMProvider.
    """
    location = await location_repo.get_by_id(payload.location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{payload.location_id}' not found.",
        )

    # 1. Fetch current weather and features
    weather_svc = WeatherService()
    snapshot = await weather_svc.get_current_snapshot(
        location_id=location.id,
        latitude=location.coordinates.latitude,
        longitude=location.coordinates.longitude,
    )
    fe_svc = FeatureEngineeringService()
    heat_features = fe_svc.prepare_heat_features(snapshot, location)

    # 2. Compute risk with traceable engine
    risk_engine = RiskEngine()
    prediction = await risk_engine.evaluate_heat_risk(location.id, heat_features)

    # 3. Request LLM explanation from AIService
    ai_svc = AIService()
    return await ai_svc.explain_risk(
        prediction=prediction,
        audience=payload.audience,
        language=payload.language,
    )
