"""
Water Stress Risk Assessment Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.repositories.location_repository import LocationRepository, get_location_repository
from app.schemas.water import WaterRiskAssessment
from app.services.feature_engineering.service import FeatureEngineeringService
from app.services.recommendations.service import RecommendationService
from app.services.risk.engine import RiskEngine
from app.services.weather.service import WeatherService

router = APIRouter(prefix="/locations", tags=["Water Stress"])


@router.get(
    "/{location_id}/water",
    response_model=WaterRiskAssessment,
    summary="Get Hyperlocal Water Stress & Supply Risk Assessment",
)
async def get_water_risk(
    location_id: str,
    location_repo: LocationRepository = Depends(get_location_repository),
) -> WaterRiskAssessment:
    location = await location_repo.get_by_id(location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{location_id}' not found.",
        )

    # Ingestion & features
    weather_svc = WeatherService()
    snapshot = await weather_svc.get_current_snapshot(
        location_id=location.id,
        latitude=location.coordinates.latitude,
        longitude=location.coordinates.longitude,
    )

    fe_svc = FeatureEngineeringService()
    water_features = fe_svc.prepare_water_features(snapshot, location)

    # Model inference
    risk_engine = RiskEngine()
    prediction = await risk_engine.evaluate_water_risk(
        location_id=location.id, features=water_features
    )

    rec_svc = RecommendationService()
    recommendations = rec_svc.generate_recommendations(prediction)
    advisory = " ".join(recommendations)

    return WaterRiskAssessment(
        prediction=prediction,
        water_stress_index=water_features.get("soil_moisture_deficit", 0.5),
        drought_stage="WATCH" if prediction.risk_score >= 40.0 else "NONE",
        supply_disruption_risk="HIGH" if prediction.risk_score >= 60.0 else "LOW",
        groundwater_status="CRITICAL" if prediction.risk_score >= 70.0 else "ADEQUATE",
        conservation_advisory=advisory,
    )
