"""
Heat Risk Assessment Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.features.weather_features import calculate_heat_index, calculate_wet_bulb_temperature
from app.repositories.location_repository import LocationRepository, get_location_repository
from app.schemas.heat import HeatRiskAssessment
from app.services.feature_engineering.service import FeatureEngineeringService
from app.services.recommendations.service import RecommendationService
from app.services.risk.engine import RiskEngine
from app.services.weather.service import WeatherService

router = APIRouter(prefix="/locations", tags=["Heat Risk"])


@router.get(
    "/{location_id}/heat",
    response_model=HeatRiskAssessment,
    summary="Get Hyperlocal Heat Risk Assessment",
)
async def get_heat_risk(
    location_id: str,
    location_repo: LocationRepository = Depends(get_location_repository),
) -> HeatRiskAssessment:
    location = await location_repo.get_by_id(location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{location_id}' not found.",
        )

    # 1. Fetch current observation via weather service
    weather_svc = WeatherService()
    snapshot = await weather_svc.get_current_snapshot(
        location_id=location.id,
        latitude=location.coordinates.latitude,
        longitude=location.coordinates.longitude,
    )

    # 2. Extract features
    fe_svc = FeatureEngineeringService()
    heat_features = fe_svc.prepare_heat_features(snapshot, location)

    # 3. Model inference (Rule baseline initially, transparently swapped to ML later)
    risk_engine = RiskEngine()
    prediction = await risk_engine.evaluate_heat_risk(
        location_id=location.id, features=heat_features
    )

    # 4. Domain-specific indicators
    hi = calculate_heat_index(snapshot.temperature_c, snapshot.relative_humidity_pct)
    wb = calculate_wet_bulb_temperature(snapshot.temperature_c, snapshot.relative_humidity_pct)

    rec_svc = RecommendationService()
    recommendations = rec_svc.generate_recommendations(prediction)
    advisory = " ".join(recommendations)

    return HeatRiskAssessment(
        prediction=prediction,
        heat_index_c=hi,
        wet_bulb_temp_c=wb,
        is_heatwave_condition=(snapshot.temperature_c >= 40.0 or hi >= 45.0),
        nighttime_recovery_adequate=True,
        advisory_recommendation=advisory,
    )
