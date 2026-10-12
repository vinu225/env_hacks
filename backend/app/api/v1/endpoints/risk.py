"""
Risk Fusion and Composite Risk Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.repositories.location_repository import LocationRepository, get_location_repository
from app.schemas.risk import CompositeRiskSummary
from app.services.feature_engineering.service import FeatureEngineeringService
from app.services.risk.engine import RiskEngine
from app.services.risk.fusion import RiskFusionService
from app.services.weather.service import WeatherService

router = APIRouter(prefix="/locations", tags=["Composite Environmental Risk"])


@router.get(
    "/{location_id}/risk",
    response_model=CompositeRiskSummary,
    summary="Get Multi-Hazard Environmental Risk Fusion",
)
async def get_composite_risk(
    location_id: str,
    location_repo: LocationRepository = Depends(get_location_repository),
) -> CompositeRiskSummary:
    location = await location_repo.get_by_id(location_id)
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{location_id}' not found.",
        )

    # 1. Fetch current weather snapshot
    weather_svc = WeatherService()
    snapshot = await weather_svc.get_current_snapshot(
        location_id=location.id,
        latitude=location.coordinates.latitude,
        longitude=location.coordinates.longitude,
    )

    # 2. Extract domain features
    fe_svc = FeatureEngineeringService()
    heat_feat = fe_svc.prepare_heat_features(snapshot, location)
    water_feat = fe_svc.prepare_water_features(snapshot, location)
    rain_feat = fe_svc.prepare_rainfall_features(snapshot, location)

    # 3. Predict domain risks
    risk_engine = RiskEngine()
    heat_pred = await risk_engine.evaluate_heat_risk(location.id, heat_feat)
    water_pred = await risk_engine.evaluate_water_risk(location.id, water_feat)
    rain_pred = await risk_engine.evaluate_rainfall_risk(location.id, rain_feat)

    # 4. Multi-hazard risk fusion
    fusion_svc = RiskFusionService()
    return fusion_svc.fuse_risks(
        location_id=location.id,
        heat_risk=heat_pred,
        water_risk=water_pred,
        rainfall_risk=rain_pred,
    )
