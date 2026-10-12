"""
Normalized Environmental Weather and Feature Endpoints.

Provides direct access to:
- Normalized meteorological observations with freshness and quality auditing
- Reusable ML-compatible feature vectors
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from app.features.builder import FeatureVectorBuilder
from app.schemas.environmental import NormalizedWeatherObservation
from app.schemas.features import EnvironmentalFeatureVector
from app.services.weather.service import WeatherService

router = APIRouter(prefix="/weather", tags=["Environmental Weather & Features"])

# Reusable service dependency
_weather_service_instance: Optional[WeatherService] = None


def get_weather_service() -> WeatherService:
    global _weather_service_instance
    if _weather_service_instance is None:
        _weather_service_instance = WeatherService()
    return _weather_service_instance


@router.get(
    "",
    response_model=NormalizedWeatherObservation,
    status_code=status.HTTP_200_OK,
    summary="Get Normalized Environmental Observation",
    response_description="Clean, validated, normalized meteorological observation with freshness audit",
)
async def get_normalized_weather(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"),
    location_id: Optional[str] = Query(default=None, description="Optional registered location ID"),
    force_refresh: bool = Query(default=False, description="Bypass cache and force fresh provider query"),
    service: WeatherService = Depends(get_weather_service),
) -> NormalizedWeatherObservation:
    """
    Retrieve real-time environmental observations normalized across any provider.
    Includes freshness seconds, data quality score, physical validation status, and standard units.
    """
    weather_data = await service.get_normalized_weather(
        latitude=latitude,
        longitude=longitude,
        location_id=location_id,
        force_refresh=force_refresh,
    )
    return weather_data.current


@router.get(
    "/features",
    response_model=EnvironmentalFeatureVector,
    status_code=status.HTTP_200_OK,
    summary="Get Normalized Environmental Feature Vector",
    response_description="Unified tabular feature matrix for online inference and offline training",
)
async def get_environmental_features(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"),
    location_id: Optional[str] = Query(default=None, description="Optional registered location ID"),
    service: WeatherService = Depends(get_weather_service),
) -> EnvironmentalFeatureVector:
    """
    Extract standardized feature vector spanning weather, heat, water, and rainfall.
    Produces identical schema representations for both online risk scoring and offline ML datasets.
    """
    weather_data = await service.get_normalized_weather(
        latitude=latitude,
        longitude=longitude,
        location_id=location_id,
    )
    return FeatureVectorBuilder.build_from_normalized_data(
        weather_data=weather_data,
        location_id=location_id,
    )
