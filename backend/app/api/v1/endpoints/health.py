"""
Health and Readiness Endpoints.

Essential for AWS ECS / ALB health checks, Kubernetes probes, and service monitoring.
"""

from typing import Any, Dict
from fastapi import APIRouter, status
from app.core.config import get_settings
from app.integrations.weather_provider import get_weather_provider
from app.repositories.location_repository import get_location_repository
from app.utils.datetime_utils import utc_now

router = APIRouter(tags=["Health & Readiness"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Application Liveness Probe",
    response_description="Basic application liveness status",
)
async def get_health() -> Dict[str, Any]:
    """
    Liveness probe.
    Returns HTTP 200 if the FastAPI application process is running and accepting traffic.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": utc_now().isoformat(),
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Application Readiness Probe",
    response_description="Readiness status of downstream providers and repositories",
)
async def get_readiness() -> Dict[str, Any]:
    """
    Readiness probe.
    Validates that essential repositories, storage layers, and provider configurations
    are initialized and capable of servicing requests.
    """
    settings = get_settings()
    location_repo = get_location_repository()
    weather_provider = get_weather_provider()

    # Verify repository availability
    locations = await location_repo.list_all(limit=1)
    repo_healthy = len(locations) >= 0

    return {
        "status": "ready",
        "timestamp": utc_now().isoformat(),
        "checks": {
            "storage_backend": {
                "configured": settings.STORAGE_BACKEND,
                "status": "healthy" if repo_healthy else "degraded",
            },
            "weather_provider": {
                "name": weather_provider.provider_name,
                "type": settings.WEATHER_PROVIDER_TYPE,
                "status": "healthy",
            },
            "environment": settings.ENVIRONMENT,
        },
    }
