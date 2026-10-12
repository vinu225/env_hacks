"""
API v1 Endpoints Package.
"""

from app.api.v1.endpoints.alerts import router as alerts_router
from app.api.v1.endpoints.assistant import router as assistant_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.heat import router as heat_router
from app.api.v1.endpoints.locations import router as locations_router
from app.api.v1.endpoints.reports import router as reports_router
from app.api.v1.endpoints.risk import router as risk_router
from app.api.v1.endpoints.water import router as water_router
from app.api.v1.endpoints.weather import router as weather_router

__all__ = [
    "health_router",
    "weather_router",
    "locations_router",
    "heat_router",
    "water_router",
    "risk_router",
    "reports_router",
    "alerts_router",
    "assistant_router",
]
