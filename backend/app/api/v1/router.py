"""
API v1 Central Router Assembly.
"""

from fastapi import APIRouter
from app.api.v1.endpoints.alerts import router as alerts_router
from app.api.v1.endpoints.assistant import router as assistant_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.heat import router as heat_router
from app.api.v1.endpoints.locations import router as locations_router
from app.api.v1.endpoints.reports import router as reports_router
from app.api.v1.endpoints.risk import router as risk_router
from app.api.v1.endpoints.water import router as water_router
from app.api.v1.endpoints.weather import router as weather_router

api_v1_router = APIRouter()

# Health & probes
api_v1_router.include_router(health_router)

# Environmental weather observations & ML-ready features
api_v1_router.include_router(weather_router)

# Geographic entities & domain risks
api_v1_router.include_router(locations_router)
api_v1_router.include_router(heat_router)
api_v1_router.include_router(water_router)
api_v1_router.include_router(risk_router)

# Community incident reports & early warnings
api_v1_router.include_router(reports_router)
api_v1_router.include_router(alerts_router)

# Natural language explanations
api_v1_router.include_router(assistant_router)
