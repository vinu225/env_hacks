"""
Alerts and Early Warning Endpoints.
"""

from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Query
from app.schemas.alert import AlertRead
from app.schemas.risk import RiskCategory, RiskLevel
from app.utils.datetime_utils import utc_now

router = APIRouter(prefix="/alerts", tags=["Alerts & Warnings"])


@router.get("", response_model=List[AlertRead], summary="List active environmental alerts")
async def list_active_alerts(
    location_id: Optional[str] = Query(default=None),
) -> List[AlertRead]:
    """
    Placeholder endpoint for real-time alerts.
    Returns simulated active warnings for foundation demonstration.
    """
    now = utc_now()
    demo_alerts = [
        AlertRead(
            id="alt_heat_delhi_01",
            location_id="loc_delhi_central",
            risk_category=RiskCategory.HEAT,
            severity=RiskLevel.SEVERE,
            title="Severe Thermal Heatwave Watch",
            summary="Apparent temperature exceeding 42°C with poor nighttime thermal dissipation.",
            triggered_at=now - timedelta(hours=2),
            expires_at=now + timedelta(hours=22),
            recommended_actions=[
                "Halt non-essential outdoor construction between 12:00-16:00.",
                "Ensure local community clinics have oral rehydration salts prepared.",
            ],
            model_source="rule-baseline-heat:1.0.0",
            is_active=True,
        ),
        AlertRead(
            id="alt_flood_mumbai_01",
            location_id="loc_mumbai_kurla",
            risk_category=RiskCategory.RAINFALL,
            severity=RiskLevel.HIGH,
            title="Urban Waterlogging and Drainage Choke Warning",
            summary="Tide peak coinciding with 20mm/hr intense rainfall in Mithi catchment.",
            triggered_at=now - timedelta(minutes=45),
            expires_at=now + timedelta(hours=6),
            recommended_actions=[
                "Deploy dewatering pump units to Kurla West rail underpasses.",
                "Divert low-chassis road transit away from LBS Marg.",
            ],
            model_source="rule-baseline-rainfall:1.0.0",
            is_active=True,
        ),
    ]

    if location_id:
        return [a for a in demo_alerts if a.location_id == location_id]
    return demo_alerts
