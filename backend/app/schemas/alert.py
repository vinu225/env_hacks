"""
Environmental Alert and Notification Schemas.

Structures deterministic and ML-triggered early warnings for public safety
and municipal administration.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.risk import RiskCategory, RiskLevel
from app.utils.datetime_utils import utc_now


class AlertRead(BaseModel):
    """Early warning alert entity."""
    id: str = Field(..., description="Unique alert identifier")
    location_id: str
    risk_category: RiskCategory
    severity: RiskLevel
    title: str = Field(..., min_length=3, max_length=200)
    summary: str = Field(..., min_length=10, max_length=1000)
    triggered_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    recommended_actions: List[str] = Field(default_factory=list)
    model_source: str = Field(..., description="Model name and version triggering the alert")
    is_active: bool = Field(default=True)

    model_config = {"from_attributes": True}
