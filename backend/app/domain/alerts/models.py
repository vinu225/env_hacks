"""
Domain Entities for Warning Alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from app.schemas.risk import RiskCategory, RiskLevel


@dataclass
class AlertEntity:
    """Core domain representation of an early-warning alert."""
    id: str
    location_id: str
    category: RiskCategory
    severity: RiskLevel
    title: str
    summary: str
    triggered_at: datetime
    expires_at: datetime
    recommended_actions: List[str] = field(default_factory=list)
    model_source: str = ""
    is_active: bool = True
