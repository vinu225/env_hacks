"""
Domain Entities for Community Environmental Incident Reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from app.schemas.report import IncidentType, ReportSeverity, VerificationStatus


@dataclass
class CommunityReportEntity:
    """Core domain model of a citizen or ground-volunteer incident report."""
    id: str
    incident_type: IncidentType
    severity: ReportSeverity
    description: str
    reported_at: datetime
    location_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    upvotes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
