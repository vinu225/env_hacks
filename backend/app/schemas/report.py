"""
Community Environmental Incident Report Schemas.

Enables crowdsourced ground-truth verification of waterlogging, dry taps,
heat distress, and canal/drainage blockages.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now


class IncidentType(str, Enum):
    """Categorization of community-reported environmental incidents."""
    WATERLOGGING = "WATERLOGGING"
    DRINKING_WATER_SHORTAGE = "DRINKING_WATER_SHORTAGE"
    PIPE_BURST_OR_LEAK = "PIPE_BURST_OR_LEAK"
    EXTREME_HEAT_DISTRESS = "EXTREME_HEAT_DISTRESS"
    CONTAMINATED_WATER = "CONTAMINATED_WATER"
    DRAINAGE_BLOCKED = "DRAINAGE_BLOCKED"
    OTHER = "OTHER"


class ReportSeverity(str, Enum):
    """Reported severity level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerificationStatus(str, Enum):
    """Verification lifecycle of community reports."""
    UNVERIFIED = "UNVERIFIED"
    COMMUNITY_UPVOTED = "COMMUNITY_UPVOTED"
    SENSOR_CORROBORATED = "SENSOR_CORROBORATED"
    RESOLVED = "RESOLVED"


class IncidentReportBase(BaseModel):
    """Core attributes of an incident report."""
    location_id: Optional[str] = Field(default=None, description="Registered location identifier")
    coordinates: Optional[Coordinates] = Field(default=None, description="Exact GPS coordinates of the incident")
    incident_type: IncidentType
    severity: ReportSeverity = ReportSeverity.MEDIUM
    description: str = Field(..., min_length=5, max_length=1000)
    reporter_id: Optional[str] = Field(default=None, description="Anonymized user identifier or null")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncidentReportCreate(IncidentReportBase):
    """Payload submitted when lodging a new community report."""
    pass


class IncidentReportRead(IncidentReportBase):
    """Stored community report with system metadata."""
    id: str = Field(..., description="Unique report ID")
    reported_at: datetime = Field(default_factory=utc_now)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    upvote_count: int = Field(default=0, ge=0)

    model_config = {"from_attributes": True}
