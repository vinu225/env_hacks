"""
Domain Entities for Risk Assessments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from app.schemas.risk import RiskCategory, RiskLevel


@dataclass
class RiskEvaluation:
    """Core domain evaluation record produced by assessment pipelines."""
    location_id: str
    category: RiskCategory
    score: float
    level: RiskLevel
    confidence: float
    model_name: str
    model_version: str
    generated_at: datetime
    valid_until: datetime
    drivers: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
