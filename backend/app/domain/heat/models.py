"""
Domain Entities for Heat Risk and Thermal Stress.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HeatStressMetrics:
    """Core domain metrics for localized heat stress evaluation."""
    dry_bulb_temp_c: float
    relative_humidity_pct: float
    heat_index_c: float
    wet_bulb_temp_c: Optional[float] = None
    excess_heat_factor: Optional[float] = None
    consecutive_hot_days: int = 0
