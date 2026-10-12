"""
Domain Entities for Water Stress and Supply Resilience.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WaterBalanceMetrics:
    """Core domain metrics for water scarcity and drought tracking."""
    soil_moisture_index: float
    dry_spell_duration_days: int
    groundwater_table_depth_m: Optional[float] = None
    reservoir_storage_pct: Optional[float] = None
    supply_shortage_hours: int = 0
