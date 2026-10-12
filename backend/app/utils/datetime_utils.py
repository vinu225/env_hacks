"""
Datetime utility functions for standardized UTC handling and timestamp formatting.
"""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Return the current time in UTC with timezone awareness."""
    return datetime.now(timezone.utc)


def format_iso(dt: Optional[datetime] = None) -> str:
    """Format datetime as ISO-8601 string in UTC."""
    if dt is None:
        dt = utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def compute_freshness_seconds(observed_at: datetime, reference_time: Optional[datetime] = None) -> float:
    """Compute the age in seconds between observation and reference time."""
    ref = reference_time or utc_now()
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    diff = (ref - observed_at).total_seconds()
    return max(0.0, diff)
