"""
Environmental Data Freshness Module.

Provides reusable mechanisms for evaluating the temporal validity, latency,
and staleness of environmental observations and forecast horizons.
"""

from datetime import datetime, timezone
from typing import Optional
from app.utils.datetime_utils import utc_now


class FreshnessEvaluator:
    """
    Evaluates observation freshness against defined SLA thresholds.
    Ensures downstream models can determine whether data is recent enough for use.
    """

    DEFAULT_MAX_AGE_SECONDS = 3600.0  # 1 hour for current observations
    FORECAST_TOLERANCE_SECONDS = 86400.0 * 7  # Up to 7 days for forecasts

    @staticmethod
    def calculate_age_seconds(observed_at: datetime, reference_time: Optional[datetime] = None) -> float:
        """
        Calculate latency/age in seconds between observed timestamp and reference time.
        Strictly enforces timezone-aware UTC comparison.
        """
        ref = reference_time or utc_now()
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return max(0.0, (ref - observed_at).total_seconds())

    @classmethod
    def is_observation_fresh(
        cls,
        observed_at: datetime,
        max_age_seconds: Optional[float] = None,
        reference_time: Optional[datetime] = None,
    ) -> bool:
        """Return True if observation is within acceptable age threshold."""
        threshold = max_age_seconds if max_age_seconds is not None else cls.DEFAULT_MAX_AGE_SECONDS
        age = cls.calculate_age_seconds(observed_at, reference_time=reference_time)
        return age <= threshold

    @classmethod
    def evaluate_quality_degradation(
        cls,
        observed_at: datetime,
        base_quality: float = 1.0,
        half_life_hours: float = 6.0,
    ) -> float:
        """
        Calculate degraded data quality score as observation ages.
        Quality decays gradually if data is stale, providing continuous confidence inputs.
        """
        age_hours = cls.calculate_age_seconds(observed_at) / 3600.0
        if age_hours <= 1.0:
            return base_quality
        decay = 0.5 ** (age_hours / half_life_hours)
        return round(max(0.1, min(base_quality, base_quality * decay)), 3)
