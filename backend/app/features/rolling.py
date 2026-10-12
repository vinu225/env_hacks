"""
Time-Series and Rolling Window Environmental Feature Calculations.

Pure, deterministic functions capable of operating on live streaming observations,
hourly forecast arrays, or historical training matrices.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence, Union
from app.utils.datetime_utils import utc_now


def get_observation_timestamp(obs: Any) -> Optional[datetime]:
    """Extract datetime from an observation object or dict."""
    if hasattr(obs, "forecast_valid_at"):
        dt = obs.forecast_valid_at
    elif hasattr(obs, "observed_at"):
        dt = obs.observed_at
    elif isinstance(obs, dict):
        dt = obs.get("forecast_valid_at") or obs.get("observed_at") or obs.get("timestamp")
    elif isinstance(obs, (tuple, list)) and len(obs) >= 2 and isinstance(obs[0], datetime):
        dt = obs[0]
    else:
        return None

    if dt is not None and isinstance(dt, datetime) and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_observation_precipitation(obs: Any) -> float:
    """Extract precipitation value in mm from an observation object or dict."""
    if hasattr(obs, "precipitation_mm"):
        return float(obs.precipitation_mm or 0.0)
    elif hasattr(obs, "precipitation_sum_mm"):
        return float(obs.precipitation_sum_mm or 0.0)
    elif isinstance(obs, dict):
        val = obs.get("precipitation_mm", obs.get("precipitation", obs.get("precipitation_sum_mm", 0.0)))
        return float(val or 0.0)
    elif isinstance(obs, (tuple, list)) and len(obs) >= 2:
        return float(obs[1] or 0.0)
    elif isinstance(obs, (int, float)):
        return float(obs)
    return 0.0


def calculate_rolling_rainfall(
    observations: Sequence[Any],
    window_hours: int,
    reference_time: Optional[datetime] = None,
) -> float:
    """
    Calculate accumulated rainfall (mm) over a rolling window of hours ending at reference_time.
    If observations are simple numeric hourly sequences without timestamps, sums the last N elements.
    """
    if not observations:
        return 0.0

    # Case A: Sequence of raw numbers
    if all(isinstance(x, (int, float)) for x in observations):
        count = min(len(observations), window_hours)
        return round(float(sum(observations[-count:])), 2)

    # Case B: Timestamped observations
    ref = reference_time or utc_now()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    window_start = ref - timedelta(hours=window_hours)

    total_rain = 0.0
    for obs in observations:
        ts = get_observation_timestamp(obs)
        if ts is not None:
            if window_start <= ts <= ref:
                total_rain += get_observation_precipitation(obs)
        else:
            # Fallback if un-timestamped element
            total_rain += get_observation_precipitation(obs)

    return round(max(0.0, total_rain), 2)


def calculate_consecutive_dry_days(
    daily_precipitation_series: Sequence[Union[float, Any]],
    rain_threshold_mm: float = 2.5,  # IMD standard meteorological dry day threshold
) -> int:
    """
    Count consecutive days without significant rainfall (< rain_threshold_mm)
    working backward from the most recent day.
    """
    if not daily_precipitation_series:
        return 0

    consecutive = 0
    # Traverse backwards from most recent day
    for day_obs in reversed(daily_precipitation_series):
        rain = get_observation_precipitation(day_obs)
        if rain < rain_threshold_mm:
            consecutive += 1
        else:
            break
    return consecutive


def calculate_consecutive_hot_days(
    daily_max_temp_series: Sequence[Union[float, Any]],
    hot_threshold_c: float = 40.0,  # IMD standard plains heatwave criterion
) -> int:
    """
    Count consecutive days with daily maximum temperature >= hot_threshold_c
    working backward from the most recent day.
    """
    if not daily_max_temp_series:
        return 0

    consecutive = 0
    for day_obs in reversed(daily_max_temp_series):
        if hasattr(day_obs, "temperature_max_c"):
            temp = float(day_obs.temperature_max_c)
        elif isinstance(day_obs, dict):
            temp = float(day_obs.get("temperature_max_c", day_obs.get("temp_max_c", 0.0)))
        elif isinstance(day_obs, (int, float)):
            temp = float(day_obs)
        else:
            temp = 0.0

        if temp >= hot_threshold_c:
            consecutive += 1
        else:
            break
    return consecutive


def calculate_temperature_anomaly(
    current_temp_c: float,
    climatological_baseline_c: Optional[float],
) -> Optional[float]:
    """
    Compute departure from normal temperature in °C.
    Returns None if baseline is unavailable, ensuring no fabrication.
    """
    if climatological_baseline_c is None:
        return None
    return round(current_temp_c - climatological_baseline_c, 2)


def calculate_rainfall_deficit_pct(
    actual_rainfall_mm: float,
    normal_rainfall_mm: Optional[float],
) -> Optional[float]:
    """
    Compute percentage departure from normal rainfall: ((actual - normal) / normal) * 100.
    Negative indicates deficit. Returns None if normal is unavailable.
    """
    if normal_rainfall_mm is None or normal_rainfall_mm <= 0.0:
        return None
    pct = ((actual_rainfall_mm - normal_rainfall_mm) / normal_rainfall_mm) * 100.0
    return round(pct, 1)
