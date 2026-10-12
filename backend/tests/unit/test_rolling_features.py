"""
Unit Tests for Time-Series and Rolling Window Environmental Calculations.
"""

from datetime import datetime, timedelta, timezone
from app.features.rolling import (
    calculate_consecutive_dry_days,
    calculate_consecutive_hot_days,
    calculate_rainfall_deficit_pct,
    calculate_rolling_rainfall,
    calculate_temperature_anomaly,
)
from app.schemas.environmental import HourlyForecastObservation


def test_calculate_rolling_rainfall_numeric():
    """Verify rolling rainfall sum over numeric arrays."""
    hourly_rain = [0.0, 2.5, 5.0, 10.0, 2.5]
    # Sum of last 3 hours = 5.0 + 10.0 + 2.5 = 17.5
    assert calculate_rolling_rainfall(hourly_rain, window_hours=3) == 17.5
    # Sum of all hours
    assert calculate_rolling_rainfall(hourly_rain, window_hours=10) == 20.0


def test_calculate_rolling_rainfall_timestamped():
    """Verify rolling rainfall calculation over timestamped forecast objects."""
    base_time = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    observations = [
        HourlyForecastObservation(
            forecast_valid_at=base_time - timedelta(hours=3),
            temperature_c=28.0,
            relative_humidity_pct=75.0,
            precipitation_mm=5.0,
            wind_speed_kmh=10.0,
        ),
        HourlyForecastObservation(
            forecast_valid_at=base_time - timedelta(hours=2),
            temperature_c=28.0,
            relative_humidity_pct=75.0,
            precipitation_mm=15.0,
            wind_speed_kmh=10.0,
        ),
        HourlyForecastObservation(
            forecast_valid_at=base_time - timedelta(hours=1),
            temperature_c=28.0,
            relative_humidity_pct=75.0,
            precipitation_mm=20.0,
            wind_speed_kmh=10.0,
        ),
    ]

    # Last 2 hours should sum 15.0 + 20.0 = 35.0 mm
    total_2h = calculate_rolling_rainfall(observations, window_hours=2, reference_time=base_time)
    assert total_2h == 35.0

    # Last 4 hours should include all 3 items = 40.0 mm
    total_4h = calculate_rolling_rainfall(observations, window_hours=4, reference_time=base_time)
    assert total_4h == 40.0


def test_calculate_consecutive_dry_days():
    """Verify consecutive dry day counting (< 2.5mm per day)."""
    # Days from past to present: [rainy, dry, dry, dry] -> 3 consecutive dry days
    series = [15.0, 1.0, 0.0, 0.4]
    assert calculate_consecutive_dry_days(series, rain_threshold_mm=2.5) == 3

    # If today was rainy: [dry, dry, rainy] -> 0 consecutive dry days
    series_rainy_today = [0.0, 0.0, 25.0]
    assert calculate_consecutive_dry_days(series_rainy_today, rain_threshold_mm=2.5) == 0


def test_calculate_consecutive_hot_days():
    """Verify consecutive hot days counting (>= 40.0°C)."""
    # Days past to present: [38.0, 41.0, 42.5, 43.0] -> 3 hot days
    temps = [38.0, 41.0, 42.5, 43.0]
    assert calculate_consecutive_hot_days(temps, hot_threshold_c=40.0) == 3


def test_temperature_and_rainfall_anomalies():
    """Verify departures from normal baseline with honest None handling."""
    # When baseline exists
    assert calculate_temperature_anomaly(current_temp_c=42.5, climatological_baseline_c=38.0) == 4.5
    # When baseline is unavailable, do NOT fabricate: returns None
    assert calculate_temperature_anomaly(current_temp_c=42.5, climatological_baseline_c=None) is None

    # Rainfall deficit %
    # Actual 50mm vs Normal 100mm = -50.0% deficit
    assert calculate_rainfall_deficit_pct(actual_rainfall_mm=50.0, normal_rainfall_mm=100.0) == -50.0
    # Unavailable normal returns None
    assert calculate_rainfall_deficit_pct(actual_rainfall_mm=50.0, normal_rainfall_mm=None) is None
