"""
Unit Tests for Open-Meteo Payload Normalization.
"""

from datetime import datetime, timezone
import pytest
from app.integrations.open_meteo import OpenMeteoWeatherProvider


def test_normalize_valid_payload(sample_open_meteo_payload):
    """Verify clean normalization of valid Open-Meteo payload."""
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(
        sample_open_meteo_payload,
        location_id="loc_delhi_central",
        requested_lat=28.6315,
        requested_lon=77.2167,
    )

    curr = normalized.current
    assert curr.source == "open_meteo"
    assert curr.location_id == "loc_delhi_central"
    assert curr.temperature_c == 38.6
    assert curr.apparent_temperature_c == 44.2
    assert curr.relative_humidity_pct == 55.0
    assert curr.precipitation_mm == 0.0
    assert curr.wind_speed_kmh == 12.8
    assert curr.observed_at.tzinfo == timezone.utc
    assert curr.ingested_at.tzinfo == timezone.utc
    assert curr.is_usable
    assert len(curr.validation_issues) == 0

    # Hourly & Daily checks
    assert len(normalized.hourly_forecast) == 3
    assert normalized.hourly_forecast[0].temperature_c == 38.6
    assert normalized.hourly_forecast[2].precipitation_mm == 1.5
    assert normalized.hourly_forecast[2].forecast_valid_at.tzinfo == timezone.utc

    assert len(normalized.daily_forecast) == 2
    assert normalized.daily_forecast[0].temperature_max_c == 42.0
    assert normalized.daily_forecast[0].precipitation_sum_mm == 1.5


def test_normalize_partial_payload():
    """Verify handling when optional hourly/daily arrays are missing."""
    minimal_payload = {
        "latitude": 19.07,
        "longitude": 72.87,
        "current": {
            "time": "2026-07-20T10:00",
            "temperature_2m": 31.0,
            "relative_humidity_2m": 82.0,
            "apparent_temperature": 38.0,
            "precipitation": 12.4,
            "wind_speed_10m": 22.0,
        },
    }
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(minimal_payload, location_id="loc_mumbai")

    assert normalized.current.temperature_c == 31.0
    assert normalized.current.precipitation_mm == 12.4
    assert len(normalized.hourly_forecast) == 0
    assert len(normalized.daily_forecast) == 0
    assert normalized.current.is_usable


def test_conversion_to_weather_snapshot(sample_open_meteo_payload):
    """Verify conversion from NormalizedWeatherObservation to standard WeatherSnapshot."""
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(sample_open_meteo_payload, location_id="loc_delhi_01")

    snapshot = normalized.current.to_weather_snapshot(location_id="loc_delhi_01")
    assert snapshot.location_id == "loc_delhi_01"
    assert snapshot.temperature_c == 38.6
    assert snapshot.apparent_temperature_c == 44.2
    assert snapshot.source == "open_meteo"
    assert snapshot.observed_at.tzinfo == timezone.utc


def test_normalize_missing_temperature_no_silent_coercion():
    """Verify missing temperature is NOT silently converted to a default value like 25.0."""
    payload = {
        "latitude": 28.6,
        "longitude": 77.2,
        "current": {
            "time": "2026-05-15T12:00",
            # temperature_2m is intentionally omitted
            "relative_humidity_2m": 60.0,
            "apparent_temperature": None,
            "precipitation": 0.0,
            "wind_speed_10m": 10.0,
        },
    }
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(payload, location_id="loc_missing_temp")
    curr = normalized.current

    assert curr.temperature_c is None
    assert curr.temperature is None
    assert not curr.is_usable
    assert any("Temperature is missing" in issue for issue in curr.validation_issues)
    assert curr.data_quality_score == 0.0


def test_normalize_invalid_physical_values():
    """Verify out-of-bounds physical values are flagged in validation_issues and degrade quality."""
    payload = {
        "latitude": 28.6,
        "longitude": 77.2,
        "current": {
            "time": "2026-05-15T12:00",
            "temperature_2m": 85.0,  # Implausible surface ground temperature
            "relative_humidity_2m": 125.0,  # Impossible humidity > 100%
            "precipitation": -5.0,  # Impossible negative precipitation
            "wind_speed_10m": -12.0,  # Impossible negative wind speed
        },
    }
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(payload, location_id="loc_invalid_vals")
    curr = normalized.current

    assert not curr.is_usable
    assert curr.temperature_c == 85.0  # Preserved for auditability, not silently altered
    assert any("physically implausible" in issue for issue in curr.validation_issues)
    assert any("Relative humidity 125.0% outside valid range" in issue for issue in curr.validation_issues)
    assert any("Precipitation cannot be negative" in issue for issue in curr.validation_issues)
    assert any("Wind speed cannot be negative" in issue for issue in curr.validation_issues)


def test_observation_property_aliases_and_freshness():
    """Verify property getters for standard names and freshness checks."""
    payload = {
        "latitude": 13.08,
        "longitude": 80.27,
        "current": {
            "time": "2026-06-01T14:00",
            "temperature_2m": 33.5,
            "relative_humidity_2m": 70.0,
            "apparent_temperature": 39.2,
            "precipitation": 2.5,
            "rain": 2.5,
            "wind_speed_10m": 15.0,
        },
    }
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(payload, location_id="loc_chennai")
    obs = normalized.current

    # Properties
    assert obs.temperature == 33.5
    assert obs.humidity == 70.0
    assert obs.apparent_temperature == 39.2
    assert obs.precipitation == 2.5
    assert obs.rain == 2.5
    assert obs.wind_speed == 15.0
    assert obs.data_quality > 0.0
    assert obs.freshness >= 0.0

    # Freshness helpers
    assert isinstance(obs.age_seconds, float)
    assert isinstance(obs.is_fresh_enough(max_age_seconds=86400.0 * 3650), bool)

