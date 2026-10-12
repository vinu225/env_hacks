"""
Unit Tests for Environmental Validation and Freshness Evaluator.
"""

from datetime import datetime, timedelta, timezone
from app.domain.environmental.freshness import FreshnessEvaluator
from app.domain.environmental.validation import EnvironmentalValidator
from app.utils.datetime_utils import utc_now


def test_coordinate_validation():
    """Verify coordinate bounds [-90, 90] and [-180, 180]."""
    assert len(EnvironmentalValidator.validate_coordinates(28.6, 77.2)) == 0
    assert len(EnvironmentalValidator.validate_coordinates(-90.0, 180.0)) == 0

    # Invalid latitude
    issues_lat = EnvironmentalValidator.validate_coordinates(95.0, 77.2)
    assert len(issues_lat) == 1
    assert "Latitude 95.0 is outside valid range" in issues_lat[0]

    # Invalid longitude
    issues_lon = EnvironmentalValidator.validate_coordinates(28.6, -185.0)
    assert len(issues_lon) == 1
    assert "Longitude -185.0 is outside valid range" in issues_lon[0]

    # Missing coordinates
    issues_none = EnvironmentalValidator.validate_coordinates(None, None)
    assert len(issues_none) == 2


def test_temperature_plausibility():
    """Verify physical temperature plausibility bounds [-60°C, 65°C]."""
    now = utc_now()
    # Plausible Indian summer temperature
    valid_payload = {
        "latitude": 28.6,
        "longitude": 77.2,
        "observed_at": now,
        "temperature_c": 44.5,
    }
    result = EnvironmentalValidator.validate_raw_weather_payload(valid_payload)
    assert result.is_valid
    assert result.is_usable
    assert result.quality_score == 1.0

    # Implausible extreme temperature (>65°C)
    implausible_payload = {
        "latitude": 28.6,
        "longitude": 77.2,
        "observed_at": now,
        "temperature_c": 98.0,
    }
    result_imp = EnvironmentalValidator.validate_raw_weather_payload(implausible_payload)
    assert not result_imp.is_valid
    assert not result_imp.is_usable
    assert any("physically implausible" in issue for issue in result_imp.issues)


def test_humidity_and_precipitation_bounds():
    """Verify relative humidity [0, 100] and non-negative precipitation."""
    now = utc_now()

    # Negative precipitation & out of bounds humidity
    bad_payload = {
        "latitude": 19.0,
        "longitude": 72.8,
        "observed_at": now,
        "temperature_c": 30.0,
        "relative_humidity_pct": 120.0,
        "precipitation_mm": -5.0,
        "wind_speed_kmh": -10.0,
    }
    result = EnvironmentalValidator.validate_raw_weather_payload(bad_payload)
    assert not result.is_valid
    assert any("Relative humidity 120.0% outside valid range" in i for i in result.issues)
    assert any("Precipitation cannot be negative" in i for i in result.issues)
    assert any("Wind speed cannot be negative" in i for i in result.issues)


def test_timezone_validation():
    """Verify rejection of naive datetimes for temporal auditability."""
    naive_dt = datetime(2026, 5, 15, 12, 0)  # naive, no tzinfo
    issues = EnvironmentalValidator.validate_timestamp(naive_dt)
    assert len(issues) == 1
    assert "must be timezone-aware (UTC)" in issues[0]

    aware_dt = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    assert len(EnvironmentalValidator.validate_timestamp(aware_dt)) == 0


def test_freshness_evaluation():
    """Verify age calculation and observation staleness checks."""
    now = utc_now()
    recent = now - timedelta(minutes=15)
    stale = now - timedelta(hours=3)

    assert FreshnessEvaluator.calculate_age_seconds(recent, reference_time=now) == 15 * 60.0
    assert FreshnessEvaluator.is_observation_fresh(recent, max_age_seconds=3600.0, reference_time=now) is True
    assert FreshnessEvaluator.is_observation_fresh(stale, max_age_seconds=3600.0, reference_time=now) is False


def test_quality_degradation_over_time():
    """Verify quality score gradually degrades as observation ages."""
    now = utc_now()
    obs_recent = now - timedelta(minutes=30)
    obs_12h = now - timedelta(hours=12)

    quality_fresh = FreshnessEvaluator.evaluate_quality_degradation(obs_recent, base_quality=1.0)
    quality_stale = FreshnessEvaluator.evaluate_quality_degradation(obs_12h, base_quality=1.0)

    assert quality_fresh == 1.0
    assert quality_stale < quality_fresh
    assert quality_stale >= 0.1
