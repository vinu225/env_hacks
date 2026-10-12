"""
Unit Tests for Meteorological and Feature Engineering Calculations.
"""

from app.features.weather_features import (
    calculate_dew_point,
    calculate_heat_index,
    calculate_wet_bulb_temperature,
)


def test_dew_point_calculation():
    """Verify dew point computation for standard atmospheric conditions."""
    dp = calculate_dew_point(temperature_c=30.0, relative_humidity_pct=70.0)
    # At 30°C and 70% RH, dew point is approximately ~24°C
    assert 23.0 <= dp <= 25.5


def test_heat_index_calculation():
    """Verify NOAA heat index formula produces higher apparent temperature in high humidity."""
    hi_humid = calculate_heat_index(temperature_c=35.0, relative_humidity_pct=75.0)
    hi_dry = calculate_heat_index(temperature_c=35.0, relative_humidity_pct=25.0)
    assert hi_humid > hi_dry
    assert hi_humid > 45.0  # dangerous heat index


def test_wet_bulb_temperature_calculation():
    """Verify Stull wet bulb temperature calculation."""
    wb = calculate_wet_bulb_temperature(temperature_c=35.0, relative_humidity_pct=60.0)
    # Wet-bulb should be less than dry-bulb in unsaturated air
    assert wb < 35.0
    assert wb > 25.0


def test_weather_features_with_none_inputs():
    """Verify that pure meteorological functions handle None inputs safely without raising exceptions."""
    assert calculate_dew_point(None, 60.0) is None
    assert calculate_dew_point(30.0, None) is None
    assert calculate_dew_point(None, None) is None

    assert calculate_heat_index(None, 60.0) is None
    assert calculate_heat_index(35.0, None) is None
    assert calculate_heat_index(None, None) is None

    # Low temperature below 27°C returns dry-bulb temperature directly
    assert calculate_heat_index(22.5, 80.0) == 22.5

    assert calculate_wet_bulb_temperature(None, 60.0) is None
    assert calculate_wet_bulb_temperature(35.0, None) is None
    assert calculate_wet_bulb_temperature(None, None) is None

