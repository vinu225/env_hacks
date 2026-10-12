"""
Pytest Fixtures for Backend Tests.
"""

from typing import Any, Dict, Generator
import os
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["STORAGE_BACKEND"] = "memory"
os.environ["WEATHER_PROVIDER_TYPE"] = "mock"

from app.core.config import Settings, get_settings
from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """TestClient fixture with application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing clean settings instance."""
    return get_settings()


@pytest.fixture
def sample_open_meteo_payload() -> Dict[str, Any]:
    """Realistic sample Open-Meteo response structure for testing."""
    return {
        "latitude": 28.625,
        "longitude": 77.25,
        "generationtime_ms": 0.045,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 216.0,
        "current_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "apparent_temperature": "°C",
            "precipitation": "mm",
            "rain": "mm",
            "wind_speed_10m": "km/h",
        },
        "current": {
            "time": "2026-05-15T14:00",
            "interval": 900,
            "temperature_2m": 38.6,
            "relative_humidity_2m": 55.0,
            "apparent_temperature": 44.2,
            "precipitation": 0.0,
            "rain": 0.0,
            "wind_speed_10m": 12.8,
        },
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "precipitation": "mm",
        },
        "hourly": {
            "time": ["2026-05-15T14:00", "2026-05-15T15:00", "2026-05-15T16:00"],
            "temperature_2m": [38.6, 39.1, 37.8],
            "apparent_temperature": [44.2, 45.0, 42.5],
            "relative_humidity_2m": [55.0, 52.0, 58.0],
            "precipitation": [0.0, 0.0, 1.5],
            "precipitation_probability": [10, 15, 60],
            "wind_speed_10m": [12.8, 14.0, 15.2],
        },
        "daily": {
            "time": ["2026-05-15", "2026-05-16"],
            "temperature_2m_max": [42.0, 41.5],
            "temperature_2m_min": [28.0, 27.2],
            "precipitation_sum": [1.5, 0.0],
            "precipitation_probability_max": [60, 20],
        },
    }
