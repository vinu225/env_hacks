"""
Weather Provider Abstraction.

Shields domain logic and risk engines from external API volatility (Open-Meteo, OpenWeather, IMD).
All weather consumption depends on this interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from app.core.config import get_settings
from app.schemas.environmental import WeatherSnapshot
from app.utils.datetime_utils import utc_now


class WeatherProvider(ABC):
    """Abstract interface for environmental and meteorological observation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the weather provider."""
        pass

    @abstractmethod
    async def get_current_weather(
        self, location_id: str, latitude: float, longitude: float
    ) -> WeatherSnapshot:
        """Retrieve current weather observation for coordinates."""
        pass

    @abstractmethod
    async def get_forecast(
        self, location_id: str, latitude: float, longitude: float, horizon_hours: int = 48
    ) -> List[WeatherSnapshot]:
        """Retrieve weather forecast snapshots up to specified horizon."""
        pass

    @abstractmethod
    async def get_historical_observations(
        self, location_id: str, latitude: float, longitude: float, days_back: int = 7
    ) -> List[WeatherSnapshot]:
        """Retrieve past observations for computing rolling metrics and dry spell durations."""
        pass


class MockWeatherProvider(WeatherProvider):
    """
    Mock Weather Provider for local development, CI/CD, and offline testing.
    Emits consistent observations with full traceability.
    """

    @property
    def provider_name(self) -> str:
        return "mock_weather_provider"

    async def get_current_weather(
        self, location_id: str, latitude: float, longitude: float
    ) -> WeatherSnapshot:
        now = utc_now()
        # Generates realistic moderate conditions for local testing
        return WeatherSnapshot(
            location_id=location_id,
            source=self.provider_name,
            observed_at=now,
            ingested_at=now,
            temperature_c=34.5,
            apparent_temperature_c=38.2,
            relative_humidity_pct=62.0,
            precipitation_mm=0.0,
            precipitation_probability_pct=15.0,
            wind_speed_kmh=12.4,
            wind_direction_deg=210.0,
            surface_pressure_hpa=1008.2,
            soil_moisture_m3_m3=0.22,
            freshness_seconds=0.0,
            data_quality_score=0.98,
            forecast_horizon_hours=0,
            metadata={"simulated": True, "station": "mock_aws_station_01"},
        )

    async def get_forecast(
        self, location_id: str, latitude: float, longitude: float, horizon_hours: int = 48
    ) -> List[WeatherSnapshot]:
        now = utc_now()
        snapshots: List[WeatherSnapshot] = []
        for hour in range(1, min(horizon_hours + 1, 49), 3):
            obs_time = now + timedelta(hours=hour)
            # Simulated diurnal cycle
            temp = 32.0 + 5.0 * ((hour % 24) / 12.0)
            snapshots.append(
                WeatherSnapshot(
                    location_id=location_id,
                    source=self.provider_name,
                    observed_at=obs_time,
                    ingested_at=now,
                    temperature_c=round(temp, 1),
                    apparent_temperature_c=round(temp + 3.0, 1),
                    relative_humidity_pct=58.0,
                    precipitation_mm=0.0 if hour < 24 else 12.5,
                    precipitation_probability_pct=20.0 if hour < 24 else 75.0,
                    wind_speed_kmh=14.0,
                    soil_moisture_m3_m3=0.21,
                    freshness_seconds=0.0,
                    data_quality_score=0.95,
                    forecast_horizon_hours=hour,
                    metadata={"simulated": True},
                )
            )
        return snapshots

    async def get_historical_observations(
        self, location_id: str, latitude: float, longitude: float, days_back: int = 7
    ) -> List[WeatherSnapshot]:
        now = utc_now()
        snapshots: List[WeatherSnapshot] = []
        for day in range(days_back, 0, -1):
            obs_time = now - timedelta(days=day)
            snapshots.append(
                WeatherSnapshot(
                    location_id=location_id,
                    source=self.provider_name,
                    observed_at=obs_time,
                    ingested_at=obs_time + timedelta(seconds=120),
                    temperature_c=36.0,
                    apparent_temperature_c=40.0,
                    relative_humidity_pct=60.0,
                    precipitation_mm=0.0,
                    precipitation_probability_pct=0.0,
                    wind_speed_kmh=10.0,
                    soil_moisture_m3_m3=0.20,
                    freshness_seconds=120.0,
                    data_quality_score=1.0,
                    forecast_horizon_hours=0,
                    metadata={"simulated": True, "archive": True},
                )
            )
        return snapshots


def get_weather_provider() -> WeatherProvider:
    """Factory creating configured WeatherProvider based on environment settings."""
    settings = get_settings()
    if settings.WEATHER_PROVIDER_TYPE == "open_meteo":
        from app.integrations.open_meteo import OpenMeteoWeatherProvider
        return OpenMeteoWeatherProvider()
    return MockWeatherProvider()
