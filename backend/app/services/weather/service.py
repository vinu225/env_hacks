"""
Weather Ingestion, Caching, and Snapshot Service.

Mediates observation retrieval from external providers (Open-Meteo, Mock).
Implements a service-level TTL cache to avoid redundant external network requests
and provides the foundation for future Redis/DynamoDB persistence.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from app.core.logging import get_logger
from app.domain.environmental.freshness import FreshnessEvaluator
from app.integrations.open_meteo import OpenMeteoWeatherProvider
from app.integrations.weather_provider import WeatherProvider, get_weather_provider
from app.schemas.environmental import (
    NormalizedWeatherData,
    NormalizedWeatherObservation,
    WeatherSnapshot,
)
from app.utils.datetime_utils import utc_now

logger = get_logger(__name__)


class WeatherService:
    """
    Service mediating weather observation retrieval, in-memory caching,
    and provider fallbacks.
    """

    DEFAULT_CACHE_TTL_SECONDS: float = 600.0  # 10 minutes cache TTL

    def __init__(
        self,
        provider: Optional[WeatherProvider] = None,
        cache_ttl_seconds: Optional[float] = None,
    ):
        self._provider = provider or get_weather_provider()
        self.cache_ttl_seconds = cache_ttl_seconds or self.DEFAULT_CACHE_TTL_SECONDS
        # In-memory cache: key -> (cached_at_utc, NormalizedWeatherData)
        self._cache: Dict[str, Tuple[datetime, NormalizedWeatherData]] = {}

    @property
    def provider(self) -> WeatherProvider:
        return self._provider

    def _make_cache_key(self, latitude: float, longitude: float, location_id: Optional[str] = None) -> str:
        """Construct normalized geographic cache key rounded to ~110m precision."""
        if location_id:
            return f"loc:{location_id}"
        return f"geo:{round(latitude, 3)}:{round(longitude, 3)}"

    def clear_cache(self) -> None:
        """Clear all cached weather entries."""
        self._cache.clear()

    async def get_normalized_weather(
        self,
        latitude: float,
        longitude: float,
        location_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> NormalizedWeatherData:
        """
        Retrieve complete normalized weather bundle (current, hourly, daily).
        Checks in-memory cache first before querying provider.
        """
        cache_key = self._make_cache_key(latitude, longitude, location_id)
        now_utc = utc_now()

        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached_at, cached_data = self._cache[cache_key]
            age = (now_utc - cached_at).total_seconds()
            if age <= self.cache_ttl_seconds:
                logger.debug(f"Serving weather from cache for {cache_key} (age: {round(age, 1)}s)")
                return cached_data

        # Fetch from provider
        logger.info(f"Fetching weather from provider '{self._provider.provider_name}' for ({latitude}, {longitude})")
        if isinstance(self._provider, OpenMeteoWeatherProvider):
            data = await self._provider.get_normalized_weather_data(
                location_id=location_id, latitude=latitude, longitude=longitude
            )
        else:
            # Fallback for MockWeatherProvider or other generic providers
            snapshot = await self._provider.get_current_weather(
                location_id=location_id or "default", latitude=latitude, longitude=longitude
            )
            forecast = await self._provider.get_forecast(
                location_id=location_id or "default", latitude=latitude, longitude=longitude, horizon_hours=48
            )
            data = NormalizedWeatherData(
                current=NormalizedWeatherObservation(
                    source=self._provider.provider_name,
                    location_id=location_id,
                    coordinates=snapshot.metadata.get("coordinates", {"latitude": latitude, "longitude": longitude}),
                    observed_at=snapshot.observed_at,
                    ingested_at=now_utc,
                    temperature_c=snapshot.temperature_c,
                    apparent_temperature_c=snapshot.apparent_temperature_c or snapshot.temperature_c,
                    relative_humidity_pct=snapshot.relative_humidity_pct,
                    precipitation_mm=snapshot.precipitation_mm,
                    wind_speed_kmh=snapshot.wind_speed_kmh,
                    freshness_seconds=snapshot.freshness_seconds,
                    data_quality_score=snapshot.data_quality_score,
                    metadata=snapshot.metadata,
                ),
                hourly_forecast=[],
                daily_forecast=[],
            )

        # Store in cache
        self._cache[cache_key] = (now_utc, data)
        return data

    async def get_current_snapshot(
        self, location_id: str, latitude: float, longitude: float
    ) -> WeatherSnapshot:
        """Fetch current weather snapshot for a location with caching."""
        weather_data = await self.get_normalized_weather(latitude=latitude, longitude=longitude, location_id=location_id)
        return weather_data.current.to_weather_snapshot(location_id=location_id)

    async def get_forecast_snapshots(
        self, location_id: str, latitude: float, longitude: float, horizon_hours: int = 48
    ) -> List[WeatherSnapshot]:
        return await self._provider.get_forecast(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            horizon_hours=horizon_hours,
        )

    async def get_recent_history(
        self, location_id: str, latitude: float, longitude: float, days: int = 7
    ) -> List[WeatherSnapshot]:
        return await self._provider.get_historical_observations(
            location_id=location_id, latitude=latitude, longitude=longitude, days_back=days
        )
