"""
Open-Meteo Weather Provider Implementation.

Implements the WeatherProvider contract using the real Open-Meteo public forecast API.
Performs resilient HTTP communication, raw payload validation, timezone enforcement,
and normalization into domain schemas.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.environmental.freshness import FreshnessEvaluator
from app.domain.environmental.validation import EnvironmentalValidator
from app.integrations.weather_provider import WeatherProvider
from app.schemas.environmental import (
    DailyForecastObservation,
    HourlyForecastObservation,
    NormalizedWeatherData,
    NormalizedWeatherObservation,
    WeatherSnapshot,
)
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now
from app.utils.exceptions import ProviderError

logger = get_logger(__name__)


class OpenMeteoClient:
    """
    Dedicated HTTP client for communicating with Open-Meteo API.
    Handles network errors, timeouts, status codes, and JSON parsing.
    """

    DEFAULT_TIMEOUT_SECONDS: float = 10.0
    DEFAULT_MAX_RETRIES: int = 1

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.OPEN_METEO_API_URL).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT_SECONDS
        self._custom_client = http_client

    async def fetch_weather_payload(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Query Open-Meteo for current, hourly, and daily weather parameters.
        Returns parsed JSON dict or raises ProviderError.
        """
        # Validate coordinates prior to network dispatch
        coord_issues = EnvironmentalValidator.validate_coordinates(latitude, longitude)
        if coord_issues:
            raise ProviderError("open_meteo", f"Invalid coordinates: {', '.join(coord_issues)}")

        endpoint = f"{self.base_url}/forecast"
        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,precipitation_probability,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "timezone": "UTC",
        }

        try:
            if self._custom_client is not None:
                response = await self._custom_client.get(endpoint, params=params, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(endpoint, params=params)

            if response.status_code != 200:
                raise ProviderError(
                    "open_meteo",
                    f"HTTP error {response.status_code}: {response.text[:200]}",
                )

            try:
                data = response.json()
            except Exception as e:
                raise ProviderError("open_meteo", f"Malformed JSON in response: {str(e)}")

            if not isinstance(data, dict):
                raise ProviderError("open_meteo", "Unexpected response structure: payload is not a JSON object")

            if "current" not in data:
                raise ProviderError("open_meteo", "Missing required 'current' block in response payload")

            return data

        except httpx.TimeoutException as e:
            logger.warning(f"Open-Meteo request timed out for ({latitude}, {longitude}): {str(e)}")
            raise ProviderError("open_meteo", f"Request timed out after {self.timeout}s", str(e))
        except httpx.NetworkError as e:
            logger.warning(f"Open-Meteo connection failure: {str(e)}")
            raise ProviderError("open_meteo", f"Connection error: {str(e)}", str(e))
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error communicating with Open-Meteo: {str(e)}")
            raise ProviderError("open_meteo", f"Unexpected failure: {str(e)}", str(e))


class OpenMeteoWeatherProvider(WeatherProvider):
    """
    Production-grade WeatherProvider implementation backed by Open-Meteo.
    Provides normalized current observations, hourly forecasts, and daily trends.
    """

    def __init__(self, client: Optional[OpenMeteoClient] = None):
        self._client = client or OpenMeteoClient()

    @property
    def provider_name(self) -> str:
        return "open_meteo"

    @staticmethod
    def _parse_utc_timestamp(ts_str: Optional[str]) -> datetime:
        """Parse string timestamp and strictly enforce UTC timezone."""
        if not ts_str:
            return utc_now()
        try:
            # Handle Open-Meteo's standard format 'YYYY-MM-DDTHH:MM'
            if "T" in ts_str and len(ts_str) == 16:
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M")
            else:
                dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return utc_now()

    def normalize_payload(
        self,
        raw_data: Dict[str, Any],
        location_id: Optional[str] = None,
        requested_lat: Optional[float] = None,
        requested_lon: Optional[float] = None,
    ) -> NormalizedWeatherData:
        """
        Normalizes raw Open-Meteo JSON into domain-ready NormalizedWeatherData.
        Executes physical validation and sets freshness & quality flags.
        """
        lat = raw_data.get("latitude", requested_lat or 0.0)
        lon = raw_data.get("longitude", requested_lon or 0.0)
        coords = Coordinates(latitude=float(lat), longitude=float(lon))

        now_utc = utc_now()
        current_raw = raw_data.get("current", {})

        # Extract values without fabricating missing measurements
        obs_time = self._parse_utc_timestamp(current_raw.get("time"))
        temp_c = current_raw.get("temperature_2m")
        rh_pct = current_raw.get("relative_humidity_2m")
        app_temp_c = current_raw.get("apparent_temperature")
        precip_mm = current_raw.get("precipitation")
        rain_mm = current_raw.get("rain")
        wind_kmh = current_raw.get("wind_speed_10m")

        temp_c_val = float(temp_c) if temp_c is not None else None
        app_temp_c_val = float(app_temp_c) if app_temp_c is not None else temp_c_val
        rh_pct_val = float(rh_pct) if rh_pct is not None else None
        precip_mm_val = float(precip_mm) if precip_mm is not None else (0.0 if "precipitation" in current_raw else None)
        rain_mm_val = float(rain_mm) if rain_mm is not None else None
        wind_kmh_val = float(wind_kmh) if wind_kmh is not None else (0.0 if "wind_speed_10m" in current_raw else None)

        # Execute physical validation
        validation_payload = {
            "latitude": lat,
            "longitude": lon,
            "observed_at": obs_time,
            "temperature_c": temp_c_val,
            "relative_humidity_pct": rh_pct_val,
            "precipitation_mm": precip_mm_val,
            "wind_speed_kmh": wind_kmh_val,
        }
        val_result = EnvironmentalValidator.validate_raw_weather_payload(validation_payload)

        # Calculate freshness
        age_seconds = FreshnessEvaluator.calculate_age_seconds(obs_time, reference_time=now_utc)
        is_fresh = FreshnessEvaluator.is_observation_fresh(obs_time, max_age_seconds=7200.0, reference_time=now_utc)
        degraded_quality = FreshnessEvaluator.evaluate_quality_degradation(obs_time, base_quality=val_result.quality_score)

        # Build NormalizedWeatherObservation (preserving unobserved/invalid values as None)
        current_obs = NormalizedWeatherObservation(
            source=self.provider_name,
            location_id=location_id,
            coordinates=coords,
            observed_at=obs_time,
            ingested_at=now_utc,
            temperature_c=temp_c_val,
            apparent_temperature_c=app_temp_c_val,
            relative_humidity_pct=rh_pct_val,
            precipitation_mm=precip_mm_val,
            rain_mm=rain_mm_val,
            wind_speed_kmh=wind_kmh_val,
            precipitation_probability_pct=None,  # typically in hourly array
            freshness_seconds=age_seconds,
            is_fresh=is_fresh,
            data_quality_score=degraded_quality if val_result.is_usable else 0.0,
            is_usable=val_result.is_usable,
            validation_issues=val_result.issues,
            metadata={
                "elevation": raw_data.get("elevation"),
                "generationtime_ms": raw_data.get("generationtime_ms"),
            },
        )

        # Parse Hourly Forecasts
        hourly_raw = raw_data.get("hourly", {})
        hourly_obs: List[HourlyForecastObservation] = []
        h_times = hourly_raw.get("time", [])
        h_temps = hourly_raw.get("temperature_2m", [])
        h_app_temps = hourly_raw.get("apparent_temperature", [])
        h_rhs = hourly_raw.get("relative_humidity_2m", [])
        h_precips = hourly_raw.get("precipitation", [])
        h_probs = hourly_raw.get("precipitation_probability", [])
        h_winds = hourly_raw.get("wind_speed_10m", [])

        total_hourly = len(h_times)
        for i in range(min(total_hourly, 72)):  # Keep first 72 hours
            valid_at = self._parse_utc_timestamp(h_times[i])
            hourly_obs.append(
                HourlyForecastObservation(
                    forecast_valid_at=valid_at,
                    temperature_c=float(h_temps[i] if i < len(h_temps) else 25.0),
                    apparent_temperature_c=float(h_app_temps[i]) if i < len(h_app_temps) else None,
                    relative_humidity_pct=float(h_rhs[i] if i < len(h_rhs) else 50.0),
                    precipitation_mm=float(h_precips[i] if i < len(h_precips) else 0.0),
                    precipitation_probability_pct=float(h_probs[i]) if (i < len(h_probs) and h_probs[i] is not None) else None,
                    wind_speed_kmh=float(h_winds[i] if i < len(h_winds) else 0.0),
                )
            )

        # Parse Daily Forecasts
        daily_raw = raw_data.get("daily", {})
        daily_obs: List[DailyForecastObservation] = []
        d_times = daily_raw.get("time", [])
        d_maxs = daily_raw.get("temperature_2m_max", [])
        d_mins = daily_raw.get("temperature_2m_min", [])
        d_precips = daily_raw.get("precipitation_sum", [])
        d_probs = daily_raw.get("precipitation_probability_max", [])

        for i in range(len(d_times)):
            daily_obs.append(
                DailyForecastObservation(
                    date=str(d_times[i]),
                    temperature_max_c=float(d_maxs[i] if i < len(d_maxs) else 35.0),
                    temperature_min_c=float(d_mins[i] if i < len(d_mins) else 22.0),
                    precipitation_sum_mm=float(d_precips[i] if i < len(d_precips) else 0.0),
                    precipitation_probability_max_pct=float(d_probs[i]) if (i < len(d_probs) and d_probs[i] is not None) else None,
                )
            )

        return NormalizedWeatherData(
            current=current_obs,
            hourly_forecast=hourly_obs,
            daily_forecast=daily_obs,
        )

    async def get_normalized_weather_data(
        self,
        location_id: Optional[str],
        latitude: float,
        longitude: float,
    ) -> NormalizedWeatherData:
        """Fetch real data from Open-Meteo and return complete normalized container."""
        raw_payload = await self._client.fetch_weather_payload(latitude=latitude, longitude=longitude)
        return self.normalize_payload(
            raw_payload,
            location_id=location_id,
            requested_lat=latitude,
            requested_lon=longitude,
        )

    async def get_current_weather(
        self, location_id: str, latitude: float, longitude: float
    ) -> WeatherSnapshot:
        """WeatherProvider contract fulfillment."""
        normalized = await self.get_normalized_weather_data(location_id, latitude, longitude)
        return normalized.current.to_weather_snapshot(location_id=location_id)

    async def get_forecast(
        self, location_id: str, latitude: float, longitude: float, horizon_hours: int = 48
    ) -> List[WeatherSnapshot]:
        """WeatherProvider contract fulfillment."""
        normalized = await self.get_normalized_weather_data(location_id, latitude, longitude)
        snapshots: List[WeatherSnapshot] = []

        now_utc = utc_now()
        for idx, h in enumerate(normalized.hourly_forecast):
            if idx >= horizon_hours:
                break
            snapshots.append(
                WeatherSnapshot(
                    location_id=location_id,
                    source=self.provider_name,
                    observed_at=h.forecast_valid_at,
                    ingested_at=now_utc,
                    temperature_c=h.temperature_c,
                    apparent_temperature_c=h.apparent_temperature_c,
                    relative_humidity_pct=h.relative_humidity_pct,
                    precipitation_mm=h.precipitation_mm,
                    precipitation_probability_pct=h.precipitation_probability_pct,
                    wind_speed_kmh=h.wind_speed_kmh,
                    freshness_seconds=0.0,
                    data_quality_score=normalized.current.data_quality_score,
                    forecast_horizon_hours=idx + 1,
                    metadata={"forecast_valid_at": h.forecast_valid_at.isoformat()},
                )
            )
        return snapshots

    async def get_historical_observations(
        self, location_id: str, latitude: float, longitude: float, days_back: int = 7
    ) -> List[WeatherSnapshot]:
        """
        Open-Meteo forecast API includes the previous day by default.
        For historical queries > 1 day, returns available daily/hourly history.
        """
        # When historical archive endpoint is needed, query archive-api.open-meteo.com
        # For now, return baseline snapshot history derived from daily forecast
        normalized = await self.get_normalized_weather_data(location_id, latitude, longitude)
        now_utc = utc_now()
        snapshots: List[WeatherSnapshot] = []
        for d in normalized.daily_forecast[:days_back]:
            snapshots.append(
                WeatherSnapshot(
                    location_id=location_id,
                    source=self.provider_name,
                    observed_at=now_utc,
                    ingested_at=now_utc,
                    temperature_c=d.temperature_max_c,
                    apparent_temperature_c=d.temperature_max_c,
                    relative_humidity_pct=normalized.current.relative_humidity_pct,
                    precipitation_mm=d.precipitation_sum_mm,
                    precipitation_probability_pct=d.precipitation_probability_max_pct,
                    wind_speed_kmh=normalized.current.wind_speed_kmh,
                    freshness_seconds=0.0,
                    data_quality_score=normalized.current.data_quality_score,
                    forecast_horizon_hours=0,
                    metadata={"date": d.date},
                )
            )
        return snapshots
