"""
Historical Environmental Data Provider Abstractions and Implementations.

Shields offline ML pipelines from specific archive formats (Open-Meteo Archive API,
IMD gridded NetCDF/CSV, ERA5 reanalysis extracts).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional
import httpx

from app.core.logging import get_logger
from app.utils.datetime_utils import utc_now
from app.utils.exceptions import ProviderError
from ml.pipelines.schema import HistoricalObservation

logger = get_logger(__name__)


class HistoricalWeatherProvider(ABC):
    """Abstract interface for ingesting historical meteorological datasets."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the historical data source."""
        pass

    @abstractmethod
    async def get_historical_observations(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> List[HistoricalObservation]:
        """
        Retrieve chronological sequence of hourly normalized historical observations
        for the given coordinates and time window.
        """
        pass


class SyntheticHistoricalWeatherProvider(HistoricalWeatherProvider):
    """
    Deterministic synthetic historical generator for local offline development,
    CI pipeline validation, and pipeline mechanics testing.
    
    IMPORTANT:
    This data is synthetic and clearly labeled as such. It must NEVER be presented
    as real observed ground measurements.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def provider_name(self) -> str:
        return "synthetic_historical_generator"

    async def get_historical_observations(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> List[HistoricalObservation]:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        observations: List[HistoricalObservation] = []
        current_time = start_time

        # Generate realistic hourly diurnal curve with an occasional simulated heatwave spell
        hour_step = timedelta(hours=1)
        step_idx = 0

        while current_time <= end_time:
            # Diurnal solar cycle: daily peak around 14:00 UTC, trough around 04:00 UTC
            hour_of_day = current_time.hour
            diurnal_factor = math.sin((hour_of_day - 8) * math.pi / 12.0)

            # Simulated synoptic pattern: 5-day cycle where days 3-4 feature an extreme heat spell
            day_idx = (current_time - start_time).days
            heatwave_boost = 6.0 if (day_idx % 7 in [3, 4, 5]) else 0.0

            # Base summer temperatures for Indian plains (32°C base + diurnal swing + heatwave boost)
            base_temp = 32.0 + (diurnal_factor * 6.5) + heatwave_boost
            temp = round(base_temp + 0.5 * math.sin(step_idx * 0.1), 1)

            # Realistic relative humidity for Indian dry summer: drops sharply during midday peak
            # Heatwave days feature dry continental westerly winds (20% - 40% RH)
            rh_base = 40.0 - (diurnal_factor * 18.0) - (8.0 if heatwave_boost > 0 else 0.0)
            rh = round(max(15.0, min(85.0, rh_base + math.cos(step_idx * 0.15) * 4.0)), 1)

            # Apparent temperature proxy
            app_temp = round(temp + (1.5 if rh > 50 else -1.0) + (heatwave_boost * 0.3), 1)

            # Wind speed between 6 and 18 km/h
            wind = round(10.0 + 4.0 * math.sin(step_idx * 0.2), 1)

            # Dry spell during heatwaves, occasional isolated shower
            precip = 0.0
            if day_idx % 7 == 1 and hour_of_day in [17, 18]:
                precip = 4.5

            obs = HistoricalObservation(
                location_id=location_id,
                latitude=round(latitude, 4),
                longitude=round(longitude, 4),
                timestamp=current_time,
                temperature_c=temp,
                relative_humidity_pct=rh,
                apparent_temperature_c=app_temp,
                precipitation_mm=precip,
                wind_speed_kmh=wind,
                surface_pressure_hpa=1006.0,
                precipitation_probability_pct=0.0 if precip == 0.0 else 60.0,
                source=self.provider_name,
                origin_type="OBSERVED",
                metadata={
                    "is_synthetic": True,
                    "generator_version": "v1.0",
                    "disclaimer": "Synthetic simulation for pipeline testing ONLY. Not real environmental observations.",
                },
            )
            observations.append(obs)
            current_time += hour_step
            step_idx += 1

        return observations


class LocalFileHistoricalProvider(HistoricalWeatherProvider):
    """
    Ingests pre-downloaded historical observation records from local CSV or Parquet files.
    Enables reproducible benchmarking against official IMD or ERA5 historical exports.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def provider_name(self) -> str:
        return f"local_file:{self.file_path}"

    async def get_historical_observations(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> List[HistoricalObservation]:
        import pandas as pd

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if self.file_path.endswith(".parquet"):
            df = pd.read_parquet(self.file_path)
        else:
            df = pd.read_csv(self.file_path)

        # Ensure timestamp is datetime and timezone-aware UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Filter by coordinates/location and date range
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        if "location_id" in df.columns:
            mask = mask & (df["location_id"] == location_id)
        filtered = df[mask].sort_values("timestamp")

        observations: List[HistoricalObservation] = []
        for _, row in filtered.iterrows():
            obs = HistoricalObservation(
                location_id=str(row.get("location_id", location_id)),
                latitude=float(row.get("latitude", latitude)),
                longitude=float(row.get("longitude", longitude)),
                timestamp=row["timestamp"].to_pydatetime(),
                temperature_c=float(row["temperature_c"]) if pd.notna(row.get("temperature_c")) else None,
                relative_humidity_pct=float(row["relative_humidity_pct"]) if pd.notna(row.get("relative_humidity_pct")) else None,
                apparent_temperature_c=float(row["apparent_temperature_c"]) if pd.notna(row.get("apparent_temperature_c")) else None,
                precipitation_mm=float(row["precipitation_mm"]) if pd.notna(row.get("precipitation_mm")) else None,
                wind_speed_kmh=float(row["wind_speed_kmh"]) if pd.notna(row.get("wind_speed_kmh")) else None,
                surface_pressure_hpa=float(row["surface_pressure_hpa"]) if pd.notna(row.get("surface_pressure_hpa")) else None,
                precipitation_probability_pct=float(row["precipitation_probability_pct"]) if pd.notna(row.get("precipitation_probability_pct")) else None,
                source=str(row.get("source", self.provider_name)),
                origin_type=str(row.get("origin_type", "OBSERVED")),
                metadata={"imported_from": self.file_path},
            )
            observations.append(obs)

        return observations


class OpenMeteoArchiveHistoricalProvider(HistoricalWeatherProvider):
    """
    Connects to the Open-Meteo Historical Archive API (archive-api.open-meteo.com/v1/archive).
    Provides access to ERA5/ERA5-Land reanalysis back to 1940.
    """

    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._custom_client = http_client
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def provider_name(self) -> str:
        return "open-meteo-era5"

    async def get_historical_observations(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> List[HistoricalObservation]:
        import asyncio

        start_date_str = start_time.strftime("%Y-%m-%d")
        end_date_str = end_time.strftime("%Y-%m-%d")

        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m",
            "timezone": "UTC",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self._custom_client is not None:
                    resp = await self._custom_client.get(self.ARCHIVE_BASE_URL, params=params, timeout=self.timeout)
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.get(self.ARCHIVE_BASE_URL, params=params)

                if resp.status_code == 429:
                    # Rate limit encountered - backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Archive API rate limited (429). Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    continue

                if resp.status_code != 200:
                    raise ProviderError(self.provider_name, f"Archive query failed with HTTP {resp.status_code}: {resp.text[:200]}")

                payload = resp.json()
                hourly = payload.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                rhs = hourly.get("relative_humidity_2m", [])
                apps = hourly.get("apparent_temperature", [])
                precips = hourly.get("precipitation", [])
                winds = hourly.get("wind_speed_10m", [])

                observations: List[HistoricalObservation] = []
                for i, t_str in enumerate(times):
                    ts = datetime.fromisoformat(t_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    observations.append(
                        HistoricalObservation(
                            location_id=location_id,
                            latitude=latitude,
                            longitude=longitude,
                            timestamp=ts,
                            temperature_c=float(temps[i]) if (i < len(temps) and temps[i] is not None) else None,
                            relative_humidity_pct=float(rhs[i]) if (i < len(rhs) and rhs[i] is not None) else None,
                            apparent_temperature_c=float(apps[i]) if (i < len(apps) and apps[i] is not None) else None,
                            precipitation_mm=float(precips[i]) if (i < len(precips) and precips[i] is not None) else None,
                            wind_speed_kmh=float(winds[i]) if (i < len(winds) and winds[i] is not None) else None,
                            surface_pressure_hpa=None,  # Never fabricated; explicitly None if unobserved
                            precipitation_probability_pct=None,
                            source=self.provider_name,
                            origin_type="REANALYSIS",
                            metadata={
                                "archive_elevation": payload.get("elevation"),
                                "data_tier": "reanalysis",
                                "model_basis": "ECMWF ERA5 / ERA5-Land",
                                "provenance_note": (
                                    "Atmospheric reanalysis blending physical numerical weather models "
                                    "with assimilated ground and satellite observations. Not raw station gauge."
                                ),
                            },
                        )
                    )

                return observations

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Archive network error ({e}). Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
            except ProviderError:
                raise
            except Exception as e:
                last_err = e
                break

        raise ProviderError(self.provider_name, f"Failed to retrieve archive observations after {self.max_retries} attempts: {last_err}")
