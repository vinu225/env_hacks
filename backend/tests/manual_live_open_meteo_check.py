"""
Manual / Standalone Live Open-Meteo Integration Test.

Run manually to test live network round-trip against the public Open-Meteo API:
    python backend/tests/manual_live_open_meteo_check.py

Excluded from automated CI/pytest runs to keep test suite deterministic and offline.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure backend directory is in sys.path when running script directly
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.integrations.open_meteo import OpenMeteoWeatherProvider


async def run_live_check():
    print("Initiating live Open-Meteo API request for Central Delhi (28.6315, 77.2167)...")
    provider = OpenMeteoWeatherProvider()
    try:
        data = await provider.get_normalized_weather_data(
            location_id="loc_delhi_central",
            latitude=28.6315,
            longitude=77.2167,
        )
        curr = data.current
        print("\n--- Live Data Received Successfully ---")
        print(f"Source: {curr.source}")
        print(f"Observed At: {curr.observed_at.isoformat()}")
        print(f"Temperature: {curr.temperature_c} {curr.units.get('temperature')}")
        print(f"Apparent Temp: {curr.apparent_temperature_c} {curr.units.get('apparent_temperature')}")
        print(f"Relative Humidity: {curr.relative_humidity_pct}%")
        print(f"Precipitation: {curr.precipitation_mm} mm")
        print(f"Wind Speed: {curr.wind_speed_kmh} km/h")
        print(f"Freshness: {round(curr.freshness_seconds, 1)} seconds old")
        print(f"Data Quality: {curr.data_quality_score}")
        print(f"Hourly Forecast Hours Returned: {len(data.hourly_forecast)}")
        print(f"Daily Forecast Days Returned: {len(data.daily_forecast)}")
        print("\nPipeline Verification: SUCCESS")
    except Exception as e:
        print(f"\nLive request failed (Network / DNS / API outage): {e}")


if __name__ == "__main__":
    asyncio.run(run_live_check())
