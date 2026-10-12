"""
Environmental Feature Vector Builder.

Constructs unified EnvironmentalFeatureVector instances for:
- Live inference API (GET /api/v1/weather/features)
- Risk model inputs
- Future offline ML feature store matrices
"""

from typing import Optional
from app.features.heat_features import (
    compute_heat_feature_vector,
    extract_heat_features,
    is_high_temperature_condition,
    is_humidity_stress_condition,
)
from app.features.rainfall_features import (
    compute_rainfall_feature_vector,
    extract_rainfall_features,
    is_heavy_rain_condition,
)
from app.features.rolling import (
    calculate_consecutive_dry_days,
    calculate_consecutive_hot_days,
    calculate_rolling_rainfall,
)
from app.features.water_features import (
    compute_water_feature_vector,
    extract_water_features,
)
from app.features.weather_features import (
    calculate_dew_point,
    calculate_heat_index,
    calculate_wet_bulb_temperature,
)
from app.schemas.environmental import NormalizedWeatherData, NormalizedWeatherObservation
from app.schemas.features import (
    DATA_VERSION,
    FEATURE_VERSION,
    EnvironmentalFeatureVector,
)
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now


class FeatureVectorBuilder:
    """Builds unified, traceable EnvironmentalFeatureVector representations."""

    @classmethod
    def build_from_normalized_data(
        cls,
        weather_data: NormalizedWeatherData,
        location_id: Optional[str] = None,
        climatological_normal_temp_c: Optional[float] = None,
        climatological_normal_rain_30d_mm: Optional[float] = None,
    ) -> EnvironmentalFeatureVector:
        curr = weather_data.current
        hourly = weather_data.hourly_forecast
        daily = weather_data.daily_forecast
        coords = curr.coordinates

        vector = EnvironmentalFeatureVector(
            location_id=location_id or curr.location_id,
            coordinates=coords,
            generated_at=utc_now(),
            data_version=DATA_VERSION,
            feature_version=FEATURE_VERSION,
            metadata={
                "source": curr.source,
                "observed_at": curr.observed_at.isoformat(),
                "data_quality_score": curr.data_quality_score,
                "is_usable": curr.is_usable,
            },
        )

        quality = curr.data_quality_score if curr.is_usable else 0.0

        # ---------------- Weather Features ----------------
        temp_avail = curr.temperature_c is not None
        vector.add_feature("temperature_c", curr.temperature_c, unit="°C", category="weather", is_available=temp_avail, data_quality=quality)
        vector.add_feature("current_temperature_c", curr.temperature_c, unit="°C", category="weather", is_available=temp_avail, data_quality=quality)

        app_temp_avail = curr.apparent_temperature_c is not None
        vector.add_feature("apparent_temperature_c", curr.apparent_temperature_c, unit="°C", category="weather", is_available=app_temp_avail, data_quality=quality)

        rh_avail = curr.relative_humidity_pct is not None
        vector.add_feature("relative_humidity_pct", curr.relative_humidity_pct, unit="%", category="weather", is_available=rh_avail, data_quality=quality)

        precip_avail = curr.precipitation_mm is not None
        vector.add_feature("precipitation_mm", curr.precipitation_mm, unit="mm", category="weather", is_available=precip_avail, data_quality=quality)

        prob_avail = curr.precipitation_probability_pct is not None
        vector.add_feature("precipitation_probability_pct", curr.precipitation_probability_pct, unit="%", category="weather", is_available=prob_avail, data_quality=quality)

        wind_avail = curr.wind_speed_kmh is not None
        vector.add_feature("wind_speed_kmh", curr.wind_speed_kmh, unit="km/h", category="weather", is_available=wind_avail, data_quality=quality)

        dew_point = calculate_dew_point(curr.temperature_c, curr.relative_humidity_pct)
        vector.add_feature("dew_point_c", dew_point, unit="°C", category="weather", is_available=(dew_point is not None), data_quality=quality)

        heat_idx = calculate_heat_index(curr.temperature_c, curr.relative_humidity_pct)
        vector.add_feature("heat_index_c", heat_idx, unit="°C", category="weather", is_available=(heat_idx is not None), data_quality=quality)

        wet_bulb = calculate_wet_bulb_temperature(curr.temperature_c, curr.relative_humidity_pct)
        vector.add_feature("wet_bulb_temperature_c", wet_bulb, unit="°C", category="weather", is_available=(wet_bulb is not None), data_quality=quality)

        # ---------------- Heat Features ----------------
        high_temp_flag = (1.0 if is_high_temperature_condition(curr.temperature_c) else 0.0) if temp_avail else None
        vector.add_feature("is_high_temperature", high_temp_flag, unit="flag", category="heat", is_available=(high_temp_flag is not None), data_quality=quality)
        vector.add_feature("high_temperature_indicator", high_temp_flag, unit="flag", category="heat", is_available=(high_temp_flag is not None), data_quality=quality)

        hum_stress_flag = (
            (1.0 if is_humidity_stress_condition(curr.temperature_c, curr.relative_humidity_pct) else 0.0)
            if (temp_avail and rh_avail)
            else None
        )
        vector.add_feature("is_humidity_stress", hum_stress_flag, unit="flag", category="heat", is_available=(hum_stress_flag is not None), data_quality=quality)
        vector.add_feature("humidity_stress_indicator", hum_stress_flag, unit="flag", category="heat", is_available=(hum_stress_flag is not None), data_quality=quality)

        hot_days = calculate_consecutive_hot_days(daily) if daily else 0
        vector.add_feature("consecutive_hot_days", float(hot_days), unit="days", category="heat", data_quality=quality)

        temp_anomaly = (curr.temperature_c - climatological_normal_temp_c) if (temp_avail and climatological_normal_temp_c is not None) else None
        vector.add_feature(
            "temperature_anomaly_c",
            round(temp_anomaly, 2) if temp_anomaly is not None else None,
            unit="°C",
            category="heat",
            is_available=(temp_anomaly is not None),
            data_quality=quality,
        )

        # ---------------- Water Features (Weather-Derived) ----------------
        recent_rain = curr.precipitation_mm
        vector.add_feature("recent_rainfall_1h_mm", recent_rain, unit="mm", category="water", is_available=precip_avail, data_quality=quality)

        # 24h, 7d, 30d rolling rainfall from hourly or daily forecast series
        rain_24h = calculate_rolling_rainfall(hourly, window_hours=24) if hourly else (curr.precipitation_mm or 0.0)
        vector.add_feature("rainfall_accumulation_24h_mm", rain_24h, unit="mm", category="water", data_quality=quality)

        # 7-day accumulation from daily forecasts or hourly
        if daily:
            rain_7d = round(sum(d.precipitation_sum_mm for d in daily[:7]), 2)
        elif hourly:
            rain_7d = calculate_rolling_rainfall(hourly, window_hours=168)
        else:
            rain_7d = curr.precipitation_mm or 0.0
        vector.add_feature("rainfall_accumulation_7d_mm", rain_7d, unit="mm", category="water", data_quality=quality)

        # 30-day accumulation is currently unavailable without historical archive
        vector.add_feature(
            "rainfall_accumulation_30d_mm",
            None,
            unit="mm",
            category="water",
            is_available=False,
            description="30-day historical archive not yet connected; marked unavailable without fabrication.",
        )

        # Rainfall deficit is unavailable without 30-day normal baseline
        vector.add_feature(
            "rainfall_deficit_pct",
            None,
            unit="%",
            category="water",
            is_available=False,
            description="Climatological rainfall deficit unavailable without 30-day normal archive.",
        )

        dry_days = calculate_consecutive_dry_days(daily) if daily else 0
        vector.add_feature("consecutive_dry_days", float(dry_days), unit="days", category="water", data_quality=quality)

        # Soil moisture is only included if actual measurement exists
        soil_moisture = curr.metadata.get("soil_moisture_m3_m3")
        vector.add_feature(
            "soil_moisture_m3_m3",
            float(soil_moisture) if soil_moisture is not None else None,
            unit="m³/m³",
            category="water",
            is_available=(soil_moisture is not None),
            description="Volumetric soil water content; explicitly None when provider does not measure it.",
        )

        # Groundwater & municipal outages are strictly marked unavailable unless ground sensors exist
        vector.add_feature(
            "groundwater_depth_m",
            None,
            unit="m",
            category="water",
            is_available=False,
            description="Groundwater telemetry requires specialized hydro-geological sensors; not fabricated.",
        )
        vector.add_feature(
            "supply_outage_hours",
            None,
            unit="hours",
            category="water",
            is_available=False,
            description="Piped municipal outage hours require utility or citizen incident report; not fabricated.",
        )

        # ---------------- Rainfall Features ----------------
        vector.add_feature("hourly_precipitation_mm", curr.precipitation_mm, unit="mm", category="rainfall", is_available=precip_avail, data_quality=quality)
        rain_6h = calculate_rolling_rainfall(hourly, window_hours=6) if hourly else 0.0
        vector.add_feature("forecast_precipitation_6h_mm", rain_6h, unit="mm", category="rainfall", data_quality=quality)
        vector.add_feature("forecast_precipitation_mm", rain_6h, unit="mm", category="rainfall", data_quality=quality)

        heavy_rain_flag = (1.0 if is_heavy_rain_condition(curr.precipitation_mm or 0.0, rain_24h) else 0.0) if precip_avail else None
        vector.add_feature("heavy_rain_indicator", heavy_rain_flag, unit="flag", category="rainfall", is_available=(heavy_rain_flag is not None), data_quality=quality)

        return vector
