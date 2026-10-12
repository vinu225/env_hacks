"""
Unit Tests for Environmental Feature Vector Builder and Schema Stability.
"""

from app.features.builder import FeatureVectorBuilder
from app.integrations.open_meteo import OpenMeteoWeatherProvider
from app.schemas.features import DATA_VERSION, FEATURE_VERSION


def test_feature_vector_schema_and_versioning(sample_open_meteo_payload):
    """Verify feature vector contains stable versioning, units, and availability flags."""
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(sample_open_meteo_payload, location_id="loc_delhi_central")

    vector = FeatureVectorBuilder.build_from_normalized_data(normalized)

    assert vector.data_version == DATA_VERSION
    assert vector.feature_version == FEATURE_VERSION
    assert vector.location_id == "loc_delhi_central"

    # Core weather features
    assert "temperature_c" in vector.features
    assert vector.features["temperature_c"].value == 38.6
    assert vector.features["temperature_c"].unit == "°C"
    assert vector.features["temperature_c"].is_available is True

    assert "heat_index_c" in vector.features
    assert vector.features["heat_index_c"].value is not None
    assert vector.features["heat_index_c"].value > 40.0

    assert "wet_bulb_temperature_c" in vector.features
    assert vector.features["wet_bulb_temperature_c"].value is not None

    # Heat indicators
    assert "is_high_temperature" in vector.features
    assert "is_humidity_stress" in vector.features
    assert "consecutive_hot_days" in vector.features

    # Water indicators & honest availability
    assert "rainfall_accumulation_24h_mm" in vector.features
    assert vector.features["rainfall_accumulation_24h_mm"].is_available is True

    # Unavailable features must NOT be fabricated
    assert "groundwater_depth_m" in vector.features
    assert vector.features["groundwater_depth_m"].value is None
    assert vector.features["groundwater_depth_m"].is_available is False

    assert "supply_outage_hours" in vector.features
    assert vector.features["supply_outage_hours"].value is None
    assert vector.features["supply_outage_hours"].is_available is False

    assert "rainfall_accumulation_30d_mm" in vector.features
    assert vector.features["rainfall_accumulation_30d_mm"].is_available is False


def test_feature_vector_to_numeric_dict(sample_open_meteo_payload):
    """Verify flat numeric dictionary conversion for ML training & inference."""
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(sample_open_meteo_payload, location_id="loc_delhi_central")
    vector = FeatureVectorBuilder.build_from_normalized_data(normalized)

    num_dict = vector.to_numeric_dict()
    assert isinstance(num_dict, dict)
    assert isinstance(num_dict["temperature_c"], float)
    assert isinstance(num_dict["heat_index_c"], float)
    assert num_dict["groundwater_depth_m"] is None


def test_complete_feature_list_presence(sample_open_meteo_payload):
    """Verify all minimum required weather, heat, water, and rainfall features are present in the schema."""
    provider = OpenMeteoWeatherProvider()
    normalized = provider.normalize_payload(sample_open_meteo_payload, location_id="loc_delhi")
    vector = FeatureVectorBuilder.build_from_normalized_data(normalized)

    # Weather Features
    weather_keys = [
        "temperature_c", "relative_humidity_pct", "apparent_temperature_c",
        "precipitation_mm", "precipitation_probability_pct", "wind_speed_kmh",
        "dew_point_c", "heat_index_c", "wet_bulb_temperature_c"
    ]
    for key in weather_keys:
        assert key in vector.features, f"Missing weather feature: {key}"

    # Heat Features
    heat_keys = [
        "current_temperature_c", "high_temperature_indicator", "is_high_temperature",
        "humidity_stress_indicator", "is_humidity_stress", "consecutive_hot_days",
        "temperature_anomaly_c"
    ]
    for key in heat_keys:
        assert key in vector.features, f"Missing heat feature: {key}"

    # Water Features
    water_keys = [
        "recent_rainfall_1h_mm", "rainfall_accumulation_7d_mm", "rainfall_accumulation_30d_mm",
        "rainfall_deficit_pct", "consecutive_dry_days", "soil_moisture_m3_m3",
        "groundwater_depth_m", "supply_outage_hours"
    ]
    for key in water_keys:
        assert key in vector.features, f"Missing water feature: {key}"

    # Rainfall Features
    rainfall_keys = [
        "hourly_precipitation_mm", "rainfall_accumulation_24h_mm",
        "forecast_precipitation_6h_mm", "forecast_precipitation_mm",
        "heavy_rain_indicator"
    ]
    for key in rainfall_keys:
        assert key in vector.features, f"Missing rainfall feature: {key}"


def test_feature_vector_with_missing_observation_fields():
    """Verify building feature vector when input observation has missing/None measurements."""
    from app.schemas.environmental import NormalizedWeatherData, NormalizedWeatherObservation
    from app.schemas.location import Coordinates
    from app.utils.datetime_utils import utc_now

    incomplete_obs = NormalizedWeatherObservation(
        source="open_meteo",
        location_id="loc_incomplete",
        coordinates=Coordinates(latitude=28.6, longitude=77.2),
        observed_at=utc_now(),
        ingested_at=utc_now(),
        temperature_c=None,  # missing
        apparent_temperature_c=None,
        relative_humidity_pct=None,  # missing
        precipitation_mm=None,
        wind_speed_kmh=None,
        is_usable=False,
        data_quality_score=0.0,
        validation_issues=["Missing essential parameter: temperature_c"],
    )
    weather_data = NormalizedWeatherData(current=incomplete_obs, hourly_forecast=[], daily_forecast=[])

    vector = FeatureVectorBuilder.build_from_normalized_data(weather_data)

    # Temperature and derived heat features must be unavailable and None, NOT fabricated
    assert vector.features["temperature_c"].value is None
    assert vector.features["temperature_c"].is_available is False
    assert vector.features["heat_index_c"].value is None
    assert vector.features["heat_index_c"].is_available is False
    assert vector.features["dew_point_c"].value is None
    assert vector.features["dew_point_c"].is_available is False
    assert vector.features["wet_bulb_temperature_c"].value is None
    assert vector.features["wet_bulb_temperature_c"].is_available is False

