"""
Integration Tests for Environmental Weather and Feature API Endpoints.
"""

from fastapi.testclient import TestClient


def test_get_normalized_weather_endpoint(client: TestClient):
    """Verify GET /api/v1/weather returns normalized meteorological observation."""
    response = client.get("/api/v1/weather?latitude=28.6315&longitude=77.2167")
    assert response.status_code == 200
    data = response.json()

    assert data["source"] in ["mock_weather_provider", "open_meteo"]
    assert "temperature_c" in data
    assert "apparent_temperature_c" in data
    assert "relative_humidity_pct" in data
    assert "precipitation_mm" in data
    assert "wind_speed_kmh" in data
    assert "observed_at" in data
    assert "ingested_at" in data
    assert "freshness_seconds" in data
    assert "is_fresh" in data
    assert "data_quality_score" in data
    assert "units" in data
    assert data["units"]["temperature"] == "°C"


def test_get_environmental_features_endpoint(client: TestClient):
    """Verify GET /api/v1/weather/features returns unified ML-ready feature vector."""
    response = client.get("/api/v1/weather/features?latitude=28.6315&longitude=77.2167&location_id=loc_delhi_central")
    assert response.status_code == 200
    data = response.json()

    assert data["data_version"] == "v1.0"
    assert data["feature_version"] == "weather-features-v1"
    assert data["location_id"] == "loc_delhi_central"
    assert "features" in data

    features = data["features"]
    assert "temperature_c" in features
    assert "apparent_temperature_c" in features
    assert "dew_point_c" in features
    assert "heat_index_c" in features
    assert "wet_bulb_temperature_c" in features
    assert "is_high_temperature" in features
    assert "is_humidity_stress" in features
    assert "rainfall_accumulation_24h_mm" in features

    # Unobserved features are explicitly unavailable
    assert "groundwater_depth_m" in features
    assert features["groundwater_depth_m"]["is_available"] is False
    assert features["groundwater_depth_m"]["value"] is None


def test_weather_endpoint_coordinate_validation(client: TestClient):
    """Verify 422 Unprocessable Entity when coordinates violate physical bounds."""
    # Latitude > 90
    response = client.get("/api/v1/weather?latitude=999.0&longitude=77.2167")
    assert response.status_code == 422

    # Longitude < -180
    response_lon = client.get("/api/v1/weather?latitude=28.6&longitude=-250.0")
    assert response_lon.status_code == 422
