"""
Unit Tests for Configuration Loading.
"""

from app.core.config import Settings


def test_settings_defaults():
    """Verify standard default configuration values."""
    settings = Settings()
    assert settings.APP_NAME == "Environmental-Intelligence-Platform"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.WEATHER_PROVIDER_TYPE in ["mock", "open_meteo", "custom"]
    assert settings.STORAGE_BACKEND in ["memory", "dynamodb"]
    assert len(settings.CORS_ORIGINS) >= 1


def test_settings_cors_parsing_from_string():
    """Verify CORS origins can parse comma-delimited strings."""
    settings = Settings(CORS_ORIGINS="http://localhost:3000,http://example.com")
    assert "http://localhost:3000" in settings.CORS_ORIGINS
    assert "http://example.com" in settings.CORS_ORIGINS
