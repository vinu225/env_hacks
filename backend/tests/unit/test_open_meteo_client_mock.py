"""
Unit Tests for OpenMeteoClient HTTP Error Handling (100% Offline with MockTransport).
"""

import httpx
import pytest
from app.integrations.open_meteo import OpenMeteoClient
from app.utils.exceptions import ProviderError


@pytest.mark.asyncio
async def test_client_successful_response():
    """Verify successful response parsing with mock transport."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "latitude=28.6" in str(request.url)
        return httpx.Response(
            200,
            json={
                "latitude": 28.6,
                "longitude": 77.2,
                "current": {
                    "time": "2026-05-15T12:00",
                    "temperature_2m": 35.0,
                },
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        payload = await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert payload["current"]["temperature_2m"] == 35.0


@pytest.mark.asyncio
async def test_client_http_500_error():
    """Verify HTTP status errors raise ProviderError."""
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(error_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        with pytest.raises(ProviderError) as exc_info:
            await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert "HTTP error 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_timeout_handling():
    """Verify timeout triggers ProviderError with clean message."""
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")

    transport = httpx.MockTransport(timeout_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        with pytest.raises(ProviderError) as exc_info:
            await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_network_error():
    """Verify network connection failures raise ProviderError."""
    def network_error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.NetworkError("Failed to resolve hostname")

    transport = httpx.MockTransport(network_error_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        with pytest.raises(ProviderError) as exc_info:
            await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert "Connection error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_malformed_json():
    """Verify malformed JSON raises ProviderError."""
    def malformed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{broken json: invalid")

    transport = httpx.MockTransport(malformed_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        with pytest.raises(ProviderError) as exc_info:
            await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert "Malformed JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_missing_current_block():
    """Verify missing required 'current' block raises ProviderError."""
    def missing_field_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"latitude": 28.6, "longitude": 77.2})

    transport = httpx.MockTransport(missing_field_handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = OpenMeteoClient(http_client=mock_http)
        with pytest.raises(ProviderError) as exc_info:
            await client.fetch_weather_payload(latitude=28.6, longitude=77.2)
        assert "Missing required 'current' block" in str(exc_info.value)
