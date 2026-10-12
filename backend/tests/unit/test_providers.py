"""
Unit Tests for External Provider Abstractions.
"""

import pytest
from app.integrations.jev_provider import MockDecisionProvider
from app.integrations.llm_provider import MockLLMProvider
from app.integrations.weather_provider import MockWeatherProvider


@pytest.mark.asyncio
async def test_weather_provider_contract():
    """Verify MockWeatherProvider returns traceable, valid snapshot."""
    provider = MockWeatherProvider()
    snapshot = await provider.get_current_weather("loc_01", 28.6, 77.2)

    assert snapshot.location_id == "loc_01"
    assert snapshot.source == "mock_weather_provider"
    assert snapshot.temperature_c > 0.0
    assert 0.0 <= snapshot.relative_humidity_pct <= 100.0
    assert snapshot.data_quality_score == 0.98

    forecast = await provider.get_forecast("loc_01", 28.6, 77.2, horizon_hours=24)
    assert len(forecast) > 0

    history = await provider.get_historical_observations("loc_01", 28.6, 77.2, days_back=3)
    assert len(history) == 3


@pytest.mark.asyncio
async def test_decision_provider_contract():
    """Verify MockDecisionProvider routes critical decisions without altering predictions."""
    provider = MockDecisionProvider()
    outcomes = await provider.evaluate_decision({"risk_level": "CRITICAL"})
    assert len(outcomes) == 1
    assert outcomes[0].action_type == "NOTIFY_MUNICIPAL_DISASTER_DESK"


@pytest.mark.asyncio
async def test_llm_provider_contract():
    """Verify MockLLMProvider provides natural language explanation without doing calculations."""
    provider = MockLLMProvider()
    explanation = await provider.generate_explanation(
        {"category": "HEAT", "risk_score": 75.0, "risk_level": "SEVERE"}
    )
    assert explanation.summary is not None
    assert "HEAT" in explanation.summary
    assert len(explanation.key_bullet_points) > 0
