"""
Unit Tests for Concrete Rule-Based Baseline Risk Models.
"""

import pytest
from app.models.heat.model import RuleBasedHeatModel
from app.models.rainfall.model import RuleBasedRainfallModel
from app.models.water.model import RuleBasedWaterModel
from app.schemas.risk import RiskCategory, RiskLevel
from app.utils.exceptions import FeatureComputationError


def test_rule_based_heat_model_high_heat():
    """Verify heat baseline model reacts appropriately to high temperature and humidity."""
    model = RuleBasedHeatModel()
    assert model.category == RiskCategory.HEAT
    assert not model.is_ml_model
    assert model.model_name == "rule-baseline-heat"

    features = {
        "temperature_c": 42.0,
        "apparent_temperature_c": 51.5,
        "wet_bulb_temperature_c": 31.8,
        "consecutive_hot_days": 4,
        "uhi_adjusted_temp_c": 44.0,
    }
    prediction = model.predict(features=features, location_id="loc_delhi_01")
    assert prediction.risk_score >= 80.0
    assert prediction.risk_level == RiskLevel.CRITICAL
    assert len(prediction.drivers) >= 3
    driver_params = [d.parameter for d in prediction.drivers]
    assert "apparent_temperature_c" in driver_params
    assert "wet_bulb_temperature_c" in driver_params


def test_rule_based_heat_model_missing_feature():
    """Verify that missing critical features raise FeatureComputationError."""
    model = RuleBasedHeatModel()
    with pytest.raises(FeatureComputationError):
        model.predict(features={"temperature_c": 30.0}, location_id="loc_01")


def test_rule_based_water_model_severe_drought():
    """Verify water baseline model detects depleted soil moisture and prolonged dry spells."""
    model = RuleBasedWaterModel()
    assert model.category == RiskCategory.WATER_STRESS
    assert not model.is_ml_model

    features = {
        "soil_moisture_index": 0.10,
        "consecutive_dry_days": 35,
        "groundwater_depth_m": 42.0,
        "supply_outage_hours": 30,
    }
    prediction = model.predict(features=features, location_id="loc_chennai_01")
    assert prediction.risk_score >= 80.0
    assert prediction.risk_level == RiskLevel.CRITICAL
    assert len(prediction.drivers) >= 3


def test_rule_based_rainfall_model_heavy_downpour():
    """Verify rainfall baseline model flags flash-flood and waterlogging conditions."""
    model = RuleBasedRainfallModel()
    assert model.category == RiskCategory.RAINFALL
    assert not model.is_ml_model

    features = {
        "precipitation_current_hour_mm": 55.0,
        "precipitation_last_24h_mm": 120.0,
        "drainage_impedance": 0.8,
        "topography_risk_multiplier": 1.4,
    }
    prediction = model.predict(features=features, location_id="loc_mumbai_01")
    assert prediction.risk_score >= 80.0
    assert prediction.risk_level == RiskLevel.CRITICAL
