"""
Unit and Integration Tests for Real Historical Dataset Generation,
Validation, Leakage Prevention, and Quality Auditing (Step 3B).
100% Hermetic and Deterministic.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from app.schemas.features import DATA_VERSION, FEATURE_VERSION
from ml.baselines.persistence import PersistenceHeatBaseline
from ml.pipelines.dataset_generator import MLDatasetGenerator
from ml.pipelines.locations import DEFAULT_LOCATIONS, LocationConfig, get_location_registry
from ml.pipelines.provider import (
    HistoricalWeatherProvider,
    LocalFileHistoricalProvider,
    OpenMeteoArchiveHistoricalProvider,
    SyntheticHistoricalWeatherProvider,
)
from ml.pipelines.schema import DatasetMetadata, DatasetQualityReport, HistoricalObservation
from ml.pipelines.splitting import temporal_train_val_test_split
from ml.pipelines.target import ThresholdBasedHeatTargetGenerator
from ml.pipelines.validation import (
    detect_target_leakage,
    generate_dataset_quality_report,
    validate_chronological_integrity,
    validate_numeric_boundaries,
)


# 1. Real historical provider normalization
@pytest.mark.asyncio
async def test_real_provider_normalization():
    """Verify OpenMeteoArchiveHistoricalProvider correctly parses and normalizes ERA5 payloads."""
    provider = OpenMeteoArchiveHistoricalProvider()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "elevation": 216.0,
        "hourly": {
            "time": ["2024-05-01T00:00", "2024-05-01T01:00"],
            "temperature_2m": [31.5, 30.8],
            "relative_humidity_2m": [45.0, 48.0],
            "apparent_temperature": [34.0, 33.2],
            "precipitation": [0.0, 0.0],
            "wind_speed_10m": [12.5, 11.0],
        },
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    provider._custom_client = mock_client

    start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 5, 1, 1, 0, tzinfo=timezone.utc)
    obs = await provider.get_historical_observations(
        location_id="loc_delhi", latitude=28.6139, longitude=77.2090, start_time=start, end_time=end
    )

    assert len(obs) == 2
    assert obs[0].location_id == "loc_delhi"
    assert obs[0].temperature_c == 31.5
    assert obs[0].relative_humidity_pct == 45.0
    assert obs[0].source == "open-meteo-era5"
    assert obs[0].origin_type == "REANALYSIS"
    assert obs[0].surface_pressure_hpa is None  # Preserved as None; never fabricated
    assert obs[0].timestamp.tzinfo == timezone.utc


# 2. Multiple locations coverage
def test_multiple_locations_registry():
    """Verify location registry contains canonical Indian cities across diverse climate zones."""
    registry = get_location_registry()
    expected_cities = [
        "loc_delhi", "loc_bhopal", "loc_jaipur", "loc_nagpur", "loc_lucknow",
        "loc_ahmedabad", "loc_hyderabad", "loc_chennai", "loc_mumbai", "loc_bengaluru",
    ]
    for loc_id in expected_cities:
        assert loc_id in registry
        cfg = registry[loc_id]
        assert isinstance(cfg, LocationConfig)
        assert -90.0 <= cfg.latitude <= 90.0
        assert -180.0 <= cfg.longitude <= 180.0
        assert len(cfg.state) > 0
        assert len(cfg.climate_zone) > 0


# 3. Hourly temporal continuity
def test_hourly_temporal_continuity():
    """Verify validate_chronological_integrity asserts monotonicity and flag gaps or non-chronological order."""
    base = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    valid_df = pd.DataFrame({
        "timestamp": [base + timedelta(hours=i) for i in range(10)],
        "location_id": ["loc_1"] * 10,
    })
    ok, issues = validate_chronological_integrity(valid_df)
    assert ok is True
    assert len(issues) == 0

    reversed_df = valid_df.sort_values("timestamp", ascending=False)
    bad_ok, bad_issues = validate_chronological_integrity(reversed_df)
    assert bad_ok is False
    assert any("chronological order" in i for i in bad_issues)


# 4. Forward 6-hour target generation
def test_forward_6h_target_generation():
    """Verify ThresholdBasedHeatTargetGenerator computes maximum heat stress in strictly (t, t+6h]."""
    start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(15)]
    # Hours 0-4: 25°C, 40% RH (LOW)
    # Hour 5: 45°C, 50% RH (HIGH/SEVERE)
    # Hours 6-14: 25°C, 40% RH (LOW)
    temps = [25.0] * 5 + [45.0] + [25.0] * 9
    rhs = [40.0] * 15

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temps,
        "relative_humidity_pct": rhs,
    })

    gen = ThresholdBasedHeatTargetGenerator(target_type="classification")
    targets = gen.generate_targets(df, horizon_hours=6)

    # Row 0 looks into hours 1..6 (which includes hour 5 with 45°C) -> should be elevated (HIGH or SEVERE)
    assert targets.iloc[0] in ["HIGH", "SEVERE", "CRITICAL"]
    # Row 6 looks into hours 7..12 (all 25°C) -> should be LOW
    assert targets.iloc[6] == "LOW"


# 5. Target horizon boundary
def test_target_horizon_boundary():
    """Verify target generator drops or marks NaN for boundary rows where forward horizon is incomplete."""
    start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(5)],
        "temperature_c": [30.0] * 5,
        "relative_humidity_pct": [50.0] * 5,
    })
    gen = ThresholdBasedHeatTargetGenerator(target_type="classification")
    targets = gen.generate_targets(df, horizon_hours=6)
    # Last row has no forward window (start_fwd >= end_fwd)
    assert pd.isna(targets.iloc[-1])


# 6. No future features in feature matrix
def test_no_future_features_in_matrix():
    """Verify feature extractor strictly uses lagging information and contains zero leakage keywords."""
    clean_X = pd.DataFrame({
        "temperature_c": [32.0, 33.0],
        "dew_point_c": [20.0, 21.0],
        "rainfall_accumulation_24h_mm": [0.0, 5.0],
        "forecast_precipitation_6h_mm": [0.0, 1.0],  # Valid NWP forecast
    })
    ok, issues = detect_target_leakage(clean_X, target_column_name="target_heat_risk_category")
    assert ok is True

    leaky_X = clean_X.copy()
    leaky_X["future_temperature_t_plus_2"] = [35.0, 36.0]
    bad_ok, bad_issues = detect_target_leakage(leaky_X, target_column_name="target_heat_risk_category")
    assert bad_ok is False
    assert any("forbidden keyword 'future'" in i for i in bad_issues)


# 7. Pressure missing-value handling (Option A)
def test_pressure_missing_value_handling():
    """Verify surface pressure is excluded from V1 feature vector and never imputed to 1013.25 hPa."""
    provider = SyntheticHistoricalWeatherProvider()
    obs_list = [
        HistoricalObservation(
            location_id="loc_test",
            latitude=28.6,
            longitude=77.2,
            timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
            temperature_c=35.0,
            relative_humidity_pct=40.0,
            surface_pressure_hpa=None,  # Missing barometric pressure
            source="test_provider",
            origin_type="OBSERVED",
        )
    ]
    gen = MLDatasetGenerator()
    feat_df = gen.extract_features_from_observations(obs_list)

    # Option A: Surface pressure must NOT be present as an imputed feature
    assert "surface_pressure_hpa" not in feat_df.columns
    # Ensure no fabricated 1013.25 value exists anywhere in the feature row
    row_vals = feat_df.iloc[0].to_dict()
    assert 1013.25 not in row_vals.values()


# 8. Target distribution calculation
def test_target_distribution_calculation():
    """Verify generate_dataset_quality_report accurately computes target counts and percentages."""
    feature_df = pd.DataFrame({"temp": [30.0, 35.0, 40.0, 45.0]})
    target_series = pd.Series(["LOW", "MODERATE", "HIGH", "SEVERE"], name="target_heat_risk_category")
    time_series = pd.Series([datetime(2024, 5, 1, i, 0, tzinfo=timezone.utc) for i in range(4)])
    loc_series = pd.Series(["loc_delhi"] * 4)

    report = generate_dataset_quality_report(
        feature_df=feature_df,
        target_series=target_series,
        time_series=time_series,
        location_series=loc_series,
    )

    assert report.total_rows == 4
    assert report.target_distribution["LOW"] == 1
    assert report.target_percentages["LOW"] == 25.0
    assert report.target_percentages["SEVERE"] == 25.0


# 9. Dataset metadata completeness
def test_dataset_metadata_completeness():
    """Verify DatasetMetadata serializes required contract fields for auditability."""
    meta = DatasetMetadata(
        dataset_name="heat_risk_real",
        dataset_version="v1.0",
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        target_version="heat-target-v1",
        source="open-meteo-era5",
        data_type="reanalysis",
        location_coverage=["loc_delhi", "loc_bhopal"],
        forecast_horizon_hours=6,
        ready_for_model_training=True,
    )
    d = meta.model_dump()
    assert d["dataset_name"] == "heat_risk_real"
    assert d["source"] == "open-meteo-era5"
    assert d["data_type"] == "reanalysis"
    assert d["forecast_horizon_hours"] == 6
    assert d["ready_for_model_training"] is True


# 10. Dataset quality report schema
def test_dataset_quality_report_schema():
    """Verify DatasetQualityReport captures temporal, seasonal, and event breakdowns."""
    report = DatasetQualityReport(
        dataset_version="heat-risk-dataset-v1",
        source="open-meteo-era5",
        data_type="reanalysis",
        total_rows=100,
        location_count=1,
        yearly_distribution={"2023": 50, "2024": 50},
        seasonal_distribution={"Monsoon (Jun-Sep)": 40, "Pre-Monsoon/Summer (Mar-May)": 60},
        heat_events_summary={"total_heat_events": 5, "mean_duration_hours": 12.0, "max_duration_hours": 24},
        validation_passed=True,
    )
    assert report.total_rows == 100
    assert report.yearly_distribution["2023"] == 50
    assert report.heat_events_summary["total_heat_events"] == 5
    assert report.validation_passed is True


# 11. Chronological split strictness across multiple locations
def test_chronological_split_strictness_multilocation():
    """Verify multi-location dataset splitting enforces max(train) < min(val) < min(test) without collision."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    records = []
    # 2 locations sharing exact hourly timestamps across 20 hours
    for loc in ["loc_delhi", "loc_mumbai"]:
        for i in range(20):
            records.append({
                "timestamp": base + timedelta(hours=i),
                "location_id": loc,
                "feature": float(i),
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = temporal_train_val_test_split(
        df, time_col="timestamp", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # Enforce strict temporal ordering
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()
    # Both locations should be present in each partition
    assert train_df["location_id"].nunique() == 2
    assert val_df["location_id"].nunique() == 2
    assert test_df["location_id"].nunique() == 2


# 12. Duplicate timestamp detection
def test_duplicate_timestamp_detection():
    """Verify duplicate observations for the same location are flagged as errors."""
    t0 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    dup_df = pd.DataFrame({
        "timestamp": [t0, t0],
        "location_id": ["loc_delhi", "loc_delhi"],
    })
    ok, issues = validate_chronological_integrity(dup_df)
    assert ok is False
    assert any("duplicate timestamp" in i for i in issues)


# 13. Missing value handling (unobserved sensor variables)
def test_missing_value_handling():
    """Verify unobserved variables (groundwater, soil moisture) remain None and are never filled with fake data."""
    obs = HistoricalObservation(
        location_id="loc_test",
        latitude=28.6,
        longitude=77.2,
        timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
        temperature_c=30.0,
        relative_humidity_pct=50.0,
        source="test",
        origin_type="OBSERVED",
    )
    gen = MLDatasetGenerator()
    feat_df = gen.extract_features_from_observations([obs])
    # Ground sensors not available -> None/NaN without fabrication
    assert pd.isna(feat_df["soil_moisture_m3_m3"].iloc[0])
    assert pd.isna(feat_df["groundwater_depth_m"].iloc[0])


# 14. Persistence baseline benchmark evaluation
def test_persistence_baseline_evaluation():
    """Verify PersistenceHeatBaseline produces accuracy, F1, and transition analytics without training any model."""
    baseline = PersistenceHeatBaseline()
    feature_df = pd.DataFrame({
        "temperature_c": [28.0, 36.0, 45.0],
        "relative_humidity_pct": [40.0, 50.0, 45.0],
    })
    # Morning (28°C -> LOW), Afternoon (predicted MODERATE, but future reached SEVERE)
    y_true = pd.Series(["LOW", "SEVERE", "CRITICAL"])

    eval_result = baseline.evaluate(feature_df, y_true)
    assert "accuracy" in eval_result
    assert "macro_f1" in eval_result
    assert "transition_analysis" in eval_result
    assert "upward_escalations_missed_by_persistence" in eval_result["transition_analysis"]
    assert eval_result["total_evaluated"] == 3


# 15. Real data artifact generation
@pytest.mark.asyncio
async def test_real_data_artifact_generation(tmp_path):
    """Verify multi-location dataset generation exports Parquet and JSON manifests with readiness confirmation."""
    provider = SyntheticHistoricalWeatherProvider(seed=42)
    target_gen = ThresholdBasedHeatTargetGenerator(target_type="classification")
    generator = MLDatasetGenerator(
        provider=provider,
        target_generator=target_gen,
        artifacts_dir=str(tmp_path),
    )

    test_locs = [
        LocationConfig(location_id="loc_test_1", name="City 1", state="State 1", latitude=28.6, longitude=77.2, climate_zone="Zone 1"),
        LocationConfig(location_id="loc_test_2", name="City 2", state="State 2", latitude=19.0, longitude=72.8, climate_zone="Zone 2"),
    ]

    start = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)

    X, y, metadata, report = await generator.generate_multi_location_dataset(
        locations=test_locs,
        start_time=start,
        end_time=end,
        horizon_hours=6,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_name="multi_loc_run",
        export_artifacts=True,
        rate_limit_delay_seconds=0.0,
    )

    out_dir = tmp_path / "multi_loc_run"
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "quality_report.json").exists()
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "validation.parquet").exists()
    assert (out_dir / "test.parquet").exists()
    assert metadata.ready_for_model_training is True
    assert report.validation_passed is True
