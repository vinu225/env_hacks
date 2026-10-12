"""
Unit Tests for Offline ML Environmental Data Pipeline.
100% Offline, Deterministic, and Hermetic.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.schemas.features import DATA_VERSION, FEATURE_VERSION
from ml.pipelines.dataset_generator import MLDatasetGenerator
from ml.pipelines.provider import (
    LocalFileHistoricalProvider,
    SyntheticHistoricalWeatherProvider,
)
from ml.pipelines.schema import (
    DatasetMetadata,
    DatasetQualityReport,
    HistoricalObservation,
)
from ml.pipelines.splitting import temporal_train_val_test_split
from ml.pipelines.target import ThresholdBasedHeatTargetGenerator
from ml.pipelines.validation import (
    detect_target_leakage,
    generate_dataset_quality_report,
    validate_chronological_integrity,
    validate_numeric_boundaries,
)


def test_historical_observation_schema():
    """Verify HistoricalObservation enforces UTC timestamps and preserves provenance."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    obs = HistoricalObservation(
        location_id="loc_delhi",
        latitude=28.6315,
        longitude=77.2167,
        timestamp=now,
        temperature_c=36.5,
        relative_humidity_pct=52.0,
        apparent_temperature_c=40.2,
        precipitation_mm=0.0,
        wind_speed_kmh=14.0,
        surface_pressure_hpa=1008.0,
        precipitation_probability_pct=10.0,
        source="open_meteo_archive",
        origin_type="OBSERVED",
        metadata={"archive_station": "DELHI_AWS_01"},
    )

    assert obs.location_id == "loc_delhi"
    assert obs.timestamp.tzinfo == timezone.utc
    assert obs.temperature_c == 36.5
    assert obs.source == "open_meteo_archive"
    assert obs.origin_type == "OBSERVED"
    assert obs.metadata["archive_station"] == "DELHI_AWS_01"


def test_target_generation_classification():
    """Verify ThresholdBasedHeatTargetGenerator produces valid classification labels."""
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(50)]

    # First 30h: baseline temperatures (28°C, 50% RH -> HI ~29°C -> LOW)
    # Next 20h: extreme heatwave temperatures (46°C, 45% RH -> HI > 54°C -> CRITICAL)
    temps = [28.0] * 30 + [46.0] * 20
    rhs = [50.0] * 30 + [45.0] * 20

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temps,
        "relative_humidity_pct": rhs,
    })

    gen = ThresholdBasedHeatTargetGenerator(target_type="classification")
    assert gen.target_name == "target_heat_risk_category"
    assert gen.target_version == "heat-target-v1"

    targets = gen.generate_targets(df, horizon_hours=24)
    assert len(targets) == 50

    # The first row (index 0) looks forward into hours 1..24 (which only contains 28°C -> LOW)
    assert targets.iloc[0] == "LOW"
    # Row at hour 25 looks forward into the 46°C heatwave block -> CRITICAL
    assert targets.iloc[25] == "CRITICAL"


def test_target_generation_regression():
    """Verify ThresholdBasedHeatTargetGenerator produces continuous scores in [0, 100]."""
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(30)]
    temps = [28.0 + (i * 0.5) for i in range(30)]
    rhs = [50.0] * 30

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temps,
        "relative_humidity_pct": rhs,
    })

    gen = ThresholdBasedHeatTargetGenerator(target_type="regression")
    assert gen.target_name == "target_heat_risk_score"
    targets = gen.generate_targets(df, horizon_hours=12)

    valid_scores = targets.dropna()
    assert len(valid_scores) > 0
    assert (valid_scores >= 0.0).all()
    assert (valid_scores <= 100.0).all()
    # Later rows with higher forward temperatures should have higher risk scores
    assert valid_scores.iloc[-15] > valid_scores.iloc[0]


def test_temporal_train_val_test_split():
    """Verify strictly chronological temporal splitting without lookahead contamination."""
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(100)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "feature_1": range(100),
    })

    train_df, val_df, test_df = temporal_train_val_test_split(
        df, time_col="timestamp", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Enforce strict chronological separation: train < val < test
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_temporal_split_invalid_ratios():
    """Verify exception when split ratios do not sum to 1.0."""
    df = pd.DataFrame({"timestamp": [datetime.now(timezone.utc)] * 5})
    with pytest.raises(ValueError) as exc:
        temporal_train_val_test_split(df, train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
    assert "must sum to 1.0" in str(exc.value)


def test_detect_target_leakage():
    """Verify target leakage detector flags forbidden target columns in feature matrix X."""
    clean_X = pd.DataFrame({
        "temperature_c": [35.0, 36.0],
        "heat_index_c": [41.0, 43.0],
        "wind_speed_kmh": [12.0, 10.0],
    })
    ok, issues = detect_target_leakage(clean_X, target_column_name="target_heat_risk_category")
    assert ok is True
    assert len(issues) == 0

    # Contaminated feature matrix containing target column
    leaky_X = clean_X.copy()
    leaky_X["target_heat_risk_category"] = ["HIGH", "HIGH"]
    leaky_ok, leaky_issues = detect_target_leakage(leaky_X, target_column_name="target_heat_risk_category")
    assert leaky_ok is False
    assert any("FATAL LEAKAGE" in issue for issue in leaky_issues)


def test_duplicate_and_chronological_detection():
    """Verify detection of non-monotonic and duplicate timestamps."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    dup_df = pd.DataFrame({
        "location_id": ["loc_1", "loc_1", "loc_1"],
        "timestamp": [now, now, now + timedelta(hours=1)],  # duplicate timestamp at index 0 and 1
    })
    ok, issues = validate_chronological_integrity(dup_df)
    assert ok is False
    assert any("duplicate timestamp" in i for i in issues)


def test_numeric_boundary_validation():
    """Verify physical bounds validation across environmental columns."""
    valid_df = pd.DataFrame({
        "temperature_c": [32.0, 44.0],
        "relative_humidity_pct": [30.0, 80.0],
        "precipitation_mm": [0.0, 12.0],
    })
    ok, issues = validate_numeric_boundaries(valid_df)
    assert ok is True
    assert len(issues) == 0

    invalid_df = pd.DataFrame({
        "temperature_c": [95.0],  # Implausible ground ambient temperature > 65°C
        "relative_humidity_pct": [-10.0],  # Impossible negative humidity
        "precipitation_mm": [-5.0],  # Impossible negative rain
    })
    bad_ok, bad_issues = validate_numeric_boundaries(invalid_df)
    assert bad_ok is False
    assert len(bad_issues) == 3


@pytest.mark.asyncio
async def test_end_to_end_sample_dataset_generation(tmp_path):
    """Verify complete ML dataset generation pipeline runs end-to-end and exports artifacts."""
    provider = SyntheticHistoricalWeatherProvider(seed=123)
    target_gen = ThresholdBasedHeatTargetGenerator(target_type="classification")
    generator = MLDatasetGenerator(
        provider=provider,
        target_generator=target_gen,
        artifacts_dir=str(tmp_path),
    )

    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=7)  # 168 hours

    X, y, metadata, report = await generator.generate_dataset(
        location_id="loc_test_city",
        latitude=28.6,
        longitude=77.2,
        start_time=start,
        end_time=end,
        horizon_hours=24,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_name="test_dataset_run",
        export_artifacts=True,
    )

    # 168 rows because the final observation has no forward horizon and is dropped
    assert len(X) == 168
    assert len(y) == 168
    assert y.name == "target_heat_risk_category"

    # Core required features must be present
    assert "temperature_c" in X.columns
    assert "heat_index_c" in X.columns
    assert "wet_bulb_temperature_c" in X.columns
    assert "consecutive_hot_days" in X.columns

    # Metadata checks
    assert metadata.data_version == DATA_VERSION
    assert metadata.feature_version == FEATURE_VERSION
    assert metadata.target_version == "heat-target-v1"
    assert metadata.is_synthetic is True
    assert "SYNTHETIC SAMPLE DATA" in metadata.provenance_note

    # Quality report checks
    assert report.total_rows == 168
    assert report.validation_passed is True

    # Artifact file checks
    out_dir = tmp_path / "test_dataset_run"
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "quality_report.json").exists()
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "val.parquet").exists()
    assert (out_dir / "test.parquet").exists()


@pytest.mark.asyncio
async def test_local_file_historical_provider(tmp_path):
    """Verify LocalFileHistoricalProvider correctly reads pre-stored observation tables."""
    now = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    sample_df = pd.DataFrame([
        {
            "location_id": "loc_bench",
            "latitude": 19.07,
            "longitude": 72.87,
            "timestamp": now.isoformat(),
            "temperature_c": 33.0,
            "relative_humidity_pct": 75.0,
            "apparent_temperature_c": 40.0,
            "precipitation_mm": 2.0,
            "wind_speed_kmh": 18.0,
            "source": "local_benchmark_file",
        }
    ])
    csv_file = tmp_path / "historical_sample.csv"
    sample_df.to_csv(csv_file, index=False)

    provider = LocalFileHistoricalProvider(str(csv_file))
    obs_list = await provider.get_historical_observations(
        location_id="loc_bench",
        latitude=19.07,
        longitude=72.87,
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    )

    assert len(obs_list) == 1
    assert obs_list[0].temperature_c == 33.0
    assert obs_list[0].relative_humidity_pct == 75.0
    assert obs_list[0].source == "local_benchmark_file"
