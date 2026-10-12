"""
Real Historical Dataset Generation and Quality Audit Script.
Uses OpenMeteoArchiveHistoricalProvider to retrieve real ERA5 reanalysis data
across 10 canonical Indian cities covering 2023-2024 (2 full calendar years).
Strictly generates datasets and audit reports without training any ML models.

Usage:
    python ml/pipelines/generate_real_dataset.py
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure repository root and backend are in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
backend_dir = repo_root / "backend"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ml.pipelines.dataset_generator import MLDatasetGenerator
from ml.pipelines.locations import DEFAULT_LOCATIONS
from ml.pipelines.provider import OpenMeteoArchiveHistoricalProvider
from ml.pipelines.target import ThresholdBasedHeatTargetGenerator


async def generate_real_historical_dataset():
    print("=" * 80)
    print("STARTING REAL HISTORICAL DATASET GENERATION (STEP 3B)")
    print("=" * 80)
    print("Source: Open-Meteo ERA5 / ERA5-Land Reanalysis (archive-api.open-meteo.com)")
    print("Locations: 10 Canonical Indian Cities")
    for loc in DEFAULT_LOCATIONS:
        print(f"  - {loc.name} ({loc.location_id}) | Lat: {loc.latitude}, Lon: {loc.longitude} | Zone: {loc.climate_zone}")

    start_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc)
    horizon_hours = 6

    print(f"\nHistorical Period: {start_time.isoformat()} to {end_time.isoformat()}")
    print(f"Forecast Target Horizon: H = {horizon_hours} hours")
    print("Target Problem: Predict maximum heat-risk severity occurring during (t, t + 6h]")
    print("Task: Ordinal multi-class classification [LOW, MODERATE, HIGH, SEVERE, CRITICAL]")
    print("Strict Constraint: ZERO ML MODEL TRAINING will take place.")

    provider = OpenMeteoArchiveHistoricalProvider(timeout=30.0, max_retries=3, retry_delay=1.0)
    target_gen = ThresholdBasedHeatTargetGenerator(target_type="classification", version="heat-target-v1")

    generator = MLDatasetGenerator(
        provider=provider,
        target_generator=target_gen,
        artifacts_dir=str(repo_root / "ml" / "datasets"),
    )

    X, y, metadata, quality_report = await generator.generate_multi_location_dataset(
        locations=DEFAULT_LOCATIONS,
        start_time=start_time,
        end_time=end_time,
        horizon_hours=horizon_hours,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_name="heat_risk_real",
        export_artifacts=True,
        rate_limit_delay_seconds=0.3,
    )

    print("\n" + "=" * 80)
    print("REAL HISTORICAL DATASET GENERATION & AUDIT COMPLETED")
    print("=" * 80)
    print(f"Total Rows: {len(X)}")
    print(f"Features in X: {len(X.columns)}")
    print(f"Target Column: {y.name}")
    print(f"Target Counts: {quality_report.target_distribution}")
    print(f"Target Percentages: {quality_report.target_percentages}")
    print(f"Row Counts per Split: {metadata.row_counts}")
    print(f"Temporal Split Timestamps:")
    for split_k, split_v in quality_report.temporal_split_timestamps.items():
        print(f"  - {split_k.upper()}: {split_v['start']} to {split_v['end']} ({split_v.get('row_count', 'N/A')} rows)")
    print(f"Validation Passed: {quality_report.validation_passed}")
    print(f"Issues Detected: {quality_report.issues if quality_report.issues else 'None (Clean)'}")
    print(f"Artifacts exported to: {repo_root / 'ml' / 'datasets' / 'heat_risk_real'}")

    return X, y, metadata, quality_report


if __name__ == "__main__":
    asyncio.run(generate_real_historical_dataset())
