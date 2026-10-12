"""
CLI / Script to generate a small synthetic sample heat training dataset.

Usage:
    python ml/pipelines/generate_sample_dataset.py
"""

import asyncio
from datetime import datetime, timedelta, timezone
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
from ml.pipelines.provider import SyntheticHistoricalWeatherProvider
from ml.pipelines.target import ThresholdBasedHeatTargetGenerator


async def main():
    print("Initiating synthetic heat training dataset generation...")
    generator = MLDatasetGenerator(
        provider=SyntheticHistoricalWeatherProvider(seed=42),
        target_generator=ThresholdBasedHeatTargetGenerator(target_type="classification"),
        artifacts_dir=str(repo_root / "ml" / "datasets"),
    )

    # 14 days of hourly synthetic observations (336 hours)
    start_time = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(days=14)

    X, y, metadata, quality_report = await generator.generate_dataset(
        location_id="loc_delhi_sample",
        latitude=28.6315,
        longitude=77.2167,
        start_time=start_time,
        end_time=end_time,
        horizon_hours=24,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_name="sample_heat_pipeline_data",
        export_artifacts=True,
    )

    print("\n--- Dataset Generation Complete ---")
    print(f"Total Rows: {len(X)}")
    print(f"Features in X: {len(X.columns)}")
    print(f"Target Column: {y.name}")
    print(f"Target Distribution: {quality_report.target_distribution}")
    print(f"Partitions: {metadata.row_counts}")
    print(f"Validation Passed: {quality_report.validation_passed}")
    print(f"Artifacts exported to: {repo_root / 'ml' / 'datasets' / 'sample_heat_pipeline_data'}")


if __name__ == "__main__":
    asyncio.run(main())
