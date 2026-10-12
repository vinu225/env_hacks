"""
Offline ML Environmental Dataset Generator.

Orchestrates the complete offline dataset pipeline:
Historical Observations
       │
       ▼
Feature Engineering (reusing backend/app/features/)
       │
       ▼
EnvironmentalFeatureVector (.to_numeric_dict())
       │
       ▼
Target Generation (HeatRiskTargetGenerator over forward window)
       │
       ▼
Data Validation & Leakage Checks
       │
       ▼
Temporal Split (Train / Val / Test)
       │
       ▼
Parquet / CSV Artifacts & Metadata Manifest
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.core.logging import get_logger
from app.features.builder import FeatureVectorBuilder
from app.schemas.environmental import (
    DailyForecastObservation,
    HourlyForecastObservation,
    NormalizedWeatherData,
    NormalizedWeatherObservation,
)
from app.schemas.features import DATA_VERSION, FEATURE_VERSION
from app.schemas.location import Coordinates
from app.utils.datetime_utils import utc_now
from ml.pipelines.provider import HistoricalWeatherProvider, SyntheticHistoricalWeatherProvider
from ml.pipelines.schema import DatasetMetadata, DatasetQualityReport, HistoricalObservation
from ml.pipelines.splitting import temporal_train_val_test_split
from ml.pipelines.target import HeatRiskTargetGenerator, ThresholdBasedHeatTargetGenerator
from ml.pipelines.validation import (
    detect_target_leakage,
    generate_dataset_quality_report,
    validate_chronological_integrity,
    validate_numeric_boundaries,
)

logger = get_logger(__name__)


class MLDatasetGenerator:
    """
    Constructs ML-ready training, validation, and test matrices from historical weather data.
    Strictly reuses the production feature engineering layer to eliminate training-serving skew.
    """

    def __init__(
        self,
        provider: Optional[HistoricalWeatherProvider] = None,
        target_generator: Optional[HeatRiskTargetGenerator] = None,
        artifacts_dir: Optional[str] = None,
    ):
        self.provider = provider or SyntheticHistoricalWeatherProvider()
        self.target_generator = target_generator or ThresholdBasedHeatTargetGenerator()
        
        # Default artifacts storage directory under ml/datasets/
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir)
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.artifacts_dir = repo_root / "ml" / "datasets"
        
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _convert_observations_to_df(self, observations: List[HistoricalObservation]) -> pd.DataFrame:
        """Convert typed historical observation records into tabular DataFrame."""
        rows = []
        for obs in observations:
            rows.append({
                "location_id": obs.location_id,
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                "timestamp": obs.timestamp,
                "temperature_c": obs.temperature_c,
                "relative_humidity_pct": obs.relative_humidity_pct,
                "apparent_temperature_c": obs.apparent_temperature_c,
                "precipitation_mm": obs.precipitation_mm,
                "wind_speed_kmh": obs.wind_speed_kmh,
                "surface_pressure_hpa": obs.surface_pressure_hpa,
                "precipitation_probability_pct": obs.precipitation_probability_pct,
                "source": obs.source,
                "origin_type": obs.origin_type,
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    def extract_features_from_observations(
        self,
        observations: List[HistoricalObservation],
        horizon_hours: int = 6,
    ) -> pd.DataFrame:
        """
        Apply identical feature engineering as online serving to each historical timestamp.
        Uses lagging window [t - 24h, t] to avoid any lookahead contamination.
        Optimized sliding window scan ensures O(N) performance over long time series.
        """
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)
        feature_rows: List[Dict[str, Any]] = []

        # Index historical records by timestamp for efficient lagged slicing
        for i, obs in enumerate(sorted_obs):
            t = obs.timestamp

            # Slice lagging 24h context for rolling features (strictly <= t)
            lag_start = t - timedelta(hours=24)
            start_idx = i
            while start_idx > 0 and sorted_obs[start_idx - 1].timestamp >= lag_start:
                start_idx -= 1
            lagged_slice = sorted_obs[start_idx : i + 1]

            # Construct mock hourly slice from lagging history to feed rolling rainfall pure functions
            mock_hourly = [
                HourlyForecastObservation(
                    forecast_valid_at=o.timestamp,
                    temperature_c=o.temperature_c if o.temperature_c is not None else 25.0,
                    relative_humidity_pct=o.relative_humidity_pct if o.relative_humidity_pct is not None else 50.0,
                    precipitation_mm=o.precipitation_mm if o.precipitation_mm is not None else 0.0,
                    wind_speed_kmh=o.wind_speed_kmh if o.wind_speed_kmh is not None else 10.0,
                )
                for o in lagged_slice
            ]

            # Current normalized observation representation
            curr_norm = NormalizedWeatherObservation(
                source=obs.source,
                location_id=obs.location_id,
                coordinates=Coordinates(latitude=obs.latitude, longitude=obs.longitude),
                observed_at=obs.timestamp,
                ingested_at=obs.timestamp,
                temperature_c=obs.temperature_c,
                apparent_temperature_c=obs.apparent_temperature_c,
                relative_humidity_pct=obs.relative_humidity_pct,
                precipitation_mm=obs.precipitation_mm,
                wind_speed_kmh=obs.wind_speed_kmh,
                precipitation_probability_pct=obs.precipitation_probability_pct,
                data_quality_score=1.0,
                is_usable=True,
                validation_issues=[],
            )

            weather_data = NormalizedWeatherData(
                current=curr_norm,
                hourly_forecast=mock_hourly,
                daily_forecast=[],
            )

            # Build standardized feature vector
            vector = FeatureVectorBuilder.build_from_normalized_data(
                weather_data=weather_data,
                location_id=obs.location_id,
            )

            # Extract numeric dictionary
            row_dict = vector.to_numeric_dict()
            # Retain indexing timestamps for alignment
            row_dict["_timestamp"] = obs.timestamp
            row_dict["_location_id"] = obs.location_id
            feature_rows.append(row_dict)

        return pd.DataFrame(feature_rows)

    async def generate_dataset(
        self,
        location_id: str,
        latitude: float,
        longitude: float,
        start_time: datetime,
        end_time: datetime,
        horizon_hours: int = 6,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        dataset_name: str = "heat_risk_dataset",
        export_artifacts: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, DatasetMetadata, DatasetQualityReport]:
        """
        Execute full end-to-end dataset generation pipeline for a single location.
        Returns (X, y, metadata, quality_report).
        """
        from ml.pipelines.locations import LocationConfig
        loc_cfg = LocationConfig(
            location_id=location_id,
            name=location_id,
            state="Unspecified",
            latitude=latitude,
            longitude=longitude,
            climate_zone="Unspecified",
        )
        return await self.generate_multi_location_dataset(
            locations=[loc_cfg],
            start_time=start_time,
            end_time=end_time,
            horizon_hours=horizon_hours,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            dataset_name=dataset_name,
            export_artifacts=export_artifacts,
            rate_limit_delay_seconds=0.0,
        )

    async def generate_multi_location_dataset(
        self,
        locations: Optional[List[Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        horizon_hours: int = 6,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        dataset_name: str = "heat_risk_real",
        export_artifacts: bool = True,
        rate_limit_delay_seconds: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.Series, DatasetMetadata, DatasetQualityReport]:
        """
        Execute full end-to-end dataset generation pipeline across multiple geographic locations.
        Enforces:
        - Per-location feature engineering and forward target generation (no cross-location contamination)
        - Chronological temporal train/val/test splitting without lookahead leakage
        - Data quality, distribution, and persistence baseline auditing
        """
        import asyncio
        from ml.baselines.persistence import PersistenceHeatBaseline
        from ml.pipelines.locations import DEFAULT_LOCATIONS, LocationConfig

        loc_list = locations or DEFAULT_LOCATIONS
        if not loc_list:
            raise ValueError("No locations provided for dataset generation.")

        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if end_time is None:
            end_time = datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        logger.info(
            f"Starting multi-location dataset generation '{dataset_name}' "
            f"across {len(loc_list)} locations from {start_time} to {end_time} (H={horizon_hours}h)"
        )

        all_clean_frames: List[pd.DataFrame] = []
        dropped_horizon_total = 0
        total_raw_observations = 0
        all_raw_obs_samples = []

        for loc in loc_list:
            loc_id = loc.location_id if isinstance(loc, LocationConfig) else getattr(loc, "location_id", str(loc))
            lat = loc.latitude if isinstance(loc, LocationConfig) else getattr(loc, "latitude", 0.0)
            lon = loc.longitude if isinstance(loc, LocationConfig) else getattr(loc, "longitude", 0.0)

            logger.info(f"Ingesting historical observations for {loc_id} ({lat}, {lon})")
            raw_obs = await self.provider.get_historical_observations(
                location_id=loc_id,
                latitude=lat,
                longitude=lon,
                start_time=start_time,
                end_time=end_time,
            )

            if len(raw_obs) < 5:
                raise ValueError(f"Insufficient observations returned for {loc_id}: {len(raw_obs)}. Need >= 5.")

            total_raw_observations += len(raw_obs)
            if len(all_raw_obs_samples) < 10:
                all_raw_obs_samples.extend(raw_obs[:2])

            obs_df = self._convert_observations_to_df(raw_obs)
            feat_df = self.extract_features_from_observations(raw_obs, horizon_hours=horizon_hours)

            # Generate forward targets for this location
            target_series = self.target_generator.generate_targets(obs_df, horizon_hours=horizon_hours)

            # Align features and target
            aligned_loc = feat_df.copy()
            aligned_loc["_target"] = target_series.values

            # Drop trailing horizon rows where forward ground-truth has not yet occurred
            trailing_nan_count = int(aligned_loc["_target"].isna().sum())
            dropped_horizon_total += trailing_nan_count
            clean_loc = aligned_loc.dropna(subset=["_target"]).copy()

            all_clean_frames.append(clean_loc)

            if rate_limit_delay_seconds > 0.0 and loc != loc_list[-1]:
                await asyncio.sleep(rate_limit_delay_seconds)

        # Concatenate all locations
        full_clean_df = pd.concat(all_clean_frames, ignore_index=True)
        target_col = self.target_generator.target_name
        y = full_clean_df["_target"].rename(target_col)

        # Separate feature matrix X
        meta_cols = ["_timestamp", "_location_id", "_target"]
        X = full_clean_df.drop(columns=meta_cols)

        # 4. Chronological Splitting
        combined_for_split = full_clean_df.copy()
        combined_for_split = combined_for_split.rename(columns={"_timestamp": "timestamp"})
        train_df, val_df, test_df = temporal_train_val_test_split(
            combined_for_split,
            time_col="timestamp",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        split_ts_map = {
            "train": {
                "start": train_df["timestamp"].min().isoformat(),
                "end": train_df["timestamp"].max().isoformat(),
                "row_count": len(train_df),
            },
            "validation": {
                "start": val_df["timestamp"].min().isoformat(),
                "end": val_df["timestamp"].max().isoformat(),
                "row_count": len(val_df),
            },
            "test": {
                "start": test_df["timestamp"].min().isoformat(),
                "end": test_df["timestamp"].max().isoformat(),
                "row_count": len(test_df),
            },
        }

        # 5. Quality, Distribution, and Leakage Audit
        time_series = full_clean_df["_timestamp"]
        location_series = full_clean_df["_location_id"]

        dropped_summary = {
            "rows_dropped_insufficient_future_horizon": dropped_horizon_total,
            "rows_dropped_invalid_observations": 0,
            "rows_dropped_duplicates": 0,
            "rows_dropped_missing_required_features": 0,
        }

        quality_report = generate_dataset_quality_report(
            feature_df=X,
            target_series=y,
            time_series=time_series,
            location_series=location_series,
            dropped_rows_summary=dropped_summary,
            temporal_split_timestamps=split_ts_map,
        )

        # Evaluate Persistence Baseline
        persistence_baseline = PersistenceHeatBaseline(target_generator=self.target_generator)
        eval_metrics = persistence_baseline.evaluate(X, y)
        quality_report.persistence_baseline_summary = eval_metrics

        if not quality_report.validation_passed:
            logger.warning(f"Dataset quality validation flagged issues: {quality_report.issues}")

        # Provenance and Metadata Manifest
        is_synthetic = "synthetic" in self.provider.provider_name.lower() or any(
            obs.metadata.get("is_synthetic", False) for obs in all_raw_obs_samples[:5]
        )

        loc_ids = [
            loc.location_id if isinstance(loc, LocationConfig) else getattr(loc, "location_id", str(loc))
            for loc in loc_list
        ]
        loc_dicts = [
            loc.model_dump() if isinstance(loc, LocationConfig) else {"location_id": str(loc)}
            for loc in loc_list
        ]

        data_tier = "synthetic" if is_synthetic else ("reanalysis" if "era5" in self.provider.provider_name.lower() else "observation")

        metadata = DatasetMetadata(
            dataset_name=dataset_name,
            dataset_version="v1.0",
            data_version=DATA_VERSION,
            feature_version=FEATURE_VERSION,
            target_version=self.target_generator.target_version,
            source=self.provider.provider_name,
            data_type=data_tier,
            location_coverage=loc_ids,
            locations=loc_dicts,
            time_range={
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            row_counts={
                "total": len(full_clean_df),
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            features_list=list(X.columns),
            target_column=target_col,
            target_type=self.target_generator.target_type,
            forecast_horizon_hours=horizon_hours,
            is_synthetic=is_synthetic,
            provider_config={
                "provider_name": self.provider.provider_name,
                "timeout": getattr(self.provider, "timeout", None),
                "max_retries": getattr(self.provider, "max_retries", None),
            },
            target_generator_config={
                "target_type": self.target_generator.target_type,
                "version": self.target_generator.target_version,
                "forecast_horizon_hours": horizon_hours,
                "categories": getattr(self.target_generator, "CATEGORIES", []),
            },
            provenance_note=(
                "SYNTHETIC SAMPLE DATA: Generated solely for testing pipeline mechanics and CI verification."
                if is_synthetic
                else (
                    "ECMWF ERA5 / ERA5-Land atmospheric reanalysis retrieved via Open-Meteo Historical Weather API. "
                    "Blends numerical weather modeling with assimilated satellite and ground observations. "
                    "Data tier: reanalysis (not direct raw ground station gauges)."
                )
            ),
            ready_for_model_training=quality_report.validation_passed,
        )

        # 6. Export artifacts if enabled
        if export_artifacts:
            self._export_dataset_artifacts(
                dataset_name=dataset_name,
                X=X,
                y=y,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                metadata=metadata,
                quality_report=quality_report,
            )

        return X, y, metadata, quality_report

    def _export_dataset_artifacts(
        self,
        dataset_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata: DatasetMetadata,
        quality_report: DatasetQualityReport,
    ) -> None:
        """Save dataset partitions to Parquet & CSV with metadata manifests."""
        target_col = metadata.target_column
        out_dir = self.artifacts_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata manifest JSON
        manifest_path = out_dir / "metadata.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        # Save quality report JSON
        report_path = out_dir / "quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(quality_report.model_dump(), f, indent=2)

        # Export split partitions (including both 'validation' and 'val' for strict compliance)
        splits = {
            "train": train_df,
            "val": val_df,
            "validation": val_df,
            "test": test_df,
        }

        meta_cols = ["timestamp", "_location_id", "_target"]
        for split_name, split_data in splits.items():
            features_split = split_data.drop(columns=[c for c in meta_cols if c in split_data.columns])
            target_split = split_data["_target"].rename(target_col)
            combined_split = features_split.copy()
            combined_split[target_col] = target_split

            # Save Parquet (preferred columnar format)
            parquet_path = out_dir / f"{split_name}.parquet"
            try:
                combined_split.to_parquet(parquet_path, index=False)
            except Exception as e:
                logger.warning(f"Could not save parquet ({e}); saving CSV instead.")

            # Save CSV for human inspection
            csv_path = out_dir / f"{split_name}.csv"
            combined_split.to_csv(csv_path, index=False)

        logger.info(f"Dataset artifacts successfully exported to {out_dir}")
