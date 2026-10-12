"""
Validation, Leakage Auditing, and Data Quality Reporting for ML Datasets.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ml.pipelines.schema import DatasetQualityReport


def validate_chronological_integrity(
    df: pd.DataFrame, time_col: str = "timestamp"
) -> Tuple[bool, List[str]]:
    """
    Ensure timestamps are strictly monotonic and contain no duplicates for identical locations.
    """
    issues: List[str] = []
    if time_col not in df.columns:
        return False, [f"Missing required time column: '{time_col}'"]

    # Check for NaNs in timestamp
    if df[time_col].isna().any():
        issues.append(f"Found {df[time_col].isna().sum()} null values in timestamp column.")

    # Check duplicates
    group_cols = ["location_id", time_col] if "location_id" in df.columns else [time_col]
    dups = df.duplicated(subset=group_cols, keep=False)
    dup_count = int(dups.sum())
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate timestamp records for location.")

    # Check monotonic order per location if location_id is present
    if "location_id" in df.columns and df["location_id"].nunique() > 1:
        for loc_id, group in df.groupby("location_id"):
            if not group[time_col].is_monotonic_increasing:
                issues.append(f"Timestamps for location '{loc_id}' in '{time_col}' are not strictly in chronological order.")
                break
    else:
        is_sorted = df[time_col].is_monotonic_increasing
        if not is_sorted:
            issues.append(f"Timestamps in '{time_col}' are not strictly in chronological order.")

    return len(issues) == 0, issues


def detect_target_leakage(
    feature_df: pd.DataFrame,
    target_column_name: str,
    prohibited_substrings: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Assert that the target variable or forward-looking horizon indicators
    do NOT exist inside the feature matrix X.
    """
    issues: List[str] = []
    forbidden = prohibited_substrings or [
        "target",
        "forward",
        "future",
        "ground_truth",
        target_column_name.lower(),
    ]

    for col in feature_df.columns:
        col_lower = str(col).lower()
        if col_lower == target_column_name.lower():
            issues.append(f"FATAL LEAKAGE: Target column '{col}' is present inside feature matrix X!")
        for pattern in forbidden:
            if pattern in col_lower and col_lower != "forecast_precipitation_6h_mm" and col_lower != "forecast_precipitation_mm":
                issues.append(f"SUSPECT LEAKAGE: Feature column '{col}' contains forbidden keyword '{pattern}'.")

    return len(issues) == 0, issues


def validate_numeric_boundaries(
    df: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """
    Verify physical plausibility bounds across known environmental features.
    """
    issues: List[str] = []

    # Ambient Temperature bounds [-60°C, 65°C]
    ambient_temp_cols = [c for c in df.columns if c in ["temperature_c", "current_temperature_c", "dew_point_c", "wet_bulb_temperature_c"]]
    for col in ambient_temp_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if (series < -60.0).any() or (series > 65.0).any():
            outliers = series[(series < -60.0) | (series > 65.0)]
            issues.append(f"Physically implausible ambient temperature in '{col}': {len(outliers)} values outside [-60, 65°C].")

    # Apparent / Heat index bounds [-60°C, 85°C] (accounting for extreme humid heat index calculations)
    apparent_cols = [c for c in df.columns if "apparent" in c.lower() or "heat_index" in c.lower()]
    for col in apparent_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if (series < -60.0).any() or (series > 85.0).any():
            outliers = series[(series < -60.0) | (series > 85.0)]
            issues.append(f"Physically implausible apparent temperature in '{col}': {len(outliers)} values outside [-60, 85°C].")

    # Humidity bounds [0%, 100%]
    rh_cols = [c for c in df.columns if "humidity" in c.lower()]
    for col in rh_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if (series < 0.0).any() or (series > 100.0).any():
            issues.append(f"Relative humidity in '{col}' contains values outside valid range [0, 100%].")

    # Precipitation non-negativity
    precip_cols = [c for c in df.columns if "precip" in c.lower() or "rain" in c.lower()]
    for col in precip_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if (series < 0.0).any():
            issues.append(f"Precipitation in '{col}' contains impossible negative values.")

    # Wind speed non-negativity
    wind_cols = [c for c in df.columns if "wind" in c.lower()]
    for col in wind_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if (series < 0.0).any():
            issues.append(f"Wind speed in '{col}' contains impossible negative values.")

    return len(issues) == 0, issues


def generate_dataset_quality_report(
    feature_df: pd.DataFrame,
    target_series: pd.Series,
    time_series: Optional[pd.Series] = None,
    location_series: Optional[pd.Series] = None,
    dropped_rows_summary: Optional[Dict[str, int]] = None,
    temporal_split_timestamps: Optional[Dict[str, Dict[str, str]]] = None,
) -> DatasetQualityReport:
    """
    Produce structured DatasetQualityReport evaluating completeness, target distribution,
    missingness, and physical validity.
    """
    total_rows = len(feature_df)
    location_count = int(location_series.nunique()) if location_series is not None else 1
    loc_dist: Dict[str, int] = {}
    if location_series is not None:
        loc_dist = {str(k): int(v) for k, v in location_series.value_counts().to_dict().items()}

    start_ts = None
    end_ts = None
    if time_series is not None and len(time_series) > 0:
        time_clean = pd.to_datetime(time_series).dropna().sort_values()
        if len(time_clean) > 0:
            start_ts = time_clean.iloc[0].isoformat()
            end_ts = time_clean.iloc[-1].isoformat()

    # Missing percentage
    missing_pct: Dict[str, float] = {}
    availability: Dict[str, bool] = {}
    for col in feature_df.columns:
        null_count = int(feature_df[col].isna().sum())
        pct = round((null_count / max(1, total_rows)) * 100.0, 2)
        missing_pct[col] = pct
        availability[col] = pct < 100.0

    # Target distribution
    target_clean = target_series.dropna()
    if pd.api.types.is_numeric_dtype(target_clean):
        target_dist: Dict[str, Any] = {
            "type": "numeric",
            "mean": float(round(target_clean.mean(), 2)),
            "std": float(round(target_clean.std(), 2)),
            "min": float(round(target_clean.min(), 2)),
            "max": float(round(target_clean.max(), 2)),
            "missing_count": int(target_series.isna().sum()),
        }
    else:
        counts = target_clean.value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in counts.items()}
        target_dist["_missing_count"] = int(target_series.isna().sum())

    # Target percentages
    target_pcts: Dict[str, float] = {}
    if not pd.api.types.is_numeric_dtype(target_clean) and total_rows > 0:
        for k, v in counts.items():
            target_pcts[str(k)] = round((v / total_rows) * 100.0, 2)

    # Temporal breakdowns if time_series is provided
    yearly_dist: Dict[str, int] = {}
    monthly_dist: Dict[str, int] = {}
    seasonal_dist: Dict[str, int] = {}
    heat_events_summary: Dict[str, Any] = {}
    target_predictability: Dict[str, Any] = {}

    if time_series is not None and len(time_series) > 0:
        dt_series = pd.to_datetime(time_series, utc=True)
        # Year distribution
        yearly_counts = dt_series.dt.year.value_counts().sort_index().to_dict()
        yearly_dist = {str(k): int(v) for k, v in yearly_counts.items()}

        # Month distribution
        month_counts = dt_series.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
        monthly_dist = {str(k): int(v) for k, v in month_counts.items()}

        # IMD Meteorological Seasons:
        # Winter: Jan-Feb (months 1, 2)
        # Pre-Monsoon / Summer: Mar-May (months 3, 4, 5)
        # Monsoon (Southwest): Jun-Sep (months 6, 7, 8, 9)
        # Post-Monsoon / Autumn: Oct-Dec (months 10, 11, 12)
        def map_season(m: int) -> str:
            if m in [1, 2]:
                return "Winter (Jan-Feb)"
            elif m in [3, 4, 5]:
                return "Pre-Monsoon/Summer (Mar-May)"
            elif m in [6, 7, 8, 9]:
                return "Monsoon (Jun-Sep)"
            else:
                return "Post-Monsoon (Oct-Dec)"

        seasons = dt_series.dt.month.map(map_season)
        seasonal_counts = seasons.value_counts().to_dict()
        seasonal_dist = {str(k): int(v) for k, v in seasonal_counts.items()}

        # Heat Event Duration Analysis
        # Defined as contiguous episodes where target is HIGH, SEVERE, or CRITICAL
        high_risk_flags = target_series.isin(["HIGH", "SEVERE", "CRITICAL"]).astype(int)
        event_durations: List[int] = []
        events_by_loc: Dict[str, int] = {}
        events_by_year: Dict[str, int] = {}

        temp_df = pd.DataFrame({
            "loc": location_series if location_series is not None else "all",
            "year": dt_series.dt.year,
            "high": high_risk_flags,
        })

        for loc_id, group in temp_df.groupby("loc"):
            curr_len = 0
            loc_events = 0
            for idx, r in group.iterrows():
                if r["high"] == 1:
                    curr_len += 1
                else:
                    if curr_len > 0:
                        event_durations.append(curr_len)
                        loc_events += 1
                        yr = str(r["year"])
                        events_by_year[yr] = events_by_year.get(yr, 0) + 1
                        curr_len = 0
            if curr_len > 0:
                event_durations.append(curr_len)
                loc_events += 1
                events_by_loc[str(loc_id)] = loc_events
            else:
                events_by_loc[str(loc_id)] = loc_events

        heat_events_summary = {
            "total_heat_events": len(event_durations),
            "mean_duration_hours": float(round(np.mean(event_durations), 2)) if event_durations else 0.0,
            "max_duration_hours": int(np.max(event_durations)) if event_durations else 0,
            "events_per_location": events_by_loc,
            "events_per_year": events_by_year,
        }

    # Baseline target predictability / proximity check
    # Check whether instantaneous heat index at t perfectly determines future 6-hour risk
    if "heat_index_c" in feature_df.columns:
        hi_col = pd.to_numeric(feature_df["heat_index_c"], errors="coerce")
        # Instantaneous proxy category at t
        def hi_to_cat(val: Optional[float]) -> str:
            if pd.isna(val):
                return "UNKNOWN"
            if val >= 54.0:
                return "CRITICAL"
            elif val >= 46.0:
                return "SEVERE"
            elif val >= 41.0:
                return "HIGH"
            elif val >= 32.0:
                return "MODERATE"
            else:
                return "LOW"

        current_proxies = hi_col.map(hi_to_cat)
        exact_matches = int((current_proxies == target_series).sum())
        match_rate = round((exact_matches / max(1, total_rows)) * 100.0, 2)
        target_predictability = {
            "current_heat_index_vs_future_target_match_pct": match_rate,
            "target_proximity_note": (
                f"Current heat index determines {match_rate}% of the forward 6-hour target label. "
                f"The remaining {round(100.0 - match_rate, 2)}% of cases represent dynamic transitions "
                f"(diurnal rises, wet-bulb escalations, or nocturnal cooling) where forward forecasting is essential."
            ),
        }

    # Validation checks
    all_issues: List[str] = []
    
    # Chronological check if timestamp available
    dup_count = 0
    if time_series is not None:
        check_df = pd.DataFrame({"timestamp": time_series})
        if location_series is not None:
            check_df["location_id"] = location_series
        chrono_ok, chrono_issues = validate_chronological_integrity(check_df)
        if not chrono_ok:
            all_issues.extend(chrono_issues)
        dup_count = int(check_df.duplicated().sum())

    # Leakage check
    leak_ok, leak_issues = detect_target_leakage(feature_df, target_series.name or "target")
    if not leak_ok:
        all_issues.extend(leak_issues)

    # Numeric boundary check
    bound_ok, bound_issues = validate_numeric_boundaries(feature_df)
    if not bound_ok:
        all_issues.extend(bound_issues)

    invalid_val_count = len(bound_issues)

    return DatasetQualityReport(
        dataset_version="heat-risk-dataset-v1",
        source="open-meteo-era5",
        data_type="reanalysis",
        total_rows=total_rows,
        location_count=location_count,
        temporal_frequency="hourly",
        observation_reanalysis_distinction=(
            "ECMWF ERA5 / ERA5-Land reanalysis via Open-Meteo Archive API "
            "(assimilated numerical models + satellite/ground observations, not raw station gauge)"
        ),
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        missing_percentage_by_column=missing_pct,
        feature_availability=availability,
        target_distribution=target_dist,
        target_percentages=target_pcts,
        location_distribution=loc_dist,
        observations_per_location=loc_dist,
        dropped_rows_summary=dropped_rows_summary or {},
        temporal_split_timestamps=temporal_split_timestamps or {},
        yearly_distribution=yearly_dist,
        monthly_distribution=monthly_dist,
        seasonal_distribution=seasonal_dist,
        heat_events_summary=heat_events_summary,
        persistence_baseline_summary={},
        target_predictability_check=target_predictability,
        duplicate_timestamp_count=dup_count,
        invalid_value_count=invalid_val_count,
        validation_passed=(len(all_issues) == 0),
        issues=all_issues,
    )
