"""
Pluggable ML Target / Label Generation for Heat Risk Prediction.

Defines scientifically defensible ground-truth formulations derived from physical
meteorological observations and authoritative human physiological thermal stress thresholds
(NOAA NWS Heat Index, Sherwood & Huber 2010, Vecellio et al. 2022, IMD heatwave criteria).

CRITICAL DESIGN PRINCIPLE:
The ML target is NEVER simply the output of the existing RuleBasedHeatModel.
That would create a model that merely learns to imitate programmatic heuristics.
Instead, the target represents verifiable, future atmospheric and physiological heat stress.

CANONICAL ML FORECASTING CONTRACT:
Given environmental information available at time t, predict the maximum heat-risk
severity occurring during the following 6 hours (t, t + 6h].
- INPUT: information available at time t
- TARGET: maximum heat-stress severity in (t, t + 6h]
- FORECAST HORIZON: 6 hours (configurable, H)
- TASK: ordinal multi-class classification (LOW, MODERATE, HIGH, SEVERE, CRITICAL)
- The target NEVER uses observations from t or earlier as target observations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from app.features.weather_features import calculate_heat_index, calculate_wet_bulb_temperature


class HeatRiskTargetGenerator(ABC):
    """Abstract interface for generating future heat risk labels from chronological observations."""

    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the target variable column."""
        pass

    @property
    @abstractmethod
    def target_version(self) -> str:
        """Traceable version of the target formulation."""
        pass

    @property
    @abstractmethod
    def target_type(self) -> Literal["classification", "regression"]:
        """Type of learning task: categorical or continuous."""
        pass

    @abstractmethod
    def generate_targets(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 6,
    ) -> pd.Series:
        """
        Generate target series aligned to df index.
        The target for row t represents thermal stress occurring strictly within the forward
        window (t, t + horizon_hours].
        """
        pass


class ThresholdBasedHeatTargetGenerator(HeatRiskTargetGenerator):
    """
    Scientifically grounded target generator using forward-looking NOAA Heat Index
    and physiological wet-bulb environmental heat-stress indicators.

    Scientific Justification & Physiological Context:
    -------------------------------------------------
    1. NOAA National Weather Service (NWS) Heat Index Thresholds:
       - Caution: 27°C - 32°C (80°F - 90°F)
       - Extreme Caution: 32°C - 41°C (90°F - 105°F) - heat exhaustion possible with prolonged exposure
       - Danger: 41°C - 54°C (105°F - 130°F) - heat exhaustion likely, heat stroke possible
       - Extreme Danger: >= 54°C (>= 130°F) - heat stroke imminent

    2. Wet-Bulb Temperature (Tw) Extreme Heat-Stress Escalation Indicators:
       - Tw >= 25.0°C: Moderate thermal strain benchmark; initial reduction in evaporative cooling efficiency.
       - Tw >= 27.5°C: High thermal strain benchmark; significant impairment of sweating efficiency during physical labor.
       - Tw >= 29.0°C: Severe environmental heat stress; substantial risk of heat exhaustion and occupational illness.
       - Tw >= 31.0°C: Critical population-level heat stress benchmark.
         Laboratory climate-chamber trials (Vecellio et al., 2022) demonstrate that under sustained physical exertion
         (metabolic rate ~150-200 W/m²), continuous evaporative heat exchange becomes uncompensable even for young,
         healthy individuals.
         IMPORTANT SCIENTIFIC DELINEATION:
         This threshold is an environmental hazard benchmark triggering municipal heat-action emergencies, outdoor labor
         suspension, and cooling shelter activation. It is NOT an absolute, universal physiological boundary claiming
         instantaneous collapse or mortality for every individual regardless of acclimatization, hydration, and shade.
         Instead, it functions as a critical escalation trigger in our population-level risk classification.

    Canonical Forecasting Contract:
    -------------------------------
    Given environmental information available at time t, predict the maximum heat-risk severity
    occurring during the following 6 hours (t, t + 6h].
    - INPUT: Information strictly available at or before time t
    - TARGET: Maximum heat-stress severity in (t, t + 6h]
    - FORECAST HORIZON: 6 hours (configurable, H)
    - TASK: Ordinal multi-class classification (LOW, MODERATE, HIGH, SEVERE, CRITICAL)
    - OBSERVATIONS: The target NEVER uses observations from t or earlier as target observations.
    """

    CATEGORIES = ["LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"]

    def __init__(
        self,
        target_type: Literal["classification", "regression"] = "classification",
        version: str = "heat-target-v1",
    ):
        self._target_type = target_type
        self._version = version

    @property
    def target_name(self) -> str:
        return "target_heat_risk_category" if self._target_type == "classification" else "target_heat_risk_score"

    @property
    def target_version(self) -> str:
        return self._version

    @property
    def target_type(self) -> Literal["classification", "regression"]:
        return self._target_type

    def _compute_instantaneous_heat_index(self, row: pd.Series) -> Optional[float]:
        t = row.get("temperature_c")
        rh = row.get("relative_humidity_pct")
        if pd.isna(t) or pd.isna(rh):
            return None
        return calculate_heat_index(float(t), float(rh))

    def _compute_instantaneous_wet_bulb(self, row: pd.Series) -> Optional[float]:
        t = row.get("temperature_c")
        rh = row.get("relative_humidity_pct")
        if pd.isna(t) or pd.isna(rh):
            return None
        return calculate_wet_bulb_temperature(float(t), float(rh))

    def map_metrics_to_category(self, hi: Optional[float], wb: Optional[float]) -> Optional[str]:
        """
        Map instantaneous or forward-maximum Heat Index and Wet-Bulb values to an ordinal category.
        Wet-bulb thresholds act as environmental escalation indicators.
        """
        if hi is None or wb is None or np.isnan(hi) or np.isnan(wb):
            return None

        # Critical escalation: Extreme Danger NOAA Heat Index or Critical Wet-Bulb limit
        if hi >= 54.0 or wb >= 31.0:
            return "CRITICAL"
        # Severe escalation: Danger NOAA Heat Index or Severe Wet-Bulb strain
        elif hi >= 46.0 or wb >= 29.0:
            return "SEVERE"
        # High escalation: Extreme Caution NOAA Heat Index or High Wet-Bulb strain
        elif hi >= 41.0 or wb >= 27.5:
            return "HIGH"
        # Moderate: Caution NOAA Heat Index or Moderate Wet-Bulb elevation
        elif hi >= 32.0 or wb >= 25.0:
            return "MODERATE"
        else:
            return "LOW"

    def map_metrics_to_score(self, hi: Optional[float], wb: Optional[float]) -> Optional[float]:
        """
        Map instantaneous or forward-maximum Heat Index and Wet-Bulb values to a 0-100 continuous score.
        """
        if hi is None or wb is None or np.isnan(hi) or np.isnan(wb):
            return None

        # Base score from Heat Index scaled between 25°C (baseline) and 55°C (extreme)
        base = ((hi - 25.0) / (55.0 - 25.0)) * 80.0
        # Additive penalty for severe wet-bulb stress above 27°C
        wb_penalty = max(0.0, (wb - 27.0) * 5.0) if wb >= 27.0 else 0.0

        score = float(np.clip(base + wb_penalty, 0.0, 100.0))
        return round(score, 2)

    def _generate_targets_single_series(
        self,
        df_location: pd.DataFrame,
        horizon_hours: int = 6,
    ) -> pd.Series:
        """
        Compute forward maximum heat stress across the next horizon_hours for a single location.
        Strictly forward-looking: index i uses elements [i + 1, i + horizon_hours].
        """
        df_sorted = df_location.sort_values("timestamp").copy()

        # Step 1: Compute instantaneous thermal indicators
        hi_series = df_sorted.apply(self._compute_instantaneous_heat_index, axis=1)
        wb_series = df_sorted.apply(self._compute_instantaneous_wet_bulb, axis=1)

        temp_df = pd.DataFrame({
            "heat_index": hi_series,
            "wet_bulb": wb_series,
        }, index=df_sorted.index)

        n_rows = len(temp_df)
        max_forward_hi = np.full(n_rows, np.nan)
        max_forward_wb = np.full(n_rows, np.nan)

        hi_vals = temp_df["heat_index"].to_numpy()
        wb_vals = temp_df["wet_bulb"].to_numpy()

        for i in range(n_rows):
            # Forward window starts at i + 1 up to i + horizon_hours + 1
            # Strictly excludes current timestamp t (at index i)
            start_fwd = i + 1
            end_fwd = min(n_rows, i + 1 + horizon_hours)
            if start_fwd < end_fwd:
                fwd_hi_slice = hi_vals[start_fwd:end_fwd]
                fwd_wb_slice = wb_vals[start_fwd:end_fwd]

                valid_hi = fwd_hi_slice[~np.isnan(fwd_hi_slice)]
                valid_wb = fwd_wb_slice[~np.isnan(fwd_wb_slice)]

                # Require that we have at least 1 valid forward observation
                if len(valid_hi) > 0:
                    max_forward_hi[i] = np.max(valid_hi)
                if len(valid_wb) > 0:
                    max_forward_wb[i] = np.max(valid_wb)

        # Step 2: Map forward maximums to target formulation
        if self._target_type == "classification":
            targets = [
                self.map_metrics_to_category(hi, wb)
                for hi, wb in zip(max_forward_hi, max_forward_wb)
            ]
            return pd.Series(targets, index=df_sorted.index, name=self.target_name)
        else:
            scores = [
                self.map_metrics_to_score(hi, wb)
                for hi, wb in zip(max_forward_hi, max_forward_wb)
            ]
            return pd.Series(scores, index=df_sorted.index, name=self.target_name)

    def generate_targets(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 6,
    ) -> pd.Series:
        """
        Generate forward targets. If df contains multiple locations, forward windows
        are strictly computed per location to prevent cross-location contamination.
        """
        if "location_id" in df.columns and df["location_id"].nunique() > 1:
            series_list = []
            for loc_id, group in df.groupby("location_id"):
                loc_series = self._generate_targets_single_series(group, horizon_hours=horizon_hours)
                series_list.append(loc_series)
            combined = pd.concat(series_list).reindex(df.index)
            combined.name = self.target_name
            return combined
        else:
            return self._generate_targets_single_series(df, horizon_hours=horizon_hours)
