# Machine Learning Architecture & Lifecycle Guide

This document specifies the end-to-end Machine Learning (ML) architecture for the Environmental Intelligence Platform, detailing how the real historical dataset pipeline, feature engineering, forward target generation, validation, and future ML model serving integrate into the existing modular monolith.

---

## 1. ML Problem Definition: Hyperlocal Heat Risk Forecasting

* **Canonical Problem**: Given environmental information available at prediction time $t$, predict the maximum heat-risk severity occurring during the following 6 hours $(t, t+6\text{h}]$.
* **Task Type**: Ordinal Classification (Primary: `LOW`, `MODERATE`, `HIGH`, `SEVERE`, `CRITICAL`) and Continuous Regression (Secondary: $0.0 - 100.0$ heat stress score).
* **Input Window**: Multi-variable environmental conditions known strictly at or before timestamp $t$.
* **Target Output**: Verifiable future thermal stress occurring strictly within $(t, t + 6\text{h}]$, grounded in authoritative NOAA Heat Index thresholds and calibrated wet-bulb physiological strain benchmarks (Vecellio et al. 2022).
* **Core Principle**: The ML target is **NEVER** trained on the outputs of the rule-based baseline model (`RuleBasedHeatModel`). That would merely learn programmatic heuristics. The target represents verifiable forward physical conditions.

---

## 2. Real Historical Dataset Ingestion Architecture

Historical observation ingestion is abstracted through [`HistoricalWeatherProvider`](file:///c:/Users/bilal/OneDrive/Desktop/New%20folder/ml/pipelines/provider.py#L22):

```text
                       [ HistoricalWeatherProvider (ABC) ]
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
[ OpenMeteoArchiveHistoricalProvider ] [ LocalFileHistoricalProvider ] [ SyntheticHistoricalWeatherProvider ]
ECMWF ERA5 Reanalysis (Real Training)  IMD / Pre-downloaded Parquet    Deterministic CI Testing ONLY
```

### Real Historical Dataset (`heat_risk_real`) Specifications:
* **Source**: `open-meteo-era5` (ECMWF ERA5 / ERA5-Land atmospheric reanalysis)
* **Data Tier**: `reanalysis` (Physical numerical models with assimilated satellite and ground observations; clearly distinguished from raw station gauges and forecasts)
* **Coverage**: 10 Canonical Indian Cities (Delhi, Bhopal, Jaipur, Nagpur, Lucknow, Ahmedabad, Hyderabad, Chennai, Mumbai, Bengaluru)
* **Period**: 2023-01-01T00:00:00Z to 2024-12-31T22:00:00Z (2 full calendar years)
* **Resolution**: Hourly (`1h` frequency)
* **Total Rows**: 175,430 rows (zero synthetic data used in production dataset)

---

## 3. Unified Feature Pipeline: Eliminating Training-Serving Skew

To eliminate training-serving skew, the identical feature engineering implementation in [`backend/app/features/`](file:///c:/Users/bilal/OneDrive/Desktop/New%20folder/backend/app/features/) serves both offline training and online live serving:

```text
[ OFFLINE HISTORICAL REANALYSIS ]               [ ONLINE REAL-TIME TELEMETRY ]
Open-Meteo ERA5 Historical Archive              Live Open-Meteo / Station Telemetry
               │                                               │
               ▼                                               ▼
[ HistoricalObservation Records ]              [ NormalizedWeatherObservation ]
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                         [ backend/app/features/ ]
                         - weather_features.py (Dew point, Heat index, Wet-bulb)
                         - heat_features.py (High temp, humidity stress, anomalies)
                         - water_features.py (7d/24h rain, dry days, deficit %)
                         - rainfall_features.py (hourly rain, heavy downpour)
                         - rolling.py (pure time-window aggregations)
                         - builder.py (FeatureVectorBuilder)
                                       │
                                       ▼
                       [ EnvironmentalFeatureVector ]
                         (data_version: "v1.0")
                         (feature_version: "weather-features-v1")
                                       │
                ┌──────────────────────┴───────────────────────┐
                ▼                                              ▼
     [ ML Training Matrix ]                        [ Online Model Serving ]
     df = pd.DataFrame([                           x = vector.to_numeric_dict()
         v.to_numeric_dict()                       prediction = model.predict(x)
         for v in historical_vectors
     ])
```

---

## 4. Scientific Target Generation & Wet-Bulb Escalation

Target generation is managed by [`ThresholdBasedHeatTargetGenerator`](file:///c:/Users/bilal/OneDrive/Desktop/New%20folder/ml/pipelines/target.py#L66):

1. **NOAA National Weather Service (NWS) Criteria**:
   * Normal: $< 27^\circ\text{C}$
   * Caution: $27^\circ\text{C} - 32^\circ\text{C}$
   * Extreme Caution: $32^\circ\text{C} - 41^\circ\text{C}$
   * Danger: $41^\circ\text{C} - 54^\circ\text{C}$
   * Extreme Danger: $\ge 54^\circ\text{C}$
2. **Wet-Bulb Temperature ($T_w$) Escalation Limits**:
   * $T_w \ge 25.0^\circ\text{C}$: Moderate evaporative impairment
   * $T_w \ge 27.5^\circ\text{C}$: High thermal strain during exertion
   * $T_w \ge 29.0^\circ\text{C}$: Severe occupational strain
   * $T_w \ge 31.0^\circ\text{C}$: Critical population-level environmental hazard benchmark (Vecellio et al. 2022). Uncompensable under physical labor; triggers municipal heat action emergency.
3. **Forward Horizon Contract**:
   Target evaluates strictly $(t, t + 6\text{h}]$, completely decoupled from the lagging feature window $[t - 24\text{h}, t]$.

---

## 5. Temporal Splitting Architecture

Random splitting (`shuffle=True`) is strictly prohibited. Chronological partitioning splits on unique timestamps:

$$\max(\text{train\_timestamp}) < \min(\text{val\_timestamp}) \le \max(\text{val\_timestamp}) < \min(\text{test\_timestamp})$$

* **Train Split (70%)**: 2023-01-01T00:00:00Z to 2024-05-26T15:00:00Z (122,800 rows)
* **Validation Split (15%)**: 2024-05-26T16:00:00Z to 2024-09-13T06:00:00Z (26,310 rows)
* **Test Split (15%)**: 2024-09-13T07:00:00Z to 2024-12-31T22:00:00Z (26,320 rows)

---

## 6. Evaluation Benchmark: Persistence Baseline (Zero ML)

The [`PersistenceHeatBaseline`](file:///c:/Users/bilal/OneDrive/Desktop/New%20folder/ml/baselines/persistence.py) provides the non-learned operational benchmark:

$$\hat{y}(t, t+6\text{h}] = y_{\text{instantaneous}}(t)$$

* **Overall Accuracy**: 82.82%
* **Macro F1-Score**: 0.5014
* **Transitions Missed**: 30,144 (17.18% of timesteps)
* **Upward Escalations Missed**: 26,149 instances (Persistence fails to foresee daytime heating)
* **Purpose**: Future Step 4 ML models must achieve superior recall on upward heat escalations compared to this baseline.

---

## 7. Artifact Manifests & Training Readiness

Dataset stored at `ml/datasets/heat_risk_real/`:
* `metadata.json` (traceable data/feature/target versions)
* `quality_report.json` (validation report with 0 issues)
* `train.parquet`, `validation.parquet`, `test.parquet` (and CSV inspection copies)
* `README.md`: Explicitly certifies `READY_FOR_MODEL_TRAINING = TRUE`.

NO ML MODEL WAS TRAINED IN STEP 3B.
Model training is reserved for Step 4 after formal user review.
