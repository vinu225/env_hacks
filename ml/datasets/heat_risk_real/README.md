# Real Historical Heat Risk Dataset (`heat_risk_real`)

## 1. Executive Summary & Provenance
* **Dataset Name**: `heat_risk_real`
* **Dataset Version**: `v1.0`
* **Data Version**: `v1.0`
* **Feature Version**: `weather-features-v1`
* **Target Version**: `heat-target-v1`
* **Source**: ECMWF ERA5 / ERA5-Land Atmospheric Reanalysis via Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`)
* **Data Tier**: `reanalysis` (Assimilated physical numerical weather models with integrated global satellite telemetry and ground observations; **not** raw uncalibrated station gauges)
* **Creation Timestamp**: `2026-10-06T14:49:02Z`
* **Total Rows**: `175,430`
* **Total Features**: `29` numeric predictors

---

## 2. Geographic Coverage
Configurable registry of 10 canonical Indian urban centers spanning diverse Köppen-Geiger / IMD climatic zones:

| Location ID | City | State / Region | Latitude | Longitude | Climate Zone | Elevation (m) | Row Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `loc_delhi` | Delhi | Delhi | 28.6139 | 77.2090 | Semi-Arid (Hot Subtropical) | 216.0 | 17,543 |
| `loc_bhopal` | Bhopal | Madhya Pradesh | 23.2599 | 77.4126 | Humid Subtropical | 527.0 | 17,543 |
| `loc_jaipur` | Jaipur | Rajasthan | 26.9124 | 75.7873 | Hot Semi-Arid | 431.0 | 17,543 |
| `loc_nagpur` | Nagpur | Maharashtra | 21.1458 | 79.0882 | Tropical Wet and Dry | 310.0 | 17,543 |
| `loc_lucknow` | Lucknow | Uttar Pradesh | 26.8467 | 80.9462 | Humid Subtropical (Gangetic Plain) | 123.0 | 17,543 |
| `loc_ahmedabad` | Ahmedabad | Gujarat | 23.0225 | 72.5714 | Hot Semi-Arid | 53.0 | 17,543 |
| `loc_hyderabad` | Hyderabad | Telangana | 17.3850 | 78.4867 | Tropical Wet and Dry | 542.0 | 17,543 |
| `loc_chennai` | Chennai | Tamil Nadu | 13.0827 | 80.2707 | Tropical Wet and Dry (Coastal) | 6.0 | 17,543 |
| `loc_mumbai` | Mumbai | Maharashtra | 19.0760 | 72.8777 | Tropical Monsoon (Coastal) | 14.0 | 17,543 |
| `loc_bengaluru` | Bengaluru | Karnataka | 12.9716 | 77.5946 | Tropical Savanna (Plateau) | 920.0 | 17,543 |

---

## 3. Temporal Coverage & Resolution
* **Temporal Resolution**: Hourly (`1h` frequency)
* **Start Timestamp**: `2023-01-01T00:00:00+00:00`
* **End Timestamp**: `2024-12-31T22:00:00+00:00`
* **Historical Span**: 2 full calendar years (731 consecutive days)
* **Yearly Distribution**:
  * `2023`: 87,600 rows (49.93%)
  * `2024`: 87,830 rows (50.07%, leap year)
* **Seasonal Distribution (IMD Criteria)**:
  * Monsoon (Jun–Sep): 58,560 rows (33.38%)
  * Pre-Monsoon / Summer (Mar–May): 44,160 rows (25.17%)
  * Post-Monsoon (Oct–Dec): 44,150 rows (25.17%)
  * Winter (Jan–Feb): 28,560 rows (16.28%)

---

## 4. Canonical Forecasting Contract & Target Formulation
* **Canonical ML Problem**: Given environmental information available at prediction time $t$, predict the maximum heat-risk severity occurring during the following 6 hours $(t, t+6\text{h}]$.
* **Input Window**: Information strictly available at or before time $t$
* **Target Window**: Strictly forward $(t, t + 6\text{h}]$
* **Target Horizon**: $H = 6$ hours (configurable)
* **Task Type**: Ordinal multi-class classification
* **Classes**: `LOW`, `MODERATE`, `HIGH`, `SEVERE`, `CRITICAL`
* **Target Column**: `target_heat_risk_category`
* **Target Grounding**:
  * `CRITICAL`: Forward $\max(HI) \ge 54.0^\circ\text{C}$ OR Forward $\max(T_w) \ge 31.0^\circ\text{C}$
  * `SEVERE`: Forward $\max(HI) \ge 46.0^\circ\text{C}$ OR Forward $\max(T_w) \ge 29.0^\circ\text{C}$
  * `HIGH`: Forward $\max(HI) \ge 41.0^\circ\text{C}$ OR Forward $\max(T_w) \ge 27.5^\circ\text{C}$
  * `MODERATE`: Forward $\max(HI) \ge 32.0^\circ\text{C}$ OR Forward $\max(T_w) \ge 25.0^\circ\text{C}$
  * `LOW`: Forward $\max(HI) < 32.0^\circ\text{C}$ AND Forward $\max(T_w) < 25.0^\circ\text{C}$

### Scientific Refinement of Wet-Bulb ($T_w$) Boundary
* $T_w \ge 31.0^\circ\text{C}$ represents an uncompensable environmental heat-stress indicator under sustained physical activity (Vecellio et al., 2022 laboratory climatic chamber trials).
* It is **not** an absolute universal claim of instantaneous personal biological mortality across all demographics.
* It operates as a population-level environmental hazard escalation benchmark triggering municipal emergency action.

---

## 5. Target Distribution & Class Imbalance Audit
Calculated across all 175,430 rows of real historical data:

| Target Class | Count | Percentage | Operational Significance |
| :--- | :--- | :--- | :--- |
| **LOW** | 105,913 | 60.37% | Safe ambient conditions; typical seasonal baseline |
| **MODERATE** | 60,332 | 34.39% | NOAA Extreme Caution; prolonged exertion risk |
| **HIGH** | 8,569 | 4.88% | NOAA Danger zone; municipal alerts advised |
| **SEVERE** | 616 | 0.35% | Severe heatwave; emergency response required |
| **CRITICAL** | 0 | 0.00% | Uncompensable thermal threshold (none observed in 2023-24 reanalysis) |
| **Total** | 175,430 | 100.00% | Zero missing target rows |

> **Audit Note on Imbalance**: The dataset reflects real physical atmospheric distributions. Severe heatwaves represent rare extreme tail events (~0.35%). In accordance with Step 3B rules, no synthetic oversampling (SMOTE) or class artificial rebalancing has been applied.

---

## 6. Chronological Partitioning (No Shuffling)
Strict temporal splitting ensures zero lookahead contamination:

| Split Partition | Timestamp Start (UTC) | Timestamp End (UTC) | Row Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `2023-01-01T00:00:00+00:00` | `2024-05-26T15:00:00+00:00` | 122,800 | 70.00% |
| **Validation** | `2024-05-26T16:00:00+00:00` | `2024-09-13T06:00:00+00:00` | 26,310 | 15.00% |
| **Test** | `2024-09-13T07:00:00+00:00` | `2024-12-31T22:00:00+00:00` | 26,320 | 15.00% |

### Verification of Boundary Isolation:
$$\max(\text{train\_timestamp}) = \text{2024-05-26T15:00:00Z} < \min(\text{val\_timestamp}) = \text{2024-05-26T16:00:00Z}$$
$$\max(\text{val\_timestamp}) = \text{2024-09-13T06:00:00Z} < \min(\text{test\_timestamp}) = \text{2024-09-13T07:00:00Z}$$

---

## 7. Feature Availability & Scientific Contract Audit
All 29 features in $X$ are strictly available at prediction time $t$ using backward-looking windows $[t - 24\text{h}, t]$. Zero forward observations are used in $X$.

### Scientific Decision on Surface Pressure (Option A - Recommended):
Surface pressure is excluded from the V1 heat-risk feature set. Missing barometric pressure is **never** silently imputed to 1013.25 hPa. Thermodynamic variables ($T$, $RH$, $T_d$, $HI$, $T_w$, Wind) completely capture thermal and evaporative heat load.

---

## 8. Persistence Baseline Benchmark (No ML Trained)
Evaluation-only heuristic: $\hat{y}(t, t+6\text{h}] = y_{\text{instantaneous}}(t)$.

* **Overall Accuracy**: 82.82%
* **Macro F1-Score**: 0.5014
* **Recall on HIGH (Severity 2)**: 39.61%
* **Recall on SEVERE (Severity 3)**: 25.16%
* **Total Dynamic Transitions**: 30,144 (17.18% of timesteps)
* **Upward Escalations Missed by Persistence**: 26,149 instances (14.91% miss rate)
* **Core Takeaway**: While persistence achieves high baseline accuracy due to class skew toward LOW, it fails catastrophically on rapid diurnal heating and heatwave intensification (missing 74.84% of SEVERE events). A forward-looking ML model must capture these transitions.

---

## 9. Reproducibility Instructions
To regenerate this identical dataset from Open-Meteo ERA5 archives:
```bash
python ml/pipelines/generate_real_dataset.py
```
This fetches all 10 locations, runs the shared production feature engineering pipeline, computes forward 6-hour labels, performs temporal splitting, and exports Parquet/CSV artifacts.

---

## 10. Audit Decision

```text
READY_FOR_MODEL_TRAINING = TRUE
```
The dataset is clean, scientifically grounded, strictly chronological, leakage-free, and fully audited.
