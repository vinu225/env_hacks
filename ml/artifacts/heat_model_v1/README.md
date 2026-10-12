# Heat Risk Model Artifact (heat_model_v1)

## 1. Overview
- **Model Type**: LightGBM Multi-Class Classifier (`LGBMClassifier`)
- **Objective**: Predict maximum atmospheric heat-risk severity occurring during the forward window $(t, t+6\text{h}]$
- **Dataset Source**: ECMWF ERA5 / ERA5-Land Reanalysis across 10 Indian cities (2023–2024)
- **Model Version**: `heat_model_v1.0.0`
- **Trained Date**: 2026-10-11T11:29:11Z
- **Selected Configuration**: `config_a_balanced_standard`

## 2. Target Classes & Empirical Coverage
- Empirical Classes: `LOW`, `MODERATE`, `HIGH`, `SEVERE`
- `CRITICAL` Class Handling: **Deterministic Safety Override Layer**
  - The 2023–2024 historical dataset contains zero empirical timestamps reaching CRITICAL severity ($HI \ge 54^\circ\text{C}$ or $T_w \ge 31^\circ\text{C}$).
  - Rather than fabricating labels, an unsuppressible rule-based safety override triggers `CRITICAL` if live observations exceed physiological survivability thresholds.

## 3. Key Performance Highlights (Validation Split)
- **LightGBM Macro-F1**: 0.5124 (vs Persistence: 0.6201, Rule-Based: 0.4105)
- **HIGH Heat Recall**: 0.5989 (vs Persistence: 0.4143)
- **SEVERE Heat Recall**: 0.5692 (vs Persistence: 0.2738)
- **Severe False Negative Count**: 140 (vs Persistence: 236)

## 4. Test Split Limitations
- The test split (Sep 13 – Dec 31, 2024) covers autumn/winter in India and contains **0 positive examples of SEVERE heat**.
- In accordance with rigorous scientific reporting, SEVERE recall on the test partition is documented as **undefined**, not 0.0.

## 5. Latency Benchmark
- Single sample prediction: mean 0.8546 ms, p95 0.8537 ms
- Fast CPU inference suitable for sub-10ms FastAPI response latency.
