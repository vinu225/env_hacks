# Machine Learning Subsystem (ml/)

Production-grade Machine Learning pipeline for hyperlocal atmospheric heat-risk forecasting across Indian metropolitan centers, built for the **AWS Bharat Builds Tour — Track 02 (Heat & Water)**.

This subsystem provides end-to-end data pipelines, time-series feature engineering, historical ERA5 reanalysis ingestion, reproducible model training, multi-baseline benchmarking, and exported production artifacts for sub-millisecond API inference.

---

## 1. Subsystem Architecture

```
ml/
├── artifacts/
│   └── heat_model_v1/                  # Serialized production model & audit manifests
│       ├── model.joblib                 # Compressed LightGBM Classifier + LabelEncoder
│       ├── model.txt                    # Raw LightGBM booster dump for multi-language inspection
│       ├── feature_schema.json          # Strict 14-feature canonical ordering and dtypes
│       ├── evaluation_report.json       # Backtest metrics, confusion matrices, Brier scores
│       ├── training_metadata.json       # Hyperparameters, split timestamps, training duration
│       └── README.md                    # Artifact metadata summary
├── baselines/
│   ├── persistence.py                   # $t \to t+6\text{h}$ Persistence baseline evaluator
│   └── __init__.py
├── datasets/
│   ├── heat_risk_real/                  # Hourly ERA5 2023–2024 partitions across 10 Indian cities
│   │   ├── train.csv / .parquet         # 2023-01-01 to 2024-05-15 (120,380 rows)
│   │   ├── val.csv / .parquet           # 2024-05-15 to 2024-09-12 (28,800 rows — peak summer)
│   │   ├── test.csv / .parquet          # 2024-09-13 to 2024-12-31 (26,380 rows — out-of-time)
│   │   ├── metadata.json                # Partition summaries, coordinates, row counts
│   │   ├── quality_report.json          # Missing value audit, freshness validation, unit checks
│   │   └── README.md                    # Dataset lineage documentation
│   └── sample_heat_pipeline_data/       # Minimal deterministic fixtures for unit testing
├── pipelines/
│   ├── dataset_generator.py             # Historical weather fetching, feature extraction & labeling
│   ├── generate_real_dataset.py         # Entrypoint to build ERA5 real dataset partitions
│   ├── generate_sample_dataset.py       # Entrypoint to build small test fixtures
│   ├── locations.py                     # 10 monitored Indian urban locations definition
│   ├── provider.py                      # Open-Meteo Historical Archive API client
│   ├── schema.py                        # Dataset schema validation and type enforcement
│   ├── splitting.py                     # Time-based splitting with 60-row chronological embargo
│   ├── target.py                        # 6-hour forward-looking heat severity target generator
│   └── validation.py                    # Physical bounds, NaN, monotonicity, and leakage auditor
├── training/
│   └── train_heat_model.py              # End-to-end LightGBM training, calibration & export
├── evaluation/                          # Extended backtesting and calibration tools
├── notebooks/                           # Exploratory data analysis (EDA) & feature experiments
└── README.md
```

---

## 2. Canonical 14-Feature Contract

The model relies on a strict 14-feature canonical contract matching the capabilities of the backend's real-time `FeatureVectorBuilder`:

| # | Feature Name | Physical Unit | Physical Valid Range | Description |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `temperature_c` | °C | $[-60.0, 65.0]$ | 2m ambient dry-bulb temperature |
| 2 | `apparent_temperature_c` | °C | $[-70.0, 75.0]$ | Perceived physiological temperature |
| 3 | `relative_humidity_pct` | % | $[0.0, 100.0]$ | Relative atmospheric moisture content |
| 4 | `dew_point_c` | °C | $[-70.0, 45.0]$ | Atmospheric saturation temperature |
| 5 | `heat_index_c` | °C | $[-20.0, 85.0]$ | Rothfusz / NOAA physiological heat index |
| 6 | `wet_bulb_temperature_c` | °C | $[-30.0, 45.0]$ | Stull (2011) wet-bulb temperature (survivability limit) |
| 7 | `wind_speed_kmh` | km/h | $[0.0, 300.0]$ | Surface wind speed |
| 8 | `consecutive_hot_days` | count | $\ge 0$ | Cumulative days where $T_{max} \ge 40^\circ\text{C}$ or $HI \ge 45^\circ\text{C}$ |
| 9 | `is_high_temperature` | boolean (0/1) | $\{0, 1\}$ | Indicator whether $T_{ambient} \ge 38.0^\circ\text{C}$ |
| 10 | `is_humidity_stress` | boolean (0/1) | $\{0, 1\}$ | Indicator whether $RH \ge 70\%$ at $T \ge 32.0^\circ\text{C}$ |
| 11 | `precipitation_mm` | mm | $\ge 0.0$ | Instantaneous precipitation |
| 12 | `rainfall_accumulation_24h_mm` | mm | $\ge 0.0$ | Cumulative rainfall over trailing 24 hours |
| 13 | `rainfall_accumulation_7d_mm` | mm | $\ge 0.0$ | Cumulative rainfall over trailing 7 days |
| 14 | `consecutive_dry_days` | count | $\ge 0$ | Cumulative days without measurable precipitation ($< 1\text{mm}$) |

---

## 3. Training & Validation Dataset (ECMWF ERA5)

Historical meteorological observations are derived from the **ECMWF ERA5 / ERA5-Land Reanalysis** via Open-Meteo Historical Archive covering 10 Indian urban centers:

1. **Delhi Central** (`28.6315°N, 77.2167°E`)
2. **Mumbai Island** (`18.9388°N, 72.8354°E`)
3. **Chennai South** (`13.0827°N, 80.2707°E`)
4. **Bengaluru East** (`12.9716°N, 77.5946°E`)
5. **Kolkata North** (`22.5726°N, 88.3639°E`)
6. **Ahmedabad West** (`23.0225°N, 72.5714°E`)
7. **Pune Central** (`18.5204°N, 73.8567°E`)
8. **Hyderabad Central** (`17.3850°N, 78.4867°E`)
9. **Jaipur North** (`26.9124°N, 75.7873°E`)
10. **Lucknow Central** (`26.8467°N, 80.9462°E`)

### Partition Strategy & Temporal Embargo
To guarantee zero lookahead leakage, datasets are strictly partitioned by time:
- **Train Split**: `2023-01-01` to `2024-05-15` (120,380 rows)
- **Embargo Window**: 6-hour gap ($6 \times 10 = 60\text{ rows}$) discarded between splits to prevent forward-looking targets from spanning partition boundaries.
- **Validation Split**: `2024-05-15` to `2024-09-12` (28,800 rows — covering the severe 2024 Indian summer heatwave).
- **Test Split (Out-of-Time)**: `2024-09-13` to `2024-12-31` (26,380 rows — post-monsoon/winter).

---

## 4. Target Definition & Multi-Class Formulation

The target variable is the maximum heat-risk category observed during the forward 6-hour window:
$$\text{Target}_t = \max_{\Delta t \in (0, 6\text{h}]} \text{RiskCategory}(t + \Delta t)$$

Empirically observed classes in ERA5 2023–2024:
* `LOW` (0): Normal thermal conditions
* `MODERATE` (1): Elevated thermal stress; caution advised
* `HIGH` (2): Severe heat stress; vulnerable populations impacted
* `SEVERE` (3): Heatwave conditions requiring emergency protocols

### Deterministic Physiological Safety Override (CRITICAL)
In the 2023–2024 dataset, zero empirical events reached the extreme physiological survivability threshold. Rather than synthesizing false labels, the architecture incorporates an unsuppressible **Deterministic Safety Override Layer**:
- If live **Wet-Bulb Temperature ($T_w$) $\ge 31.0^\circ\text{C}$** (Vecellio et al., 2022 human survivability limit) $\implies$ **FORCED `CRITICAL`** (Score $\ge 80$).
- If live **NOAA Heat Index ($HI$) $\ge 54.0^\circ\text{C}$** (Extreme Danger threshold) $\implies$ **FORCED `CRITICAL`** (Score $\ge 80$).
- The ML model operates within this safety envelope and is structurally barred from downgrading a lethal physical state.

---

## 5. Model Architecture & Training Highlights

- **Algorithm**: LightGBM Multi-Class Classifier (`LGBMClassifier`)
- **Loss Function**: Multi-class logarithmic loss (`multi_logloss`)
- **Class Balancing**: `class_weight='balanced'` to prevent majority class (`LOW`) from dominating minority heatwave classes (`SEVERE`).
- **Regularization**: `colsample_bytree=0.8`, `subsample=0.8`, `learning_rate=0.05`, `n_estimators=300`.

### Benchmark Comparison (Validation Set — Summer 2024)

| Metric | Persistence Baseline | Rule-Based Model | **LightGBM Model (heat_model_v1)** |
| :--- | :---: | :---: | :---: |
| **Macro F1 Score** | 0.6201 | 0.4105 | **0.5124** |
| **HIGH Heat Recall** | 41.43% | 72.10% | **59.89%** |
| **SEVERE Heat Recall** | 27.38% | 100.00% (high FP) | **56.92%** |
| **Severe False Negatives** | 236 | 0 (excessive alarms) | **140** (balanced precision/recall) |
| **Inference Latency (p95)**| — | 0.12 ms | **0.85 ms** |

### Test Split Nuance
The out-of-time test partition (Sep 13 – Dec 31, 2024) falls during the Indian autumn and winter months. Consequently, empirical occurrences of SEVERE heat were 0. In adherence to scientific reporting integrity, SEVERE recall on this split is reported as `undefined`, and the model's true heatwave recall is evaluated on the summer validation partition.

---

## 6. How to Reproduce Dataset & Training

### 1. Ingest Historical Data & Generate Partitions
```bash
python ml/pipelines/generate_real_dataset.py
```
Outputs audited CSV and Parquet files into `ml/datasets/heat_risk_real/` with full quality validation reports.

### 2. Train and Benchmark the Model
```bash
python ml/training/train_heat_model.py
```
This runs the full pipeline:
1. Verifies dataset integrity and enforces the 60-row embargo.
2. Trains the LightGBM classifier.
3. Evaluates against Persistence and Rule-Based baselines.
4. Serializes `model.joblib`, `model.txt`, and audit manifests into `ml/artifacts/heat_model_v1/`.

### 3. Verify ML Model in Backend Integration
```bash
HEAT_MODEL_TYPE=ml python -m pytest backend/tests/unit/test_ml_heat_model_integration.py -v
```
