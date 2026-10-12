# Environmental Intelligence Platform (Bharat Builds Tour — Track 02: Heat and Water)

An environmental intelligence platform designed to assess hyperlocal heat risk, water stress, rainfall/flood risks, groundwater depletion, municipal supply disruption, and community-reported environmental incidents across Indian urban centers.

---

## Architecture Overview

The system uses a **Modular Monolith** architecture designed for cloud deployment on AWS:

```
External Environmental / Weather Providers
(Open-Meteo API, IMD, IoT Stations, Mock Provider)
                     │
                     ▼
          [ Resilient HTTP Client ]
(OpenMeteoClient with timeouts, network error trapping, status check)
                     │
                     ▼
             [ Data Ingestion ]
                     │
                     ▼
          [ Physical Data Validation ]
 (Coordinates [-90,90], temp [-60,65°C], humidity [0,100%], non-neg precip & wind)
                     │
                     ▼
         [ Temporal Freshness Audit ]
  (Age in seconds, is_fresh flag, quality degradation penalty)
                     │
                     ▼
          [ Normalization Layer ]
(NormalizedWeatherObservation, HourlyForecastObservation, DailyForecastObservation;
 Standardized units: °C, mm, %, km/h; strictly timezone-aware UTC)
                     │
                     ▼
          [ In-Memory TTL Cache ]
   (WeatherService caching by geographic key to prevent redundant external calls)
                     │
                     ▼
        [ Feature Engineering Layer ]
 (FeatureVectorBuilder: pure deterministic functions for weather, heat,
  weather-derived water stress, and rainfall features)
                     │
                     ▼
    ┌────────────────┴────────────────┐
    ▼                                 ▼
[ Online Serving / API ]     [ Offline ML Datasets ]
(GET /api/v1/weather/features) (ml/training/ feature matrix)
    │                                 │
    └────────────────┬────────────────┘
                     │
                     ▼
       [ Rule / ML Model Inference ]
 (RiskModel abstraction: Rule-based baseline or ML model.
  Generates scores 0-100, confidence, drivers & evidence)
                     │
                     ▼
             [ Risk Engines ]
 (Domain evaluators: Heat, Water, Rainfall)
                     │
                     ▼
              [ Risk Fusion ]
 (Multi-hazard composite index & dominant hazard isolation)
                     │
                     ▼
             [ Decision Layer ]
 (Bounded policy evaluation via DecisionProvider / future Jev)
                     │
                     ▼
    ┌───────────────────────────────┬───────────────────────────────┐
    ▼                               ▼                               ▼
[ Actionable Alerts ]     [ Deterministic Guidance ]     [ Natural-Language Advisories ]
(Public & Municipal)       (Proven Safety Measures)        (Multilingual LLM Summaries)
```

---

## Directory Structure

```
├── backend/
│   ├── app/
│   │   ├── api/v1/          # Endpoints (health, ready, weather, features, locations, heat, water, risk, reports, alerts, assistant)
│   │   ├── core/            # Strongly-typed config, JSON structured logging, security hooks
│   │   ├── domain/          # Core domain models (locations, environmental validation & freshness, heat, water, rainfall, risk, reports, alerts)
│   │   ├── features/        # Feature engineering (weather, heat, water, rainfall, rolling window, feature builder)
│   │   ├── models/          # RiskModel abstraction, Rule-Based baselines, and Model Predictors
│   │   ├── services/        # Weather (with TTL caching), feature engineering, risk engine, risk fusion, decision, AI, recommendations
│   │   ├── integrations/    # Provider abstractions (OpenMeteoClient, OpenMeteoWeatherProvider, MockWeatherProvider, DecisionProvider, LLMProvider)
│   │   ├── repositories/    # Storage abstractions (Location, Risk, Report) with in-memory & DynamoDB compatibility
│   │   ├── schemas/         # Pydantic contracts for environmental observations, feature vectors, predictions, and reports
│   │   ├── workers/         # Background task worker extension point
│   │   ├── utils/           # Timezone-aware datetime tools, domain exceptions
│   │   └── main.py          # FastAPI application factory with correlation ID tracing & error handling
│   ├── tests/
│   │   ├── unit/            # Unit tests (config, validation, freshness, normalization, mock HTTP client, features, rolling, rule baselines)
│   │   ├── integration/     # Integration tests (/health, /ready, /weather, /features, /locations, /heat, /water, /risk, /reports)
│   │   └── manual_live_open_meteo_check.py # Standalone manual verification script for live Open-Meteo API
│   ├── pyproject.toml
│   └── requirements.txt
├── ml/
│   ├── datasets/            # Datasets for ML training
│   ├── notebooks/           # EDA and feature prototyping notebooks
│   ├── training/            # Model training scripts (XGBoost, Random Forest, LightGBM)
│   ├── evaluation/          # Backtesting and model calibration scripts
│   ├── inference/           # Batch inference utilities
│   ├── pipelines/           # ML pipelines (SageMaker Pipelines target)
│   ├── artifacts/           # Serialized model weights and manifest metadata
│   └── README.md
├── frontend/
│   ├── src/                 # React + TypeScript web application source
│   └── README.md
├── infra/
│   └── README.md            # AWS target architecture (ECS/App Runner, DynamoDB, S3, Bedrock)
├── data/sample/             # Sample Indian urban locations and weather observation JSON payloads
├── docs/architecture/       # Comprehensive architectural diagrams, ML integration guide, and ADRs
│   ├── system-overview.md
│   ├── ml-architecture.md
│   └── decisions.md
├── .env.example             # Strongly typed environment configuration template
├── .gitignore
└── README.md
```

---

## Quickstart & Local Execution

### 1. Configure Environment
```bash
cp .env.example .env
```
Default configuration is preset for offline local development without external credentials:
```env
WEATHER_PROVIDER_TYPE=mock
STORAGE_BACKEND=memory
```
To enable live real-time environmental data via Open-Meteo:
```env
WEATHER_PROVIDER_TYPE=open_meteo
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Run FastAPI Application
```bash
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to inspect and execute interactive API endpoints.

### 4. Run Pytest Suite (Deterministic & 100% Offline)
```bash
python -m pytest backend/tests -v
```

### 5. Optional: Run Standalone Live Open-Meteo Verification
To test live end-to-end network round-trip against the public Open-Meteo API:
```bash
python backend/tests/manual_live_open_meteo_check.py
```

---

## Example API Queries

### Query Real Normalized Weather
```bash
curl -X GET "http://localhost:8000/api/v1/weather?latitude=28.6315&longitude=77.2167"
```

### Query Unified ML-Ready Feature Vector
```bash
curl -X GET "http://localhost:8000/api/v1/weather/features?latitude=28.6315&longitude=77.2167&location_id=loc_delhi_central"
```

---

## Heat-Risk ML Integration (Phase E)

### 1. Model Architecture & Artifacts
The platform features a trained **LightGBM Multi-Class Classifier** (`MLHeatModel`) forecasting forward 6-hour maximum heat-risk severity:
- **Artifact Path**: `ml/artifacts/heat_model_v1/` (`model.joblib`, `model.txt`, `feature_schema.json`, `evaluation_report.json`, `training_metadata.json`)
- **Model Version**: `heat_model_v1.0.0`
- **Trained Classes**: `LOW`, `MODERATE`, `HIGH`, `SEVERE` (empirical classes from ERA5 2023–2024 reanalysis across 10 Indian cities)

### 2. Strict Precedence Hierarchy
Prediction execution strictly follows the safety hierarchy:
$$\textbf{Deterministic Safety Override} \;\longrightarrow\; \textbf{Validated ML Prediction} \;\longrightarrow\; \textbf{Rule-Based Fallback} \;\longrightarrow\; \textbf{Safe Error Response}$$

1. **Safety Override (Authoritative)**:
   - Heat Index $\ge 54.0^\circ\text{C}$ (NOAA Extreme Danger threshold) $\implies$ forced `CRITICAL` (score $\ge 80$).
   - Wet-Bulb Temperature $\ge 31.0^\circ\text{C}$ (Vecellio 2022 survivability limit) $\implies$ forced `CRITICAL` (score $\ge 80$).
   - *ML predictions can NEVER downgrade a critical environmental state.*
2. **Validated ML Prediction**:
   - Evaluates the 14-feature canonical vector via `LGBMClassifier`.
   - Validates probability vector bounds and maps output monotonically to 0–100 scores.
3. **Rule-Based Fallback**:
   - If model artifact is missing or throws an inference exception, system automatically falls back to `RuleBasedHeatModel` and flags `fallback_active=True` in evidence.
4. **Safe Error Response**:
   - On non-finite (`NaN`/`Inf`) or missing essential inputs (`temperature_c`), raises `FeatureComputationError` to return standard 422/400 errors without fabricating predictions.

### 3. Canonical 14-Feature Contract
`temperature_c`, `apparent_temperature_c`, `relative_humidity_pct`, `dew_point_c`, `heat_index_c`, `wet_bulb_temperature_c`, `wind_speed_kmh`, `consecutive_hot_days`, `is_high_temperature`, `is_humidity_stress`, `precipitation_mm`, `rainfall_accumulation_24h_mm`, `rainfall_accumulation_7d_mm`, `consecutive_dry_days`.

### 4. Configuration & Rollback
Switching between ML and Rule-Based baselines is controlled via environment variable:
```env
# Enable ML model (Default for demonstration)
HEAT_MODEL_TYPE=ml

# Instant rollback to deterministic Rule-Based baseline
HEAT_MODEL_TYPE=rule_based
```

### 5. Running and Testing
```bash
# Launch FastAPI server with ML Heat Model active
HEAT_MODEL_TYPE=ml python -m uvicorn app.main:app --app-dir backend --port 8000

# Execute complete test suite (109 tests passed)
python -m pytest backend/tests -v
```

### 6. Known Evaluation Limitations
- **Zero Test SEVERE Support**: The out-of-time test partition (Sep 13 – Dec 31, 2024) covers autumn/winter across India, so empirical SEVERE heat occurrences were 0. Severe recall is reported as `undefined` on test, and benchmarked on the summer validation partition (56.92% recall).
- **CRITICAL Class Handling**: ERA5 2023–2024 data contains 0 empirical CRITICAL events; CRITICAL capability is governed by deterministic physiological safety overrides rather than learned classification.

---

## Containerization & AWS Cloud Deployment (Phase F)

### 1. Production Docker Image
The application includes a production-oriented container build (`Dockerfile`) configured for AWS ECS Fargate:
- **Base Image**: `python:3.13-slim`
- **Native Runtime**: Includes `libgomp1` (LightGBM OpenMP) and `curl` (Healthcheck probe)
- **Security**: Runs under non-root user `appuser` (UID/GID isolated)
- **Port**: Exposes `8000`, binds Uvicorn to `0.0.0.0`
- **Model Baked In**: Automatically embeds `ml/artifacts/heat_model_v1/` with zero external storage dependency for inference
- **Healthcheck**: Configured against `/api/v1/health`

### 2. Local Container Commands
```bash
# Build production image
docker build -t heat-risk-api:latest .

# Run container with ML model active
docker run -d --name heat-risk-api -p 8000:8000 heat-risk-api:latest

# Verify health probe
curl -i http://localhost:8000/api/v1/health

# Verify live heat prediction
curl -s http://localhost:8000/api/v1/locations/loc_delhi_central/heat

# Stop and clean up
docker stop heat-risk-api && docker rm heat-risk-api
```

Or run via Docker Compose:
```bash
docker compose up -d
docker compose logs -f
docker compose down
```

### 3. AWS ECS Fargate Deployment
Complete step-by-step instructions for Amazon ECR push and ECS Fargate deployment are documented in [`infra/deploy_aws.md`](infra/deploy_aws.md) with ready-to-use task definitions in [`infra/ecs-task-definition.json`](infra/ecs-task-definition.json).
