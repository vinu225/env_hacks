# Environmental Intelligence Platform — Backend (backend/)

Production-grade modular monolith backend built with **Python 3.13**, **FastAPI**, **Pydantic v2**, and **Uvicorn** for the **AWS Bharat Builds Tour Track 02 (Heat and Water)**.

Provides sub-millisecond environmental risk modeling, multi-hazard risk fusion, real-time weather ingestion with physical bounds validation, and cloud-ready storage decoupling (In-Memory / DynamoDB).

---

## 1. Modular Monolith Architecture

The backend pipeline processes telemetry through ten strictly isolated, deterministic stages:

```
[ External Weather API / Mock ]
              │
              ▼
    [ Resilient Ingestion ]  ───► OpenMeteoClient with retries, timeouts & network error trapping
              │
              ▼
   [ Physical Validation ]   ───► Validates coordinates, temp [-60,65°C], humidity [0,100%], precip >= 0
              │
              ▼
  [ Freshness Verification ] ───► Calculates age_seconds, is_fresh flag, quality degradation penalty
              │
              ▼
    [ Normalization Layer ]  ───► NormalizedWeatherObservation in standard SI units (°C, mm, km/h, UTC)
              │
              ▼
      [ In-Memory Cache ]    ───► WeatherService TTL cache keyed by rounded coordinate hash
              │
              ▼
   [ Feature Engineering ]   ───► FeatureVectorBuilder computing 14 canonical environmental features
              │
              ▼
  [ Model Inference Layer ]  ───► LightGBM ML Classifier (MLHeatModel) with Rule-Based Fallback
              │
              ▼
     [ Risk Fusion Engine ]  ───► Multi-hazard composite scoring (0-100) & dominant hazard isolation
              │
              ▼
    [ Decision & Alerts ]    ───► Bounded municipal intervention policy and active alert generation
              │
              ▼
   [ AI Advisory Assistant ] ───► Multilingual plain-language citizen guidance via AWS Bedrock / Mock
```

---

## 2. Directory Layout

```
backend/
├── app/
│   ├── api/
│   │   └── v1/                  # FastAPI routers
│   │       ├── health.py        # /health and /ready probes
│   │       ├── weather.py       # /weather and /weather/features
│   │       ├── locations.py     # /locations and /locations/{id}/*
│   │       ├── reports.py       # /reports (citizen incident logging & upvoting)
│   │       ├── alerts.py        # /alerts (active early warnings)
│   │       └── assistant.py     # /assistant/explain (AI advisory panel)
│   ├── core/
│   │   ├── config.py            # Pydantic BaseSettings strongly typed configuration
│   │   ├── logging.py           # Structured JSON logging with correlation IDs
│   │   └── security.py          # CORS, rate limiting, and security middleware
│   ├── domain/
│   │   ├── environmental/       # Observations, physical bounds validator, freshness rules
│   │   ├── locations/           # Location entity, urban zones, and ward metadata
│   │   ├── heat/                # Heat risk entity, NOAA Heat Index, Stull wet-bulb formulas
│   │   ├── water/               # Water stress entity, precipitation deficit, dry spells
│   │   ├── rainfall/            # Rainfall accumulation, IMD intensity scale, inundation
│   │   ├── risk/                # Composite risk summary, risk levels, driver representations
│   │   ├── reports/             # Community incident reports and upvote models
│   │   └── alerts/              # Environmental alert schemas and severity tiers
│   ├── features/
│   │   ├── builder.py           # Canonical 14-feature vector computation engine
│   │   └── rolling.py           # 24-hour and 7-day rolling window aggregators
│   ├── models/
│   │   ├── base.py              # RiskModel abstract contract (predict, evaluate, explain)
│   │   ├── heat/                # MLHeatModel (LightGBM) & RuleBasedHeatModel baseline
│   │   ├── water/               # RuleBasedWaterModel & weather-derived stress models
│   │   └── rainfall/            # RuleBasedRainfallModel & inundation vulnerability models
│   ├── services/
│   │   ├── weather.py           # Ingestion, validation, TTL caching orchestrator
│   │   ├── features.py          # Feature vector assembly service
│   │   ├── heat.py              # Hyperlocal heat evaluation service
│   │   ├── water.py             # Hyperlocal water stress evaluation service
│   │   ├── fusion.py            # Composite multi-hazard risk synthesis engine
│   │   ├── alerts.py            # Alert lifecycle management
│   │   ├── reports.py           # Incident crowdsourcing & corroboration service
│   │   └── ai.py                # Plain-language advisory generation service
│   ├── integrations/
│   │   ├── weather/             # OpenMeteoClient, MockWeatherProvider, WeatherProvider protocol
│   │   └── llm/                 # BedrockLLMProvider, MockLLMProvider, LLMProvider protocol
│   ├── repositories/            # In-Memory & DynamoDB storage repositories
│   ├── schemas/                 # Pydantic DTOs for request validation and response serialization
│   └── main.py                  # FastAPI application factory with lifespan handlers
├── tests/
│   ├── unit/                    # 100+ deterministic offline unit tests
│   ├── integration/             # End-to-end API route tests with TestClient
│   └── conftest.py              # Reusable fixtures and test environment presets
├── pyproject.toml
└── requirements.txt
```

---

## 3. Environment Configuration (.env)

All configurations are strictly typed using Pydantic `BaseSettings`:

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `PROJECT_NAME` | string | `Environmental Intelligence Platform` | System identifier |
| `ENVIRONMENT` | string | `development` | Deployment environment (`development`, `staging`, `production`) |
| `LOG_LEVEL` | string | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `WEATHER_PROVIDER_TYPE` | string | `mock` | Weather ingestion source (`mock`, `open_meteo`) |
| `STORAGE_BACKEND` | string | `memory` | Persistence layer (`memory`, `dynamodb`) |
| `HEAT_MODEL_TYPE` | string | `ml` | Heat risk engine (`ml` for LightGBM, `rule_based` for fallback) |
| `AI_PROVIDER_TYPE` | string | `mock` | Advisory LLM engine (`mock`, `bedrock`) |
| `OPEN_METEO_BASE_URL` | string | `https://api.open-meteo.com/v1` | Open-Meteo endpoint |
| `OPEN_METEO_TIMEOUT_SECONDS` | float | `5.0` | Ingestion HTTP timeout |
| `CACHE_TTL_SECONDS` | int | `300` | In-memory weather cache expiration (5 minutes) |
| `CORS_ORIGINS` | string | `["http://localhost:5173", "http://localhost:3000"]` | Allowed CORS origins for frontend |

---

## 4. Getting Started

### 1. Prerequisites
* Python 3.13+
* Pip / Virtualenv

### 2. Configure Environment
```bash
cp .env.example .env
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run Development Server
```bash
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

### 5. Interactive Documentation
Once the server is running, visit:
* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 5. Core API Endpoints

### Health & Diagnostics
* `GET /api/v1/health`: Liveness probe returning application version, uptime, and status.
* `GET /api/v1/ready`: Readiness probe verifying storage backend and weather provider connectivity.

### Meteorological Ingestion & Feature Engineering
* `GET /api/v1/weather?latitude=28.6315&longitude=77.2167`: Ingests, bounds-checks, and normalizes live weather.
* `GET /api/v1/weather/features?latitude=28.6315&longitude=77.2167`: Assembles the canonical 14-feature vector.

### Hyperlocal Risk Assessments
* `GET /api/v1/locations`: Lists all 10 monitored Indian urban metropolitan centers.
* `GET /api/v1/locations/{id}/heat`: Returns 6-hour forward heat risk, NOAA Heat Index, Stull wet-bulb, and ML model drivers.
* `GET /api/v1/locations/{id}/water`: Returns weather-derived water stress, dry spell count, and precipitation deficit.
* `GET /api/v1/locations/{id}/risk`: Synthesizes multi-hazard composite index (0–100) and isolates dominant hazard.

### Citizen Reporting & Crowdsourced Ground-Truth
* `GET /api/v1/reports`: Retrieves list of community-reported environmental incidents.
* `POST /api/v1/reports`: Lodges a citizen incident (`DRY_TAP`, `WATERLOGGING`, `PIPE_LEAK`, `EXTREME_HEAT_DISTRESS`).
* `POST /api/v1/reports/{id}/upvote`: Corroborates a community incident report to increase confidence.

### Early Warning Alerts & AI Advisories
* `GET /api/v1/alerts`: Returns active public health and municipal early warning notices.
* `POST /api/v1/assistant/explain`: Generates plain-language public advisory guidelines.

---

## 6. Example API Requests & Responses

### 1. Unified 14-Feature Canonical Vector
```bash
curl -s "http://localhost:8000/api/v1/weather/features?latitude=28.6315&longitude=77.2167&location_id=loc_delhi_central"
```
```json
{
  "location_id": "loc_delhi_central",
  "timestamp": "2026-10-15T18:30:00Z",
  "temperature_c": 36.4,
  "apparent_temperature_c": 41.2,
  "relative_humidity_pct": 62.0,
  "dew_point_c": 27.8,
  "heat_index_c": 44.5,
  "wet_bulb_temperature_c": 29.1,
  "wind_speed_kmh": 12.4,
  "consecutive_hot_days": 3,
  "is_high_temperature": 0,
  "is_humidity_stress": 0,
  "precipitation_mm": 0.0,
  "rainfall_accumulation_24h_mm": 0.0,
  "rainfall_accumulation_7d_mm": 2.5,
  "consecutive_dry_days": 5
}
```

### 2. Hyperlocal Heat Risk Assessment (LightGBM ML Active)
```bash
curl -s "http://localhost:8000/api/v1/locations/loc_delhi_central/heat"
```
```json
{
  "location_id": "loc_delhi_central",
  "category": "HEAT",
  "risk_score": 74.2,
  "risk_level": "SEVERE",
  "confidence": 0.88,
  "model_name": "heat_model_v1",
  "model_version": "1.0.0",
  "drivers": [
    {
      "parameter": "heat_index_c",
      "impact_level": "HIGH",
      "observed_value": 44.5,
      "threshold_reference": "41.0°C (Danger Threshold)",
      "description": "NOAA Heat Index exceeds Danger threshold"
    },
    {
      "parameter": "wet_bulb_temperature_c",
      "impact_level": "HIGH",
      "observed_value": 29.1,
      "threshold_reference": "29.0°C (High Caution)",
      "description": "Wet-bulb temperature approaching critical survivability envelope"
    }
  ],
  "evidence": {
    "override_triggered": false,
    "fallback_active": false,
    "empirical_class": "SEVERE"
  }
}
```

---

## 7. Automated Testing Suite

The backend contains a deterministic, 100% offline test suite testing all layers:

```bash
# Run complete test suite
python -m pytest backend/tests -v

# Run unit tests only
python -m pytest backend/tests/unit -v

# Run integration tests only
python -m pytest backend/tests/integration -v

# Verify ML Heat Model loading and inference
python -m pytest backend/tests/unit/test_ml_heat_model_integration.py -v
```

### Manual Live Verification Script
To test live end-to-end network round-trip against the public Open-Meteo API:
```bash
python backend/tests/manual_live_open_meteo_check.py
```
