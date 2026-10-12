# System Architecture Overview

**Project**: Environmental Intelligence Platform  
**Hackathon**: AWS Bharat Builds Tour — Track 02: Heat and Water  
**Architecture Style**: Clean Modular Monolith (Cloud-Ready for AWS)  

---

## 1. End-to-End Real Environmental Data Pipeline

The platform processes hyperlocal environmental data through an explicit, auditable, multi-stage pipeline:

```
External Environmental / Weather Providers
(Open-Meteo Forecast API, IMD, IoT Stations, Mock Provider)
                     │
                     ▼
          [ Resilient HTTP Client ]
(OpenMeteoClient with timeouts, network error trapping, status check)
                     │
                     ▼
             [ Data Ingestion ]
  (Capture source, timestamps, latency, raw payload)
                     │
                     ▼
          [ Physical Data Validation ]
 (Validate coordinates [-90,90], [-180,180], temp [-60,65°C],
  humidity [0,100%], non-negative precipitation & wind)
                     │
                     ▼
         [ Temporal Freshness Audit ]
  (Calculate age_seconds, is_fresh flag, quality degradation penalty)
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
 (Domain evaluators: HeatEngine, WaterEngine, RainfallEngine)
                     │
                     ▼
              [ Risk Fusion ]
 (Multi-hazard composite index, dominant hazard isolation)
                     │
                     ▼
             [ Decision Layer ]
 (Bounded policies & routing via DecisionProvider / future Jev)
                     │
                     ▼
    ┌───────────────────────────────┬───────────────────────────────┐
    ▼                               ▼                               ▼
[ Actionable Alerts ]     [ Deterministic Guidance ]     [ Natural-Language Advisories ]
(Public & Municipal)       (Proven Safety Measures)        (Multilingual LLM Summaries)
```

---

## 2. Real Environmental Data Pipeline Details

### 2.1 Open-Meteo Weather Provider (`OpenMeteoWeatherProvider`)
* Built behind the existing `WeatherProvider` interface.
* Queries live surface and forecast observations across any global coordinates (not hardcoded to specific cities):
  * **Current**: Temperature (2m), Apparent Temperature, Relative Humidity (2m), Precipitation, Rain, Wind Speed (10m).
  * **Hourly Forecast**: Temperature, Apparent Temperature, Humidity, Precipitation, Precipitation Probability, Wind Speed.
  * **Daily Forecast**: Max Temperature, Min Temperature, Precipitation Sum, Max Precipitation Probability.
* Configured with strict UTC timezone enforcement (`timezone="UTC"`).

### 2.2 Dedicated HTTP Client (`OpenMeteoClient`)
* Centralized client utilizing `httpx.AsyncClient`.
* Enforces sensible timeouts (default: 10.0 seconds).
* Comprehensive error handling:
  * Connection / DNS errors (`httpx.NetworkError`) -> `ProviderError`
  * Timeouts (`httpx.TimeoutException`) -> `ProviderError`
  * HTTP non-200 responses -> `ProviderError`
  * Malformed JSON / unexpected schema -> `ProviderError`
* Configurable base URL via environment settings (`OPEN_METEO_API_URL`). No API key required for Open-Meteo.

### 2.3 Physical Validation (`EnvironmentalValidator`)
* Enforces real-world physical limits before data enters feature engineering or risk scoring:
  * Latitude: `[-90.0, 90.0]`
  * Longitude: `[-180.0, 180.0]`
  * Temperature: `[-60.0°C, 65.0°C]`
  * Humidity: `[0.0%, 100.0%]`
  * Precipitation: `>= 0.0 mm` (flags physically impossible negative values)
  * Wind Speed: `>= 0.0 km/h`
* Rejects naive datetimes; requires strictly timezone-aware UTC timestamps.
* Generates `ValidationResult` with `is_valid`, `is_usable`, and detailed issues list. Does **NOT** silently sanitize or fabricate fake numbers.

### 2.4 Temporal Freshness & Quality (`FreshnessEvaluator`)
* Reusable evaluator calculating `age_seconds = (now_utc - observed_at).total_seconds()`.
* Distinctly distinguishes:
  * `observed_at`: Sensor/provider observation timestamp.
  * `ingested_at`: Platform ingestion timestamp.
  * `forecast_valid_at`: Targeted future projection validity time.
* Calculates continuous quality degradation factor as observation ages past SLA boundaries.

### 2.5 Feature Versioning & Availability Honesty
* Every feature vector records:
  * `data_version = "v1.0"`
  * `feature_version = "weather-features-v1"`
* **Water Risk Honesty**: Weather data alone cannot prove a municipal water shortage. The model is explicitly designated as **"weather-derived water stress risk"**.
* Features that lack telemetry (groundwater depth, municipal water cuts, 30-day historical archives) are explicitly tagged as `is_available = False` and set to `None`. They are never fabricated.

---

## 3. Core Architectural Pillars

### 3.1 Modular Monolith
* Zero network overhead during hackathon development.
* All components separated behind strict interfaces (`domain`, `features`, `models`, `services`, `integrations`, `repositories`, `schemas`).
* Can be extracted into AWS Lambda functions or standalone microservices later without touching business logic.

### 3.2 Provider Abstractions
* Domain and service layers depend on `WeatherProvider`, `DecisionProvider`, and `LLMProvider`.
* Swapping between `OpenMeteoWeatherProvider` and `MockWeatherProvider` is driven entirely by environment configuration (`WEATHER_PROVIDER_TYPE=open_meteo` or `mock`).
* Automated unit test suite runs 100% offline without live internet access.

### 3.3 Strict Separation of Concerns
* **Deterministic Rules & ML**: Pure numerical risk scoring, feature calculation, and confidence evaluation.
* **Jev (via `DecisionProvider`)**: Bounded policy evaluation, municipal routing, and workflow triggers.
* **LLM (via `LLMProvider`)**: Plain-language multilingual advisory generation, report summarization, and natural citizen dialogue.
