# Architecture Decision Records (ADR)

This document catalogs key architectural decisions, context, trade-offs, and consequences for the Environmental Intelligence Platform.

---

## ADR-001: Modular Monolith over Microservices

* **Status**: Accepted
* **Context**: The project is in foundational development for a hackathon. Hyperlocal risk assessment involves multiple domain areas (heat, water, rainfall, community incidents, alerts).
* **Decision**: Adopt a clean modular monolith architecture in Python (FastAPI) instead of individual microservices.
* **Reasoning**:
  * Eliminates distributed network overhead, RPC latency, deployment orchestration complexity, and distributed transaction management.
  * Allows rapid domain iteration with compile-time type safety via Pydantic and in-process function calls.
  * Modules are organized behind strict interfaces (`domain`, `features`, `services`, `repositories`, `models`), enabling any component to be extracted into a standalone service or AWS Lambda function in the future if required.
* **Consequences**: Single deployment artifact; straightforward testing; low operational overhead.

---

## ADR-002: Abstract Base Class Contract (`RiskModel`) for ML Decoupling

* **Status**: Accepted
* **Context**: ML models require time, data acquisition, and backtesting. The platform needs to deliver end-to-end functionality immediately without blocking on ML training or faking ML outputs.
* **Decision**: Establish the `RiskModel` abstract base class with a strict prediction contract (`predict(features, location_id) -> RiskPrediction`).
* **Reasoning**:
  * Allows launching immediately with calibrated, transparent rule-based baselines (`RuleBasedHeatModel`, `RuleBasedWaterModel`, `RuleBasedRainfallModel`).
  * Enables future ML models (`MLHeatModel`, `MLWaterModel`) to replace baselines without touching API routes, database schemas, frontend dashboards, or decision logic.
  * Decouples the application codebase from specific ML frameworks (Scikit-learn, XGBoost, LightGBM, PyTorch).
* **Consequences**: No "fake ML"; transparent model versioning and metrics; seamless future upgrades.

---

## ADR-003: Strict Separation of Concerns: Deterministic Rules vs. ML vs. Jev vs. LLM

* **Status**: Accepted
* **Context**: LLMs and rule engines are often misused to generate numerical calculations or untraceable predictions.
* **Decision**: Enforce rigid boundaries across the system:
  1. **Deterministic Rules & ML**: Strictly responsible for numerical risk scoring, feature calculation, and confidence evaluation.
  2. **Jev (via `DecisionProvider`)**: Strictly responsible for bounded policy evaluation, municipal routing, and workflow triggers.
  3. **LLM (via `LLMProvider`)**: Strictly responsible for plain-language multilingual advisory generation, report summarization, and natural citizen dialogue.
* **Reasoning**:
  * Prevents dangerous hallucinations in public health and flood advisories.
  * Ensures 100% mathematical auditability for all numerical scores and thresholds.
* **Consequences**: LLMs never compute risk scores; Jev never computes weather indicators; ML never makes arbitrary workflow decisions.

---

## ADR-004: Decoupled Provider and Storage Abstractions

* **Status**: Accepted
* **Context**: The system must run locally and in offline CI/CD without requiring AWS credentials, paid weather API subscriptions, or external network connectivity.
* **Decision**: Introduce provider abstractions (`WeatherProvider`, `DecisionProvider`, `LLMProvider`) and repository abstractions (`LocationRepository`, `RiskRepository`, `ReportRepository`).
* **Reasoning**:
  * Local development and unit tests execute using in-memory repositories and mock providers.
  * Cloud deployment simply configures the appropriate adapter (Amazon DynamoDB, Amazon S3, Open-Meteo, AWS Bedrock) without changing domain code.
* **Consequences**: 100% test pass rate offline; zero credentials required to evaluate the repository.

---

## ADR-005: Mandatory Prediction Traceability Contract

* **Status**: Accepted
* **Context**: Environmental risk scores dictate civil defense actions, water rations, and medical alerts. Predictions without provenance are unacceptable.
* **Decision**: Every `RiskPrediction` must store: `location_id`, `risk_score`, `risk_level`, `confidence`, `generated_at`, `valid_until`, `model_name`, `model_version`, `drivers`, `evidence`, and `data_version`.
* **Reasoning**:
  * Enables complete auditing, retrospective backtesting, and model debugging.
  * Explains *why* a risk level was elevated to municipal officers and citizens.
* **Consequences**: Full transparency and auditability across all risk categories.
