# Environmental Intelligence Platform — Technical Documentation

Comprehensive architectural specifications, machine learning problem formulations, data contracts, and Architectural Decision Records (ADRs) for the **AWS Bharat Builds Tour — Track 02 (Heat & Water)**.

---

## Documentation Index

### 1. Architecture Specifications (`docs/architecture/`)
* [System Architecture Overview](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/architecture/system-overview.md):
  - End-to-end data pipeline: Resilient Ingestion $\to$ Physical Validation $\to$ Temporal Freshness $\to$ Normalization $\to$ Feature Engineering $\to$ Risk Fusion $\to$ Decision Layer $\to$ Actionable Advisories.
  - Multi-hazard composite risk index mathematical formulation and dominant threat isolation.
  - Decoupled storage repository design (In-Memory for testing, DynamoDB for cloud deployment).
* [Machine Learning Pipeline Architecture](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/architecture/ml-architecture.md):
  - Forward 6-hour maximum heat-risk target formulation ($\max_{\Delta t \in (0,6\text{h}]} \text{RiskCategory}(t + \Delta t)$).
  - Time-series cross-validation strategy with strict 60-row chronological embargoing to prevent split-crossing lookahead leakage.
  - LightGBM multi-class classifier training and benchmark comparisons against Persistence and Rule-Based baselines.
  - Deterministic physiological safety overrides ($T_w \ge 31^\circ\text{C}$ and $HI \ge 54^\circ\text{C}$) that cannot be downgraded by learned inference.
* [Architectural Decision Records (ADRs)](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/architecture/decisions.md):
  - ADR-001: Selection of Clean Modular Monolith architecture.
  - ADR-002: Real-time Ingestion via Open-Meteo with local in-memory fallback.
  - ADR-003: Pure-function Feature Engineering with strict physical validation bounds.
  - ADR-004: Decoupled Multi-Hazard Risk Fusion and Monotonic Score Mapping (0–100).
  - ADR-005: Dual-path decision making: bounded municipal policies and AWS Bedrock LLM advisories.

### 2. Machine Learning & Data Contracts (`docs/ml/`)
* [Data Contract Specification](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/ml/data-contract.md):
  - Canonical 14-feature contract matching real-time serving and offline training partitions.
  - Physical limits, measurement units (°C, %, km/h, mm), and temporal freshness decay rules.
  - Schema drift detection and serialization standards (JSON, Parquet).
* [Dataset Quality Audit Report](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/ml/dataset-audit.md):
  - Empirical audit of ECMWF ERA5 reanalysis across 10 Indian cities (Delhi, Mumbai, Chennai, Bengaluru, Kolkata, Ahmedabad, Pune, Hyderabad, Jaipur, Lucknow) from 2023–2024.
  - Class distribution analysis: empirical coverage of `LOW`, `MODERATE`, `HIGH`, `SEVERE` classes and rationale for deterministic handling of `CRITICAL`.
  - Missing value analysis (0.00% missing values verified across all training partitions).
* [Heat Risk Problem Formulation](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/docs/ml/heat-risk-problem.md):
  - Formal mathematical definition of forward-looking thermal stress.
  - Human physiological survivability limits (NOAA Heat Index and Vecellio 2022 wet-bulb limits).
  - Multi-class target categorization vs regression score mapping.

---

## Subsystem READMEs

* [Root Platform Overview](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/README.md)
* [Frontend Web Dashboard](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/frontend/README.md)
* [Backend Modular Monolith](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/backend/README.md)
* [Machine Learning Subsystem](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/ml/README.md)
* [Infrastructure & AWS Deployment](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/infra/README.md)
* [AWS ECS Deployment Guide](file:///c:/LATEST/Environmental_Hacks/New%20folder%20(2)/New%20folder/infra/deploy_aws.md)
