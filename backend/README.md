# Environmental Intelligence Platform — Backend

Production-grade modular monolith backend built with Python 3.13, FastAPI, Pydantic, and Uvicorn for AWS Bharat Builds Tour Track 02 (Heat and Water).

---

## Getting Started

### 1. Prerequisites
* Python 3.13+
* Pip

### 2. Environment Configuration
Copy the configuration template:
```bash
cp .env.example .env
```
Default settings are configured for 100% offline local development:
```env
WEATHER_PROVIDER_TYPE=mock
STORAGE_BACKEND=memory
```
To enable live real-time environmental data via Open-Meteo, set:
```env
WEATHER_PROVIDER_TYPE=open_meteo
```

### 3. Installation
Install dependencies:
```bash
pip install -r backend/requirements.txt
```

### 4. Running the Development Server
From the project root:
```bash
python -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```
Or from the `backend/` directory:
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### 5. Interactive API Documentation
Once running, open your browser:
* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Liveness Probe**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
* **Readiness Probe**: [http://localhost:8000/api/v1/ready](http://localhost:8000/api/v1/ready)

---

## Running the Automated Test Suite

Execute pytest across all unit and integration tests (100% deterministic & offline):
```bash
python -m pytest backend/tests -v
```

### Optional: Manual Live Open-Meteo API Check
To verify live network round-trip against the public Open-Meteo service:
```bash
python backend/tests/manual_live_open_meteo_check.py
```

---

## Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Liveness health check with timestamp & version |
| `GET` | `/api/v1/ready` | Readiness check verifying storage & weather providers |
| `GET` | `/api/v1/weather` | Retrieve normalized meteorological observations with freshness audit |
| `GET` | `/api/v1/weather/features` | Unified ML-ready feature vector (heat, water, rainfall) |
| `GET` | `/api/v1/locations` | List monitored urban areas (Delhi, Mumbai, Chennai, etc.) |
| `GET` | `/api/v1/locations/{id}/heat` | Hyperlocal heat risk, heat index, and wet-bulb evaluation |
| `GET` | `/api/v1/locations/{id}/water` | Weather-derived water stress risk and dry spell evaluation |
| `GET` | `/api/v1/locations/{id}/risk` | Multi-hazard composite risk fusion synthesis |
| `GET` | `/api/v1/reports` | List community environmental incident reports |
| `POST`| `/api/v1/reports` | Lodge a new crowdsourced incident report |
| `POST`| `/api/v1/reports/{id}/upvote` | Upvote and corroborate a community report |
| `GET` | `/api/v1/alerts` | List active environmental early warnings |
| `POST`| `/api/v1/assistant/explain` | Generate plain-language advisory via AI service |

---

## Example Queries

### Normalized Weather Observation
```bash
curl -X GET "http://localhost:8000/api/v1/weather?latitude=28.6315&longitude=77.2167"
```

### Environmental Feature Vector
```bash
curl -X GET "http://localhost:8000/api/v1/weather/features?latitude=28.6315&longitude=77.2167&location_id=loc_delhi_central"
```
