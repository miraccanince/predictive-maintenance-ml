# SentinelX - Complete Technical Overview

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Technology Stack](#2-technology-stack)
3. [Data Pipeline](#3-data-pipeline)
4. [Machine Learning](#4-machine-learning)
5. [API Architecture](#5-api-architecture)
6. [Database Design](#6-database-design)
7. [Docker Infrastructure](#7-docker-infrastructure)
8. [LLM Agent](#8-llm-agent)
9. [Dashboard](#9-dashboard)
10. [Testing](#10-testing)
11. [File Structure](#11-file-structure)

---

## 1. Project Summary

### What is SentinelX?

SentinelX is an **AI-powered predictive maintenance platform** for industrial IoT. It predicts machine failures before they happen, explains why, and provides actionable recommendations.

### The 9 Implementation Stages

| Stage | What We Built | Key Technology |
|-------|---------------|----------------|
| **1** | Data Ingestion | Streaming pipeline with quality gates |
| **2** | Feature Engineering | 124 features: rolling windows, lags, cross-domain |
| **3** | Model Training | XGBoost + SMOTE for imbalanced data |
| **4** | Anomaly Detection | PyTorch autoencoder baseline |
| **5** | Evaluation | PR-AUC optimization, threshold calibration |
| **6** | Diagnostic Agent | SHAP explainability + rule-based logic |
| **7** | API + Database | FastAPI + PostgreSQL persistence |
| **8** | Dashboard | Streamlit CEO visualization |
| **9** | LLM Agent | Ollama integration for generative diagnostics |

### Key Metrics Achieved

- **1.4% False Positive Rate** (vs 54% industry baseline)
- **97% Alert Fatigue Reduction**
- **$120.9M Annual Savings** projection
- **124 Engineered Features** from 19 raw inputs

---

## 2. Technology Stack

### Backend
```
┌─────────────────────────────────────────────────────────┐
│                    PYTHON BACKEND                        │
├─────────────────────────────────────────────────────────┤
│  FastAPI 0.109     │ Async web framework for REST API   │
│  Uvicorn           │ ASGI server (4 workers production) │
│  SQLAlchemy 2.0    │ ORM for database operations        │
│  Pydantic 2.5      │ Data validation & serialization    │
│  aiohttp 3.9       │ Async HTTP client for Ollama       │
└─────────────────────────────────────────────────────────┘
```

### Machine Learning
```
┌─────────────────────────────────────────────────────────┐
│                  ML / DATA SCIENCE                       │
├─────────────────────────────────────────────────────────┤
│  XGBoost 2.0       │ Gradient boosting classifier       │
│  SHAP 0.44         │ Model explainability               │
│  scikit-learn 1.4  │ Preprocessing, SMOTE, metrics      │
│  pandas 2.1        │ Data manipulation                  │
│  numpy 1.26        │ Numerical computing                │
│  PyArrow 15.0      │ Parquet file I/O                   │
└─────────────────────────────────────────────────────────┘
```

### Database
```
┌─────────────────────────────────────────────────────────┐
│                     DATABASE                             │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL 15     │ Primary data store                 │
│  JSONB columns     │ Flexible schema for raw_data/SHAP  │
│  psycopg2          │ PostgreSQL adapter                 │
└─────────────────────────────────────────────────────────┘
```

### Frontend / Visualization
```
┌─────────────────────────────────────────────────────────┐
│                    DASHBOARD                             │
├─────────────────────────────────────────────────────────┤
│  Streamlit 1.31    │ Python-native dashboards           │
│  Plotly 5.18       │ Interactive charts                 │
│  Altair 5.2        │ Declarative visualizations         │
└─────────────────────────────────────────────────────────┘
```

### Infrastructure
```
┌─────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────┤
│  Docker            │ Containerization                   │
│  Docker Compose    │ Multi-service orchestration        │
│  Ollama            │ Local LLM inference (llama3.2)     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

### Data Flow Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   SENSORS    │────▶│   QUALITY    │────▶│   FEATURE    │────▶│   MODEL      │
│   (Raw)      │     │   GATES      │     │   STORE      │     │   PREDICT    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 19 raw       │     │ Validate     │     │ 124 features │     │ Probability  │
│ metrics      │     │ ranges       │     │ computed     │     │ 0.0 - 1.0    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Raw Input Metrics (19 total)

**System Metrics (10):**
| Metric | Unit | Description |
|--------|------|-------------|
| air_temperature_K | Kelvin | Ambient temperature |
| process_temperature_K | Kelvin | Machine operating temp |
| rotational_speed_rpm | RPM | Spindle speed |
| torque_Nm | Newton-meters | Cutting force |
| tool_wear_min | Minutes | Tool usage time |
| vibration_mm_s | mm/s | Vibration amplitude |
| pressure_psi | PSI | Hydraulic pressure |
| network_latency_ms | ms | Edge controller latency |
| edge_processing_time_ms | ms | Local compute time |
| fuzzy_pid_output | 0-1 | Control system output |

**Application Metrics (9):**
| Metric | Unit | Description |
|--------|------|-------------|
| error_rate_pct | % | Application error rate |
| cpu_utilization_pct | % | CPU usage |
| memory_utilization_pct | % | RAM usage |
| disk_io_wait_ms | ms | Storage latency |
| packet_loss_pct | % | Network packet loss |
| api_response_latency_ms | ms | API response time |
| http_5xx_count | count | Server errors |
| queue_depth | count | Job queue length |
| request_throughput_rps | req/s | Requests per second |

### Feature Engineering (124 features)

**1. Rolling Statistics (windows: 6, 12, 36 observations)**
```python
# For each raw feature, compute:
feature_mean_6   = rolling_mean(window=6)   # 30 min avg
feature_std_6    = rolling_std(window=6)    # 30 min volatility
feature_mean_12  = rolling_mean(window=12)  # 1 hour avg
feature_std_12   = rolling_std(window=12)   # 1 hour volatility
feature_mean_36  = rolling_mean(window=36)  # 3 hour avg
feature_std_36   = rolling_std(window=36)   # 3 hour volatility
```

**2. Lag Features (T-1, T-2, T-6)**
```python
# Historical values for trend detection:
feature_lag_1 = value at T-1 (5 min ago)
feature_lag_2 = value at T-2 (10 min ago)
feature_lag_6 = value at T-6 (30 min ago)
```

**3. Cross-Domain Features (physics-based)**
```python
# Hybrid features capturing physical interactions:
thermal_efficiency_idx = cpu_utilization / air_temperature
instantaneous_power_W = rpm * torque * (2π/60)
power_anomaly_score = |actual_power - expected_power| / expected_power
error_under_stress = error_rate * cpu_utilization / 100
```

### Feature Store (Real-Time)

The `FeatureStore` class maintains per-machine history:

```python
class FeatureStore:
    def __init__(self, buffer_size=36):
        self.buffers = {}  # Dict[machine_id, deque]

    def update(self, machine_id, observation):
        # Add new observation to machine's buffer
        # Compute rolling features from buffer history
        # Return feature vector for model
```

**Cold Start Handling:**
- If < 6 observations: Synthesize rolling stats based on value extremity
- If ≥ 6 observations: Compute real rolling mean/std

---

## 4. Machine Learning

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      XGBoost CLASSIFIER                          │
├─────────────────────────────────────────────────────────────────┤
│  Input:  124 features (normalized)                               │
│  Output: P(failure) ∈ [0, 1]                                    │
│                                                                  │
│  Key Hyperparameters:                                            │
│  ├── n_estimators: 100                                          │
│  ├── max_depth: 6                                               │
│  ├── learning_rate: 0.1                                         │
│  ├── scale_pos_weight: auto (handles imbalance)                 │
│  └── eval_metric: aucpr (Precision-Recall AUC)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Handling Class Imbalance

**Problem:** Only 3.4% of observations are failures (97:3 ratio)

**Solution: SMOTE (Synthetic Minority Over-sampling)**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Now ~50:50 ratio for training
```

### Threshold Calibration

Three operational thresholds for different use cases:

| Threshold | Probability | Use Case |
|-----------|-------------|----------|
| **Ultra-Safe** | ≥ 0.94 | High-value equipment (minimize false alarms) |
| **Balanced** | ≥ 0.63 | General production (balance precision/recall) |
| **Sensitive** | ≥ 0.22 | Safety-critical (catch all failures) |

### SHAP Explainability

```python
import shap

# Create explainer from trained model
explainer = shap.TreeExplainer(model)

# Compute SHAP values for prediction
shap_values = explainer.shap_values(features)

# Interpretation:
# - Positive SHAP → feature INCREASES failure risk
# - Negative SHAP → feature DECREASES failure risk
# - Magnitude → importance of contribution
```

**Feature Groups for Aggregation:**
```python
FEATURE_GROUPS = {
    "mechanical": ["vibration", "torque", "tool_wear", ...],
    "thermal": ["air_temperature", "process_temperature", ...],
    "network": ["network_latency", "packet_loss", ...],
    "software": ["error_rate", "cpu_utilization", ...],
    "power": ["power_anomaly_score", "fuzzy_pid_output"]
}
```

---

## 5. API Architecture

### FastAPI Application Structure

```
src/api/
├── main.py          # FastAPI app, endpoints, feature store
└── database.py      # SQLAlchemy models, CRUD operations
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict` | Run prediction with diagnosis |
| GET | `/alerts` | Query historical alerts |
| POST | `/alerts/{id}/acknowledge` | Mark alert as seen |
| GET | `/fatigue` | Alert fatigue metrics |

### Prediction Flow

```python
@app.post("/predict")
async def predict(request: PredictRequest, db: Session):
    # 1. Engineer features from raw input
    features = engineer_features(request.system, request.application)

    # 2. Run diagnostic engine (XGBoost + SHAP + LLM)
    result = await diagnostic_engine.diagnose_with_llm(features)

    # 3. Persist alert to database
    alert = create_alert(db, result)

    # 4. Return response
    return PredictResponse(
        alert_id=alert.id,
        risk_level=result.risk_level,
        failure_probability=result.failure_probability,
        root_cause=result.primary_failure_mode,
        markdown_report=result.to_markdown()
    )
```

### Request/Response Models

```python
class SystemMetrics(BaseModel):
    machine_id: str
    air_temperature_K: float
    process_temperature_K: float
    rotational_speed_rpm: float
    torque_Nm: float
    tool_wear_min: float
    vibration_mm_s: Optional[float] = 15.0
    # ... more fields

class PredictRequest(BaseModel):
    system: SystemMetrics
    application: ApplicationLogs

class PredictResponse(BaseModel):
    alert_id: int
    risk_level: str
    failure_probability: float
    root_cause: str
    markdown_report: str
    json_summary: dict
```

---

## 6. Database Design

### PostgreSQL Schema

```sql
-- Main alerts table
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,

    -- Model outputs
    failure_prob FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    root_cause VARCHAR(50),
    confidence VARCHAR(20),

    -- Reports
    report_md TEXT,                    -- Markdown diagnostic
    raw_data JSONB,                    -- Original input
    shap_values JSONB,                 -- SHAP analysis

    -- Alert fatigue tracking
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    false_positive BOOLEAN,
    action_taken TEXT,

    -- Metadata
    model_version VARCHAR(50),
    threshold_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for query performance
CREATE INDEX ix_alerts_machine_date ON alerts(machine_id, DATE(timestamp));
CREATE INDEX ix_alerts_risk_time ON alerts(risk_level, created_at DESC);
```

### Why JSONB?

```python
# Flexible storage for complex nested data:

raw_data = {
    "system": {
        "machine_id": "M1",
        "air_temperature_K": 310.5,
        # ... all input metrics
    },
    "application": {
        "error_rate_pct": 15.2,
        # ... all app metrics
    }
}

shap_values = {
    "top_contributors": [
        {"feature": "error_rate_pct", "shap_value": 1.234, "direction": "increases_risk"},
        # ... top 10 features
    ],
    "feature_group_impacts": {
        "software": 2.45,
        "mechanical": -0.32,
        # ... all groups
    }
}
```

### SQLAlchemy Models

```python
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    machine_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)

    failure_prob = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    root_cause = Column(String(50))

    report_md = Column(Text)
    raw_data = Column(JSONB)      # PostgreSQL JSONB
    shap_values = Column(JSONB)   # PostgreSQL JSONB

    acknowledged = Column(Boolean, default=False)
    # ... more fields
```

### CRUD Operations

```python
def create_alert(db: Session, machine_id: str, ...) -> Alert:
    alert = Alert(
        machine_id=machine_id,
        timestamp=timestamp,
        failure_prob=failure_prob,
        risk_level=risk_level,
        raw_data=raw_data,      # Automatically serialized to JSONB
        shap_values=shap_values
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

def get_alerts(db: Session, machine_id: str = None, limit: int = 100):
    query = db.query(Alert)
    if machine_id:
        query = query.filter(Alert.machine_id == machine_id)
    return query.order_by(Alert.created_at.desc()).limit(limit).all()
```

---

## 7. Docker Infrastructure

### Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE STACK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │     db      │     │     api     │     │  dashboard  │      │
│   │  postgres   │◀───▶│   fastapi   │◀───▶│  streamlit  │      │
│   │   :5432     │     │   :8000     │     │   :8501     │      │
│   └─────────────┘     └──────┬──────┘     └─────────────┘      │
│         │                    │                                   │
│         │                    ▼                                   │
│         │            ┌─────────────┐                            │
│         │            │   ollama    │  (host machine)            │
│         │            │  :11434     │                            │
│         │            └─────────────┘                            │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │   volume    │  (persistent data)                            │
│   │ postgres-   │                                               │
│   │   data      │                                               │
│   └─────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### docker-compose.yml Explained

```yaml
services:
  # ============================================
  # DATABASE (PostgreSQL 15)
  # ============================================
  db:
    image: postgres:15-alpine    # Small, fast Alpine image
    container_name: sentinelx-db
    restart: unless-stopped      # Auto-restart on failure

    environment:
      POSTGRES_USER: sentinelx
      POSTGRES_PASSWORD: sentinelx_secure_2024
      POSTGRES_DB: sentinelx

    volumes:
      - sentinelx-postgres-data:/var/lib/postgresql/data
      # ↑ Named volume persists data across restarts

    ports:
      - "5432:5432"              # Expose for DBeaver, psql

    healthcheck:                 # Docker knows when DB is ready
      test: ["CMD-SHELL", "pg_isready -U sentinelx"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ============================================
  # API (FastAPI + XGBoost + SHAP + LLM)
  # ============================================
  api:
    build:
      context: ..                # Build from parent directory
      dockerfile: docker/Dockerfile.api
      target: production         # Multi-stage build

    container_name: sentinelx-api

    environment:
      # Database connection (uses container name as hostname)
      DB_HOST: db                # "db" resolves to postgres container
      DB_PORT: 5432
      DB_NAME: sentinelx
      DB_USER: sentinelx
      DB_PASS: sentinelx_secure_2024

      # LLM Configuration
      OLLAMA_URL: http://host.docker.internal:11434
      # ↑ Special hostname to reach host machine from container
      OLLAMA_MODEL: llama3.2
      OLLAMA_TIMEOUT: 60

    ports:
      - "8000:8000"

    depends_on:
      db:
        condition: service_healthy  # Wait for DB healthcheck

  # ============================================
  # DASHBOARD (Streamlit)
  # ============================================
  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard

    container_name: sentinelx-dashboard

    environment:
      DB_HOST: db
      API_URL: http://api:8000   # Inter-container communication

    ports:
      - "8501:8501"

    depends_on:
      - api
      - db

# Named volume for data persistence
volumes:
  sentinelx-postgres-data:
    name: sentinelx-postgres-data
```

### Dockerfile.api (Multi-Stage Build)

```dockerfile
# Stage 1: Base dependencies
FROM python:3.11-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM base as production
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 3: Development (with reload)
FROM base as development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Docker Commands Cheatsheet

```bash
# Start everything
docker-compose up --build -d

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop everything
docker-compose down

# Reset database (delete volume)
docker-compose down -v

# Restart single service
docker-compose restart api

# Enter container shell
docker exec -it sentinelx-api bash

# Database backup
docker run --rm \
  -v sentinelx-postgres-data:/data \
  -v $(pwd):/backup \
  alpine tar cvf /backup/db-backup.tar /data
```

---

## 8. LLM Agent

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM DIAGNOSTIC PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. XGBoost Predict ──▶ probability (0.0 - 1.0)                │
│                                                                  │
│  2. SHAP Analysis ──▶ top 5 feature contributions               │
│                                                                  │
│  3. Prompt Builder ──▶ context-rich prompt with:                │
│     ├── Failure probability & risk level                        │
│     ├── Top 5 SHAP features + values + directions               │
│     ├── Feature group aggregations                              │
│     ├── Raw sensor values                                       │
│     └── Hybrid feature explanations                             │
│                                                                  │
│  4. Ollama API Call ──▶ POST http://host.docker.internal:11434 │
│                                                                  │
│  5. Response Parser ──▶ Extract JSON + Markdown                 │
│                                                                  │
│  6. Fallback ──▶ If Ollama fails, use rule-based diagnosis      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### System Prompt (Senior Reliability Engineer)

```python
LLM_SYSTEM_PROMPT = """You are a Senior Reliability Engineer at a
high-tech manufacturing plant. Your goal is to analyze SHAP
explainability values and predict the physical root cause of
machine failures.

CRITICAL EXPERTISE:
1. SHAP values quantify each feature's contribution
2. Positive SHAP = INCREASES risk; Negative = DECREASES risk
3. You correlate hardware signals with software symptoms

FAILURE MECHANICS:
- Torque × Tool Wear > 3000: Mechanical stress
- High Temperature + Network Latency: Thermal throttling
- Error Rate > 5% + CPU > 80%: Software instability

OUTPUT FORMAT:
Part 1 - JSON: {"risk_level", "root_cause", "confidence", "action"}
Part 2 - Markdown: Diagnostic report for engineers
"""
```

### Example Prompt Sent to LLM

```
MACHINE HEALTH ANALYSIS REQUEST

XGBoost PREDICTION:
- Failure Probability: 97.30%
- Risk Level: CRITICAL

TOP 5 SHAP CONTRIBUTORS:
1. error_rate_pct
   - Current Value: 28.0000
   - SHAP Value: +3.2234 (INCREASES failure risk)

2. disk_io_wait_ms
   - Current Value: 48.0000
   - SHAP Value: +1.2040 (INCREASES failure risk)
...

FEATURE GROUP IMPACTS:
- SOFTWARE: +4.234 (increases risk)
- MECHANICAL: +1.123 (increases risk)
...

KEY RAW SENSOR VALUES:
- vibration_mm_s: 48.00 mm/s
- torque_Nm: 100.00 Nm
- error_rate_pct: 28.00 %
- cpu_utilization_pct: 98.00 %
...
```

### Fallback Mechanism

```python
async def diagnose_with_llm(self, features, machine_id, timestamp):
    # Always compute rule-based result first
    fallback_result = self._rule_based_diagnosis(features)

    # Try LLM
    try:
        llm_response = await self._call_ollama(prompt)
        if llm_response:
            return self._parse_llm_response(llm_response)
    except:
        pass

    # Fallback to rule-based if LLM fails
    logger.warning("Ollama unavailable, using rule-based")
    return fallback_result
```

---

## 9. Dashboard

### Streamlit Application

```
src/dashboard/
└── dashboard.py    # CEO Dashboard
```

### Features

| Section | Description |
|---------|-------------|
| **💰 Savings Counter** | Real-time $120M annual savings projection |
| **📉 Alert Fatigue Monitor** | 1.4% FP rate vs 54% baseline |
| **📊 Risk Distribution** | Pie chart by risk level |
| **🔍 Root Cause Analysis** | Bar chart by failure mode |
| **⏰ 24-Hour Timeline** | Recent alerts timeline |
| **🧪 Live Prediction** | Interactive prediction form |

### Live Prediction Form

```python
# Scenario presets
scenarios = {
    'normal': {
        'air_temp': 305.0, 'torque': 40.0, 'tool_wear': 100.0,
        'error_rate': 2.0, 'cpu_util': 55.0, 'network_latency': 20.0
    },
    'stress': {
        'air_temp': 330.0, 'torque': 70.0, 'tool_wear': 200.0,
        'error_rate': 15.0, 'cpu_util': 88.0, 'network_latency': 60.0
    },
    'failure': {
        'air_temp': 365.0, 'torque': 100.0, 'tool_wear': 280.0,
        'error_rate': 28.0, 'cpu_util': 98.0, 'network_latency': 120.0
    }
}
```

---

## 10. Testing

### Test Types

| Type | Tool | Location |
|------|------|----------|
| **E2E API** | Playwright | `tests/e2e/api.spec.ts` |
| **E2E Dashboard** | Playwright | `tests/e2e/dashboard.spec.ts` |
| **Unit Tests** | pytest | `tests/unit/` |

### Playwright E2E Tests

```typescript
// tests/e2e/api.spec.ts
test('health endpoint returns healthy', async ({ request }) => {
    const response = await request.get('http://localhost:8000/health');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe('healthy');
});

test('predict endpoint returns valid response', async ({ request }) => {
    const response = await request.post('http://localhost:8000/predict', {
        data: testPayload
    });
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.risk_level).toMatch(/critical|high|moderate|low|nominal/);
});
```

### Running Tests

```bash
# Install Playwright
npm install
npx playwright install

# Run E2E tests (requires Docker stack running)
npm test

# Run specific test file
npx playwright test tests/e2e/api.spec.ts

# Run with UI
npx playwright test --ui
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "system": {"machine_id": "M1", "air_temperature_K": 365, ...},
    "application": {"error_rate_pct": 28, ...}
  }'

# LLM Agent demo
OLLAMA_URL=http://localhost:11434 python src/ml/agent.py --llm
```

---

## 11. File Structure

```
SentinelX/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app, /predict endpoint
│   │   └── database.py          # SQLAlchemy models, CRUD
│   │
│   ├── ml/
│   │   ├── agent.py             # DiagnosticEngine (SHAP + LLM)
│   │   ├── train_model.py       # XGBoost training
│   │   ├── feature_engineer.py  # Feature transformation
│   │   └── evaluate_and_scale.py
│   │
│   ├── pipeline/
│   │   ├── stream_manager.py    # Data ingestion
│   │   └── generate_synthetic_data.py
│   │
│   └── dashboard/
│       └── dashboard.py         # Streamlit CEO dashboard
│
├── docker/
│   ├── docker-compose.yml       # 3-service stack
│   ├── Dockerfile.api           # API container
│   └── Dockerfile.dashboard     # Dashboard container
│
├── models/
│   ├── model.joblib             # Trained XGBoost
│   ├── feature_names.json       # 124 feature names
│   └── evaluation_results.json  # Thresholds
│
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Cleaned data
│   └── features/                # Feature matrices
│       └── feature_matrix.parquet
│
├── tests/
│   └── e2e/
│       ├── api.spec.ts
│       └── dashboard.spec.ts
│
├── docs/
│   └── TECHNICAL_OVERVIEW.md    # This file
│
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js (Playwright)
├── playwright.config.ts
└── README.md
```

---

## Quick Reference

### Start the System
```bash
cd docker && docker-compose up --build -d
```

### Access Points
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| Database | localhost:5432 |

### Check Logs
```bash
docker-compose logs -f api       # API logs
docker-compose logs -f dashboard # Dashboard logs
```

### Test LLM Agent
```bash
# Ensure Ollama is running
ollama serve

# Run demo
OLLAMA_URL=http://localhost:11434 python src/ml/agent.py --llm
```

---

*Last updated: January 2026*
