# SentinelX: Predictive Maintenance Platform

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20Ready-FF9900?logo=amazon-aws)](https://aws.amazon.com/ecs/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Playwright-65%20Tests-45ba4b?logo=playwright)](https://playwright.dev/)

SentinelX is a production-grade AIOps platform for predictive maintenance. It ingests dual-domain telemetry (hardware sensors + application logs), engineers temporal features, trains an imbalanced classifier, and serves real-time failure predictions with SHAP-based explanations through a FastAPI service backed by PostgreSQL.

The core problem it addresses: industrial monitoring systems generate too many false alerts. Operators start ignoring them. Real failures get missed. SentinelX brings the false positive rate from the 54% industry average down to 1.4% by combining cross-domain feature fusion, proper imbalance handling, and explainable predictions that operators can actually trust.

The implementation is grounded in four research papers covering hybrid ML architectures, imbalanced classification, alert fatigue, and real-time data quality. See the [Literature](#literature) section for details.

---

## Project Summary

The pipeline is built across 9 stages:

| Stage | Component | What It Does |
|-------|-----------|--------------|
| 1 | Data Ingestion | Streaming pipeline with quality gates (Paper 4) |
| 2 | Feature Engineering | 105+ features: rolling windows, lags, cross-domain fusion |
| 3 | Model Training | XGBoost + SMOTE for 10:1 class imbalance |
| 4 | Anomaly Detection | PyTorch autoencoder baseline comparison |
| 5 | Evaluation | PR-AUC optimization, threshold calibration at 500K scale |
| 6 | Diagnostic Agent | SHAP-powered explainability engine |
| 7 | API + Persistence | FastAPI service + PostgreSQL with alert fatigue tracking |
| 8 | Dashboard | Streamlit visualization of alert metrics and model performance |
| 9 | LLM Agent | Ollama-powered generative diagnostics with chain-of-thought |

**Results:**
- 1.4% false positive rate (vs 54% industry baseline — Paper 3)
- 97% reduction in alert fatigue
- 95% failure detection recall
- 65 end-to-end tests via Playwright (API + Dashboard)
- Docker-ready with a documented AWS ECS migration path

---

## Quick Start

```bash
git clone <repository>
cd SentinelX
docker-compose -f docker/docker-compose.yml up --build
```

Three services start:

| Service | URL | Purpose |
|---------|-----|---------|
| API | http://localhost:8000 | FastAPI inference engine |
| Dashboard | http://localhost:8501 | Streamlit metrics dashboard |
| Database | localhost:5432 | PostgreSQL alert storage |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SentinelX Architecture                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐   │
│   │   Sensors   │────▶│   FastAPI   │────▶│      PostgreSQL         │   │
│   │   (Input)   │     │   + XGBoost │     │   (Alert Storage)       │   │
│   └─────────────┘     │   + SHAP    │     └─────────────────────────┘   │
│                       │   + LLM     │                │                   │
│                       └──────┬──────┘                │                   │
│                              │                       │                   │
│   ┌─────────────┐            │                       ▼                   │
│   │   Ollama    │◀───────────┤           ┌─────────────────────────┐    │
│   │  llama3.2   │            │           │   Streamlit Dashboard   │    │
│   └─────────────┘            ▼           └─────────────────────────┘    │
│                       ┌─────────────┐                                    │
│                       │  Diagnostic │                                    │
│                       │   Engine    │                                    │
│                       └─────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | File | Description |
|-------|------|-------------|
| 1 | `src/pipeline/stream_manager.py` | Data ingestion with quality gates |
| 2 | `src/ml/feature_engineer.py` | Rolling windows, lags, cross-domain features |
| 3 | `src/ml/train_model.py` | XGBoost + SMOTE for imbalanced data |
| 4 | `src/ml/autoencoder.py` | Anomaly detection baseline |
| 5 | `src/ml/evaluate_and_scale.py` | PR-AUC evaluation, threshold optimization |
| 6 | `src/ml/agent.py` | SHAP-powered diagnostic engine |
| 7 | `src/api/main.py` + `database.py` | FastAPI service + PostgreSQL |
| 8 | `src/dashboard/dashboard.py` | Alert metrics and model performance dashboard |

---

## LLM-Powered Diagnostics (Stage 9)

The diagnostic agent wraps XGBoost predictions with SHAP analysis and sends them to a local Ollama instance for natural-language root cause generation. It falls back to rule-based diagnosis if Ollama is unavailable.

```
XGBoost Predict → SHAP Analysis → Prompt Builder → Ollama (llama3.2)
                                                          │
                                                    Parser (JSON + MD)
                                                    │         │
                                               Database    Dashboard
```

| Feature | Description |
|---------|-------------|
| Async integration | Non-blocking Ollama API calls via `aiohttp` |
| Chain-of-thought | LLM reasons through SHAP values before concluding |
| Structured output | JSON for database, Markdown for human-readable report |
| Context injection | Top 5 SHAP features with physical meanings and sensor values |
| Automatic fallback | Rule-based diagnosis if Ollama is unavailable |
| Version tracking | `2.0-llm-llama3.2` vs `1.0-rule-based` |

### Enabling LLM Features

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

The API detects Ollama on startup and uses it automatically.

### Rule-Based vs LLM Comparison

| Aspect | Rule-Based (v1.0) | LLM-Powered (v2.0) |
|--------|-------------------|-------------------|
| Speed | ~50ms | ~10-30s |
| Consistency | Deterministic | Temperature-controlled (0.3) |
| Flexibility | Fixed patterns | Reasons about novel patterns |
| Explainability | Template-based | Natural language grounded in SHAP |
| Availability | Always | Requires Ollama |

### Environment Variables

```bash
OLLAMA_URL=http://host.docker.internal:11434  # Docker to host
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=60
```

---

## Docker

### Services

```yaml
services:
  db:         # PostgreSQL 15
    port: 5432

  api:        # FastAPI + XGBoost + SHAP
    port: 8000
    depends_on: db

  dashboard:  # Streamlit
    port: 8501
    depends_on: api, db
```

### Common Commands

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up --build

# Run in background
docker-compose -f docker/docker-compose.yml up -d --build

# Follow logs
docker-compose -f docker/docker-compose.yml logs -f api
docker-compose -f docker/docker-compose.yml logs -f dashboard

# Stop
docker-compose -f docker/docker-compose.yml down

# Stop and reset database
docker-compose -f docker/docker-compose.yml down -v

# Rebuild a single service
docker-compose -f docker/docker-compose.yml up --build api
```

---

## API Reference

### Health Check
```bash
curl http://localhost:8000/health
```

### Submit Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "system": {
      "machine_id": "M1",
      "air_temperature_K": 310.5,
      "process_temperature_K": 318.2,
      "rotational_speed_rpm": 1550,
      "torque_Nm": 52.3,
      "tool_wear_min": 180,
      "vibration_mm_s": 28.5,
      "pressure_psi": 125.0
    },
    "application": {
      "error_rate_pct": 15.2,
      "cpu_utilization_pct": 85.0,
      "memory_utilization_pct": 72.0,
      "disk_io_wait_ms": 25.0,
      "packet_loss_pct": 2.5,
      "api_response_latency_ms": 250.0,
      "http_5xx_count": 12,
      "queue_depth": 45,
      "request_throughput_rps": 85.0
    }
  }'
```

### List Alerts
```bash
curl "http://localhost:8000/alerts?risk_level=high&limit=10"
```

### Alert Fatigue Metrics
```bash
curl "http://localhost:8000/fatigue?days=30"
```

Full interactive docs at http://localhost:8000/docs.

---

## AWS Migration

SentinelX is structured as a digital twin of production AWS infrastructure. The local Docker setup mirrors the AWS deployment exactly — migrating is a matter of changing environment variables, not code.

```
LOCAL                           AWS
──────────────────────────────────────────────────────
Docker PostgreSQL      →   Amazon RDS PostgreSQL
Docker API container   →   ECS Fargate task
Docker Dashboard       →   ECS Fargate task
docker-compose         →   ECS service + ALB
localhost              →   Route53 + ACM (HTTPS)
.env file              →   Secrets Manager
```

### Migration Steps

1. **Push images to ECR**
   ```bash
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

   docker build -f docker/Dockerfile.api -t sentinelx-api .
   docker tag sentinelx-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-api:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-api:latest

   docker build -f docker/Dockerfile.dashboard -t sentinelx-dashboard .
   docker tag sentinelx-dashboard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-dashboard:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-dashboard:latest
   ```

2. **Create RDS instance**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier sentinelx-prod \
     --db-instance-class db.t3.medium \
     --engine postgres --engine-version 15 \
     --allocated-storage 100 --storage-type gp3 \
     --multi-az --storage-encrypted
   ```

3. **Store secrets**
   ```bash
   aws secretsmanager create-secret \
     --name sentinelx/prod/database \
     --secret-string '{"DB_HOST":"...rds.amazonaws.com","DB_PASS":"..."}'
   ```

4. **Deploy to ECS** — create cluster, task definitions referencing ECR images, services with ALB target groups, and configure auto-scaling.

See [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) for the full walkthrough.

---

## Dashboard

The Streamlit dashboard at http://localhost:8501 includes:

**Alert fatigue monitor** — tracks the 1.4% vs 54% FP rate comparison, acknowledgment rate, and false positive labeling over configurable time windows.

**Root cause breakdown** — distribution by failure mode (mechanical, thermal, software), risk level counts, and a 24-hour alert timeline.

**Live prediction demo** — interactive form for submitting predictions, with real-time SHAP explanations and instant database persistence.

---

## Project Structure

```
SentinelX/
├── src/
│   ├── api/                      FastAPI service
│   │   ├── main.py               Endpoints, inference, feature store
│   │   └── database.py           SQLAlchemy models + CRUD
│   ├── ml/
│   │   ├── agent.py              Diagnostic engine (SHAP + LLM)
│   │   ├── train_model.py        XGBoost + SMOTE training
│   │   ├── feature_engineer.py   Feature transformation pipeline
│   │   ├── autoencoder.py        Anomaly detection baseline
│   │   └── evaluate_and_scale.py PR-AUC + threshold optimization
│   ├── pipeline/
│   │   ├── stream_manager.py     Data ingestion + quality gates
│   │   ├── generate_synthetic_data.py
│   │   └── analyze_logs.py
│   └── dashboard/
│       └── dashboard.py
│
├── data/
│   ├── raw/                      Source CSVs (system_metrics, application_logs)
│   ├── raw_500k/                 500K row version
│   ├── parquet/                  Processed parquet (100K)
│   ├── parquet_500k/             Processed parquet (500K)
│   └── features/                 Engineered feature matrix
│
├── models/                       Trained artifacts (joblib, JSON metadata)
├── notebooks/                    01 through 05 — exploration to deployment
├── tests/
│   ├── test_api.py
│   ├── test_agent.py
│   └── e2e/                      Playwright tests (api.spec.ts, dashboard.spec.ts)
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── docs/                         AWS deployment guide, portfolio, technical design
├── papers/                       Reference research papers (Papers 1–4)
├── scripts/
│   └── run-tests.sh
├── .github/workflows/            CI (ci.yml) and CD (cd.yml)
├── TECHNICAL_DESIGN.md           End-to-end engineering reference document
├── requirements.txt
├── pyproject.toml
└── playwright.config.ts
```

---

## Literature

SentinelX is built on four papers, each addressing a different layer of the pipeline. The full PDFs are in `papers/`.

---

**Paper 1 — A Hybrid Machine Learning Approach**

This paper introduces a hybrid architecture combining Self-Organizing Feature Maps (SOFM) and Support Vector Machines (SVM) for multi-modal anomaly detection. The central argument is that hardware and software signals carry complementary failure information — neither domain alone is sufficient. Fusing them at the feature level, rather than at the decision level, captures cross-domain causal chains that single-modality models miss entirely.

In SentinelX this translates directly into the cross-domain feature set: `thermal_efficiency_idx` (CPU utilization normalized by temperature), `power_anomaly_score` (deviation from expected mechanical power curve), and `error_under_stress` (error rate conditioned on high torque). These features encode physical relationships between hardware state and software behavior that standard tabular features cannot represent. The SHAP analysis confirms these cross-domain features rank consistently in the top-15 most influential predictors.

---

**Paper 2 — Anomaly Detection with XGBoost and SMOTE for Imbalanced Industrial Data**

This paper addresses the class imbalance problem in industrial fault detection, where failure events typically represent less than 5% of observations. It benchmarks several oversampling and algorithmic approaches and recommends XGBoost with SMOTE applied strictly within cross-validation folds. The key finding is that applying SMOTE before splitting — a common shortcut — produces optimistic performance estimates because synthetic minority samples generated from the full dataset contaminate the validation set.

SentinelX implements this protocol exactly: SMOTE is called inside each fold of Stratified K-Fold CV, never on the full dataset. The paper also informs the choice of `sampling_strategy=0.5` (not full balance at 1.0) to preserve the model's awareness that failures are rare, and `scale_pos_weight=1.0` in XGBoost to avoid double-correcting for imbalance when SMOTE is already applied. PR-AUC is used as the primary evaluation metric throughout, as the paper demonstrates that ROC-AUC is misleading at this class ratio.

---

**Paper 3 — Alert Fatigue in Industrial Monitoring Systems**

This paper quantifies the alert fatigue problem in production monitoring environments. It establishes that the industry average false positive rate in anomaly detection systems sits around 54%, and that at this level operators begin ignoring alerts systematically — with 73% of alerts going unacknowledged. The downstream effect is that genuine failures get missed not because the model failed to detect them, but because the operator no longer trusts the system. The paper proposes SHAP-based explanations as a trust-building mechanism: when operators understand why an alert fired, they engage with it more reliably.

This directly shapes SentinelX's design goals (1.4% FP target), its database schema (the `acknowledged`, `false_positive`, and `action_taken` columns exist specifically to measure and track operator response), and the `/fatigue` API endpoint that surfaces acknowledgment rates and FP rates over time. The SHAP diagnostic report attached to every prediction is the implementation of the paper's explainability recommendation.

---

**Paper 4 — Deep Learning-Based Real-Time Data Quality Assessment**

This paper argues that data quality assessment must happen in-stream rather than as a batch post-processing step. By the time a batch quality check catches a corrupted sensor reading, the model has already ingested and acted on it. The paper proposes physics-based bounds (not statistical outlier thresholds) as quality gates — values outside physically possible ranges are definitively sensor errors, not anomalies. It also introduces Population Stability Index (PSI) as a lightweight, stateless distribution shift detector suitable for streaming contexts, with PSI > 0.2 as the threshold for triggering retraining.

SentinelX's `QualityGate` class implements the in-stream flagging pattern: rows are annotated with a bitwise quality flag rather than dropped, preserving rare failure events that would otherwise be discarded. The `DriftMonitor` class implements PSI per column against a fixed baseline, logging a warning when any column exceeds the threshold. The chunk boundary lookback buffer — which ensures rolling window features computed at chunk boundaries have the same historical context as features computed mid-chunk — is a direct response to the paper's requirement that streaming feature computation produce identical results to batch computation.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker run -d --name sentinelx-db \
  -e POSTGRES_USER=sentinelx \
  -e POSTGRES_PASSWORD=sentinelx_secure_2024 \
  -e POSTGRES_DB=sentinelx \
  -p 5432:5432 postgres:15

# Initialize database schema
python src/api/database.py --init

# Start API
python src/api/main.py

# Start Dashboard (separate terminal)
streamlit run src/dashboard/dashboard.py
```

### Running Tests

```bash
# Quick health check
curl http://localhost:8000/health

# Prediction test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json

# LLM agent demo (requires Ollama running)
OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=llama3.2 python src/ml/agent.py --llm
```

---

## License

MIT License
