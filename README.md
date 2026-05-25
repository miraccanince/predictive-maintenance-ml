# SentinelX: Predictive Maintenance Platform

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20Ready-FF9900?logo=amazon-aws)](https://aws.amazon.com/ecs/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Playwright-65%20Tests-45ba4b?logo=playwright)](https://playwright.dev/)

SentinelX is a production-grade AIOps platform for predictive maintenance. It ingests dual-domain telemetry (hardware sensors + application logs), engineers temporal features, trains an imbalanced classifier, and serves real-time failure predictions with SHAP-based explanations through a FastAPI service backed by PostgreSQL.

The core problem it addresses: industrial monitoring systems generate too many false alerts. Operators start ignoring them. Real failures get missed. SentinelX brings the false positive rate from the 54% industry average down to 1.4% by combining cross-domain feature fusion, proper imbalance handling, and explainable predictions that operators can actually trust.

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
| 8 | Dashboard | Streamlit visualization of savings and alert metrics |
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

## Value Proposition

Based on Paper 3 cost assumptions ($5,000 per missed failure, $500 per unnecessary maintenance):

```
Industry Baseline (54% False Positive Rate)
  Operators ignore 73% of alerts (alert fatigue)
  Missed failures cost: $5,000 each
  Unnecessary maintenance: $500 each

SentinelX (1.4% False Positive Rate)
  97% reduction in false alarms
  95% failure detection rate (recall)
  SHAP explanations build operator trust

  NET ANNUAL SAVINGS: $120,900,000
```

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
| 8 | `src/dashboard/dashboard.py` | Metrics and savings dashboard |

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

**Savings counter** — real-time calculation of avoided costs versus industry baseline, with annual projection based on current alert rate.

**Alert fatigue monitor** — tracks the 1.4% vs 54% FP rate comparison, acknowledgment rate, and Paper 3 compliance metrics over configurable time windows.

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

## Research Foundation

| Paper | Contribution | Implementation |
|-------|--------------|----------------|
| Paper 1 | SOFM+SVM hybrid | Cross-domain feature fusion (thermal_efficiency_idx, power_anomaly_score) |
| Paper 2 | XGBoost+SMOTE | Imbalanced classification, SMOTE inside CV loop |
| Paper 3 | Alert fatigue | 1.4% FP rate, SHAP for operator trust |
| Paper 4 | Real-time quality | Streaming with in-stream quality gates and PSI drift detection |

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
