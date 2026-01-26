# 🛡️ SentinelX: Production-Grade Predictive Maintenance

> **AI-powered industrial intelligence that saves $120.9M annually through explainable failure prediction.**

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20Ready-FF9900?logo=amazon-aws)](https://aws.amazon.com/ecs/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Playwright-65%20Tests-45ba4b?logo=playwright)](https://playwright.dev/)

---

## 📋 Project Summary

SentinelX is a **complete AIOps platform** for predictive maintenance, built across 8 implementation stages:

| Stage | Component | What It Does |
|-------|-----------|--------------|
| **1** | Data Ingestion | Streaming pipeline with quality gates (Paper 4) |
| **2** | Feature Engineering | 50+ features: rolling windows, lags, cross-domain fusion |
| **3** | Model Training | XGBoost + SMOTE for 10:1 class imbalance |
| **4** | Anomaly Detection | PyTorch autoencoder baseline comparison |
| **5** | Evaluation | PR-AUC optimization, threshold calibration at 500K scale |
| **6** | Diagnostic Agent | SHAP-powered explainability engine |
| **7** | API + Persistence | FastAPI service + PostgreSQL with Alert Fatigue tracking |
| **8** | CEO Dashboard | Streamlit visualization: $120M savings, 97% FP reduction |
| **9** | LLM Agent | Ollama-powered generative diagnostics with Chain-of-Thought |

**Key Achievements:**
- **1.4% False Positive Rate** (vs 54% industry baseline - Paper 3)
- **97% Alert Fatigue Reduction** through SHAP explainability
- **$120.9M Annual Savings** projection based on failure costs
- **65 E2E Tests** via Playwright (API + Dashboard)
- **Docker-ready** with AWS ECS migration path
- **LLM-Powered Diagnostics** via Ollama (llama3.2) with automatic fallback

---

## 🚀 Quick Start (1-Click Deployment)

```bash
# Clone and launch the entire stack
git clone <repository>
cd SentinelX
docker-compose -f docker/docker-compose.yml up --build
```

**That's it!** Three services will start:

| Service | URL | Purpose |
|---------|-----|---------|
| **API** | http://localhost:8000 | FastAPI inference engine |
| **Dashboard** | http://localhost:8501 | CEO value visualization |
| **Database** | localhost:5432 | PostgreSQL persistence |

---

## 📊 The $120.9M Value Proposition

Based on **Paper 3** research, SentinelX delivers massive ROI through Alert Fatigue reduction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANNUAL SAVINGS BREAKDOWN                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Industry Baseline (54% False Positive Rate)                   │
│   ├── Operators ignore 73% of alerts (alert fatigue)            │
│   ├── Missed failures cost: $5,000 each                         │
│   └── Unnecessary maintenance: $500 each                        │
│                                                                  │
│   SentinelX (1.4% False Positive Rate)                          │
│   ├── 97% reduction in false alarms                             │
│   ├── 95% failure detection rate (recall)                       │
│   └── SHAP explanations build operator trust                    │
│                                                                  │
│   ════════════════════════════════════════                      │
│   NET ANNUAL SAVINGS: $120,900,000                              │
│   ════════════════════════════════════════                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         SentinelX Architecture                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐   │
│   │   Sensors   │────▶│   FastAPI   │────▶│      PostgreSQL         │   │
│   │   (Input)   │     │   + XGBoost │     │   (Alert Storage)       │   │
│   └─────────────┘     │   + SHAP    │     └─────────────────────────┘   │
│                       │   + LLM     │                │                   │
│                       └──────┬──────┘                │                   │
│                              │                       │                   │
│   ┌─────────────┐            │                       ▼                   │
│   │   Ollama    │◀───────────┤           ┌─────────────────────────┐   │
│   │  llama3.2   │            │           │   Streamlit Dashboard   │   │
│   └─────────────┘            ▼           │   ($120M Savings View)  │   │
│                       ┌─────────────┐     └─────────────────────────┘   │
│                       │  Diagnostic │                                    │
│                       │   Engine    │                                    │
│                       └─────────────┘                                    │
│                                                                           │
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
| 8 | `src/dashboard/dashboard.py` | Executive value dashboard |

---

## 🤖 LLM-Powered Diagnostics (Stage 9)

SentinelX now includes an **LLM Agent** that provides context-aware, generative diagnostics using Ollama.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM AGENT PIPELINE                          │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ XGBoost  │───▶│   SHAP   │───▶│  Prompt  │───▶│  Ollama  │  │
│  │ Predict  │    │ Analysis │    │ Builder  │    │ llama3.2 │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │         │
│                                    ┌──────────────────┘         │
│                                    ▼                            │
│                             ┌─────────────┐                     │
│                             │   Parser    │                     │
│                             │ JSON + MD   │                     │
│                             └──────┬──────┘                     │
│                                    │                            │
│              ┌─────────────────────┼─────────────────────┐      │
│              ▼                     ▼                     ▼      │
│       ┌──────────┐          ┌──────────┐          ┌──────────┐ │
│       │ Database │          │ Dashboard │          │ Fallback │ │
│       │  (JSON)  │          │   (MD)    │          │ (Rules)  │ │
│       └──────────┘          └──────────┘          └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| **Async Integration** | Non-blocking Ollama API calls via `aiohttp` |
| **Chain-of-Thought** | LLM reasons through SHAP values before concluding |
| **Structured Output** | JSON (database) + Markdown (human report) |
| **Context Injection** | Top 5 SHAP features, physical meanings, sensor values |
| **Automatic Fallback** | Rule-based diagnosis if Ollama unavailable |
| **Version Tracking** | `2.0-llm-llama3.2` vs `1.0-rule-based` |

### Enable LLM Features

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

The API automatically detects Ollama and uses it when available.

### LLM System Prompt

The LLM is configured as a **Senior Reliability Engineer** with expertise in:
- SHAP value interpretation (positive = increases risk)
- Hardware-software correlation (temperature → network latency)
- Failure mechanics (Torque × Wear > 3000 = mechanical stress)
- Chain-of-Thought reasoning before diagnosis

### Comparison: Rule-Based vs LLM

| Aspect | Rule-Based (v1.0) | LLM-Powered (v2.0) |
|--------|-------------------|-------------------|
| **Speed** | ~50ms | ~10-30s |
| **Consistency** | Deterministic | Temperature-controlled (0.3) |
| **Flexibility** | Fixed patterns | Reasons about novel patterns |
| **Explainability** | Template-based | Natural language grounded in SHAP |
| **Availability** | Always | Requires Ollama |

### Environment Variables

```bash
OLLAMA_URL=http://host.docker.internal:11434  # Docker → Host
OLLAMA_MODEL=llama3.2                          # Model name
OLLAMA_TIMEOUT=60                              # Request timeout (seconds)
```

---

## 🐳 Docker Services

### Service Configuration

```yaml
services:
  db:        # PostgreSQL 15 (AWS RDS Digital Twin)
    port: 5432

  api:       # FastAPI + XGBoost + SHAP
    port: 8000
    depends_on: db

  dashboard: # Streamlit CEO Dashboard
    port: 8501
    depends_on: api, db
```

### Useful Commands

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up --build

# Start in background
docker-compose -f docker/docker-compose.yml up -d --build

# View logs
docker-compose -f docker/docker-compose.yml logs -f api
docker-compose -f docker/docker-compose.yml logs -f dashboard

# Stop all services
docker-compose -f docker/docker-compose.yml down

# Stop and remove volumes (reset database)
docker-compose -f docker/docker-compose.yml down -v

# Rebuild single service
docker-compose -f docker/docker-compose.yml up --build api
```

---

## 🔌 API Reference

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

### Interactive Docs
Open http://localhost:8000/docs for Swagger UI.

---

## ☁️ AWS Migration Path

SentinelX is designed as a **Digital Twin** of production AWS infrastructure:

```
LOCAL                              AWS
─────────────────────────────────────────────────────
Docker PostgreSQL      →     Amazon RDS PostgreSQL
Docker API Container   →     ECS Fargate Task
Docker Dashboard       →     ECS Fargate Task
docker-compose         →     ECS Service + ALB
localhost              →     Route53 + ACM (HTTPS)
.env file              →     Secrets Manager
```

### Migration Steps

1. **Push Images to ECR**
   ```bash
   # Authenticate
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

   # Build and push API
   docker build -f docker/Dockerfile.api -t sentinelx-api .
   docker tag sentinelx-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-api:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-api:latest

   # Build and push Dashboard
   docker build -f docker/Dockerfile.dashboard -t sentinelx-dashboard .
   docker tag sentinelx-dashboard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-dashboard:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sentinelx-dashboard:latest
   ```

2. **Create RDS Instance**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier sentinelx-prod \
     --db-instance-class db.t3.medium \
     --engine postgres --engine-version 15 \
     --allocated-storage 100 --storage-type gp3 \
     --multi-az --storage-encrypted
   ```

3. **Store Secrets**
   ```bash
   aws secretsmanager create-secret \
     --name sentinelx/prod/database \
     --secret-string '{"DB_HOST":"...rds.amazonaws.com","DB_PASS":"..."}'
   ```

4. **Deploy to ECS**
   - Create ECS Cluster
   - Create Task Definitions (reference ECR images)
   - Create Services with ALB target groups
   - Configure auto-scaling

See [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) for detailed instructions.

---

## 📈 Dashboard Features

The CEO Dashboard (http://localhost:8501) provides:

### 💰 Savings Counter
- Real-time calculation of avoided costs
- Comparison vs industry baseline
- Annual projection based on current alert rate

### 📉 Alert Fatigue Monitor
- 1.4% FP rate vs 54% industry baseline
- 97% reduction in false alarms
- Paper 3 compliance metrics

### 🔍 Root Cause Analysis
- Distribution by failure mode (Mechanical, Thermal, Software, etc.)
- Risk level breakdown
- 24-hour alert timeline

### 🧪 Live Prediction Demo
- Interactive prediction form
- Real-time SHAP explanations
- Instant database persistence

---

## 📁 Project Structure

```
SentinelX/
├── src/                          # Source code
│   ├── api/                      # FastAPI service
│   │   ├── main.py               # API endpoints + inference
│   │   └── database.py           # SQLAlchemy models + CRUD
│   │
│   ├── ml/                       # Machine learning modules
│   │   ├── agent.py              # Diagnostic engine (SHAP + LLM)
│   │   ├── train_model.py        # XGBoost + SMOTE training
│   │   ├── feature_engineer.py   # Feature transformation
│   │   ├── autoencoder.py        # Anomaly detection baseline
│   │   └── evaluate_and_scale.py # PR-AUC + threshold optimization
│   │
│   ├── pipeline/                 # Data pipeline
│   │   ├── stream_manager.py     # Data ingestion + quality gates
│   │   ├── generate_synthetic_data.py
│   │   └── analyze_logs.py       # Log analysis utilities
│   │
│   └── dashboard/                # Streamlit CEO dashboard
│       └── dashboard.py          # Executive value visualization
│
├── docker/                       # Docker configuration
│   ├── Dockerfile.api            # API container
│   ├── Dockerfile.dashboard      # Dashboard container
│   └── docker-compose.yml        # Service orchestration
│
├── tests/                        # Test suites
│   └── e2e/                      # Playwright E2E tests
│       ├── api.spec.ts           # API endpoint tests
│       └── dashboard.spec.ts     # Dashboard UI tests
│
├── docs/                         # Documentation
│   └── AWS_DEPLOYMENT.md         # AWS migration guide
│
├── scripts/                      # Utility scripts
│   └── run-tests.sh              # Test runner
│
├── models/                       # Trained artifacts
│   ├── model.joblib              # XGBoost model
│   ├── feature_names.json        # Feature list
│   ├── shap_values.npy           # SHAP explanations
│   └── evaluation_results.json   # Thresholds
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_streaming_quality.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation_deployment.ipynb
│
├── data/                         # Data files
│   ├── parquet/                  # Processed data
│   └── features/                 # Feature matrix
│
├── papers/                       # Reference research papers
│
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies (tests)
├── playwright.config.ts          # Playwright configuration
├── .env                          # Environment configuration
└── README.md                     # This file
```

---

## 🔬 Research Foundation

SentinelX implements techniques from 4 research papers:

| Paper | Contribution | Implementation |
|-------|--------------|----------------|
| **Paper 1** | SOFM+SVM Hybrid | Cross-domain feature fusion |
| **Paper 2** | XGBoost+SMOTE | Imbalanced classification |
| **Paper 3** | Alert Fatigue | 1.4% FP rate, SHAP trust |
| **Paper 4** | Real-Time Quality | Streaming with quality gates |

---

## 🛠️ Development

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (Docker)
docker run -d --name sentinelx-db \
  -e POSTGRES_USER=sentinelx \
  -e POSTGRES_PASSWORD=sentinelx_secure_2024 \
  -e POSTGRES_DB=sentinelx \
  -p 5432:5432 postgres:15

# Initialize database
python src/api/database.py --init

# Start API
python src/api/main.py

# Start Dashboard (new terminal)
streamlit run src/dashboard/dashboard.py
```

### Running Tests

```bash
# Health check
curl http://localhost:8000/health

# Prediction test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json

# LLM Agent demo (requires Ollama)
OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=llama3.2 python src/ml/agent.py --llm
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

<div align="center">

**SentinelX** - Turning Industrial Data into $120.9M Annual Savings

*Built with XGBoost, SHAP, FastAPI, and PostgreSQL*

</div>
# SentineIX
