# SentinelX AWS Deployment Strategy

## The "Digital Twin" Architecture

Your local development environment is a **Digital Twin** of production AWS infrastructure. This means:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL (Development)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │   Browser    │────▶│   FastAPI    │────▶│  PostgreSQL (Docker)     │   │
│   │   /curl      │     │   uvicorn    │     │  localhost:5432          │   │
│   └──────────────┘     │   :8000      │     └──────────────────────────┘   │
│                        └──────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SAME CODE
                                    │ Change only .env
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AWS (Production)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│   │     ALB      │────▶│  ECS Fargate │────▶│      Amazon RDS          │   │
│   │  (HTTPS)     │     │  (Container) │     │  PostgreSQL 15           │   │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘   │
│         │                      │                       │                     │
│         │                      │                       │                     │
│   ┌─────▼──────┐        ┌─────▼──────┐         ┌─────▼──────┐              │
│   │   Route53  │        │    ECR     │         │  Secrets   │              │
│   │  (DNS)     │        │  (Images)  │         │  Manager   │              │
│   └────────────┘        └────────────┘         └────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Is a Digital Twin

| Component | Local | AWS | Change Required |
|-----------|-------|-----|-----------------|
| Database | Docker PostgreSQL | Amazon RDS PostgreSQL | `.env` only |
| API Server | uvicorn | ECS Fargate | Dockerfile (same) |
| Load Balancer | N/A | Application Load Balancer | AWS Console |
| Secrets | `.env` file | Secrets Manager | ECS task definition |
| Connection Pool | SQLAlchemy QueuePool | Same | None |
| JSONB Support | PostgreSQL 15 | PostgreSQL 15 | None |

**The code is IDENTICAL.** Only configuration changes.

---

## Step-by-Step AWS Deployment

### Phase 1: Infrastructure Setup (One-Time)

#### 1.1 Create RDS PostgreSQL Instance

```bash
# Using AWS CLI
aws rds create-db-instance \
  --db-instance-identifier sentinelx-prod \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.4 \
  --master-username sentinelx_admin \
  --master-user-password <GENERATE_SECURE_PASSWORD> \
  --allocated-storage 100 \
  --storage-type gp3 \
  --vpc-security-group-ids sg-xxxxxxxx \
  --db-subnet-group-name sentinelx-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted \
  --tags Key=Project,Value=SentinelX
```

**Why these settings?**
- `db.t3.medium`: Burstable, cost-effective for initial deployment
- `multi-az`: High availability (automatic failover)
- `storage-encrypted`: Compliance requirement
- `gp3`: Best price/performance for IOPS

#### 1.2 Store Credentials in Secrets Manager

```bash
aws secretsmanager create-secret \
  --name sentinelx/prod/database \
  --secret-string '{
    "DB_HOST": "sentinelx-prod.xxxxx.us-east-1.rds.amazonaws.com",
    "DB_PORT": "5432",
    "DB_NAME": "sentinelx",
    "DB_USER": "sentinelx_admin",
    "DB_PASS": "<YOUR_SECURE_PASSWORD>"
  }'
```

#### 1.3 Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name sentinelx \
  --image-scanning-configuration scanOnPush=true
```

### Phase 2: Container Setup

#### 2.1 Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with production settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### 2.2 Create requirements.txt

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
python-dotenv==1.0.0
pydantic==2.5.3
pandas==2.1.4
numpy==1.26.3
xgboost==2.0.3
shap==0.44.1
joblib==1.3.2
pyarrow==15.0.0
```

#### 2.3 Build and Push to ECR

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t sentinelx .
docker tag sentinelx:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/sentinelx:latest

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/sentinelx:latest
```

### Phase 3: ECS Deployment

#### 3.1 Create ECS Task Definition

```json
{
  "family": "sentinelx-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789:role/sentinelx-task-role",
  "containerDefinitions": [
    {
      "name": "sentinelx-api",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/sentinelx:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "secrets": [
        {
          "name": "DB_HOST",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:sentinelx/prod/database:DB_HOST::"
        },
        {
          "name": "DB_PORT",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:sentinelx/prod/database:DB_PORT::"
        },
        {
          "name": "DB_NAME",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:sentinelx/prod/database:DB_NAME::"
        },
        {
          "name": "DB_USER",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:sentinelx/prod/database:DB_USER::"
        },
        {
          "name": "DB_PASS",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:sentinelx/prod/database:DB_PASS::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentinelx",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 3.2 Create ECS Service

```bash
aws ecs create-service \
  --cluster sentinelx-cluster \
  --service-name sentinelx-api \
  --task-definition sentinelx-api:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-zzz],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=sentinelx-api,containerPort=8000"
```

### Phase 4: Load Balancer & DNS

#### 4.1 Create Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name sentinelx-alb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-alb \
  --scheme internet-facing \
  --type application

# Create target group
aws elbv2 create-target-group \
  --name sentinelx-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx \
  --target-type ip \
  --health-check-path /health

# Create HTTPS listener (requires ACM certificate)
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:... \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

---

## Cost Estimation

| Component | Specification | Monthly Cost (US-East-1) |
|-----------|---------------|--------------------------|
| RDS PostgreSQL | db.t3.medium, Multi-AZ, 100GB | ~$70 |
| ECS Fargate | 2 tasks, 1vCPU, 2GB RAM | ~$60 |
| ALB | + data transfer | ~$25 |
| ECR | 10GB storage | ~$1 |
| Secrets Manager | 5 secrets | ~$2 |
| CloudWatch Logs | 10GB/month | ~$5 |
| **Total** | | **~$163/month** |

**Scaling costs:**
- Each additional Fargate task: ~$30/month
- RDS scale up to db.r6g.large: +$200/month

---

## Environment Variable Mapping

| Local (.env) | AWS (Secrets Manager) |
|--------------|----------------------|
| `DB_HOST=localhost` | `DB_HOST=sentinelx-prod.xxxxx.rds.amazonaws.com` |
| `DB_PORT=5432` | `DB_PORT=5432` |
| `DB_NAME=postgres` | `DB_NAME=sentinelx` |
| `DB_USER=postgres` | `DB_USER=sentinelx_admin` |
| `DB_PASS=sentinelx123` | `DB_PASS=<32-char-random>` |

---

## Monitoring & Alerting

### CloudWatch Alarms

```bash
# High CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "SentinelX-HighCPU" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:...:sentinelx-alerts

# Database connections alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "SentinelX-DBConnections" \
  --metric-name DatabaseConnections \
  --namespace AWS/RDS \
  --statistic Average \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:...:sentinelx-alerts
```

---

## Quick Migration Checklist

- [ ] Create RDS PostgreSQL instance
- [ ] Store credentials in Secrets Manager
- [ ] Create ECR repository
- [ ] Build and push Docker image
- [ ] Create ECS cluster and task definition
- [ ] Create ALB with HTTPS listener
- [ ] Create ECS service with load balancer
- [ ] Configure Route53 DNS (optional)
- [ ] Set up CloudWatch alarms
- [ ] Test `/health` endpoint
- [ ] Run production smoke test

**Total deployment time: ~2 hours** (mostly waiting for RDS provisioning)
