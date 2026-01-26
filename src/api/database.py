"""
SentinelX Database Layer (Stage 7)
==================================

SQLAlchemy ORM models and database operations for the SentinelX alerts system.

Architecture:
    FastAPI → Database Layer → PostgreSQL (Local) → [AWS RDS in Production]

This is a "Digital Twin" of production infrastructure:
- Local PostgreSQL container mimics AWS RDS behavior exactly
- Same connection pooling, same constraints, same JSONB operations
- Migration to AWS = change DB_HOST in .env (that's it!)

Reference: Paper 3 - Store metadata to analyze 'Alert Fatigue' patterns:
- How many alerts per machine/day?
- What's the false positive rate by root cause?
- Which thresholds generate the most noise?
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Boolean, Index, func, text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelX.Database")

# =============================================================================
# 1. DATABASE CONFIGURATION
# =============================================================================
# WHY environment variables + dotenv?
# - 12-Factor App compliance (config in environment, not code)
# - Same code runs local (Docker) and production (RDS)
# - Secrets never committed to git (.env in .gitignore)
#
# AWS Strategy:
# - Local: DB_HOST=localhost (Docker container)
# - AWS: DB_HOST=sentinelx.xxxxx.us-east-1.rds.amazonaws.com
# - All other code stays IDENTICAL

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sentinelx")
DB_USER = os.getenv("DB_USER", "sentinelx")
DB_PASS = os.getenv("DB_PASS", "sentinelx_dev_2024")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# =============================================================================
# 2. ENGINE & SESSION FACTORY
# =============================================================================
# WHY connection pooling (QueuePool)?
# - Production FastAPI handles concurrent requests
# - Each request needs a DB connection
# - Creating new connections is SLOW (TCP handshake, auth)
# - Pool reuses connections: 10ms → 0.1ms per query
#
# Pool Configuration (matches AWS RDS best practices):
# - pool_size=5: Baseline connections (matches RDS min_connections)
# - max_overflow=10: Burst capacity (15 total max)
# - pool_pre_ping=True: Detect stale connections (RDS has timeout)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Prevents "connection reset" errors
    echo=False  # Set True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# =============================================================================
# 3. ORM MODELS
# =============================================================================
# Reference: Paper 3 - Alert Fatigue Analysis requires rich metadata

class Alert(Base):
    """
    SentinelX Alert Record

    Stores every prediction + diagnostic report for:
    1. Real-time alerting (Markdown report)
    2. Historical analysis (Alert Fatigue metrics)
    3. Model retraining (raw_data for feature drift detection)

    Paper 3 Compliance:
    - acknowledged: Track which alerts were acted upon
    - false_positive: Label for FP rate calculation
    - root_cause: Group alerts by diagnosed cause
    - created_at: Time-series analysis of alert frequency
    """
    __tablename__ = "alerts"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core Identification
    machine_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Model Outputs
    failure_prob = Column(Float, nullable=False)  # Raw probability 0-1
    risk_level = Column(String(20), nullable=False)  # nominal/low/moderate/high/critical
    root_cause = Column(String(50), nullable=True)  # Diagnosed failure mode
    confidence = Column(String(20), nullable=True)  # Diagnosis confidence

    # Human-Readable Report
    report_md = Column(Text, nullable=False)  # Full Markdown diagnostic

    # Machine-Readable Data (for retraining & analysis)
    raw_data = Column(JSONB, nullable=False)  # Original input metrics
    shap_values = Column(JSONB, nullable=True)  # SHAP explanation

    # Alert Fatigue Tracking (Paper 3)
    acknowledged = Column(Boolean, default=False)  # Was this alert seen?
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)

    false_positive = Column(Boolean, nullable=True)  # Labeled after fact
    action_taken = Column(Text, nullable=True)  # What maintenance was done?

    # Metadata
    model_version = Column(String(50), default="1.0")
    threshold_used = Column(String(50), nullable=True)  # Which threshold config
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Composite indexes for common queries
    __table_args__ = (
        # Alert Fatigue query: alerts per machine per day
        Index('ix_alerts_machine_date', machine_id, func.date(timestamp)),
        # FP analysis: filter by root cause + FP label
        Index('ix_alerts_rootcause_fp', root_cause, false_positive),
        # Time-series: recent high-risk alerts
        Index('ix_alerts_risk_time', risk_level, created_at.desc()),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "failure_prob": self.failure_prob,
            "risk_level": self.risk_level,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "acknowledged": self.acknowledged,
            "false_positive": self.false_positive,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AlertFatigueMetrics(Base):
    """
    Aggregated metrics for Alert Fatigue analysis (Paper 3).

    Precomputed daily aggregates to avoid expensive queries:
    - Total alerts per machine per day
    - Alerts by risk level
    - Acknowledgment rate
    - False positive rate

    WHY precompute?
    - Production query: "Show me last 30 days of alert metrics"
    - Without aggregates: Scan millions of rows → 10+ seconds
    - With aggregates: 30 rows → 5ms
    """
    __tablename__ = "alert_fatigue_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    machine_id = Column(String(50), nullable=False, index=True)

    # Volume metrics
    total_alerts = Column(Integer, default=0)
    critical_count = Column(Integer, default=0)
    high_count = Column(Integer, default=0)
    moderate_count = Column(Integer, default=0)
    low_count = Column(Integer, default=0)
    nominal_count = Column(Integer, default=0)

    # Quality metrics
    acknowledged_count = Column(Integer, default=0)
    false_positive_count = Column(Integer, default=0)
    true_positive_count = Column(Integer, default=0)

    # Root cause distribution
    root_cause_distribution = Column(JSONB, nullable=True)

    __table_args__ = (
        Index('ix_fatigue_date_machine', date, machine_id, unique=True),
    )


# =============================================================================
# 4. DATABASE OPERATIONS (CRUD)
# =============================================================================

def get_db() -> Session:
    """
    Dependency injection for FastAPI routes.

    Usage:
        @app.post("/predict")
        def predict(db: Session = Depends(get_db)):
            ...

    The 'yield' pattern ensures connection is returned to pool after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables if they don't exist.

    Safe to call multiple times (CREATE IF NOT EXISTS).
    In production, use Alembic migrations for schema changes.
    """
    logger.info(f"Initializing database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def create_alert(
    db: Session,
    machine_id: str,
    timestamp: datetime,
    failure_prob: float,
    risk_level: str,
    root_cause: Optional[str],
    confidence: Optional[str],
    report_md: str,
    raw_data: Dict[str, Any],
    shap_values: Optional[Dict[str, Any]] = None,
    threshold_used: Optional[str] = None,
    model_version: str = "1.0"
) -> Alert:
    """
    Create a new alert record.

    Args:
        db: Database session
        machine_id: Machine identifier
        timestamp: Time of prediction
        failure_prob: Model output probability
        risk_level: Classified risk level
        root_cause: Diagnosed failure mode
        confidence: Diagnosis confidence
        report_md: Full Markdown report
        raw_data: Original input metrics (for retraining)
        shap_values: SHAP explanation dict
        threshold_used: Which threshold configuration
        model_version: Model version string

    Returns:
        Created Alert object with ID assigned
    """
    alert = Alert(
        machine_id=machine_id,
        timestamp=timestamp,
        failure_prob=failure_prob,
        risk_level=risk_level,
        root_cause=root_cause,
        confidence=confidence,
        report_md=report_md,
        raw_data=raw_data,
        shap_values=shap_values,
        threshold_used=threshold_used,
        model_version=model_version
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    logger.info(f"Alert created: id={alert.id}, machine={machine_id}, risk={risk_level}")
    return alert


def get_alerts(
    db: Session,
    machine_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Alert]:
    """
    Query alerts with optional filters.

    Args:
        db: Database session
        machine_id: Filter by machine (optional)
        risk_level: Filter by risk level (optional)
        limit: Max results
        offset: Pagination offset

    Returns:
        List of Alert objects
    """
    query = db.query(Alert)

    if machine_id:
        query = query.filter(Alert.machine_id == machine_id)
    if risk_level:
        query = query.filter(Alert.risk_level == risk_level)

    return query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()


def get_alert_fatigue_summary(
    db: Session,
    machine_id: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]:
    """
    Calculate Alert Fatigue metrics (Paper 3).

    Returns summary of alert volume and quality over recent period.

    Args:
        db: Database session
        machine_id: Filter by machine (optional)
        days: Number of days to analyze

    Returns:
        Dict with fatigue metrics
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = db.query(Alert).filter(Alert.created_at >= cutoff)
    if machine_id:
        query = query.filter(Alert.machine_id == machine_id)

    alerts = query.all()

    if not alerts:
        return {"total_alerts": 0, "message": "No alerts in period"}

    # Volume metrics
    total = len(alerts)
    by_risk = {}
    for alert in alerts:
        by_risk[alert.risk_level] = by_risk.get(alert.risk_level, 0) + 1

    # Quality metrics
    acknowledged = sum(1 for a in alerts if a.acknowledged)
    labeled = [a for a in alerts if a.false_positive is not None]
    fp_count = sum(1 for a in labeled if a.false_positive)

    # Alerts per day (fatigue indicator)
    alerts_per_day = total / days

    # Paper 3: Industry benchmark is 54% FP rate
    fp_rate = (fp_count / len(labeled) * 100) if labeled else None

    return {
        "period_days": days,
        "total_alerts": total,
        "alerts_per_day": round(alerts_per_day, 1),
        "by_risk_level": by_risk,
        "acknowledged_count": acknowledged,
        "acknowledgment_rate": round(acknowledged / total * 100, 1) if total else 0,
        "false_positive_count": fp_count,
        "false_positive_rate": round(fp_rate, 1) if fp_rate else "Not enough labeled data",
        "paper3_benchmark_fp_rate": 54.0,  # Industry average
    }


def acknowledge_alert(
    db: Session,
    alert_id: int,
    acknowledged_by: str,
    action_taken: Optional[str] = None,
    is_false_positive: Optional[bool] = None
) -> Optional[Alert]:
    """
    Mark an alert as acknowledged (for Alert Fatigue tracking).

    Args:
        db: Database session
        alert_id: Alert ID to acknowledge
        acknowledged_by: User/system that acknowledged
        action_taken: Description of maintenance action
        is_false_positive: Label if known

    Returns:
        Updated Alert or None if not found
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        return None

    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = acknowledged_by

    if action_taken:
        alert.action_taken = action_taken
    if is_false_positive is not None:
        alert.false_positive = is_false_positive

    db.commit()
    db.refresh(alert)

    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    return alert


# =============================================================================
# 5. HEALTH CHECK
# =============================================================================

def check_db_health() -> Dict[str, Any]:
    """
    Database health check for /health endpoint.

    Returns:
        Dict with connection status and basic stats
    """
    try:
        db = SessionLocal()
        # Simple query to verify connection
        result = db.execute(text("SELECT 1")).fetchone()

        # Get table stats
        alert_count = db.query(Alert).count()

        db.close()

        return {
            "status": "healthy",
            "database": f"{DB_HOST}:{DB_PORT}/{DB_NAME}",
            "connection_pool": f"{engine.pool.size()}/{engine.pool.size() + engine.pool.overflow()}",
            "alert_count": alert_count
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# =============================================================================
# 6. CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelX Database Management")
    parser.add_argument("--init", action="store_true", help="Initialize database tables")
    parser.add_argument("--health", action="store_true", help="Check database health")

    args = parser.parse_args()

    if args.init:
        init_db()
        print("Database initialized successfully!")
    elif args.health:
        health = check_db_health()
        print(json.dumps(health, indent=2))
    else:
        parser.print_help()
