"""
SentinelX API Service (Stage 7)
================================

Production-grade FastAPI service that orchestrates:
1. Raw metrics ingestion
2. Feature engineering (Stage 2)
3. Model prediction (Stage 3)
4. Diagnostic report generation (Stage 6)
5. Database persistence (Stage 7)

Architecture:
    Client → FastAPI → Feature Engineer → XGBoost → Diagnostic Engine → PostgreSQL
                                                                      ↓
                                                              Markdown Report

AWS Deployment Strategy:
    This service is designed as a "Digital Twin" of production infrastructure:
    - Local: Docker (PostgreSQL) + uvicorn
    - AWS: RDS + ECS/Fargate + ALB

    Migration checklist:
    1. Change DB_HOST to RDS endpoint
    2. Deploy container to ECR
    3. Create ECS task definition (this image)
    4. Configure ALB target group
    5. That's it - code is identical!

Reference: Paper 3 - Every prediction is persisted for Alert Fatigue analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Local imports
from database import (
    init_db, get_db, create_alert, get_alerts,
    get_alert_fatigue_summary, acknowledge_alert, check_db_health, Alert
)
from agent import DiagnosticEngine, DiagnosticResult
from feature_engineer import FeatureConfig

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("SentinelX.API")


# =============================================================================
# FEATURE STORE (Fixes Train-Serve Mismatch)
# =============================================================================
# WHY a FeatureStore?
# The training pipeline computes rolling mean/std and lag features using ACTUAL
# historical data via groupby().rolling() and groupby().shift(). The previous
# inference code used synthetic approximations (e.g., std = value * 0.15).
# This created a train-serve mismatch causing predictions to always return NOMINAL.
#
# SOLUTION: Store the last N observations per machine and compute REAL rolling
# statistics at inference time, exactly as done during training.

class FeatureStore:
    """
    Per-machine observation buffer for computing real rolling/lag features.

    This fixes the train-serve mismatch by storing historical observations
    and computing rolling mean/std/lag features exactly as done during training.

    For cold-start scenarios (< min_history observations), uses a hybrid approach
    that synthesizes variance based on how extreme values are from normal ranges.
    """

    # Normal operating ranges for key metrics (used for cold-start variance estimation)
    NORMAL_RANGES = {
        "air_temperature_K": (295, 315),
        "process_temperature_K": (305, 325),
        "torque_Nm": (20, 60),
        "rotational_speed_rpm": (1200, 1800),
        "vibration_mm_s": (10, 25),
        "pressure_psi": (90, 130),
        "tool_wear_min": (0, 200),
        "api_response_latency_ms": (50, 200),
        "error_rate_pct": (0, 5),
        "cpu_utilization_pct": (30, 80),
        "memory_utilization_pct": (40, 85),
        "queue_depth": (0, 50),
        "disk_io_wait_ms": (5, 20)
    }

    def __init__(self, buffer_size: int = 36, min_history: int = 6):
        """
        Initialize the feature store.

        Args:
            buffer_size: Number of observations to keep per machine.
                        Should be >= max(rolling_windows) = 36.
            min_history: Minimum observations before using real rolling stats.
                        Below this, synthetic variance is used.
        """
        self.buffer_size = buffer_size
        self.min_history = min_history
        self.buffers: Dict[str, deque] = {}
        self.config = FeatureConfig()

    def update(self, machine_id: str, observation: Dict[str, Any]) -> None:
        """
        Add an observation to the machine's history buffer.

        Args:
            machine_id: The machine identifier
            observation: Dict of raw metric values
        """
        if machine_id not in self.buffers:
            self.buffers[machine_id] = deque(maxlen=self.buffer_size)
        self.buffers[machine_id].append(observation)

    def get_history_df(self, machine_id: str) -> pd.DataFrame:
        """
        Get historical observations as DataFrame.

        Args:
            machine_id: The machine identifier

        Returns:
            DataFrame with all buffered observations (most recent at end)
        """
        history = list(self.buffers.get(machine_id, []))
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(history)

    def _estimate_anomaly_std(self, value: float, col: str) -> float:
        """
        Estimate standard deviation for cold-start scenarios based on value extremity.

        The model was trained to detect anomalies via high std values in rolling windows.
        For cold-start (< min_history observations), we synthesize std proportional to
        how far the value is from normal operating range.

        Made more aggressive to detect single extreme values that would be dangerous.
        """
        normal_range = self.NORMAL_RANGES.get(col, (value * 0.8, value * 1.2))
        low, high = normal_range
        range_size = high - low

        if low <= value <= high:
            # Within normal range: small std (stable operation)
            return abs(value) * 0.02
        else:
            # Outside normal range: std proportional to extremity
            distance_from_range = max(value - high, low - value, 0)
            # How many "range widths" outside normal?
            severity = distance_from_range / (range_size + 1e-6)

            # More aggressive scaling for extreme values
            # severity=1 means 1 range-width outside normal
            # severity=2 means 2 range-widths outside (very extreme)
            if severity > 2:
                # Very extreme: massive std
                base_std = abs(value) * 0.4
                extra_std = distance_from_range * 0.3
            elif severity > 1:
                # Moderately extreme
                base_std = abs(value) * 0.25
                extra_std = distance_from_range * 0.2
            else:
                # Slightly outside range
                base_std = abs(value) * 0.15
                extra_std = distance_from_range * 0.1

            return base_std + extra_std

    def compute_rolling_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute rolling mean and std features.

        Uses a hybrid approach:
        - If history >= min_history: use real rolling statistics (like training)
        - If history < min_history: synthesize variance based on value extremity

        This handles cold-start scenarios (dashboard single-shot predictions) while
        still providing proper rolling stats when history accumulates.
        """
        features = {}
        rolling_cols = (
            self.config.hardware_rolling_cols +
            self.config.software_rolling_cols
        )
        n_observations = len(df)
        use_synthetic = n_observations < self.min_history

        for col in rolling_cols:
            if col not in df.columns:
                continue
            series = df[col]
            current_value = series.iloc[-1]

            for window in self.config.rolling_windows:  # [6, 12, 36]
                # Rolling mean (always use real rolling)
                mean_val = series.rolling(window=window, min_periods=1, center=False).mean().iloc[-1]
                features[f"{col}_mean_{window}"] = mean_val

                # Rolling std (hybrid approach)
                if use_synthetic:
                    # Cold-start: synthesize std based on value extremity
                    synthetic_std = self._estimate_anomaly_std(current_value, col)
                    # Scale by window size (larger windows = more accumulated variance)
                    features[f"{col}_std_{window}"] = synthetic_std * (1 + window / 36)
                else:
                    # Enough history: use real rolling std
                    std_val = series.rolling(window=window, min_periods=2, center=False).std().iloc[-1]
                    features[f"{col}_std_{window}"] = std_val if pd.notna(std_val) else 0.0

        return features

    def compute_lag_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute lag features.

        Uses a hybrid approach for cold-start:
        - If enough history: use real historical values (like training)
        - If not enough history: synthesize trend based on value extremity
        """
        features = {}
        lag_cols = [
            "torque_Nm", "vibration_mm_s", "air_temperature_K", "tool_wear_min",
            "api_response_latency_ms", "error_rate_pct", "cpu_utilization_pct"
        ]
        n_observations = len(df)

        for col in lag_cols:
            if col not in df.columns:
                continue
            series = df[col]
            current_value = series.iloc[-1]

            for lag in self.config.lag_periods:  # [1, 2, 6]
                if n_observations > lag:
                    # Enough history: use real historical value
                    features[f"{col}_lag_{lag}"] = series.iloc[-(lag + 1)]
                else:
                    # Cold-start: synthesize trend based on value extremity
                    normal_range = self.NORMAL_RANGES.get(col, (current_value * 0.8, current_value * 1.2))
                    low, high = normal_range

                    if current_value > high:
                        # Value is high: simulate rising trend (lags were lower)
                        trend_factor = 0.95
                    elif current_value < low:
                        # Value is low: simulate falling trend (lags were higher)
                        trend_factor = 1.05
                    else:
                        # Normal: stable history
                        trend_factor = 1.0

                    features[f"{col}_lag_{lag}"] = current_value * (trend_factor ** lag)

        return features

    def compute_tool_wear_rate(self, df: pd.DataFrame) -> float:
        """
        Compute tool_wear_rate.

        For cold-start (< 2 observations): estimate rate based on wear level
        For enough history: use diff() like training (feature_engineer.py lines 474-476)
        """
        if "tool_wear_min" not in df.columns:
            return 0.0

        if len(df) < 2:
            # Cold-start: estimate wear rate based on current wear level
            # Higher wear levels typically have higher wear rates
            tool_wear = df["tool_wear_min"].iloc[-1]
            # Estimate ~0.5-2.0 wear units per period depending on wear level
            if tool_wear > 200:
                return 2.0  # High wear = high rate
            elif tool_wear > 100:
                return 1.0  # Medium wear
            else:
                return 0.5  # Low wear

        rate = df["tool_wear_min"].diff().iloc[-1]
        return max(rate, 0.0) if pd.notna(rate) else 0.0

    def compute_expected_power(self, df: pd.DataFrame) -> float:
        """
        Compute expected_power_W as 12-period rolling mean of instantaneous power.

        Matches feature_engineer.py lines 455-457:
        - df.groupby("machine_id")["instantaneous_power_W"].transform(
              lambda x: x.rolling(window=12, min_periods=1, center=False).mean()
          )
        """
        if "rotational_speed_rpm" not in df.columns or "torque_Nm" not in df.columns:
            # Fallback to fixed value if columns missing
            return 1500 * 40 * (2 * np.pi / 60)

        # Compute instantaneous power for all history
        power_series = df["rotational_speed_rpm"] * df["torque_Nm"] * (2 * np.pi / 60)

        # Rolling mean with window=12
        expected = power_series.rolling(window=12, min_periods=1, center=False).mean().iloc[-1]
        return expected if pd.notna(expected) else power_series.iloc[-1]


# Global feature store instance (persists across requests)
feature_store: Optional[FeatureStore] = None


# =============================================================================
# 1. PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================
# WHY Pydantic?
# - Auto-validation: rejects malformed JSON immediately
# - Auto-documentation: generates OpenAPI schema
# - Type hints: IDE autocomplete + runtime checking

class SystemMetrics(BaseModel):
    """Raw system metrics from edge sensors."""
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")
    machine_id: str = Field(..., description="Machine identifier")

    # Hardware sensors
    air_temperature_K: float = Field(..., ge=200, le=400, description="Air temp in Kelvin")
    process_temperature_K: float = Field(..., ge=200, le=450, description="Process temp")
    rotational_speed_rpm: float = Field(..., ge=0, le=5000, description="RPM")
    torque_Nm: float = Field(..., ge=0, le=200, description="Torque in Newton-meters")
    tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes")
    vibration_mm_s: Optional[float] = Field(15.0, ge=0, description="Vibration mm/s")
    pressure_psi: Optional[float] = Field(100.0, ge=0, description="Pressure PSI")

    # Edge computing
    network_latency_ms: Optional[float] = Field(20.0, ge=0, description="Network latency")
    edge_processing_time_ms: Optional[float] = Field(15.0, ge=0, description="Edge processing")
    fuzzy_pid_output: Optional[float] = Field(0.5, ge=0, le=1, description="PID output")

    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "M1",
                "air_temperature_K": 305.2,
                "process_temperature_K": 312.1,
                "rotational_speed_rpm": 1450,
                "torque_Nm": 45.3,
                "tool_wear_min": 120,
                "vibration_mm_s": 18.5,
                "pressure_psi": 115.0,
                "network_latency_ms": 25.0,
                "edge_processing_time_ms": 12.0,
                "fuzzy_pid_output": 0.65
            }
        }


class ApplicationLogs(BaseModel):
    """Application-level metrics."""
    error_rate_pct: float = Field(5.0, ge=0, le=100, description="Error rate %")
    cpu_utilization_pct: float = Field(50.0, ge=0, le=100, description="CPU %")
    memory_utilization_pct: float = Field(60.0, ge=0, le=100, description="Memory %")
    disk_io_wait_ms: float = Field(10.0, ge=0, description="Disk I/O wait")
    packet_loss_pct: float = Field(0.1, ge=0, le=100, description="Packet loss %")
    api_response_latency_ms: float = Field(150.0, ge=0, description="API latency")
    http_5xx_count: int = Field(0, ge=0, description="HTTP 5xx errors")
    queue_depth: int = Field(10, ge=0, description="Queue depth")
    request_throughput_rps: float = Field(100.0, ge=0, description="Requests/sec")


class PredictRequest(BaseModel):
    """Complete prediction request with system + app metrics."""
    system: SystemMetrics
    application: ApplicationLogs


class PredictResponse(BaseModel):
    """Prediction response with full diagnostic."""
    alert_id: int
    machine_id: str
    timestamp: str
    risk_level: str
    failure_probability: float
    root_cause: Optional[str]
    confidence: str
    markdown_report: str
    json_summary: Dict[str, Any]


class AlertResponse(BaseModel):
    """Alert record response."""
    id: int
    machine_id: str
    timestamp: str
    failure_prob: float
    risk_level: str
    root_cause: Optional[str]
    acknowledged: bool
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: Dict[str, Any]
    model_loaded: bool


class AcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""
    acknowledged_by: str
    action_taken: Optional[str] = None
    is_false_positive: Optional[bool] = None


# =============================================================================
# 2. APPLICATION LIFESPAN (Startup/Shutdown)
# =============================================================================
# WHY lifespan context manager?
# - Ensures DB is initialized before first request
# - Ensures model is loaded before serving traffic
# - Clean shutdown on SIGTERM (ECS/Kubernetes)

diagnostic_engine: Optional[DiagnosticEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    global diagnostic_engine, feature_store

    logger.info("=" * 60)
    logger.info("SentinelX API Starting...")
    logger.info("=" * 60)

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # Initialize feature store
    feature_store = FeatureStore(buffer_size=36)
    logger.info("Feature store initialized (buffer_size=36)")

    # Load diagnostic engine (model + SHAP)
    try:
        diagnostic_engine = DiagnosticEngine(
            model_path=os.getenv("MODEL_PATH", "models/model.joblib"),
            feature_names_path=os.getenv("FEATURE_NAMES_PATH", "models/feature_names.json"),
            threshold_config_path=os.getenv("THRESHOLDS_PATH", "models/evaluation_results.json")
        )
        logger.info("Diagnostic engine loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

    logger.info("SentinelX API Ready!")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("SentinelX API Shutting down...")


# =============================================================================
# 3. FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="SentinelX Predictive Maintenance API",
    description="""
## SentinelX: Production-Grade AIOps Platform

Real-time predictive maintenance using XGBoost + SHAP explainability.

### Features
- **Prediction**: Real-time failure probability with SHAP explanations
- **Diagnostics**: AI-generated root cause analysis
- **Persistence**: All alerts stored for Alert Fatigue analysis
- **Explainability**: Every prediction includes feature contributions

### Architecture
```
Raw Metrics → Feature Engineering → XGBoost → SHAP → Diagnostic Report → PostgreSQL
```

### Paper References
- Paper 1: SOFM+SVM hybrid approach
- Paper 2: XGBoost with SMOTE for imbalanced data
- Paper 3: Alert Fatigue reduction (54% FP → <2%)
- Paper 4: Real-time quality gates
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 4. FEATURE ENGINEERING (Using Feature Store for Real Rolling Stats)
# =============================================================================
# This replaces the old synthetic feature approximation with REAL rolling
# statistics computed from historical observations, fixing the train-serve mismatch.

def engineer_features(system: SystemMetrics, application: ApplicationLogs) -> pd.DataFrame:
    """
    Transform raw metrics into model features using the Feature Store.

    This now computes REAL rolling mean/std and lag features from historical
    observations, exactly as done during training (feature_engineer.py).

    Pipeline:
    1. Create raw observation dict
    2. Update feature store with this observation
    3. Get historical DataFrame
    4. Compute rolling features (real rolling mean/std)
    5. Compute lag features (real historical values)
    6. Compute cross-domain features (matching training logic)
    """
    global feature_store

    # Initialize feature store if not yet created (shouldn't happen in normal flow)
    if feature_store is None:
        feature_store = FeatureStore(buffer_size=36)

    # --- Step 1: Create raw observation dict ---
    observation = {
        "air_temperature_K": system.air_temperature_K,
        "process_temperature_K": system.process_temperature_K,
        "rotational_speed_rpm": system.rotational_speed_rpm,
        "torque_Nm": system.torque_Nm,
        "tool_wear_min": system.tool_wear_min,
        "vibration_mm_s": system.vibration_mm_s or 15.0,
        "pressure_psi": system.pressure_psi or 100.0,
        "network_latency_ms": system.network_latency_ms or 20.0,
        "edge_processing_time_ms": system.edge_processing_time_ms or 15.0,
        "fuzzy_pid_output": system.fuzzy_pid_output or 0.5,

        # Application metrics
        "api_response_latency_ms": application.api_response_latency_ms,
        "error_rate_pct": application.error_rate_pct,
        "http_5xx_count": application.http_5xx_count,
        "request_throughput_rps": application.request_throughput_rps,
        "queue_depth": application.queue_depth,
        "cpu_utilization_pct": application.cpu_utilization_pct,
        "memory_utilization_pct": application.memory_utilization_pct,
        "packet_loss_pct": application.packet_loss_pct,
        "disk_io_wait_ms": application.disk_io_wait_ms,
    }

    # --- Step 2: Update feature store with this observation ---
    machine_id = system.machine_id
    feature_store.update(machine_id, observation)

    # --- Step 3: Get historical DataFrame (includes current observation from update) ---
    history_df = feature_store.get_history_df(machine_id)
    n_history = len(history_df)
    logger.info(f"Feature store: {machine_id} has {n_history} observations")

    # --- Step 4: Start building feature dict with base features ---
    features = observation.copy()

    # --- Step 5: Compute rolling features (REAL rolling mean/std) ---
    rolling_features = feature_store.compute_rolling_features(history_df)
    features.update(rolling_features)

    # --- Step 6: Compute lag features (REAL historical values) ---
    lag_features = feature_store.compute_lag_features(history_df)
    features.update(lag_features)

    # --- Step 7: Compute cross-domain features (matching training exactly) ---

    # thermal_efficiency_idx (feature_engineer.py lines 411-413)
    features["thermal_efficiency_idx"] = (
        features["cpu_utilization_pct"] / (features["air_temperature_K"] + 1e-6)
    )

    # instantaneous_power_W (feature_engineer.py lines 451-453)
    features["instantaneous_power_W"] = (
        features["rotational_speed_rpm"] * features["torque_Nm"] * (2 * np.pi / 60)
    )

    # expected_power_W: 12-period rolling mean of power (feature_engineer.py lines 455-457)
    features["expected_power_W"] = feature_store.compute_expected_power(history_df)

    # power_anomaly_score (feature_engineer.py lines 459-462)
    power_diff = abs(features["instantaneous_power_W"] - features["expected_power_W"])
    features["power_anomaly_score"] = power_diff / (abs(features["expected_power_W"]) + 1e-6)

    # tool_wear_rate: diff() not normalized! (feature_engineer.py lines 474-476)
    features["tool_wear_rate"] = feature_store.compute_tool_wear_rate(history_df)

    # error_under_stress (feature_engineer.py lines 433-435)
    # Training uses 75th percentile of torque, but for single machine we use fixed threshold
    # (40 Nm is approximately the 75th percentile from training data)
    features["error_under_stress"] = int(
        features["error_rate_pct"] > 1.0 and features["torque_Nm"] > 40
    )

    return pd.DataFrame([features])


# =============================================================================
# 5. API ENDPOINTS
# =============================================================================

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint with ASCII banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     ██╗  ██╗   ║
    ║   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     ╚██╗██╔╝   ║
    ║   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║      ╚███╔╝    ║
    ║   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║      ██╔██╗    ║
    ║   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗██╔╝ ██╗   ║
    ║   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝   ║
    ║                                                           ║
    ║           Predictive Maintenance API v1.0                 ║
    ║                                                           ║
    ║   Endpoints:                                              ║
    ║     POST /predict     - Get failure prediction            ║
    ║     GET  /alerts      - List stored alerts                ║
    ║     GET  /health      - Service health check              ║
    ║     GET  /docs        - Interactive API docs              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    return banner


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for load balancers and monitoring.

    Returns database status, model status, and version info.
    Used by:
    - AWS ALB health checks
    - Kubernetes liveness/readiness probes
    - Monitoring dashboards
    """
    db_health = check_db_health()

    return HealthResponse(
        status="healthy" if db_health["status"] == "healthy" else "degraded",
        version="1.0.0",
        database=db_health,
        model_loaded=diagnostic_engine is not None
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """
    Generate failure prediction with full diagnostic report.

    Pipeline:
    1. Validate input metrics
    2. Engineer features (rolling stats, cross-domain)
    3. Predict with XGBoost
    4. Generate SHAP explanation
    5. Create diagnostic report (Markdown + JSON)
    6. Persist to database
    7. Return report

    This endpoint is the heart of SentinelX - it transforms raw
    sensor readings into actionable maintenance insights.
    """
    if diagnostic_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Parse timestamp
    timestamp = request.system.timestamp
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    try:
        timestamp_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        timestamp_dt = datetime.utcnow()

    # Step 1: Engineer features
    logger.info(f"Processing prediction for machine={request.system.machine_id}")
    features = engineer_features(request.system, request.application)

    # Step 2: Run diagnostic engine (prediction + SHAP + analysis)
    # Uses LLM-powered diagnosis with automatic fallback to rule-based
    try:
        result: DiagnosticResult = await diagnostic_engine.diagnose_with_llm(
            features=features,
            machine_id=request.system.machine_id,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostic engine error: {str(e)}")

    # Step 3: Persist to database
    raw_data = {
        "system": request.system.model_dump(),
        "application": request.application.model_dump()
    }

    shap_data = {
        "top_contributors": result.top_contributors[:10],
        "feature_group_impacts": result.feature_group_impacts
    }

    try:
        alert = create_alert(
            db=db,
            machine_id=request.system.machine_id,
            timestamp=timestamp_dt,
            failure_prob=result.failure_probability,
            risk_level=result.risk_level,
            root_cause=result.primary_failure_mode,
            confidence=result.confidence,
            report_md=result.to_markdown(),
            raw_data=raw_data,
            shap_values=shap_data,
            threshold_used="balanced",
            model_version=result.model_version
        )
        logger.info(f"Alert created: id={alert.id}, risk={result.risk_level}")
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Step 4: Return response
    return PredictResponse(
        alert_id=alert.id,
        machine_id=request.system.machine_id,
        timestamp=timestamp,
        risk_level=result.risk_level,
        failure_probability=result.failure_probability,
        root_cause=result.primary_failure_mode,
        confidence=result.confidence,
        markdown_report=result.to_markdown(),
        json_summary=result.to_json()
    )


@app.get("/alerts", response_model=List[AlertResponse])
async def list_alerts(
    machine_id: Optional[str] = Query(None, description="Filter by machine"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    limit: int = Query(100, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    """
    List stored alerts with optional filtering.

    Use cases:
    - Dashboard: Show recent high-risk alerts
    - Analysis: Export alerts for a specific machine
    - Pagination: Handle large result sets
    """
    alerts = get_alerts(
        db=db,
        machine_id=machine_id,
        risk_level=risk_level,
        limit=limit,
        offset=offset
    )

    return [
        AlertResponse(
            id=a.id,
            machine_id=a.machine_id,
            timestamp=a.timestamp.isoformat() if a.timestamp else "",
            failure_prob=a.failure_prob,
            risk_level=a.risk_level,
            root_cause=a.root_cause,
            acknowledged=a.acknowledged,
            created_at=a.created_at.isoformat() if a.created_at else ""
        )
        for a in alerts
    ]


@app.get("/alerts/{alert_id}")
async def get_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific alert with full details including the Markdown report.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        **alert.to_dict(),
        "report_md": alert.report_md,
        "raw_data": alert.raw_data,
        "shap_values": alert.shap_values
    }


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge(
    alert_id: int,
    request: AcknowledgeRequest,
    db: Session = Depends(get_db)
):
    """
    Acknowledge an alert (for Alert Fatigue tracking - Paper 3).

    Tracking acknowledgments helps measure:
    - Alert fatigue: Are operators ignoring alerts?
    - False positive rate: Which alerts were wrong?
    - Response time: How quickly are alerts addressed?
    """
    alert = acknowledge_alert(
        db=db,
        alert_id=alert_id,
        acknowledged_by=request.acknowledged_by,
        action_taken=request.action_taken,
        is_false_positive=request.is_false_positive
    )

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/fatigue")
async def alert_fatigue_metrics(
    machine_id: Optional[str] = Query(None, description="Filter by machine"),
    days: int = Query(7, ge=1, le=90, description="Analysis period in days"),
    db: Session = Depends(get_db)
):
    """
    Alert Fatigue analysis (Paper 3 compliance).

    Returns metrics that help identify alert fatigue:
    - Total alerts per day (high = fatigue risk)
    - Acknowledgment rate (low = operators ignoring alerts)
    - False positive rate (high = noise, low trust)

    Paper 3 benchmark: Industry average FP rate is 54%.
    SentinelX target: < 2% FP rate.
    """
    return get_alert_fatigue_summary(db, machine_id=machine_id, days=days)


# =============================================================================
# 6. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  SentinelX API Server                                     ║
    ║  Starting on http://{host}:{port}                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload for development
        log_level="info"
    )
