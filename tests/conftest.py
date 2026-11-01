"""
Pytest configuration and shared fixtures for SentinelX tests.
"""

import json
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def predict_payload():
    """Valid /predict request body matching SystemMetrics + ApplicationLogs."""
    return {
        "system": {
            "machine_id": "TEST-001",
            "air_temperature_K": 300.0,
            "process_temperature_K": 310.0,
            "rotational_speed_rpm": 1500,
            "torque_Nm": 40.0,
            "tool_wear_min": 100,
            "vibration_mm_s": 15.0,
            "pressure_psi": 100.0,
        },
        "application": {
            "error_rate_pct": 0.5,
            "cpu_utilization_pct": 45.0,
            "memory_utilization_pct": 60.0,
            "disk_io_wait_ms": 10.0,
            "packet_loss_pct": 0.1,
            "api_response_latency_ms": 150.0,
            "http_5xx_count": 0,
            "queue_depth": 10,
            "request_throughput_rps": 100.0,
        },
    }


@pytest.fixture
def high_risk_predict_payload():
    """/predict body with values skewed toward a failure prediction."""
    return {
        "system": {
            "machine_id": "TEST-002",
            "air_temperature_K": 320.0,
            "process_temperature_K": 340.0,
            "rotational_speed_rpm": 2800,
            "torque_Nm": 75.0,
            "tool_wear_min": 220,
            "vibration_mm_s": 85.0,
            "pressure_psi": 160.0,
        },
        "application": {
            "error_rate_pct": 8.5,
            "cpu_utilization_pct": 95.0,
            "memory_utilization_pct": 88.0,
            "disk_io_wait_ms": 60.0,
            "packet_loss_pct": 3.0,
            "api_response_latency_ms": 600.0,
            "http_5xx_count": 12,
            "queue_depth": 80,
            "request_throughput_rps": 20.0,
        },
    }


@pytest.fixture
def feature_names():
    """Load feature names from model config."""
    feature_path = Path(__file__).parent.parent / "models" / "feature_names.json"
    if feature_path.exists():
        with open(feature_path) as f:
            return json.load(f)
    return None


@pytest.fixture
def model_config():
    """Load training config."""
    config_path = Path(__file__).parent.parent / "models" / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None
