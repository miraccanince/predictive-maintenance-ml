"""
Pytest configuration and shared fixtures for SentinelX tests.
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_sensor_data():
    """Sample sensor input matching API schema."""
    return {
        "machine_id": "TEST-001",
        "timestamp": "2024-01-26T12:00:00Z",
        "sensors": {
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40.0,
            "tool_wear_min": 100
        },
        "system": {
            "cpu_utilization_percent": 45.0,
            "memory_usage_percent": 60.0,
            "disk_io_bytes_per_sec": 50000000,
            "error_rate_per_min": 0.5,
            "network_latency_ms": 25.0,
            "edge_processing_time_ms": 12.5,
            "fuzzy_pid_output": 0.4
        },
        "quality": {
            "type_encoded": 1,
            "vibration_rms": 0.35
        },
        "environment": {
            "ambient_humidity_percent": 45.0,
            "power_supply_voltage": 230.0,
            "coolant_flow_rate_lpm": 2.5,
            "spindle_load_percent": 55.0
        }
    }


@pytest.fixture
def high_risk_sensor_data():
    """Sensor data that should trigger high-risk prediction."""
    return {
        "machine_id": "TEST-002",
        "timestamp": "2024-01-26T12:00:00Z",
        "sensors": {
            "air_temperature_k": 320.0,  # High temp
            "process_temperature_k": 340.0,  # Very high
            "rotational_speed_rpm": 2800,  # High speed
            "torque_nm": 75.0,  # High torque
            "tool_wear_min": 220  # High wear
        },
        "system": {
            "cpu_utilization_percent": 95.0,  # High CPU
            "memory_usage_percent": 88.0,  # High memory
            "disk_io_bytes_per_sec": 150000000,
            "error_rate_per_min": 8.5,  # High errors
            "network_latency_ms": 150.0,  # High latency
            "edge_processing_time_ms": 75.0,
            "fuzzy_pid_output": 0.9
        },
        "quality": {
            "type_encoded": 2,
            "vibration_rms": 0.85  # High vibration
        },
        "environment": {
            "ambient_humidity_percent": 75.0,
            "power_supply_voltage": 215.0,  # Low voltage
            "coolant_flow_rate_lpm": 1.2,  # Low flow
            "spindle_load_percent": 92.0  # High load
        }
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
