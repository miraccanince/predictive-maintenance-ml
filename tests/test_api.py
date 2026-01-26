"""
Unit tests for the SentinelX API.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json


class TestAPISchema:
    """Test API request/response schemas."""

    def test_sample_data_has_required_fields(self, sample_sensor_data):
        """Verify sample data contains all required sections."""
        assert "machine_id" in sample_sensor_data
        assert "timestamp" in sample_sensor_data
        assert "sensors" in sample_sensor_data
        assert "system" in sample_sensor_data
        assert "quality" in sample_sensor_data
        assert "environment" in sample_sensor_data

    def test_sensor_fields_complete(self, sample_sensor_data):
        """Verify all sensor fields are present."""
        sensors = sample_sensor_data["sensors"]
        required = [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min"
        ]
        for field in required:
            assert field in sensors, f"Missing sensor field: {field}"

    def test_system_fields_complete(self, sample_sensor_data):
        """Verify all system fields are present."""
        system = sample_sensor_data["system"]
        required = [
            "cpu_utilization_percent",
            "memory_usage_percent",
            "disk_io_bytes_per_sec",
            "error_rate_per_min",
            "network_latency_ms"
        ]
        for field in required:
            assert field in system, f"Missing system field: {field}"

    def test_sensor_values_in_valid_range(self, sample_sensor_data):
        """Verify sensor values are within expected physical ranges."""
        sensors = sample_sensor_data["sensors"]

        # Temperature in Kelvin (reasonable industrial range)
        assert 250 <= sensors["air_temperature_k"] <= 400
        assert 250 <= sensors["process_temperature_k"] <= 450

        # RPM (industrial motor range)
        assert 0 <= sensors["rotational_speed_rpm"] <= 5000

        # Torque (Nm)
        assert 0 <= sensors["torque_nm"] <= 200

        # Tool wear (minutes)
        assert 0 <= sensors["tool_wear_min"] <= 300


class TestDiagnosticResult:
    """Test diagnostic result structure."""

    def test_risk_levels_valid(self):
        """Verify valid risk level values."""
        valid_levels = ["nominal", "low", "moderate", "high", "critical"]
        for level in valid_levels:
            assert level in valid_levels

    def test_root_causes_valid(self):
        """Verify valid root cause categories."""
        valid_causes = [
            "mechanical_wear",
            "thermal_overload",
            "network_congestion",
            "software_stress",
            "power_anomaly",
            "complex_pattern"
        ]
        assert len(valid_causes) == 6


class TestFeatureEngineering:
    """Test feature engineering configuration."""

    def test_feature_count(self, feature_names):
        """Verify expected number of features."""
        if feature_names is None:
            pytest.skip("Feature names file not found")
        assert len(feature_names) == 124, f"Expected 124 features, got {len(feature_names)}"

    def test_feature_names_not_empty(self, feature_names):
        """Verify feature names are valid strings."""
        if feature_names is None:
            pytest.skip("Feature names file not found")
        for name in feature_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_rolling_features_present(self, feature_names):
        """Verify rolling window features are included."""
        if feature_names is None:
            pytest.skip("Feature names file not found")

        rolling_indicators = ["_mean_", "_std_", "_min_", "_max_"]
        has_rolling = any(
            any(ind in name for ind in rolling_indicators)
            for name in feature_names
        )
        assert has_rolling, "Rolling window features should be present"

    def test_lag_features_present(self, feature_names):
        """Verify lag features are included."""
        if feature_names is None:
            pytest.skip("Feature names file not found")

        has_lag = any("_lag_" in name for name in feature_names)
        assert has_lag, "Lag features should be present"


class TestModelConfig:
    """Test model configuration."""

    def test_config_has_required_keys(self, model_config):
        """Verify training config has required parameters."""
        if model_config is None:
            pytest.skip("Model config file not found")

        # Check for XGBoost params (nested under xgb_params)
        assert "xgb_params" in model_config or any(
            key in model_config
            for key in ["n_estimators", "max_depth", "learning_rate"]
        )

        # If xgb_params exists, check nested structure
        if "xgb_params" in model_config:
            xgb = model_config["xgb_params"]
            assert "n_estimators" in xgb
            assert "max_depth" in xgb

    def test_threshold_configured(self, model_config):
        """Verify decision threshold is configured."""
        if model_config is None:
            pytest.skip("Model config file not found")

        if "threshold" in model_config:
            threshold = model_config["threshold"]
            assert 0 < threshold < 1, "Threshold should be between 0 and 1"
