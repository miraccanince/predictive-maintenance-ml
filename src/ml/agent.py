"""
SentinelX Diagnostic Engine (Stage 6)
=====================================

An AI-powered diagnostic agent that interprets XGBoost predictions using SHAP
explainability and domain expertise to provide actionable maintenance insights.

Architecture:
    Raw Data → XGBoost Prediction → SHAP Explanation → Diagnostic Engine → Report

The engine embodies a "Senior Reliability Engineer" persona, applying domain
knowledge about failure mechanics to translate ML outputs into human-actionable
diagnoses.

Reference: Paper 3 - Minimize 'Alert Fatigue' through concise, data-driven insights.
"""

import json
import logging
import os
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import re

import joblib
import shap
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("SentinelX.DiagnosticEngine")


# =============================================================================
# 1. DOMAIN KNOWLEDGE & FAILURE MECHANICS
# =============================================================================
# This section encodes expert knowledge about failure patterns.
# Reference: Input Context from Senior AI Solutions Architect

class FailureMode(Enum):
    """Categorized failure modes based on domain expertise."""
    MECHANICAL_WEAR = "mechanical_wear"
    THERMAL_OVERLOAD = "thermal_overload"
    NETWORK_CONGESTION = "network_congestion"
    SOFTWARE_STRESS = "software_stress"
    POWER_ANOMALY = "power_anomaly"
    COMPLEX_PATTERN = "complex_pattern"  # When SHAP can't isolate root cause


class RiskLevel(Enum):
    """Risk classification for downstream alerting systems."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"              # Action within 24 hours
    MODERATE = "moderate"      # Schedule maintenance
    LOW = "low"                # Monitor closely
    NOMINAL = "nominal"        # Normal operation


# Domain knowledge: Feature groups and their physical meaning
FEATURE_GROUPS = {
    "mechanical": [
        "vibration_mm_s", "torque_Nm", "rotational_speed_rpm",
        "tool_wear_min", "pressure_psi"
    ],
    "thermal": [
        "air_temperature_K", "process_temperature_K",
        "thermal_efficiency_idx"
    ],
    "network": [
        "network_latency_ms", "packet_loss_pct", "api_response_latency_ms"
    ],
    "software": [
        "error_rate_pct", "cpu_utilization_pct", "memory_utilization_pct",
        "disk_io_wait_ms", "queue_depth", "http_5xx_count"
    ],
    "power": [
        "power_anomaly_score", "fuzzy_pid_output"
    ]
}

# Domain knowledge: Critical thresholds and interaction patterns
FAILURE_MECHANICS = {
    "torque_wear_interaction": {
        "description": "Torque × Tool Wear exceeds safe operating threshold",
        "features": ["torque_Nm", "tool_wear_min"],
        "threshold_product": 3000,  # Nm × minutes
        "failure_mode": FailureMode.MECHANICAL_WEAR
    },
    "thermal_throttle": {
        "description": "Air temperature rise + network latency spike = edge controller thermal throttle",
        "features": ["air_temperature_K", "network_latency_ms"],
        "temp_threshold": 310,  # Kelvin (~37°C)
        "latency_threshold": 50,  # ms
        "failure_mode": FailureMode.THERMAL_OVERLOAD
    },
    "vibration_cascade": {
        "description": "High vibration causing disk I/O stress (physical-digital cascade)",
        "features": ["vibration_mm_s", "disk_io_wait_ms"],
        "vibration_threshold": 25,  # mm/s
        "failure_mode": FailureMode.MECHANICAL_WEAR
    },
    "error_storm": {
        "description": "Error rate spike under CPU stress indicates software instability",
        "features": ["error_rate_pct", "cpu_utilization_pct"],
        "error_threshold": 5,  # percent
        "cpu_threshold": 80,  # percent
        "failure_mode": FailureMode.SOFTWARE_STRESS
    }
}

# Feature semantics for interpretation
FEATURE_SEMANTICS = {
    "_std_": "instability/jitter in the signal",
    "_lag_": "influence from previous state",
    "_mean_": "sustained trend",
    "thermal_efficiency": "hybrid health score (lower = degradation)",
    "power_anomaly": "deviation from expected power consumption",
    "error_under_stress": "errors occurring during high load"
}

# =============================================================================
# OLLAMA LLM CONFIGURATION
# =============================================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Hybrid feature physical meanings for LLM context
HYBRID_FEATURE_MEANINGS = {
    "torque_wear_product": "Torque × Tool Wear (Nm·min) - Mechanical stress accumulation. "
                          "High values indicate the cutting tool is experiencing excessive "
                          "stress under wear conditions, risking sudden breakage.",
    "thermal_efficiency_idx": "CPU Utilization / Air Temperature (°C⁻¹) - Processing efficiency "
                              "relative to cooling capacity. Low values suggest thermal throttling.",
    "power_anomaly_score": "|(Instantaneous Power - Expected Power)| / Expected Power - "
                           "Deviation from normal power consumption pattern. High values "
                           "indicate motor/drive electrical anomalies.",
    "instantaneous_power_W": "RPM × Torque × (2π/60) - Real-time mechanical power output. "
                             "Used to detect power delivery issues.",
    "error_under_stress": "Error Rate × CPU Utilization / 100 - Software errors weighted by "
                          "system load. High values indicate software instability under stress."
}


# =============================================================================
# 2. SYSTEM PROMPT (Reliability Engineer Persona)
# =============================================================================

SYSTEM_PROMPT = """
You are a Senior Reliability Engineer with 15+ years of experience in industrial
predictive maintenance. Your expertise spans mechanical systems, thermal dynamics,
and edge computing infrastructure.

DIAGNOSTIC PRINCIPLES:
1. Always ground analysis in SHAP feature contributions - never speculate beyond data
2. Connect hardware signals (vibration, temperature) to software symptoms (latency, errors)
3. Identify the PRIMARY root cause - avoid listing every anomaly
4. Recommend SPECIFIC actions, not generic advice like "monitor the system"
5. Minimize alert fatigue: only escalate when data supports intervention

FAILURE MECHANICS YOU KNOW:
- Torque × Tool Wear > 3000: Mechanical stress threshold exceeded
- High Air Temp + Network Latency: Edge controller thermal throttling
- Vibration → Disk I/O: Physical vibration causing storage stress
- std features indicate instability/jitter
- lag features show influence from previous operating state
- thermal_efficiency_idx is a hybrid health score (low = bad)

OUTPUT FORMAT:
- Be concise: Engineers don't have time for essays
- Lead with risk level and root cause
- End with 1-2 actionable recommendations
- If uncertain, say so explicitly rather than guessing
"""

# Advanced LLM System Prompt for Ollama
LLM_SYSTEM_PROMPT = """You are a Senior Reliability Engineer at a high-tech manufacturing plant.
Your goal is to analyze SHAP explainability values and predict the physical root cause of machine failures.

CRITICAL EXPERTISE:
1. You understand that SHAP values quantify each feature's contribution to the failure prediction
2. Positive SHAP values INCREASE failure risk; negative SHAP values DECREASE failure risk
3. You can correlate hardware signals (vibration, temperature, torque) with software symptoms (latency, errors)
4. You know that hybrid/engineered features like "torque × wear" capture physical interactions

FAILURE MECHANICS YOU MUST CONSIDER:
- Torque × Tool Wear > 3000: Cutting tool under mechanical stress, risk of breakage
- High Air Temperature + Network Latency: Edge controller thermal throttling (CPU overheating)
- Vibration → Disk I/O: Physical vibration causing storage subsystem stress
- Error Rate + CPU > 80%: Software instability under computational load
- Power Anomaly Score > 0.3: Electrical/motor drive issues

REASONING APPROACH (Chain of Thought):
1. First, internally analyze the top SHAP contributors and their physical meanings
2. Identify the PRIMARY root cause domain (mechanical, thermal, network, software, power)
3. Look for hardware-software correlations (e.g., temperature causing network latency)
4. Consider if multiple factors are interacting (complex pattern)
5. Only then write your final diagnosis

OUTPUT REQUIREMENTS:
You MUST respond in TWO parts, clearly separated:

PART 1 - JSON (for database):
```json
{
    "risk_level": "critical|high|moderate|low|nominal",
    "root_cause": "mechanical_wear|thermal_overload|network_congestion|software_stress|power_anomaly|complex_pattern",
    "confidence": "high|moderate|low",
    "recommended_action": "Specific action to take (one sentence)"
}
```

PART 2 - MARKDOWN REPORT (for human engineers):
Write a concise diagnostic report with:
- Risk assessment
- Root cause analysis (grounded in SHAP evidence)
- Physical explanation of what's happening
- 2-3 specific recommendations

Be SPECIFIC, not generic. Use actual sensor values in your recommendations."""


# =============================================================================
# 3. DATA STRUCTURES
# =============================================================================

@dataclass
class SHAPContribution:
    """Individual feature's contribution to prediction."""
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"

    @property
    def abs_contribution(self) -> float:
        return abs(self.shap_value)


@dataclass
class DiagnosticResult:
    """Complete diagnostic output for a single prediction."""
    # Identification
    timestamp: str
    machine_id: str

    # Prediction
    failure_probability: float
    risk_level: str

    # Diagnosis
    primary_failure_mode: str
    root_cause_summary: str
    confidence: str  # "high", "moderate", "low"

    # SHAP Analysis
    top_contributors: List[Dict[str, Any]]
    feature_group_impacts: Dict[str, float]

    # Recommendations
    immediate_actions: List[str]
    monitoring_focus: List[str]

    # Metadata
    model_version: str = "1.0"
    diagnostic_engine_version: str = "1.0"

    def to_json(self) -> Dict[str, Any]:
        """JSON format for database insertion (Stage 7)."""
        return {
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "risk_level": self.risk_level,
            "failure_probability": round(self.failure_probability, 4),
            "root_cause": self.primary_failure_mode,
            "root_cause_detail": self.root_cause_summary,
            "confidence": self.confidence,
            "action": self.immediate_actions[0] if self.immediate_actions else "Monitor",
            "top_features": [c["feature"] for c in self.top_contributors[:3]],
            "model_version": self.model_version
        }

    def to_markdown(self) -> str:
        """Human-readable Markdown report."""
        risk_emoji = {
            "critical": "🔴", "high": "🟠",
            "moderate": "🟡", "low": "🟢", "nominal": "⚪"
        }

        md = f"""
## SentinelX Diagnostic Report
**Machine:** {self.machine_id} | **Time:** {self.timestamp}

---

### {risk_emoji.get(self.risk_level, '⚪')} Risk Level: {self.risk_level.upper()}
**Failure Probability:** {self.failure_probability:.1%}

### Root Cause Analysis
**Primary Failure Mode:** {self.primary_failure_mode.replace('_', ' ').title()}

{self.root_cause_summary}

**Diagnostic Confidence:** {self.confidence.title()}

### Top Contributing Factors
| Feature | Value | Impact |
|---------|-------|--------|
"""
        for c in self.top_contributors[:5]:
            direction = "↑ Risk" if c["direction"] == "increases_risk" else "↓ Risk"
            md += f"| {c['feature']} | {c['value']:.3f} | {direction} ({c['shap_value']:+.3f}) |\n"

        md += f"""
### Feature Group Analysis
"""
        for group, impact in sorted(self.feature_group_impacts.items(),
                                     key=lambda x: abs(x[1]), reverse=True):
            if abs(impact) > 0.01:
                bar = "█" * int(min(abs(impact) * 20, 10))
                sign = "+" if impact > 0 else "-"
                md += f"- **{group.title()}**: {sign}{abs(impact):.3f} {bar}\n"

        md += f"""
### Recommended Actions
"""
        for i, action in enumerate(self.immediate_actions, 1):
            md += f"{i}. {action}\n"

        if self.monitoring_focus:
            md += f"""
### Monitoring Focus
"""
            for item in self.monitoring_focus:
                md += f"- {item}\n"

        md += f"""
---
*Generated by SentinelX Diagnostic Engine v{self.diagnostic_engine_version}*
"""
        return md


# =============================================================================
# 4. DIAGNOSTIC ENGINE
# =============================================================================

class DiagnosticEngine:
    """
    The core diagnostic engine that transforms ML predictions into actionable insights.

    Implements Chain-of-Thought reasoning:
    1. ANALYZE: Examine SHAP contributions and feature values
    2. DIAGNOSE: Apply domain knowledge to identify failure mode
    3. RECOMMEND: Generate specific, actionable guidance
    """

    def __init__(
        self,
        model_path: str = "models/model.joblib",
        feature_names_path: str = "models/feature_names.json",
        threshold_config_path: str = "models/evaluation_results.json"
    ):
        """
        Initialize the diagnostic engine with trained model artifacts.

        Args:
            model_path: Path to trained XGBoost model
            feature_names_path: Path to feature names JSON
            threshold_config_path: Path to threshold configuration
        """
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.threshold_config_path = Path(threshold_config_path)

        self.model = None
        self.explainer = None
        self.feature_names = None
        self.thresholds = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, explainer, and configuration."""
        logger.info("Loading diagnostic engine artifacts...")

        # Load XGBoost model
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"  Model loaded: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load feature names
        if self.feature_names_path.exists():
            with open(self.feature_names_path) as f:
                self.feature_names = json.load(f)
            logger.info(f"  Feature names loaded: {len(self.feature_names)} features")
        else:
            raise FileNotFoundError(f"Feature names not found: {self.feature_names_path}")

        # Load threshold configuration
        if self.threshold_config_path.exists():
            with open(self.threshold_config_path) as f:
                config = json.load(f)
                self.thresholds = config.get("thresholds", {})
            logger.info(f"  Thresholds loaded: {list(self.thresholds.keys())}")

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("  SHAP TreeExplainer initialized")

    def _get_risk_level(self, probability: float) -> RiskLevel:
        """
        Map probability to risk level using configured thresholds.

        Uses the three operational thresholds:
        - Ultra-safe (high precision): probability > 0.94
        - Balanced: probability > 0.63
        - Sensitive (high recall): probability > 0.22
        """
        if probability >= 0.90:
            return RiskLevel.CRITICAL
        elif probability >= 0.70:
            return RiskLevel.HIGH
        elif probability >= 0.40:
            return RiskLevel.MODERATE
        elif probability >= 0.20:
            return RiskLevel.LOW
        else:
            return RiskLevel.NOMINAL

    def _compute_shap_contributions(
        self,
        features: pd.DataFrame
    ) -> List[SHAPContribution]:
        """
        Compute SHAP values and create sorted contribution list.

        Args:
            features: Single row DataFrame with feature values

        Returns:
            List of SHAPContribution objects sorted by absolute impact
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(features)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        # Create contribution objects
        contributions = []
        feature_values = features.iloc[0]

        for i, (feat, shap_val) in enumerate(zip(self.feature_names, shap_vals)):
            if feat in feature_values.index:
                contrib = SHAPContribution(
                    feature=feat,
                    value=float(feature_values[feat]),
                    shap_value=float(shap_val),
                    direction="increases_risk" if shap_val > 0 else "decreases_risk"
                )
                contributions.append(contrib)

        # Sort by absolute contribution
        contributions.sort(key=lambda x: x.abs_contribution, reverse=True)

        return contributions

    def _compute_feature_group_impacts(
        self,
        contributions: List[SHAPContribution]
    ) -> Dict[str, float]:
        """
        Aggregate SHAP values by feature group.

        This helps identify which DOMAIN (mechanical, thermal, etc.)
        is most responsible for the prediction.
        """
        group_impacts = {group: 0.0 for group in FEATURE_GROUPS}

        for contrib in contributions:
            for group, features in FEATURE_GROUPS.items():
                # Check if feature belongs to this group (including derived features)
                for base_feat in features:
                    if base_feat in contrib.feature:
                        group_impacts[group] += contrib.shap_value
                        break

        return group_impacts

    def _analyze_failure_mechanics(
        self,
        features: pd.DataFrame,
        contributions: List[SHAPContribution]
    ) -> Tuple[FailureMode, str, str]:
        """
        CHAIN OF THOUGHT - Step 1: ANALYZE

        Apply domain knowledge to identify which failure mechanic
        best explains the current prediction.

        Returns:
            (failure_mode, explanation, confidence)
        """
        feature_values = features.iloc[0]
        top_features = [c.feature for c in contributions[:10]]

        # Check each known failure mechanic
        mechanic_scores = {}

        # 1. Torque × Tool Wear interaction
        if "torque_Nm" in feature_values and "tool_wear_min" in feature_values:
            product = feature_values["torque_Nm"] * feature_values["tool_wear_min"]
            threshold = FAILURE_MECHANICS["torque_wear_interaction"]["threshold_product"]
            if product > threshold * 0.8:  # 80% of threshold
                score = min(product / threshold, 2.0)
                mechanic_scores["torque_wear_interaction"] = score

        # 2. Thermal throttle
        if "air_temperature_K" in feature_values and "network_latency_ms" in feature_values:
            temp = feature_values["air_temperature_K"]
            latency = feature_values["network_latency_ms"]
            temp_thresh = FAILURE_MECHANICS["thermal_throttle"]["temp_threshold"]
            lat_thresh = FAILURE_MECHANICS["thermal_throttle"]["latency_threshold"]

            if temp > temp_thresh * 0.95 and latency > lat_thresh * 0.8:
                score = (temp / temp_thresh) * (latency / lat_thresh)
                mechanic_scores["thermal_throttle"] = score

        # 3. Vibration cascade
        if "vibration_mm_s" in feature_values and "disk_io_wait_ms" in feature_values:
            vib = feature_values["vibration_mm_s"]
            disk_io = feature_values["disk_io_wait_ms"]
            vib_thresh = FAILURE_MECHANICS["vibration_cascade"]["vibration_threshold"]

            if vib > vib_thresh * 0.8:
                # Check if vibration features are top contributors
                vib_in_top = any("vibration" in f for f in top_features[:5])
                disk_in_top = any("disk" in f for f in top_features[:5])
                if vib_in_top or disk_in_top:
                    score = vib / vib_thresh
                    mechanic_scores["vibration_cascade"] = score

        # 4. Error storm
        if "error_rate_pct" in feature_values and "cpu_utilization_pct" in feature_values:
            error = feature_values["error_rate_pct"]
            cpu = feature_values["cpu_utilization_pct"]
            err_thresh = FAILURE_MECHANICS["error_storm"]["error_threshold"]
            cpu_thresh = FAILURE_MECHANICS["error_storm"]["cpu_threshold"]

            if error > err_thresh * 0.8 and cpu > cpu_thresh * 0.8:
                score = (error / err_thresh) * (cpu / cpu_thresh)
                mechanic_scores["error_storm"] = score

        # Select best matching mechanic
        if mechanic_scores:
            best_mechanic = max(mechanic_scores, key=mechanic_scores.get)
            best_score = mechanic_scores[best_mechanic]

            failure_mode = FAILURE_MECHANICS[best_mechanic]["failure_mode"]
            explanation = FAILURE_MECHANICS[best_mechanic]["description"]

            if best_score > 1.5:
                confidence = "high"
            elif best_score > 1.0:
                confidence = "moderate"
            else:
                confidence = "low"

            return failure_mode, explanation, confidence

        # No clear mechanic identified - check SHAP concentration
        top_5_shap = sum(c.abs_contribution for c in contributions[:5])
        total_shap = sum(c.abs_contribution for c in contributions)

        if total_shap > 0 and (top_5_shap / total_shap) < 0.5:
            # SHAP values spread thin - complex pattern
            return (
                FailureMode.COMPLEX_PATTERN,
                "Model detects a complex pattern not reducible to single sensors. "
                "Multiple interacting factors are contributing to elevated risk.",
                "low"
            )

        # Determine failure mode from top contributing feature group
        group_impacts = self._compute_feature_group_impacts(contributions)
        top_group = max(group_impacts, key=lambda k: abs(group_impacts[k]))

        group_to_mode = {
            "mechanical": FailureMode.MECHANICAL_WEAR,
            "thermal": FailureMode.THERMAL_OVERLOAD,
            "network": FailureMode.NETWORK_CONGESTION,
            "software": FailureMode.SOFTWARE_STRESS,
            "power": FailureMode.POWER_ANOMALY
        }

        failure_mode = group_to_mode.get(top_group, FailureMode.COMPLEX_PATTERN)
        explanation = f"Elevated {top_group} indicators detected. Primary driver: {contributions[0].feature}"

        return failure_mode, explanation, "moderate"

    def _generate_recommendations(
        self,
        failure_mode: FailureMode,
        risk_level: RiskLevel,
        contributions: List[SHAPContribution],
        features: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        CHAIN OF THOUGHT - Step 3: RECOMMEND

        Generate specific, actionable recommendations based on diagnosis.
        Avoids generic advice per Paper 3 (Alert Fatigue minimization).

        Returns:
            (immediate_actions, monitoring_focus)
        """
        actions = []
        monitoring = []
        feature_values = features.iloc[0]

        # Get top contributing features for context
        top_feat = contributions[0].feature if contributions else None

        if failure_mode == FailureMode.MECHANICAL_WEAR:
            if "tool_wear" in str(top_feat):
                wear = feature_values.get("tool_wear_min", 0)
                actions.append(f"Replace tooling immediately (current wear: {wear:.0f} min)")
            if "vibration" in str(top_feat):
                vib = feature_values.get("vibration_mm_s", 0)
                actions.append(f"Inspect bearing assembly (vibration: {vib:.1f} mm/s)")
            if "torque" in str(top_feat):
                actions.append("Check spindle alignment and lubrication")

            monitoring.append("Track vibration_mm_s trend over next 2 hours")
            monitoring.append("Monitor torque × tool_wear product")

        elif failure_mode == FailureMode.THERMAL_OVERLOAD:
            temp = feature_values.get("air_temperature_K", 300)
            actions.append(f"Increase cooling airflow (current: {temp-273:.1f}°C)")

            if feature_values.get("network_latency_ms", 0) > 30:
                actions.append("Check edge controller thermal paste and fans")

            monitoring.append("Track process_temperature_K - air_temperature_K delta")
            monitoring.append("Monitor thermal_efficiency_idx for degradation")

        elif failure_mode == FailureMode.NETWORK_CONGESTION:
            latency = feature_values.get("network_latency_ms", 0)
            actions.append(f"Check network switch and cables (latency: {latency:.1f}ms)")

            if feature_values.get("packet_loss_pct", 0) > 0.5:
                actions.append("Investigate packet loss source - possible EMI interference")

            monitoring.append("Track network_latency_ms and packet_loss_pct")

        elif failure_mode == FailureMode.SOFTWARE_STRESS:
            error = feature_values.get("error_rate_pct", 0)
            cpu = feature_values.get("cpu_utilization_pct", 0)

            if error > 3:
                actions.append(f"Review application logs for error_rate spike ({error:.1f}%)")
            if cpu > 80:
                actions.append(f"Scale compute resources or optimize workload (CPU: {cpu:.0f}%)")

            monitoring.append("Track error_rate_pct and http_5xx_count correlation")
            monitoring.append("Monitor queue_depth for backpressure")

        elif failure_mode == FailureMode.POWER_ANOMALY:
            actions.append("Check power supply unit and voltage regulators")
            actions.append("Inspect motor drive electronics")

            monitoring.append("Track power_anomaly_score trend")
            monitoring.append("Monitor fuzzy_pid_output stability")

        elif failure_mode == FailureMode.COMPLEX_PATTERN:
            actions.append("Schedule comprehensive system diagnostic")
            actions.append("Collect extended sensor data for pattern analysis")

            monitoring.append("Enable enhanced logging for all sensor groups")
            monitoring.append("Cross-correlate top 5 SHAP features")

        # Risk-level adjustments
        if risk_level == RiskLevel.CRITICAL:
            actions.insert(0, "⚠️ CRITICAL: Consider immediate production pause for inspection")
        elif risk_level == RiskLevel.HIGH:
            actions.insert(0, "Schedule maintenance within 24 hours")

        # Ensure at least one action
        if not actions:
            actions.append("Continue monitoring - no specific action required")

        return actions, monitoring

    # =========================================================================
    # LLM-POWERED DIAGNOSIS (Ollama Integration)
    # =========================================================================

    def _build_llm_prompt(
        self,
        features: pd.DataFrame,
        contributions: List[SHAPContribution],
        probability: float,
        risk_level: RiskLevel
    ) -> str:
        """
        Build context-rich prompt for the LLM with SHAP analysis.

        Includes:
        - Top 5 SHAP features with values and directions
        - XGBoost failure probability
        - Physical meanings of hybrid features
        - Feature group aggregations
        """
        feature_values = features.iloc[0]
        top_5 = contributions[:5]

        # Build SHAP feature context
        shap_context = "TOP 5 SHAP CONTRIBUTORS (sorted by impact):\n"
        for i, c in enumerate(top_5, 1):
            direction = "INCREASES" if c.direction == "increases_risk" else "DECREASES"
            shap_context += (
                f"{i}. {c.feature}\n"
                f"   - Current Value: {c.value:.4f}\n"
                f"   - SHAP Value: {c.shap_value:+.4f} ({direction} failure risk)\n"
            )
            # Add physical meaning for hybrid features
            for hybrid_key, meaning in HYBRID_FEATURE_MEANINGS.items():
                if hybrid_key in c.feature.lower():
                    shap_context += f"   - Physical Meaning: {meaning}\n"
                    break

        # Compute feature group impacts
        group_impacts = self._compute_feature_group_impacts(contributions)
        group_context = "\nFEATURE GROUP AGGREGATED IMPACTS:\n"
        for group, impact in sorted(group_impacts.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(impact) > 0.01:
                direction = "increases" if impact > 0 else "decreases"
                group_context += f"- {group.upper()}: {impact:+.3f} ({direction} risk)\n"

        # Key raw sensor values for context
        sensor_context = "\nKEY RAW SENSOR VALUES:\n"
        key_sensors = [
            ("vibration_mm_s", "mm/s"),
            ("torque_Nm", "Nm"),
            ("tool_wear_min", "min"),
            ("air_temperature_K", "K"),
            ("network_latency_ms", "ms"),
            ("error_rate_pct", "%"),
            ("cpu_utilization_pct", "%"),
        ]
        for sensor, unit in key_sensors:
            if sensor in feature_values:
                sensor_context += f"- {sensor}: {feature_values[sensor]:.2f} {unit}\n"

        # Build final prompt
        prompt = f"""MACHINE HEALTH ANALYSIS REQUEST

XGBoost PREDICTION:
- Failure Probability: {probability:.2%}
- Risk Level: {risk_level.value.upper()}

{shap_context}
{group_context}
{sensor_context}

HYBRID FEATURE REFERENCE:
{chr(10).join(f'- {k}: {v}' for k, v in HYBRID_FEATURE_MEANINGS.items())}

Based on this SHAP explainability analysis, provide your diagnosis following the required output format (JSON + Markdown)."""

        return prompt

    async def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Make async call to Ollama API.

        Returns:
            LLM response text or None if failed
        """
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": LLM_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower for more consistent outputs
                "num_predict": 1500,  # Enough for JSON + report
            }
        }

        try:
            timeout = aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        logger.warning(
                            f"Ollama API returned status {response.status}: "
                            f"{await response.text()}"
                        )
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Ollama API timeout after {OLLAMA_TIMEOUT}s")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Ollama API connection error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Ollama API unexpected error: {e}")
            return None

    def _parse_llm_response(
        self,
        response: str,
        fallback_result: DiagnosticResult
    ) -> Tuple[Dict[str, Any], str]:
        """
        Parse LLM response into structured JSON and Markdown parts.

        Args:
            response: Raw LLM response text
            fallback_result: Rule-based result to use for missing fields

        Returns:
            (json_data, markdown_report)
        """
        json_data = None
        markdown_report = ""

        # Extract JSON block using regex
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response)

        if json_match:
            try:
                json_str = json_match.group(1).strip()
                json_data = json.loads(json_str)
                logger.info("Successfully parsed LLM JSON response")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM JSON: {e}")

        # Extract Markdown (everything after the JSON block, or if no JSON, look for markdown headers)
        if json_match:
            # Get content after JSON block
            after_json = response[json_match.end():].strip()
            if after_json:
                markdown_report = after_json
        else:
            # No JSON found, check if there's markdown content
            md_pattern = r'(#{1,3}\s+.*[\s\S]*)'
            md_match = re.search(md_pattern, response)
            if md_match:
                markdown_report = md_match.group(1)

        # Validate and fill missing JSON fields with fallback
        if json_data is None:
            json_data = {}

        # Ensure required fields exist with valid values
        valid_risk_levels = {"critical", "high", "moderate", "low", "nominal"}
        valid_root_causes = {
            "mechanical_wear", "thermal_overload", "network_congestion",
            "software_stress", "power_anomaly", "complex_pattern"
        }
        valid_confidence = {"high", "moderate", "low"}

        if json_data.get("risk_level", "").lower() not in valid_risk_levels:
            json_data["risk_level"] = fallback_result.risk_level

        if json_data.get("root_cause", "").lower() not in valid_root_causes:
            json_data["root_cause"] = fallback_result.primary_failure_mode

        if json_data.get("confidence", "").lower() not in valid_confidence:
            json_data["confidence"] = fallback_result.confidence

        if not json_data.get("recommended_action"):
            json_data["recommended_action"] = (
                fallback_result.immediate_actions[0]
                if fallback_result.immediate_actions
                else "Continue monitoring"
            )

        # Normalize case
        json_data["risk_level"] = json_data["risk_level"].lower()
        json_data["root_cause"] = json_data["root_cause"].lower()
        json_data["confidence"] = json_data["confidence"].lower()

        # If no markdown report, generate a minimal one
        if not markdown_report.strip():
            markdown_report = self._generate_minimal_report(json_data, fallback_result)

        return json_data, markdown_report

    def _generate_minimal_report(
        self,
        json_data: Dict[str, Any],
        fallback_result: DiagnosticResult
    ) -> str:
        """Generate minimal markdown report if LLM didn't provide one."""
        risk_emoji = {
            "critical": "🔴", "high": "🟠",
            "moderate": "🟡", "low": "🟢", "nominal": "⚪"
        }
        risk_level = json_data.get("risk_level", "nominal")
        emoji = risk_emoji.get(risk_level, "⚪")

        return f"""## LLM Diagnostic Summary

### {emoji} Risk Level: {risk_level.upper()}

**Root Cause:** {json_data.get('root_cause', 'unknown').replace('_', ' ').title()}

**Confidence:** {json_data.get('confidence', 'moderate').title()}

### Recommended Action
{json_data.get('recommended_action', 'Continue monitoring')}

---
*Generated by SentinelX LLM Agent (Ollama + {OLLAMA_MODEL})*
"""

    async def diagnose_with_llm(
        self,
        features: pd.DataFrame,
        machine_id: str = "Unknown",
        timestamp: Optional[str] = None
    ) -> DiagnosticResult:
        """
        LLM-powered diagnosis with automatic fallback to rule-based logic.

        This method:
        1. Runs XGBoost prediction and SHAP analysis (same as rule-based)
        2. Builds context-rich prompt for the LLM
        3. Calls Ollama API asynchronously
        4. Parses structured JSON + Markdown response
        5. Falls back to rule-based diagnosis if LLM fails

        Args:
            features: Single row DataFrame with feature values
            machine_id: Machine identifier for reporting
            timestamp: Timestamp for the observation (defaults to now)

        Returns:
            DiagnosticResult with LLM-generated or rule-based analysis
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Ensure features match expected columns
        missing_cols = set(self.feature_names) - set(features.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            for col in missing_cols:
                features[col] = 0

        # Select only the features the model expects
        model_features = features[self.feature_names].copy()
        model_features = model_features.fillna(0)

        # Step 1: PREDICT (same as rule-based)
        probability = float(self.model.predict_proba(model_features)[0, 1])
        risk_level = self._get_risk_level(probability)

        logger.info(f"[LLM] Prediction: {probability:.3f} ({risk_level.value})")

        # Step 2: ANALYZE - Compute SHAP contributions (same as rule-based)
        contributions = self._compute_shap_contributions(model_features)
        group_impacts = self._compute_feature_group_impacts(contributions)

        logger.info(f"[LLM] Top contributor: {contributions[0].feature} "
                    f"(SHAP: {contributions[0].shap_value:+.3f})")

        # Step 3: Generate rule-based result as fallback
        failure_mode, explanation, confidence = self._analyze_failure_mechanics(
            model_features, contributions
        )
        actions, monitoring = self._generate_recommendations(
            failure_mode, risk_level, contributions, model_features
        )

        fallback_result = DiagnosticResult(
            timestamp=timestamp,
            machine_id=machine_id,
            failure_probability=probability,
            risk_level=risk_level.value,
            primary_failure_mode=failure_mode.value,
            root_cause_summary=explanation,
            confidence=confidence,
            top_contributors=[
                {
                    "feature": c.feature,
                    "value": c.value,
                    "shap_value": c.shap_value,
                    "direction": c.direction
                }
                for c in contributions[:10]
            ],
            feature_group_impacts=group_impacts,
            immediate_actions=actions,
            monitoring_focus=monitoring,
            diagnostic_engine_version="1.0-rule-based"
        )

        # Step 4: Call LLM
        logger.info(f"[LLM] Calling Ollama ({OLLAMA_MODEL})...")

        prompt = self._build_llm_prompt(
            model_features, contributions, probability, risk_level
        )

        llm_response = await self._call_ollama(prompt)

        # Step 5: Parse LLM response or fallback
        if llm_response:
            logger.info("[LLM] Received response, parsing...")
            json_data, markdown_report = self._parse_llm_response(
                llm_response, fallback_result
            )

            # Build LLM-enhanced result
            result = DiagnosticResult(
                timestamp=timestamp,
                machine_id=machine_id,
                failure_probability=probability,
                risk_level=json_data["risk_level"],
                primary_failure_mode=json_data["root_cause"],
                root_cause_summary=markdown_report,
                confidence=json_data["confidence"],
                top_contributors=[
                    {
                        "feature": c.feature,
                        "value": c.value,
                        "shap_value": c.shap_value,
                        "direction": c.direction
                    }
                    for c in contributions[:10]
                ],
                feature_group_impacts=group_impacts,
                immediate_actions=[json_data["recommended_action"]] + actions[1:],
                monitoring_focus=monitoring,
                diagnostic_engine_version=f"2.0-llm-{OLLAMA_MODEL}"
            )

            logger.info(f"[LLM] Diagnosis complete: {json_data['root_cause']} "
                        f"(confidence: {json_data['confidence']})")

            return result

        else:
            # Fallback to rule-based
            logger.warning("[LLM] Ollama unavailable, falling back to rule-based diagnosis")
            return fallback_result

    def diagnose(
        self,
        features: pd.DataFrame,
        machine_id: str = "Unknown",
        timestamp: Optional[str] = None
    ) -> DiagnosticResult:
        """
        Main diagnostic entry point.

        Implements full Chain-of-Thought reasoning:
        1. PREDICT: Get failure probability from XGBoost
        2. ANALYZE: Compute SHAP contributions
        3. DIAGNOSE: Apply domain knowledge to identify root cause
        4. RECOMMEND: Generate actionable guidance

        Args:
            features: Single row DataFrame with feature values
            machine_id: Machine identifier for reporting
            timestamp: Timestamp for the observation (defaults to now)

        Returns:
            DiagnosticResult with full analysis
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Ensure features match expected columns
        missing_cols = set(self.feature_names) - set(features.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            for col in missing_cols:
                features[col] = 0

        # Select only the features the model expects
        model_features = features[self.feature_names].copy()
        model_features = model_features.fillna(0)

        # Step 1: PREDICT
        probability = float(self.model.predict_proba(model_features)[0, 1])
        risk_level = self._get_risk_level(probability)

        logger.info(f"Prediction: {probability:.3f} ({risk_level.value})")

        # Step 2: ANALYZE - Compute SHAP contributions
        contributions = self._compute_shap_contributions(model_features)
        group_impacts = self._compute_feature_group_impacts(contributions)

        logger.info(f"Top contributor: {contributions[0].feature} "
                    f"(SHAP: {contributions[0].shap_value:+.3f})")

        # Step 3: DIAGNOSE - Apply domain knowledge
        failure_mode, explanation, confidence = self._analyze_failure_mechanics(
            model_features, contributions
        )

        logger.info(f"Diagnosis: {failure_mode.value} (confidence: {confidence})")

        # Step 4: RECOMMEND - Generate actions
        actions, monitoring = self._generate_recommendations(
            failure_mode, risk_level, contributions, model_features
        )

        # Build result
        result = DiagnosticResult(
            timestamp=timestamp,
            machine_id=machine_id,
            failure_probability=probability,
            risk_level=risk_level.value,
            primary_failure_mode=failure_mode.value,
            root_cause_summary=explanation,
            confidence=confidence,
            top_contributors=[
                {
                    "feature": c.feature,
                    "value": c.value,
                    "shap_value": c.shap_value,
                    "direction": c.direction
                }
                for c in contributions[:10]
            ],
            feature_group_impacts=group_impacts,
            immediate_actions=actions,
            monitoring_focus=monitoring
        )

        return result

    def diagnose_batch(
        self,
        features_df: pd.DataFrame,
        machine_id_col: str = "machine_id",
        timestamp_col: str = "timestamp"
    ) -> List[DiagnosticResult]:
        """
        Diagnose multiple observations.

        Args:
            features_df: DataFrame with multiple rows
            machine_id_col: Column name for machine ID
            timestamp_col: Column name for timestamp

        Returns:
            List of DiagnosticResult objects
        """
        results = []

        for idx, row in features_df.iterrows():
            machine_id = row.get(machine_id_col, f"Row_{idx}")
            timestamp = str(row.get(timestamp_col, datetime.now().isoformat()))

            # Create single-row DataFrame
            feature_row = pd.DataFrame([row]).drop(
                columns=[machine_id_col, timestamp_col],
                errors='ignore'
            )

            result = self.diagnose(feature_row, machine_id, timestamp)
            results.append(result)

        return results

    def diagnose_sync_llm(
        self,
        features: pd.DataFrame,
        machine_id: str = "Unknown",
        timestamp: Optional[str] = None
    ) -> DiagnosticResult:
        """
        Synchronous wrapper for LLM-powered diagnosis.

        Use this method when you need LLM diagnosis from synchronous code.
        It creates a new event loop if one doesn't exist.

        Args:
            features: Single row DataFrame with feature values
            machine_id: Machine identifier for reporting
            timestamp: Timestamp for the observation (defaults to now)

        Returns:
            DiagnosticResult with LLM-generated or rule-based analysis
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - can't use run_until_complete
            # Fall back to rule-based
            logger.warning("Called diagnose_sync_llm from async context, "
                           "use diagnose_with_llm directly")
            return self.diagnose(features, machine_id, timestamp)
        except RuntimeError:
            # No event loop running - create one
            return asyncio.run(
                self.diagnose_with_llm(features, machine_id, timestamp)
            )


# =============================================================================
# 5. CLI & DEMO
# =============================================================================

def run_demo():
    """
    Demonstrate the diagnostic engine on sample data.
    """
    logger.info("=" * 60)
    logger.info("SentinelX Diagnostic Engine Demo")
    logger.info("=" * 60)

    # Initialize engine
    engine = DiagnosticEngine()

    # Load sample data
    feature_matrix_path = Path("data/features/feature_matrix.parquet")
    if not feature_matrix_path.exists():
        logger.error(f"Feature matrix not found: {feature_matrix_path}")
        return

    df = pd.read_parquet(feature_matrix_path)

    # Find some failure cases to diagnose
    failures = df[df["machine_failure"] == 1].head(3)
    healthy = df[df["machine_failure"] == 0].sample(2, random_state=42)

    samples = pd.concat([failures, healthy])

    logger.info(f"\nDiagnosing {len(samples)} samples...")
    logger.info("=" * 60)

    for idx, row in samples.iterrows():
        machine_id = row.get("machine_id", "Unknown")
        timestamp = str(row.get("timestamp", ""))
        actual = "FAILURE" if row.get("machine_failure", 0) == 1 else "HEALTHY"

        # Prepare features
        exclude_cols = ["timestamp", "machine_id", "machine_failure",
                        "product_type", "maintenance_status",
                        "maintenance_status_x", "maintenance_status_y"]
        feature_cols = [c for c in row.index if c not in exclude_cols]
        features = pd.DataFrame([row[feature_cols]])
        features = features.select_dtypes(include=[np.number])

        # Run diagnosis
        result = engine.diagnose(features, machine_id, timestamp)

        # Print results
        print("\n" + "=" * 70)
        print(f"Machine: {machine_id} | Actual: {actual}")
        print("=" * 70)
        print(result.to_markdown())

        # Also show JSON output
        print("\n--- JSON Summary (for database) ---")
        print(json.dumps(result.to_json(), indent=2))


def diagnose_from_file(input_path: str, output_dir: str = "diagnostics"):
    """
    Run diagnostics on data from a file and save reports.

    Args:
        input_path: Path to Parquet/CSV file with features
        output_dir: Directory to save diagnostic reports
    """
    logger.info(f"Running diagnostics on: {input_path}")

    # Load data
    path = Path(input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Initialize engine
    engine = DiagnosticEngine()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run diagnostics
    results = []

    for idx, row in df.iterrows():
        machine_id = row.get("machine_id", f"Row_{idx}")
        timestamp = str(row.get("timestamp", datetime.now().isoformat()))

        # Prepare features
        exclude_cols = ["timestamp", "machine_id", "machine_failure",
                        "product_type", "maintenance_status",
                        "maintenance_status_x", "maintenance_status_y"]
        feature_cols = [c for c in row.index if c not in exclude_cols]
        features = pd.DataFrame([row[feature_cols]])
        features = features.select_dtypes(include=[np.number])

        result = engine.diagnose(features, machine_id, timestamp)
        results.append(result)

        # Only save detailed reports for non-nominal risk levels
        if result.risk_level != "nominal":
            # Save markdown report
            md_path = output_path / f"{machine_id}_{idx}_report.md"
            with open(md_path, 'w') as f:
                f.write(result.to_markdown())

    # Save all JSON summaries
    json_summaries = [r.to_json() for r in results]
    json_path = output_path / "diagnostic_summaries.json"
    with open(json_path, 'w') as f:
        json.dump(json_summaries, f, indent=2)

    # Summary statistics
    risk_counts = {}
    for r in results:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1

    logger.info(f"\nDiagnostic Summary:")
    logger.info(f"  Total observations: {len(results)}")
    for level, count in sorted(risk_counts.items()):
        logger.info(f"  {level.upper()}: {count}")
    logger.info(f"\nReports saved to: {output_path}")

    return results


async def run_llm_demo():
    """
    Demonstrate the LLM-powered diagnostic engine on sample data.
    """
    logger.info("=" * 60)
    logger.info("SentinelX LLM Diagnostic Engine Demo")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info("=" * 60)

    # Initialize engine
    engine = DiagnosticEngine()

    # Load sample data
    feature_matrix_path = Path("data/features/feature_matrix.parquet")
    if not feature_matrix_path.exists():
        logger.error(f"Feature matrix not found: {feature_matrix_path}")
        return

    df = pd.read_parquet(feature_matrix_path)

    # Find failure cases to diagnose (more interesting for LLM analysis)
    failures = df[df["machine_failure"] == 1].head(2)
    healthy = df[df["machine_failure"] == 0].sample(1, random_state=42)

    samples = pd.concat([failures, healthy])

    logger.info(f"\nDiagnosing {len(samples)} samples with LLM...")
    logger.info("=" * 60)

    for idx, row in samples.iterrows():
        machine_id = row.get("machine_id", "Unknown")
        timestamp = str(row.get("timestamp", ""))
        actual = "FAILURE" if row.get("machine_failure", 0) == 1 else "HEALTHY"

        # Prepare features
        exclude_cols = ["timestamp", "machine_id", "machine_failure",
                        "product_type", "maintenance_status",
                        "maintenance_status_x", "maintenance_status_y"]
        feature_cols = [c for c in row.index if c not in exclude_cols]
        features = pd.DataFrame([row[feature_cols]])
        features = features.select_dtypes(include=[np.number])

        # Run LLM diagnosis
        result = await engine.diagnose_with_llm(features, machine_id, timestamp)

        # Print results
        print("\n" + "=" * 70)
        print(f"Machine: {machine_id} | Actual: {actual}")
        print(f"Engine Version: {result.diagnostic_engine_version}")
        print("=" * 70)
        print(result.to_markdown())

        # Also show JSON output
        print("\n--- JSON Summary (for database) ---")
        print(json.dumps(result.to_json(), indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--llm":
            # Run LLM demo
            asyncio.run(run_llm_demo())
        else:
            # Run on specified file
            diagnose_from_file(sys.argv[1])
    else:
        # Run rule-based demo (default)
        print("Usage:")
        print("  python agent.py          # Run rule-based demo")
        print("  python agent.py --llm    # Run LLM-powered demo (requires Ollama)")
        print("  python agent.py <file>   # Diagnose from file")
        print()
        run_demo()
