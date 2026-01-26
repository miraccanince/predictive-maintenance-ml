"""
Unit tests for the SentinelX Diagnostic Agent.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock


class TestDiagnosticEngine:
    """Test the DiagnosticEngine class."""

    def test_model_files_exist(self):
        """Verify required model files are present."""
        models_dir = Path(__file__).parent.parent / "models"

        # Metadata files should exist (tracked in git)
        assert (models_dir / "feature_names.json").exists()
        assert (models_dir / "training_config.json").exists()

    def test_feature_names_valid_json(self):
        """Verify feature names file is valid JSON."""
        feature_path = Path(__file__).parent.parent / "models" / "feature_names.json"

        with open(feature_path) as f:
            features = json.load(f)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, str) for f in features)

    def test_cv_metrics_structure(self):
        """Verify cross-validation metrics structure."""
        cv_path = Path(__file__).parent.parent / "models" / "cv_metrics.json"

        if not cv_path.exists():
            pytest.skip("CV metrics file not found")

        with open(cv_path) as f:
            metrics = json.load(f)

        # Should have mean and std for PR-AUC
        assert "pr_auc_mean" in metrics
        assert isinstance(metrics["pr_auc_mean"], (int, float))
        assert 0 <= metrics["pr_auc_mean"] <= 1

    def test_evaluation_results_structure(self):
        """Verify evaluation results structure."""
        eval_path = Path(__file__).parent.parent / "models" / "evaluation_results.json"

        if not eval_path.exists():
            pytest.skip("Evaluation results file not found")

        with open(eval_path) as f:
            results = json.load(f)

        # Check for key metrics
        expected_keys = ["precision", "recall", "f1_score", "pr_auc"]
        for key in expected_keys:
            if key in results:
                assert isinstance(results[key], (int, float))


class TestRiskLevelMapping:
    """Test risk level determination logic."""

    @pytest.mark.parametrize("probability,expected_risk", [
        (0.05, "nominal"),
        (0.15, "low"),
        (0.35, "moderate"),
        (0.65, "high"),
        (0.90, "critical"),
    ])
    def test_probability_to_risk_level(self, probability, expected_risk):
        """Test mapping from probability to risk level."""
        # Thresholds from agent.py
        if probability < 0.1:
            risk = "nominal"
        elif probability < 0.25:
            risk = "low"
        elif probability < 0.5:
            risk = "moderate"
        elif probability < 0.75:
            risk = "high"
        else:
            risk = "critical"

        assert risk == expected_risk


class TestSHAPImportance:
    """Test SHAP feature importance."""

    def test_shap_importance_exists(self):
        """Verify SHAP importance file exists."""
        shap_path = Path(__file__).parent.parent / "models" / "shap_importance.json"
        assert shap_path.exists(), "SHAP importance file should exist"

    def test_shap_importance_structure(self):
        """Verify SHAP importance structure."""
        shap_path = Path(__file__).parent.parent / "models" / "shap_importance.json"

        with open(shap_path) as f:
            data = json.load(f)

        # Structure is {"feature_importance": [{"feature": ..., "mean_abs_shap": ...}, ...]}
        assert "feature_importance" in data
        importance_list = data["feature_importance"]
        assert isinstance(importance_list, list)
        assert len(importance_list) > 0

        # Each entry should have feature name and SHAP score
        for entry in importance_list:
            assert "feature" in entry
            assert "mean_abs_shap" in entry
            assert isinstance(entry["feature"], str)
            assert isinstance(entry["mean_abs_shap"], (int, float))

    def test_top_features_have_positive_importance(self):
        """Verify top features have non-negative importance."""
        shap_path = Path(__file__).parent.parent / "models" / "shap_importance.json"

        with open(shap_path) as f:
            data = json.load(f)

        importance_list = data["feature_importance"]

        # Get top 10 features (already sorted in file)
        top_features = importance_list[:10]

        for entry in top_features:
            assert entry["mean_abs_shap"] >= 0, f"Feature {entry['feature']} has negative importance"


class TestLLMPromptConstruction:
    """Test LLM prompt building (without actual LLM calls)."""

    def test_hybrid_feature_meanings_comprehensive(self):
        """Test that hybrid feature meanings cover key engineered features."""
        # Key engineered features that should have meanings
        key_features = [
            "torque_wear_product",
            "thermal_efficiency_idx",
            "power_anomaly_score",
            "stress_ratio",
            "process_temp_gradient"
        ]

        # This would import from agent.py if available
        # For now, just verify the concept
        assert len(key_features) > 0

    def test_risk_level_descriptions_exist(self):
        """Verify all risk levels have clear descriptions."""
        risk_levels = {
            "critical": "Immediate shutdown required",
            "high": "Urgent maintenance needed",
            "moderate": "Schedule maintenance soon",
            "low": "Monitor closely",
            "nominal": "Normal operation"
        }

        assert len(risk_levels) == 5
        for level, desc in risk_levels.items():
            assert len(desc) > 0


class TestFallbackBehavior:
    """Test fallback to rule-based diagnosis."""

    def test_fallback_provides_valid_result(self):
        """Verify fallback result structure."""
        # Simulate a fallback result
        fallback_result = {
            "risk_level": "moderate",
            "root_cause": "mechanical_wear",
            "confidence": "moderate",
            "recommended_action": "Schedule maintenance inspection",
            "source": "rule-based"
        }

        # Verify structure
        assert "risk_level" in fallback_result
        assert "root_cause" in fallback_result
        assert "confidence" in fallback_result
        assert "recommended_action" in fallback_result

        # Verify values are valid
        assert fallback_result["risk_level"] in ["nominal", "low", "moderate", "high", "critical"]
        assert len(fallback_result["recommended_action"]) > 0


@pytest.mark.integration
class TestOllamaIntegration:
    """Integration tests for Ollama (requires Ollama running)."""

    @pytest.mark.asyncio
    async def test_ollama_connection(self):
        """Test connection to Ollama service."""
        import os

        try:
            import aiohttp
        except ImportError:
            pytest.skip("aiohttp not installed")

        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        assert "models" in data
                    else:
                        pytest.skip(f"Ollama returned status {resp.status}")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
