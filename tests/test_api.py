"""
Tests for the SentinelX FastAPI service (src/api/main.py), using FastAPI's
TestClient against the real app and its real Pydantic schemas.

Two tiers:
- Fast, no external services: request validation and the "model not
  loaded" guard, exercised WITHOUT running the app's lifespan (so no
  PostgreSQL/model-file dependency is needed — these test the
  validation layer and guard clause, not full prediction).
- `@pytest.mark.integration`: full lifespan (real Postgres + the
  trained model), exercised end-to-end. Self-skips if the database
  isn't reachable or the model artifacts aren't present, matching this
  repo's existing convention (see test_agent.py's Ollama check and
  tests/e2e, which already require `docker-compose up -d`).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# main.py uses bare `from agent import ...` / `from feature_engineer
# import ...` / `from database import ...`, which only resolve when
# src/api and src/ml are both on sys.path — mirroring how
# docker/Dockerfile.api flattens these files into one directory.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "api"))
sys.path.insert(0, str(REPO_ROOT / "src" / "ml"))

from main import app  # noqa: E402

MODELS_DIR = Path(__file__).parent.parent / "models"


def _db_reachable() -> bool:
    try:
        with TestClient(app):
            return True
    except Exception:
        return False


class TestRequestValidation:
    """
    These make requests without entering the app as a context manager,
    so FastAPI's lifespan (DB init, model load) never runs. Pydantic
    validation happens before any endpoint code, so these are valid
    tests of the request schema regardless of DB/model state.
    """

    client = TestClient(app)

    def test_predict_rejects_missing_required_field(self):
        payload = {"system": {"machine_id": "X"}, "application": {}}
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_out_of_range_temperature(self, predict_payload):
        predict_payload["system"]["air_temperature_K"] = 9999.0
        response = self.client.post("/predict", json=predict_payload)
        assert response.status_code == 422

    def test_predict_rejects_negative_torque(self, predict_payload):
        predict_payload["system"]["torque_Nm"] = -5.0
        response = self.client.post("/predict", json=predict_payload)
        assert response.status_code == 422

    def test_predict_accepts_well_formed_payload_through_validation(self, predict_payload):
        """
        Without lifespan, diagnostic_engine is never loaded, so this
        exercises the explicit 503 guard in predict() rather than a
        full prediction — proving the payload itself passes validation
        and reaches application code.
        """
        response = self.client.post("/predict", json=predict_payload)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_acknowledge_rejects_missing_acknowledged_by(self):
        response = self.client.post("/alerts/1/acknowledge", json={})
        assert response.status_code == 422


@pytest.mark.integration
class TestPredictEndToEnd:
    """Full lifespan: real Postgres + the trained model."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self):
        required = ["model.joblib", "feature_names.json", "evaluation_results.json"]
        if not all((MODELS_DIR / name).exists() for name in required):
            pytest.skip("Model artifacts not present — run train_model.py first")
        if not _db_reachable():
            pytest.skip("PostgreSQL not reachable — start it with docker-compose up -d")

    def test_health_reports_model_loaded(self):
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is True

    def test_predict_flags_cold_start_on_first_call(self, predict_payload):
        with TestClient(app) as client:
            response = client.post("/predict", json=predict_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["is_cold_start"] is True
        assert body["history_observations"] == 1
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert "alert_id" in body

    def test_predict_clears_cold_start_after_enough_history(self, predict_payload):
        with TestClient(app) as client:
            for _ in range(6):
                response = client.post("/predict", json=predict_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["is_cold_start"] is False
        assert body["history_observations"] >= 6

    def test_predict_persists_a_retrievable_alert(self, predict_payload):
        with TestClient(app) as client:
            created = client.post("/predict", json=predict_payload).json()
            fetched = client.get(f"/alerts/{created['alert_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["machine_id"] == predict_payload["system"]["machine_id"]
