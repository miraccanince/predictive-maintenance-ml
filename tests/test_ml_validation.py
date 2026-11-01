"""
Focused tests for the leakage-safe temporal split and holdout evaluation
introduced in train_model.py (see models/evaluation_results.json's
`note` field for the synthetic-data caveat these tests do not remove).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# train_model.py is run directly as a script in this repo's convention
# (e.g. `python src/ml/train_model.py`), with src/ml on sys.path.
SRC_ML = Path(__file__).parent.parent / "src" / "ml"
sys.path.insert(0, str(SRC_ML))

import train_model  # noqa: E402

MODELS_DIR = Path(__file__).parent.parent / "models"


def _make_synthetic_df(n_machines=3, rows_per_machine=100, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(n_machines):
        machine_id = f"M{m+1}"
        timestamps = pd.date_range("2024-01-01", periods=rows_per_machine, freq="5min")
        data = {
            "timestamp": timestamps,
            "machine_id": machine_id,
            "product_type": "L",
            "machine_failure": rng.integers(0, 2, rows_per_machine),
            "failure_TWF": 0,
            "failure_HDF": 0,
            "failure_PWF": 0,
            "failure_OSF": 0,
            "failure_RNF": 0,
            "maintenance_status": "Normal",
            "anomaly_score": rng.random(rows_per_machine),
            "is_anomaly": 0,
        }
        for i in range(n_features):
            data[f"feature_{i}"] = rng.normal(size=rows_per_machine)
        frames.append(pd.DataFrame(data))
    # Shuffle rows across machines/time to prove the split re-sorts correctly.
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def config():
    return train_model.TrainConfig(holdout_frac=0.2, n_folds=5)


class TestSplitIntegrity:
    def test_holdout_is_strictly_after_dev_per_machine(self, config):
        df = _make_synthetic_df()
        df_dev, df_holdout = train_model.split_dev_holdout(df, config)

        for machine_id in df[config.group_col].unique():
            dev_times = df_dev.loc[df_dev[config.group_col] == machine_id, config.timestamp_col]
            holdout_times = df_holdout.loc[
                df_holdout[config.group_col] == machine_id, config.timestamp_col
            ]
            assert len(dev_times) and len(holdout_times)
            assert dev_times.max() < holdout_times.min()

    def test_dev_and_holdout_partition_without_overlap(self, config):
        df = _make_synthetic_df()
        df_dev, df_holdout = train_model.split_dev_holdout(df, config)

        assert set(df_dev.index).isdisjoint(set(df_holdout.index))
        assert len(df_dev) + len(df_holdout) == len(df)

    def test_holdout_fraction_is_approximately_requested(self, config):
        df = _make_synthetic_df(rows_per_machine=200)
        df_dev, df_holdout = train_model.split_dev_holdout(df, config)

        actual_frac = len(df_holdout) / len(df)
        assert abs(actual_frac - config.holdout_frac) < 0.05

    def test_temporal_folds_partition_dev_without_overlap(self, config):
        df = _make_synthetic_df()
        df_dev, _ = train_model.split_dev_holdout(df, config)
        fold_ids = train_model.make_temporal_folds(df_dev, config)

        assert len(fold_ids) == len(df_dev)
        assert set(np.unique(fold_ids)) == set(range(config.n_folds))

    def test_temporal_folds_are_chronologically_contiguous_per_machine(self, config):
        df = _make_synthetic_df()
        df_dev, _ = train_model.split_dev_holdout(df, config)
        fold_ids = train_model.make_temporal_folds(df_dev, config)
        df_dev = df_dev.copy()
        df_dev["_fold"] = fold_ids

        for machine_id in df_dev[config.group_col].unique():
            machine_rows = df_dev[df_dev[config.group_col] == machine_id].sort_values(
                config.timestamp_col
            )
            # Fold id must be non-decreasing along the machine's own timeline
            # (each fold is one contiguous time block, not scattered rows).
            assert machine_rows["_fold"].is_monotonic_increasing

    def test_no_future_timestamp_touches_an_earlier_fold(self, config):
        """
        A later (higher-index) fold for a machine must never start before
        an earlier fold ends — this is the property that makes an
        embargo gap unnecessary, since backward-only features computed
        once over the full timeline cannot encode a later block's values
        into an earlier block.
        """
        df = _make_synthetic_df()
        df_dev, _ = train_model.split_dev_holdout(df, config)
        fold_ids = train_model.make_temporal_folds(df_dev, config)
        df_dev = df_dev.copy()
        df_dev["_fold"] = fold_ids

        for machine_id in df_dev[config.group_col].unique():
            machine_rows = df_dev[df_dev[config.group_col] == machine_id].sort_values(
                config.timestamp_col
            )
            boundaries = machine_rows.groupby("_fold")[config.timestamp_col].agg(["min", "max"])
            boundaries = boundaries.sort_index()
            for i in range(len(boundaries) - 1):
                assert boundaries.iloc[i]["max"] < boundaries.iloc[i + 1]["min"]

    def test_forward_chaining_cv_trains_only_on_earlier_blocks(self, config):
        """
        train_with_cv evaluates fold_idx in range(1, n_folds), training
        on fold_ids < fold_idx and validating on fold_ids == fold_idx.
        This test checks that slicing rule directly: for every evaluated
        fold, every training row's timestamp must precede every
        validation row's timestamp, per machine — i.e. training always
        precedes validation, never the reverse.
        """
        df = _make_synthetic_df()
        df_dev, _ = train_model.split_dev_holdout(df, config)
        fold_ids = train_model.make_temporal_folds(df_dev, config)

        for fold_idx in range(1, config.n_folds):
            train_mask = fold_ids < fold_idx
            val_mask = fold_ids == fold_idx
            df_train = df_dev[train_mask]
            df_val = df_dev[val_mask]
            assert len(df_train) and len(df_val)

            for machine_id in df_val[config.group_col].unique():
                train_times = df_train.loc[
                    df_train[config.group_col] == machine_id, config.timestamp_col
                ]
                val_times = df_val.loc[
                    df_val[config.group_col] == machine_id, config.timestamp_col
                ]
                assert len(train_times) and len(val_times)
                assert train_times.max() < val_times.min()


class TestHoldoutEvaluationArtifacts:
    """
    These require an actual trained run (models/*.json + model.joblib).
    They self-skip if that run hasn't happened, matching this repo's
    existing convention (see test_agent.py's Ollama check).
    """

    @pytest.mark.integration
    def test_saved_evaluation_is_labeled_synthetic_and_untuned(self):
        eval_path = MODELS_DIR / "evaluation_results.json"
        if not eval_path.exists():
            pytest.skip("No evaluation_results.json — run train_model.py first")

        with open(eval_path) as f:
            results = json.load(f)

        assert results["synthetic_data_evaluation"] is True
        assert "synthetic" in results["note"].lower()
        assert results["decision_threshold"] == 0.5
        for key in ("pr_auc", "f1", "precision", "recall", "confusion_matrix", "n_holdout_rows"):
            assert key in results
        cm = results["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == results["n_holdout_rows"]

    @pytest.mark.integration
    def test_saved_model_reproduces_saved_metrics(self):
        """
        The headline requirement: the final saved XGBoost model and
        evaluation_results.json must come from the SAME run. Verified
        by re-scoring the holdout with exactly the saved model
        (inference only, no retraining) and checking it reproduces the
        saved metrics.
        """
        import joblib

        required = ["model.joblib", "evaluation_results.json", "feature_names.json"]
        if not all((MODELS_DIR / name).exists() for name in required):
            pytest.skip("Full artifact set not present — run train_model.py first")

        config = train_model.TrainConfig()
        df = train_model.load_raw_matrix(config)
        _, df_holdout = train_model.split_dev_holdout(df, config)

        model = joblib.load(MODELS_DIR / "model.joblib")
        recomputed = train_model.evaluate_on_holdout(model, df_holdout, config)

        with open(MODELS_DIR / "evaluation_results.json") as f:
            saved = json.load(f)

        assert recomputed["n_holdout_rows"] == saved["n_holdout_rows"]
        for metric in ("pr_auc", "f1", "precision", "recall"):
            assert recomputed[metric] == pytest.approx(saved[metric], abs=1e-6)
        assert recomputed["confusion_matrix"] == saved["confusion_matrix"]

    @pytest.mark.integration
    def test_feature_count_is_124_without_autoencoder(self):
        features_path = MODELS_DIR / "feature_names.json"
        if not features_path.exists():
            pytest.skip("No feature_names.json — run train_model.py first")

        with open(features_path) as f:
            feature_names = json.load(f)

        assert "autoencoder_score" not in feature_names
        assert len(feature_names) == 124
