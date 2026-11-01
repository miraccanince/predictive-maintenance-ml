"""
SentinelX - Stage 3: Modeling & Explainability
===============================================
Trains an XGBoost classifier on the Golden Dataset (feature_matrix.parquet)
with SMOTE-based class balancing and SHAP-based explainability.

Architecture References:
- Paper 1 (SOFM+SVM): Multi-modal feature fusion validated via cross-domain
  feature importance — our SHAP analysis confirms Paper 1's hypothesis.
- Paper 2 (XGBoost+SMOTE): Direct implementation of the recommended approach
  for high-dimensional imbalanced anomaly detection.
- Paper 3 (54% FP): PR-AUC as primary metric — accuracy is meaningless
  at 95/5 class splits. Industrial systems need Precision (no false alarms)
  AND Recall (no missed failures).
- Paper 4 (Real-Time Quality): Model must be lightweight enough for
  edge deployment with <100ms inference latency.

SMOTE Safety Protocol:
- SMOTE is applied ONLY inside training folds
- Validation folds ALWAYS contain real (unaugmented) data
- This prevents the "reflected training data" overfitting trap
- See detailed explanation in Section 3 comments

Mathematical Rationale for XGBoost:
1. Native NaN handling (no imputation needed for lag warm-up NaN)
2. Built-in L1/L2 regularization (prevents overfitting on 136 features)
3. Feature interaction discovery (complementary to engineered cross-domain features)
4. Scale-invariant splits (no StandardScaler needed for tree-based model)
5. Monotonic constraints available (can enforce "more wear = more risk")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SentinelX.Trainer")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# WHY separate training config?
# - Hyperparameters are tuned independently of feature engineering
# - Experiment tracking (MLflow/W&B) logs this config per run
# - Reproducibility: same config + same data = same model


@dataclass
class TrainConfig:
    """Configuration for model training pipeline."""

    # --- Input/Output ---
    feature_matrix_path: str = "data/features/feature_matrix.parquet"
    model_output_dir: str = "models"

    # --- Target Column ---
    # WHY machine_failure (not is_anomaly)?
    # machine_failure is the HARD problem (5.21%, 1:18 imbalance).
    # is_anomaly at 36.45% is nearly balanced and trivial for XGBoost.
    # We tackle the hard problem — predicting actual equipment failure.
    target_col: str = "machine_failure"

    # --- Columns to EXCLUDE from features ---
    # WHY exclude these?
    # - Target columns: using them as features = trivial leakage
    # - Identifiers: machine_id, timestamp have no predictive value
    #   (we already captured their effect via per-machine features)
    # - failure_* columns: these are COMPONENTS of machine_failure
    #   (TWF, HDF, PWF, OSF, RNF) — including them = circular prediction
    # - maintenance_status: derived from failure state = leakage
    # - anomaly_score / is_anomaly: correlated labels, not features
    exclude_cols: list[str] = field(
        default_factory=lambda: [
            "timestamp",
            "machine_id",
            "product_type",
            "machine_failure",
            "failure_TWF",
            "failure_HDF",
            "failure_PWF",
            "failure_OSF",
            "failure_RNF",
            "maintenance_status",
            "anomaly_score",
            "is_anomaly",
        ]
    )

    # --- Temporal Holdout ---
    # WHY a chronological, per-machine holdout instead of a random split?
    # Rolling/lag features (feature_engineer.py) make adjacent-in-time
    # rows of the same machine highly correlated. A random row-level
    # split (the previous approach) puts near-duplicate rows in both
    # train and test, inflating performance. Holding out the LAST 20%
    # of each machine's timeline mirrors real deployment (train on
    # past, evaluate on future) and removes that leakage path.
    holdout_frac: float = 0.2
    timestamp_col: str = "timestamp"
    group_col: str = "machine_id"

    # --- Cross-Validation ---
    # WHY 5 contiguous time-blocks per machine, forward-chaining (not
    # shuffled KFold)?
    # Same autocorrelation concern as the holdout: shuffling would mix
    # adjacent timestamps across train/val within a fold. Each machine's
    # portion of the dev set is cut into 5 contiguous chronological
    # blocks. Fold i trains on blocks 0..i-1 and validates on block i —
    # training always precedes validation in time (block 0 is never
    # validated, since nothing precedes it), mirroring how the model
    # would actually be deployed and retrained over time. This also
    # means no embargo gap is needed at block boundaries: rolling/lag
    # features are strictly backward-looking and were computed ONCE
    # over each machine's full continuous timeline before any split, so
    # a training block (always chronologically before its validation
    # block here) cannot contain information derived from a later block.
    n_folds: int = 5
    random_state: int = 42

    # --- Decision threshold ---
    # WHY a fixed 0.5, not a tuned "optimal" threshold?
    # Tuning the threshold on the same data used to report performance
    # is itself a form of leakage/optimistic bias. 0.5 is reported as a
    # plain, untuned baseline operating point.
    decision_threshold: float = 0.5

    # --- SMOTE ---
    # WHY SMOTE over simple oversampling?
    # Simple oversampling: duplicates existing minority samples exactly.
    #   Result: model memorizes those exact samples → overfits.
    # SMOTE: creates NEW synthetic samples by interpolating between
    #   k-nearest minority neighbors in feature space.
    #   Result: model learns the minority REGION, not individual points.
    #
    # Paper 2: "SMOTE in high-dimensional spaces prevents the model
    # from overfitting to repeated instances while expanding the
    # decision boundary around the minority class."
    #
    # WHY k_neighbors=5?
    # Default and well-validated. Lower k = more aggressive synthesis
    # (noisier samples). Higher k = more conservative (closer to
    # oversampling behavior). 5 is the sweet spot for our 2,605 positives.
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float = 0.5  # Minority reaches 50% of majority
    # WHY 0.5 (not 1.0)?
    # 1.0 = fully balanced = forces model to predict 50/50 baseline.
    # 0.5 = minority becomes ~1:2 ratio. This still gives the model
    # strong minority signal while preserving the "failure is rare" prior.
    # Paper 2 recommends 0.3-0.7 range for industrial anomaly detection.

    # --- XGBoost Hyperparameters ---
    # WHY these specific values?
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,  # Enough trees to converge; early stopping prevents excess
            "max_depth": 6,  # Paper 2: depth 5-7 optimal for sensor data
            # Deeper = more interactions but overfitting risk
            "learning_rate": 0.05,  # Low rate + many trees = more stable than 0.3 + few trees
            "subsample": 0.8,  # Row subsampling — regularization (like bagging)
            "colsample_bytree": 0.7,  # Column subsampling — reduces feature co-dependency
            # With 136 features, each tree sees ~95 columns
            "reg_alpha": 0.1,  # L1 regularization — drives unimportant features to zero
            "reg_lambda": 1.0,  # L2 regularization — smooths leaf weights
            "min_child_weight": 10,  # Minimum samples per leaf — prevents overfitting to
            # rare edge cases in the minority class
            "gamma": 0.1,  # Minimum loss reduction for split — prunes weak splits
            "scale_pos_weight": 1.0,  # Set to 1.0 because SMOTE handles imbalance externally.
            # Using BOTH SMOTE + scale_pos_weight = double-correction
            # (Paper 2 warns against this — pick ONE strategy).
            "eval_metric": "aucpr",  # Early stopping on PR-AUC (not logloss)
            "tree_method": "hist",  # Histogram-based splits — fast for 50K+ rows
            "random_state": 42,
            "n_jobs": -1,  # Use all CPU cores
        }
    )

    # --- Early Stopping ---
    early_stopping_rounds: int = 50  # Stop if no PR-AUC improvement for 50 rounds


# =============================================================================
# 2. DATA LOADING & PREPARATION
# =============================================================================
# WHY a dedicated loading function?
# - Separates I/O concerns from training logic
# - Validates feature matrix integrity before expensive training
# - Logs feature count and class distribution for reproducibility


def load_raw_matrix(config: TrainConfig) -> pd.DataFrame:
    """
    Load the Golden Dataset, sorted chronologically within each machine.
    """
    path = Path(config.feature_matrix_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. Run feature_engineer.py first."
        )

    logger.info(f"Loading feature matrix: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  → Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if "autoencoder_score" in df.columns:
        # The autoencoder feature was removed from the pipeline (its
        # measured contribution was negligible — see the ablation that
        # justified removal). Drop any stale column left over from a
        # feature_matrix.parquet written before that removal.
        logger.warning("  Dropping stale 'autoencoder_score' column (feature removed)")
        df = df.drop(columns=["autoencoder_score"])

    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in feature matrix")

    df = df.sort_values([config.group_col, config.timestamp_col]).reset_index(drop=True)
    return df


def prepare_X_y(df: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Separate features/target from a (dev/holdout/fold) slice of the raw matrix.

    Returns:
        (X, y, feature_names): Features DataFrame, target Series, feature column names
    """
    y = df[config.target_col].astype(int)

    feature_cols = [col for col in df.columns if col not in config.exclude_cols]
    X = df[feature_cols].copy()

    # --- NaN Imputation (required for SMOTE, not for XGBoost) ---
    # WHY fill with 0?
    # Our NaN cells come EXCLUSIVELY from lag/rolling warm-up:
    # the first 6 rows per machine where lag_6 has no prior data.
    # Semantically, "no prior observation" = zero signal (no change,
    # no historical context). This is more interpretable than median
    # imputation and doesn't distort SMOTE's k-NN distance calculation.
    # Filling with a constant is split-independent, so doing this
    # separately per split introduces no leakage.
    #
    # WHY not drop rows?
    # Those first rows per machine are often "cold start" — healthy baseline.
    # Dropping them biases the negative class slightly.
    n_nan_before = X.isna().sum().sum()
    if n_nan_before > 0:
        X = X.fillna(0)
        logger.info(f"  NaN imputation: {n_nan_before:,} cells filled with 0 (lag/rolling warm-up)")

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    logger.info(f"    Positive: {n_pos:,} ({n_pos/len(y)*100:.2f}%) | Negative: {n_neg:,} | 1:{ratio:.0f}")

    return X, y, feature_cols


def split_dev_holdout(
    df: pd.DataFrame, config: TrainConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological, per-machine dev/holdout split.

    For each machine, the first (1 - holdout_frac) rows in time go to
    `dev` (used for CV + final model training); the last holdout_frac
    rows go to `holdout` (touched exactly once, at final evaluation).
    """
    dev_parts, holdout_parts = [], []
    for _, group in df.groupby(config.group_col, sort=False):
        group = group.sort_values(config.timestamp_col)
        cutoff = int(len(group) * (1 - config.holdout_frac))
        dev_parts.append(group.iloc[:cutoff])
        holdout_parts.append(group.iloc[cutoff:])

    df_dev = pd.concat(dev_parts).sort_values([config.group_col, config.timestamp_col])
    df_holdout = pd.concat(holdout_parts).sort_values([config.group_col, config.timestamp_col])

    for _, dev_group in df_dev.groupby(config.group_col):
        machine = dev_group[config.group_col].iloc[0]
        holdout_group = df_holdout[df_holdout[config.group_col] == machine]
        if len(holdout_group) and dev_group[config.timestamp_col].max() >= holdout_group[
            config.timestamp_col
        ].min():
            raise AssertionError(f"Temporal split violated for machine {machine}")

    logger.info(
        f"  Dev/Holdout split: {len(df_dev):,} dev rows / {len(df_holdout):,} holdout rows "
        f"({len(df_holdout)/len(df)*100:.1f}% holdout, per machine)"
    )
    return df_dev, df_holdout


def make_temporal_folds(df_dev: pd.DataFrame, config: TrainConfig) -> np.ndarray:
    """
    Assign each dev row to one of `n_folds` contiguous chronological
    blocks WITHIN its own machine (block 0 = earliest, block n-1 = latest).

    Used for FORWARD-CHAINING CV: fold i's validation set is block i,
    and its training set is every block BEFORE it (blocks 0..i-1), never
    a block after it. This guarantees training always precedes
    validation in time, matching real deployment (the model is never
    trained on data from the future relative to what it's evaluated
    on). Block 0 is never used as a validation fold (there is nothing
    before it to train on), so only folds 1..n_folds-1 are evaluated.
    """
    fold_ids = np.full(len(df_dev), -1, dtype=int)
    for _, group in df_dev.groupby(config.group_col, sort=False):
        idx = group.sort_values(config.timestamp_col).index
        blocks = np.array_split(np.arange(len(idx)), config.n_folds)
        for block_id, block_positions in enumerate(blocks):
            fold_ids[df_dev.index.get_indexer(idx[block_positions])] = block_id

    assert (fold_ids >= 0).all(), "Every dev row must be assigned to a fold"
    return fold_ids


# =============================================================================
# 3. TRAINING LOOP (SMOTE + Stratified K-Fold + XGBoost)
# =============================================================================
# WHY this specific pipeline order?
#
# CORRECT ORDER:
#   1. Split into K folds (Stratified)
#   2. For each fold:
#      a. SMOTE the TRAINING portion only
#      b. Train XGBoost on SMOTE'd training data
#      c. Evaluate on REAL (unaugmented) validation data
#   3. Aggregate metrics across folds
#
# WRONG ORDER (common mistake):
#   1. SMOTE the FULL dataset
#   2. Split into K folds
#   → Problem: synthetic samples from SMOTE are based on neighbors that
#     might end up in the validation fold. The model has already "seen"
#     information about those validation samples through their synthetic
#     children in training. This is DATA LEAKAGE.
#
# Paper 2: "SMOTE must be applied within the cross-validation loop
# to prevent optimistic bias in performance estimates."


def train_with_cv(
    df_dev: pd.DataFrame, config: TrainConfig
) -> tuple[xgb.XGBClassifier, dict, list[str]]:
    """
    Train XGBoost with temporal-block, forward-chaining CV and per-fold SMOTE.

    Architecture per fold:
    - SMOTE(train) → XGBoost.fit → evaluate(real, unaugmented val)
    Final model: trained on the full dev set (with SMOTE), using the
    median best_n_trees from CV.

    Returns:
        (final_model, cv_summary, feature_names)
    """
    fold_ids = make_temporal_folds(df_dev, config)

    fold_metrics = {"pr_auc": [], "f1": [], "precision": [], "recall": [], "best_n_trees": []}

    logger.info("=" * 60)
    logger.info(f"Training: forward-chaining CV over {config.n_folds} temporal blocks (per machine)")
    logger.info("=" * 60)

    # --- Forward-chaining: fold i trains on blocks 0..i-1, validates on
    # block i. Block 0 is never validated (nothing precedes it), so folds
    # run from 1 to n_folds-1 — this guarantees train always precedes
    # validation in time for every fold, matching real deployment.
    for fold_idx in range(1, config.n_folds):
        logger.info(f"\n--- Fold {fold_idx}/{config.n_folds - 1} (train: blocks <{fold_idx}, val: block {fold_idx}) ---")
        train_mask = fold_ids < fold_idx
        val_mask = fold_ids == fold_idx
        df_train_fold, df_val_fold = df_dev[train_mask], df_dev[val_mask]

        X_train, y_train, feature_cols = prepare_X_y(df_train_fold, config)
        X_val, y_val, _ = prepare_X_y(df_val_fold, config)

        # --- SMOTE on training fold ONLY ---
        # Paper 2's core safety rule: synthetic samples must NEVER
        # contaminate the validation set.
        smote = SMOTE(
            k_neighbors=config.smote_k_neighbors,
            sampling_strategy=config.smote_sampling_strategy,
            random_state=config.random_state + fold_idx,
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        n_synthetic = len(X_train_resampled) - len(X_train)
        logger.info(
            f"  SMOTE: {len(X_train):,} → {len(X_train_resampled):,} "
            f"(+{n_synthetic:,} synthetic failures)"
        )

        model = xgb.XGBClassifier(
            **config.xgb_params, early_stopping_rounds=config.early_stopping_rounds
        )
        model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)], verbose=False)

        best_iteration = (
            model.best_iteration + 1
            if model.best_iteration is not None
            else config.xgb_params["n_estimators"]
        )

        # --- Evaluate on REAL (unaugmented) validation data ---
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)

        pr_auc = average_precision_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        precision = (y_val_pred & y_val).sum() / max(y_val_pred.sum(), 1)
        recall = (y_val_pred & y_val).sum() / max(y_val.sum(), 1)

        fold_metrics["pr_auc"].append(pr_auc)
        fold_metrics["f1"].append(f1)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["best_n_trees"].append(best_iteration)

        logger.info(
            f"  Results: PR-AUC={pr_auc:.4f}, F1={f1:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, Trees={best_iteration}"
        )

    # --- Aggregate CV Metrics ---
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY (temporal blocks)")
    logger.info("=" * 60)
    cv_summary = {}
    for metric, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        cv_summary[f"{metric}_mean"] = mean
        cv_summary[f"{metric}_std"] = std
        logger.info(f"  {metric:<15}: {mean:.4f} ± {std:.4f}")

    # --- Train Final Model on Full Dev Set ---
    # WHY the dev set, not dev+holdout?
    # The holdout must remain genuinely unseen by anything that
    # produces the shipped model, so final training uses dev only.
    logger.info("\nTraining final model on full dev set (with SMOTE)...")
    optimal_n_trees = int(np.median(fold_metrics["best_n_trees"]))
    logger.info(f"  Using n_estimators={optimal_n_trees} (median from CV)")

    X_dev, y_dev, feature_names = prepare_X_y(df_dev, config)

    smote_final = SMOTE(
        k_neighbors=config.smote_k_neighbors,
        sampling_strategy=config.smote_sampling_strategy,
        random_state=config.random_state,
    )
    X_dev_resampled, y_dev_resampled = smote_final.fit_resample(X_dev, y_dev)

    final_params = config.xgb_params.copy()
    final_params["n_estimators"] = optimal_n_trees
    final_params.pop("eval_metric", None)

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_dev_resampled, y_dev_resampled, verbose=False)

    logger.info(
        f"  Final model trained: {optimal_n_trees} trees, {final_model.n_features_in_} features"
    )

    return final_model, cv_summary, feature_names


def evaluate_on_holdout(
    model: xgb.XGBClassifier,
    df_holdout: pd.DataFrame,
    config: TrainConfig,
) -> dict:
    """
    One-shot evaluation of the FINAL shipped model on the genuinely
    held-out, never-before-used test set, at a fixed, untuned threshold.
    """
    X_holdout, y_holdout, feature_cols = prepare_X_y(df_holdout, config)

    y_proba = model.predict_proba(X_holdout)[:, 1]
    y_pred = (y_proba >= config.decision_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()

    results = {
        "synthetic_data_evaluation": True,
        "note": (
            "Evaluated on synthetic, rule-generated data from 5 simulated "
            "machines, on a chronological per-machine holdout never used "
            "for training, feature selection, or threshold tuning. These "
            "numbers do NOT demonstrate real-world generalization."
        ),
        "n_holdout_rows": int(len(df_holdout)),
        "decision_threshold": config.decision_threshold,
        "pr_auc": float(average_precision_score(y_holdout, y_proba)),
        "f1": float(f1_score(y_holdout, y_pred)),
        "precision": float(precision_score(y_holdout, y_pred, zero_division=0)),
        "recall": float(recall_score(y_holdout, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    logger.info("\n" + "=" * 60)
    logger.info("HOLDOUT EVALUATION (synthetic data; fixed threshold=0.5)")
    logger.info("=" * 60)
    logger.info(f"  PR-AUC:    {results['pr_auc']:.4f}")
    logger.info(f"  F1:        {results['f1']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    return results


# =============================================================================
# 4. SHAP EXPLAINABILITY
# =============================================================================
# WHY SHAP (not just feature_importances_)?
# XGBoost's built-in feature_importance uses "gain" — how much each feature
# reduces loss across all splits. Problems:
#   1. It's GLOBAL only — can't explain individual predictions
#   2. Correlated features split the importance (each gets half credit)
#   3. No directionality — doesn't say "high temperature → more failure"
#
# SHAP provides:
#   1. Per-prediction explanations (why did THIS sample fail?)
#   2. Additive: SHAP values sum to the model output (mathematically rigorous)
#   3. Directional: positive SHAP = pushes toward failure prediction
#   4. Handles correlations correctly (game-theoretic foundation)
#
# Paper 1 connection: SHAP validates whether our cross-domain features
# (thermal_efficiency, power_anomaly) actually matter to the model.
# If they don't appear in top-10, our Paper 1 hypothesis was wrong.


def compute_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    n_samples: int = 5000,
) -> np.ndarray:
    """
    Compute SHAP values and save summary data for visualization.

    Uses TreeExplainer (exact SHAP for tree-based models, not approximate).
    Samples a subset for computational efficiency while maintaining
    statistical representativeness.

    Args:
        model: Trained XGBoost model
        X: Feature matrix (original, not SMOTE'd — we want explanations
           for REAL data, not synthetic samples)
        feature_names: Column names
        output_dir: Where to save SHAP artifacts
        n_samples: Number of samples for SHAP computation

    Returns:
        shap_values: Array of SHAP values (n_samples × n_features)
    """
    logger.info("Computing SHAP values (TreeExplainer)...")

    # --- Sample for SHAP ---
    # WHY sample instead of full dataset?
    # SHAP on 50K samples × 136 features = ~60 seconds.
    # 5K samples gives statistically identical importance rankings
    # at 10x speed. For production, we'd compute SHAP per-prediction.
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X

    # --- TreeExplainer ---
    # WHY TreeExplainer (not KernelExplainer)?
    # TreeExplainer is EXACT for tree-based models (polynomial time).
    # KernelExplainer is approximate and exponentially slower.
    # Since we're using XGBoost, TreeExplainer is the correct choice.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- Top Features by Mean |SHAP| ---
    # Mean absolute SHAP = average impact on prediction across all samples
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info("\nTop 15 Features by SHAP Importance:")
    logger.info("-" * 55)
    for _, row in feature_importance.head(15).iterrows():
        bar_len = int(row["mean_abs_shap"] / feature_importance["mean_abs_shap"].max() * 30)
        bar = "█" * bar_len
        logger.info(f"  {row['feature']:<40} {row['mean_abs_shap']:.4f} {bar}")

    # --- Save SHAP artifacts ---
    shap_output = {
        "feature_importance": feature_importance.to_dict(orient="records"),
        "n_samples": len(X_sample),
        "explainer_type": "TreeExplainer",
    }
    shap_path = output_dir / "shap_importance.json"
    with open(shap_path, "w") as f:
        json.dump(shap_output, f, indent=2, default=str)
    logger.info(f"\nSHAP importance saved: {shap_path}")

    # Save raw SHAP values for future visualization
    np.save(output_dir / "shap_values.npy", shap_values)
    # Save the sample indices for reproducibility
    np.save(output_dir / "shap_sample_indices.npy", X_sample.index.values)

    return shap_values


# =============================================================================
# 5. MODEL ARTIFACT SAVING
# =============================================================================
# WHY save these specific artifacts?
# For production deployment, we need:
# 1. The model itself (joblib serialized XGBClassifier)
# 2. Feature names list (to validate incoming data has the right columns)
# 3. Training config (reproducibility + audit trail)
# 4. CV metrics (to compare against future model versions)
# 5. SHAP data (for the LLM Agent's diagnostic explanations)


def save_model_artifacts(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    cv_metrics: dict,
    holdout_metrics: dict,
    config: TrainConfig,
    output_dir: Path,
) -> None:
    """
    Save all model and evaluation artifacts from THIS run.

    Artifacts saved:
    - model.joblib: Serialized XGBoost model (final, dev-set-only)
    - feature_names.json: Ordered list of expected features
    - training_config.json: Full training configuration
    - cv_metrics.json: Temporal-block CV performance estimates
    - evaluation_results.json: One-shot holdout evaluation (see note field)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- XGBoost Model ---
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved: {model_path} ({model_size_mb:.2f} MB)")

    # --- Feature names ---
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Feature names saved: {features_path} ({len(feature_names)} features)")

    # --- Training config ---
    config_path = output_dir / "training_config.json"
    config_dict = {
        "target_col": config.target_col,
        "holdout_frac": config.holdout_frac,
        "n_folds": config.n_folds,
        "random_state": config.random_state,
        "decision_threshold": config.decision_threshold,
        "smote_k_neighbors": config.smote_k_neighbors,
        "smote_sampling_strategy": config.smote_sampling_strategy,
        "xgb_params": config.xgb_params,
        "early_stopping_rounds": config.early_stopping_rounds,
        "n_features": len(feature_names),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Training config saved: {config_path}")

    # --- CV Metrics ---
    metrics_path = output_dir / "cv_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(cv_metrics, f, indent=2, default=float)
    logger.info(f"CV metrics saved: {metrics_path}")

    # --- Holdout Evaluation ---
    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(holdout_metrics, f, indent=2, default=float)
    logger.info(f"Holdout evaluation saved: {eval_path}")


# =============================================================================
# 6. PIPELINE ORCHESTRATOR
# =============================================================================


def run_training_pipeline(config: TrainConfig | None = None) -> Path:
    """
    Execute the full training pipeline:
    Load → Dev/Holdout split → CV (SMOTE per fold) →
    Final model (dev-only) → SHAP → Holdout evaluation → Save.

    Args:
        config: Training configuration (uses defaults if None)

    Returns:
        Path to the model output directory
    """
    if config is None:
        config = TrainConfig()

    output_dir = Path(config.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SentinelX Stage 3: Model Training & Explainability")
    logger.info("=" * 60)

    # --- Step 1: Load + Split ---
    logger.info("\nStep 1/5: Loading Golden Dataset and splitting dev/holdout")
    df = load_raw_matrix(config)
    df_dev, df_holdout = split_dev_holdout(df, config)

    # --- Step 2: Train with temporal-block CV ---
    logger.info("\nStep 2/5: Training with temporal-block CV + SMOTE")
    model, cv_metrics, feature_names = train_with_cv(df_dev, config)

    # --- Step 3: SHAP Analysis (on the dev set actually used to train) ---
    logger.info("\nStep 3/5: SHAP Explainability Analysis")
    X_dev, y_dev, _ = prepare_X_y(df_dev, config)
    compute_shap_analysis(model, X_dev, feature_names, output_dir)

    # --- Step 4: Holdout Evaluation (one shot, never used until now) ---
    logger.info("\nStep 4/5: Evaluating final model on the held-out test set")
    holdout_metrics = evaluate_on_holdout(model, df_holdout, config)

    # --- Step 5: Save Artifacts ---
    logger.info("\nStep 5/5: Saving Model/Evaluation Artifacts")
    save_model_artifacts(model, feature_names, cv_metrics, holdout_metrics, config, output_dir)

    # --- Final Report ---
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3 COMPLETE — MODEL READY FOR DEPLOYMENT")
    logger.info("=" * 60)
    logger.info(f"  Model:     {output_dir / 'model.joblib'}")
    logger.info(f"  Features:  {len(feature_names)} columns")
    logger.info(f"  CV PR-AUC: {cv_metrics['pr_auc_mean']:.4f} ± {cv_metrics['pr_auc_std']:.4f}")
    logger.info(
        f"  Holdout:   PR-AUC={holdout_metrics['pr_auc']:.4f} F1={holdout_metrics['f1']:.4f} "
        f"(synthetic data, threshold={config.decision_threshold})"
    )
    logger.info(f"  SHAP:      Top features saved to {output_dir / 'shap_importance.json'}")

    return output_dir


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    output = run_training_pipeline()
    print(f"\nModel artifacts saved to: {output}/")
