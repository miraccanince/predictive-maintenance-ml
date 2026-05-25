"""
SentinelX - Stage 3b: Deep Learning Anomaly Scoring (Autoencoder)
==================================================================
Implements a multi-layer Autoencoder that learns the "geometry of normalcy"
from healthy system data, then scores ALL observations by their
reconstruction error — creating a powerful non-linear anomaly signal.

Architecture References:
- Paper 1 (SOFM+SVM): Self-Organizing Maps also learn latent representations.
  Our Autoencoder is the modern deep-learning analog — learning the manifold
  of normal system states in a continuous latent space.
- Paper 4 (Real-Time Quality): Deep learning-based anomaly scoring.
  "Reconstruction error captures multi-variate deviations that linear
  methods and tree-based splits cannot express."

Why Autoencoder + XGBoost (Hybrid Approach):
- XGBoost: Axis-aligned splits. Catches "if feature_A > threshold."
- Autoencoder: Learns curved manifold geometry. Catches "the COMBINATION
  of features has drifted from the normal operating region, even though
  each individual feature looks acceptable."
- Together: XGBoost uses the autoencoder's reconstruction error as a feature,
  gaining access to non-linear pattern detection it couldn't learn alone.

Latent Space Noise Reduction:
- Input: 124 features (many correlated, some noisy from sensor jitter)
- Bottleneck: 16 dimensions (forced compression)
- The network CANNOT memorize 124 dims in 16 — it must learn the
  essential structure (signal) and discard the rest (noise).
- This is equivalent to non-linear PCA: a principled dimensionality
  reduction that preserves the "true degrees of freedom" of healthy
  system behavior.

Data Leakage Prevention:
- StandardScaler fitted on HEALTHY TRAINING data only
- Autoencoder trained on HEALTHY data only (machine_failure == 0)
- Validation split: healthy data further split 80/20 for early stopping
- Scoring: applied to ALL data (healthy + failure) — no leakage
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SentinelX.Autoencoder")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder training and scoring."""

    # --- Input/Output ---
    feature_matrix_path: str = "data/features/feature_matrix.parquet"
    model_output_dir: str = "models"

    # --- Target (for splitting healthy vs failure data) ---
    target_col: str = "machine_failure"

    # --- Columns to exclude (same as train_model.py) ---
    exclude_cols: list = field(
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

    # --- Architecture ---
    # WHY 124 → 64 → 32 → 16 → 32 → 64 → 124?
    # Paper 4: "Progressive compression forces the network to learn
    # hierarchical abstractions of the input space."
    #
    # Layer rationale:
    # - 124 → 64: First compression. Removes obvious redundancies
    #   (e.g., torque_mean_6 and torque_mean_12 are nearly identical)
    # - 64 → 32: Intermediate. Captures interaction patterns.
    # - 32 → 16: Bottleneck. The "essence" of the system state.
    #   WHY 16? With 5 machines × 3 operating modes ≈ 15 natural clusters.
    #   16 dimensions can represent this plus margin for transitions.
    # - Decoder mirrors encoder for symmetric reconstruction.
    encoder_dims: list = field(default_factory=lambda: [64, 32, 16])
    decoder_dims: list = field(default_factory=lambda: [32, 64])
    # Activation: ReLU for hidden layers, none for output (linear reconstruction)

    # --- Training ---
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5  # L2 regularization on weights
    val_split: float = 0.2  # 20% of healthy data for validation
    early_stopping_patience: int = 10  # Stop if val_loss doesn't improve
    random_state: int = 42

    # --- Device ---
    # Use MPS (Apple Silicon) if available, else CPU
    # For production: would be CUDA
    device: str = "auto"

    # --- Scoring ---
    # After training, we compute MSE per row.
    # Optionally normalize scores to [0, 1] range.
    normalize_scores: bool = True


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================
# WHY a symmetric Encoder-Decoder?
# Paper 1: The SOFM learns a topology-preserving map of input space.
# Similarly, our autoencoder learns a smooth, continuous mapping from
# 124D input → 16D latent → 124D reconstruction.
#
# WHY ReLU (not Sigmoid/Tanh)?
# - Our features span arbitrary ranges (already standardized to ~N(0,1))
# - ReLU doesn't saturate for large values → better gradient flow
# - Faster training convergence than Sigmoid (no vanishing gradient)
#
# WHY Batch Normalization?
# - Stabilizes training with our mixed-scale features
# - Acts as mild regularization (reduces need for dropout)
# - Paper 4: "BN layers enable deeper networks for real-time processing"


class Autoencoder(nn.Module):
    """
    Multi-layer Autoencoder for anomaly detection via reconstruction error.

    Architecture: Input → Encoder (progressive compression) → Latent Space
    → Decoder (progressive expansion) → Reconstructed Input

    Anomaly Signal: MSE(input, reconstruction) — high for unseen failure modes.
    """

    def __init__(self, input_dim: int, config: AutoencoderConfig):
        super().__init__()

        # --- Build Encoder ---
        encoder_layers = []
        prev_dim = input_dim
        for dim in config.encoder_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Build Decoder ---
        decoder_layers = []
        for dim in config.decoder_dims:
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = dim
        # Final layer: linear activation (reconstruct arbitrary values)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        logger.info(
            f"  Autoencoder architecture: {input_dim} → "
            f"{' → '.join(map(str, config.encoder_dims))} → "
            f"{' → '.join(map(str, config.decoder_dims))} → {input_dim}"
        )
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation (for future clustering/viz)."""
        return self.encoder(x)


# =============================================================================
# 3. TRAINING PIPELINE
# =============================================================================
# WHY train on healthy data ONLY?
# Paper 4: "Anomaly detection via reconstruction assumes the model
# has ONLY seen normal patterns. Failure modes produce high MSE
# because they lie outside the learned manifold."
#
# If we trained on ALL data (including failures), the autoencoder would
# ALSO learn to reconstruct failures well → low MSE for failures →
# no anomaly signal. The key is: the network has never seen failure,
# so it can't reproduce it faithfully.
#
# DATA LEAKAGE NOTE:
# The scaler and model see ONLY healthy training data.
# Scores are computed on ALL data (including failures) AFTER training.
# This is analogous to fitting a normalcy model, then deploying it
# to score new (potentially anomalous) data — no leakage.


def train_autoencoder(
    X: pd.DataFrame, y: pd.Series, config: AutoencoderConfig
) -> tuple[Autoencoder, StandardScaler, dict]:
    """
    Train autoencoder on healthy data only.

    Strategy:
    1. Filter to healthy samples (machine_failure == 0)
    2. Split healthy data into train/val (80/20)
    3. Fit StandardScaler on healthy training data
    4. Train autoencoder with early stopping on validation MSE
    5. Return model, scaler, and training metrics

    Args:
        X: Feature matrix (all samples)
        y: Target column (for filtering healthy samples)
        config: Autoencoder configuration

    Returns:
        (model, scaler, metrics): Trained model, fitted scaler, training history
    """
    # --- Device Selection ---
    if config.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    logger.info(f"  Device: {device}")

    # --- Filter healthy data ---
    healthy_mask = y == 0
    X_healthy = X[healthy_mask].copy()
    n_healthy = len(X_healthy)
    n_failure = (~healthy_mask).sum()
    logger.info(
        f"  Healthy samples for training: {n_healthy:,} "
        f"(excluded {n_failure:,} failure samples)"
    )

    # --- Train/Val split (within healthy data) ---
    np.random.seed(config.random_state)
    val_size = int(n_healthy * config.val_split)
    indices = np.random.permutation(n_healthy)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_train_raw = X_healthy.iloc[train_idx]
    X_val_raw = X_healthy.iloc[val_idx]

    logger.info(f"  Train/Val split: {len(X_train_raw):,} / {len(X_val_raw):,}")

    # --- StandardScaler (fitted on healthy training ONLY) ---
    # WHY StandardScaler?
    # Neural networks expect inputs near N(0,1). Our features span:
    # - Power: 0-15000 W
    # - Temperature: 250-350 K
    # - Percentages: 0-100
    # Without scaling, high-magnitude features dominate the loss,
    # and the network ignores low-magnitude but informative features.
    #
    # WHY fit only on healthy training data?
    # If we fit on ALL data, failure patterns influence the mean/std.
    # The scaler would "know" about failure distributions — subtle leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    logger.info(f"  Scaler fitted on {len(X_train_raw):,} healthy training samples")

    # --- Convert to PyTorch tensors ---
    train_tensor = torch.FloatTensor(X_train_scaled)
    val_tensor = torch.FloatTensor(X_val_scaled)

    train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target (reconstruction)
    val_dataset = TensorDataset(val_tensor, val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # --- Initialize Model ---
    input_dim = X_train_scaled.shape[1]
    model = Autoencoder(input_dim, config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    # --- Training Loop with Early Stopping ---
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    logger.info(
        f"  Training: {config.epochs} max epochs, " f"patience={config.early_stopping_patience}"
    )

    for epoch in range(config.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{config.epochs}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}"
                f"{' *best*' if patience_counter == 0 else ''}"
            )

        if patience_counter >= config.early_stopping_patience:
            logger.info(
                f"  Early stopping at epoch {epoch+1} "
                f"(no improvement for {config.early_stopping_patience} epochs)"
            )
            break

    # --- Restore best model ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    metrics = {
        "best_val_loss": best_val_loss,
        "final_epoch": epoch + 1,
        "n_healthy_train": len(X_train_raw),
        "n_healthy_val": len(X_val_raw),
        "input_dim": input_dim,
        "latent_dim": config.encoder_dims[-1],
    }

    logger.info(
        f"  Training complete: best_val_loss={best_val_loss:.6f} "
        f"at epoch {epoch + 1 - patience_counter}"
    )

    return model, scaler, metrics


# =============================================================================
# 4. ANOMALY SCORING
# =============================================================================
# WHY MSE as the anomaly score?
# The autoencoder has learned to reconstruct "healthy" patterns faithfully.
# When a failure-state sample arrives:
# - Its feature combination is OUTSIDE the learned healthy manifold
# - The decoder can't map it back accurately from the 16D bottleneck
# - Result: high MSE between input and reconstruction
#
# This MSE captures NON-LINEAR multi-variate anomalies that XGBoost's
# axis-aligned splits cannot express.
#
# MEMORY EFFICIENCY for 500K+ rows:
# We score in batches (same batch_size as training) to avoid
# loading 500K × 124 features × 4 bytes = ~250MB into GPU at once.


def compute_anomaly_scores(
    model: Autoencoder,
    scaler: StandardScaler,
    X: pd.DataFrame,
    config: AutoencoderConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Compute per-row reconstruction error (anomaly score) for ALL samples.

    Strategy:
    - Scale ALL data using the healthy-fitted scaler
    - Reconstruct via the autoencoder
    - MSE per row = anomaly score
    - Higher MSE = more anomalous (farther from healthy manifold)

    Args:
        model: Trained autoencoder
        scaler: Scaler fitted on healthy training data
        X: Full feature matrix (healthy + failures)
        config: Configuration
        device: PyTorch device

    Returns:
        anomaly_scores: Array of per-row MSE values
    """
    logger.info("  Computing reconstruction errors for all samples...")

    X_scaled = scaler.transform(X)
    tensor_data = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    all_mse = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)
            # Per-row MSE (not batch-average)
            mse = ((batch_x - reconstructed) ** 2).mean(dim=1)
            all_mse.append(mse.cpu().numpy())

    scores = np.concatenate(all_mse)

    # --- Normalize to [0, 1] if configured ---
    # WHY normalize?
    # Raw MSE values depend on the scaler's scale. Normalizing to [0,1]
    # makes the feature interpretable and scale-compatible with other
    # features in the XGBoost model.
    if config.normalize_scores:
        min_score = scores.min()
        max_score = scores.max()
        scores = (scores - min_score) / (max_score - min_score + 1e-8)
        logger.info(f"  Scores normalized to [0, 1] (raw range: {min_score:.6f} - {max_score:.6f})")

    return scores


# =============================================================================
# 5. PIPELINE ORCHESTRATOR
# =============================================================================


def run_autoencoder_pipeline(config: AutoencoderConfig | None = None) -> Path:
    """
    Execute the full autoencoder pipeline: Train → Score → Integrate.

    Steps:
    1. Load feature matrix
    2. Train autoencoder on healthy data
    3. Score ALL data with reconstruction error
    4. Save updated feature matrix with autoencoder_score column
    5. Save model artifacts (for production inference)

    Args:
        config: Autoencoder configuration

    Returns:
        Path to updated feature matrix
    """
    if config is None:
        config = AutoencoderConfig()

    output_dir = Path(config.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SentinelX Stage 3b: Autoencoder Anomaly Scoring")
    logger.info("=" * 60)

    # --- Step 1: Load Data ---
    logger.info("\nStep 1/5: Loading feature matrix")
    path = Path(config.feature_matrix_path)
    df = pd.read_parquet(path)
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    y = df[config.target_col].astype(int)
    feature_cols = [col for col in df.columns if col not in config.exclude_cols]
    X = df[feature_cols].copy().fillna(0)
    logger.info(f"  Features: {len(feature_cols)} columns")

    # --- Step 2: Train Autoencoder ---
    logger.info("\nStep 2/5: Training autoencoder on healthy data")
    model, scaler, metrics = train_autoencoder(X, y, config)

    # --- Step 3: Device setup for scoring ---
    if config.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # --- Step 4: Score all data ---
    logger.info("\nStep 3/5: Computing anomaly scores")
    scores = compute_anomaly_scores(model, scaler, X, config, device)

    # --- Validate scores ---
    healthy_scores = scores[y == 0]
    failure_scores = scores[y == 1]
    logger.info(
        f"  Healthy samples:  mean={healthy_scores.mean():.4f}, " f"std={healthy_scores.std():.4f}"
    )
    logger.info(
        f"  Failure samples:  mean={failure_scores.mean():.4f}, " f"std={failure_scores.std():.4f}"
    )
    separation = (failure_scores.mean() - healthy_scores.mean()) / (healthy_scores.std() + 1e-8)
    logger.info(
        f"  Separation (Cohen's d): {separation:.2f} "
        f"({'STRONG' if separation > 1.0 else 'MODERATE' if separation > 0.5 else 'WEAK'})"
    )

    # --- Step 5: Integrate into feature matrix ---
    logger.info("\nStep 4/5: Updating Golden Dataset with autoencoder_score")
    df["autoencoder_score"] = scores
    updated_path = Path(config.feature_matrix_path)
    df.to_parquet(updated_path, engine="pyarrow", index=False)
    logger.info(f"  Updated feature matrix saved: {updated_path}")
    logger.info(f"  New shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Step 6: Save model artifacts ---
    logger.info("\nStep 5/5: Saving autoencoder artifacts")

    # Save PyTorch model
    torch_path = output_dir / "autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": metrics["input_dim"],
            "config": {
                "encoder_dims": config.encoder_dims,
                "decoder_dims": config.decoder_dims,
            },
        },
        torch_path,
    )
    logger.info(f"  Model: {torch_path} ({torch_path.stat().st_size/1024:.1f} KB)")

    # Save scaler
    scaler_path = output_dir / "autoencoder_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler: {scaler_path}")

    # Save metrics
    metrics_path = output_dir / "autoencoder_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    logger.info(f"  Metrics: {metrics_path}")

    # --- Final Report ---
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3b COMPLETE — Autoencoder Anomaly Score Ready")
    logger.info("=" * 60)
    logger.info(
        f"  Architecture: {metrics['input_dim']} → {metrics['latent_dim']} → {metrics['input_dim']}"
    )
    logger.info(f"  Best val_loss: {metrics['best_val_loss']:.6f}")
    logger.info(f"  Score separation (Cohen's d): {separation:.2f}")
    logger.info("  Golden Dataset updated: +1 column (autoencoder_score)")

    return updated_path


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    output = run_autoencoder_pipeline()
    print(f"\nUpdated feature matrix: {output}")
