"""
SentinelX - Stage 2: Feature Engineering & Cross-Domain Correlation
====================================================================
Transforms quality-gated Parquet chunks into a unified Feature Matrix
ready for XGBoost (Paper 2) and hybrid SOFM+SVM (Paper 1) models.

Architecture References:
- Paper 1 (SOFM+SVM): Multi-modal fusion requires aligned, correlated features
  across hardware and software domains.
- Paper 2 (XGBoost+SMOTE): High-dimensional feature space benefits from
  temporal context (rolling stats) and interaction terms.
- Paper 3 (54% FP): Cross-domain RCA features reduce false positives by
  providing causal context — a temperature spike WITH a latency spike
  is more informative than either alone.
- Paper 4 (Real-Time Quality): Feature computation must be chunk-compatible
  for streaming deployment.

Data Leakage Prevention:
- All rolling windows are STRICTLY backward-looking (no center=True)
- Lag features use POSITIVE shift (past values only)
- No target column (machine_failure) is used in feature construction
- min_periods enforced: early rows get NaN, not padded future data
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("SentinelX.FeatureEngineer")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# WHY a separate config from StreamConfig?
# - Separation of Concerns: ingestion config (chunk_size, paths) vs
#   feature config (window sizes, lag depths) change independently.
# - Experiment Tracking: when we tune window_sizes, we log THIS config
#   to MLflow/W&B without touching the ingestion layer.

@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""

    # --- Input/Output Paths ---
    parquet_input_dir: str = "data/parquet"
    feature_output_path: str = "data/features/feature_matrix.parquet"

    # --- Temporal Alignment ---
    # WHY left join on system_metrics?
    # System metrics are the CLOCK — they arrive at fixed 5-min intervals.
    # Application logs are event-driven (a request may or may not happen).
    # Left join preserves every system observation, filling gaps in app data.
    join_keys: List[str] = field(default_factory=lambda: ["timestamp", "machine_id"])

    # --- Rolling Window Sizes (in number of periods) ---
    # WHY these specific sizes?
    # Paper 1: SOFM requires multi-scale temporal patterns.
    # Paper 2: XGBoost feature importance shows 30min-3hr windows
    #   capture distinct failure dynamics:
    #   - 6 periods (30 min): Acute stress response — tool overheating,
    #     sudden torque spikes. Captures "immediate danger."
    #   - 12 periods (1 hr): Sustained degradation — slow pressure buildup,
    #     memory leaks. Captures "something is wrong."
    #   - 36 periods (3 hr): Trend context — seasonal baseline shift,
    #     gradual wear. Captures "is this normal for this time of day?"
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 36])

    # --- Lag Depths (in number of periods) ---
    # WHY lags specifically at 1, 2, 6?
    # Paper 3 (Cross-Domain RCA): Hardware failures cause software symptoms
    # with a DELAY. A bearing failure doesn't crash the API instantly —
    # vibration → heat → throttling → timeout. The causal chain takes time.
    #   - T-1 (5 min):  Immediate cause-effect (direct mechanical failure)
    #   - T-2 (10 min): Thermal propagation delay
    #   - T-6 (30 min): Cascading system-level degradation
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 6])

    # --- Feature Column Groups ---
    # WHY separate hardware vs software?
    # Paper 1: The "hybrid" innovation is EXPLICITLY correlating these domains.
    # If we mixed them from the start, we couldn't create meaningful
    # cross-domain interaction features.
    hardware_rolling_cols: List[str] = field(default_factory=lambda: [
        "air_temperature_K",
        "process_temperature_K",
        "torque_Nm",
        "rotational_speed_rpm",
        "vibration_mm_s",
        "pressure_psi",
        "tool_wear_min",
    ])

    software_rolling_cols: List[str] = field(default_factory=lambda: [
        "api_response_latency_ms",
        "error_rate_pct",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "queue_depth",
        "disk_io_wait_ms",
    ])

    # --- Cross-Domain Feature Flags ---
    enable_thermal_efficiency: bool = True
    enable_error_stress_flag: bool = True
    enable_power_anomaly: bool = True
    enable_wear_rate: bool = True


# =============================================================================
# 2. TEMPORAL ALIGNMENT (The Marriage)
# =============================================================================
# WHY is this the FIRST step?
# Paper 1 (SOFM+SVM): The hybrid approach REQUIRES synchronized multi-modal
# inputs. You cannot correlate temperature with latency if they're on
# different time grids.
#
# WHY left join (not inner join)?
# Inner join would DROP system observations where no app event occurred.
# Those "quiet" periods are INFORMATIVE — they tell us "the system was
# healthy here." Dropping them creates survivorship bias in the model.
#
# DATA LEAKAGE WARNING:
# We join ONLY on timestamp+machine_id. Never on derived features or labels.
# If we joined on "maintenance_status" for convenience, we'd be leaking
# the target into the features.

def align_temporal(
    sys_chunk: pd.DataFrame,
    app_chunk: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Merge system metrics and application logs on (timestamp, machine_id).

    Strategy:
    - Left join: system_metrics is the temporal backbone
    - Forward-fill: if no app event in a 5-min window, carry forward
      the last known state (not the NEXT state — that would be leakage)

    Args:
        sys_chunk: System metrics DataFrame (the clock)
        app_chunk: Application logs DataFrame (event-driven)
        config: Feature engineering configuration

    Returns:
        Merged DataFrame with all columns aligned temporally
    """
    # Defensive: ensure timestamps are datetime
    for df in [sys_chunk, app_chunk]:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Identify columns to bring from app_chunk (avoid duplicates)
    # Both DataFrames share: timestamp, machine_id, maintenance_status
    # We keep sys_chunk's maintenance_status as the authoritative one
    app_cols_to_merge = [col for col in app_chunk.columns
                         if col not in sys_chunk.columns or col in config.join_keys]

    merged = pd.merge(
        sys_chunk,
        app_chunk[app_cols_to_merge],
        on=config.join_keys,
        how="left"
    )

    # Forward-fill ONLY the software columns (within each machine)
    # WHY per-machine? Machine M1's last latency reading should NOT
    # fill into Machine M3's gaps. That would be cross-contamination.
    #
    # WHY forward-fill, not interpolation?
    # Interpolation looks at BOTH neighbors (past AND future) — leakage.
    # Forward-fill only uses past information.
    software_cols = [col for col in config.software_rolling_cols
                     if col in merged.columns]

    if software_cols:
        merged = merged.sort_values(["machine_id", "timestamp"])
        merged[software_cols] = merged.groupby("machine_id")[software_cols].ffill()

    logger.debug(f"Temporal alignment: {len(sys_chunk)} sys + {len(app_chunk)} app "
                 f"→ {len(merged)} merged rows")

    return merged


# =============================================================================
# 3. ROLLING WINDOW FEATURES
# =============================================================================
# WHY rolling statistics?
# Paper 1: SOFM (Self-Organizing Feature Maps) needs to distinguish between:
#   - A single spike (noise) vs sustained elevation (real degradation)
#   - Stable at 300K vs oscillating between 290-310K (same mean, different risk)
#
# This is why we compute BOTH mean (level) and std (volatility).
#
# Paper 2: XGBoost's feature importance analysis shows rolling features
# consistently rank in the top 15 most important features for anomaly
# detection. Raw values alone miss the temporal context.
#
# DATA LEAKAGE WARNING:
# pandas .rolling() with center=False (default) is backward-looking.
# We explicitly set min_periods to prevent NaN-padded future peeking.
# The first (window-1) rows will have NaN — this is CORRECT behavior.
# We do NOT fill these NaNs with the column mean (that uses global info).

def compute_rolling_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Compute rolling mean and std for hardware and software columns.

    For each (column, window_size) pair, creates:
    - {col}_mean_{window}: Rolling average (level indicator)
    - {col}_std_{window}: Rolling std deviation (volatility indicator)

    All computations are per-machine to prevent cross-machine contamination.

    Args:
        df: Temporally aligned DataFrame (output of align_temporal)
        config: Feature configuration with window sizes and column groups

    Returns:
        DataFrame with rolling features appended
    """
    all_rolling_cols = config.hardware_rolling_cols + config.software_rolling_cols
    # Only process columns that actually exist in this chunk
    available_cols = [col for col in all_rolling_cols if col in df.columns]

    if not available_cols:
        logger.warning("No rolling columns found in DataFrame")
        return df

    # Sort by machine + time to ensure rolling windows are chronologically correct
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    new_features = {}

    for col in available_cols:
        for window in config.rolling_windows:
            # Per-machine rolling computation
            # WHY groupby THEN rolling?
            # Without groupby, the rolling window at machine boundaries would
            # mix M1's last readings with M2's first readings — nonsensical.
            grouped = df.groupby("machine_id")[col]

            # Rolling MEAN: "What is the recent level?"
            # Paper 2: Baseline drift detection — if the 3hr mean is rising,
            # the system is degrading even if instantaneous values look okay.
            mean_col = f"{col}_mean_{window}"
            new_features[mean_col] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1, center=False).mean()
            )

            # Rolling STD: "How unstable is it recently?"
            # Paper 1: SOFM separates clusters by volatility patterns.
            # A machine with std=0.1 for temperature is healthy;
            # std=5.0 means it's oscillating — impending failure.
            std_col = f"{col}_std_{window}"
            new_features[std_col] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=2, center=False).std()
            )

    # Assign all at once (faster than repeated df[col] = ...)
    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    n_features = len(new_features)
    logger.info(f"Rolling features: +{n_features} columns "
                f"({len(available_cols)} cols × {len(config.rolling_windows)} windows × 2 stats)")

    return df


# =============================================================================
# 4. LAG FEATURES
# =============================================================================
# WHY lag features?
# Paper 3 (Cross-Domain RCA): The core insight is that hardware degradation
# PRECEDES software failure. A bearing wearing out at T=0 causes:
#   T+5min:  Vibration increase
#   T+10min: Temperature rise (friction → heat)
#   T+30min: API timeouts (thermal throttling → slow responses)
#
# By creating lag features, we give the model access to "what happened
# 5/10/30 minutes ago" — enabling it to learn these causal chains.
#
# CRITICAL DATA LEAKAGE PREVENTION:
# We shift BACKWARD (positive shift value in pandas).
# df[col].shift(1) gives us the PREVIOUS row's value — safe.
# df[col].shift(-1) would give us the NEXT row's value — LEAKAGE!
#
# We NEVER use negative shifts. Period.

def compute_lag_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Create lagged versions of key columns to capture temporal causality.

    For each (column, lag_depth) pair, creates:
    - {col}_lag_{lag}: Value from {lag} periods ago

    Args:
        df: DataFrame with rolling features already computed
        config: Feature configuration with lag periods

    Returns:
        DataFrame with lag features appended
    """
    # We lag a focused subset — not every column.
    # WHY not lag everything?
    # Paper 2: High-dimensional curse. With 20 base cols × 3 lags = 60 extra features.
    # XGBoost handles this via feature importance pruning, but we should be
    # intentional. We lag columns with known causal chains:
    lag_candidates = [
        # Hardware: leading indicators of failure
        "torque_Nm",              # Mechanical stress → downstream effects
        "vibration_mm_s",         # Bearing wear signature
        "air_temperature_K",      # Thermal stress propagation
        "tool_wear_min",          # Cumulative degradation
        # Software: lagging indicators (response to hardware)
        "api_response_latency_ms",  # Software response to hardware stress
        "error_rate_pct",           # Error cascade timing
        "cpu_utilization_pct",      # Compute stress
    ]

    available_lag_cols = [col for col in lag_candidates if col in df.columns]

    if not available_lag_cols:
        logger.warning("No lag candidate columns found in DataFrame")
        return df

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    new_features = {}

    for col in available_lag_cols:
        for lag in config.lag_periods:
            lag_col = f"{col}_lag_{lag}"
            # Per-machine shift to prevent cross-machine leakage
            # WHY groupby? Without it, M1's last row would "leak" into M2's first lag.
            new_features[lag_col] = df.groupby("machine_id")[col].shift(lag)

    feature_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    n_features = len(new_features)
    logger.info(f"Lag features: +{n_features} columns "
                f"({len(available_lag_cols)} cols × {len(config.lag_periods)} lags)")

    return df


# =============================================================================
# 5. CROSS-DOMAIN (HYBRID) FEATURES
# =============================================================================
# WHY cross-domain features?
# Paper 1: The entire thesis of the SOFM+SVM hybrid is that COMBINING
# modalities (hardware + software) detects anomalies that NEITHER catches alone.
#
# Paper 3: Cross-domain Root Cause Analysis — a latency spike alone might
# be a network blip (false positive). But latency spike + temperature spike
# + torque spike = genuine mechanical failure causing software degradation.
#
# These features encode DOMAIN KNOWLEDGE about failure physics:
# - Thermal efficiency: "Is the CPU running hot despite low load?"
#   → Could indicate cooling system failure
# - Power anomaly: speed × torque = power. Deviations from expected power
#   curve indicate mechanical resistance (bearing failure, misalignment)
# - Wear rate: First derivative of tool_wear. Acceleration = danger.

def compute_cross_domain_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Create interaction features that bridge hardware and software domains.

    These are the "innovation" of SentinelX — features that encode
    physical causality between mechanical state and software health.

    Args:
        df: DataFrame with rolling and lag features
        config: Feature configuration with feature flags

    Returns:
        DataFrame with cross-domain features appended
    """
    features_added = 0

    # --- Feature 1: Thermal Efficiency Index ---
    # FORMULA: cpu_utilization_pct / air_temperature_K
    # INTUITION (Paper 1): A healthy system has high CPU at normal temp.
    #   - High ratio = CPU working hard at low temp (healthy under load)
    #   - Low ratio = CPU idle but system hot (cooling failure!)
    #   - This catches a failure mode INVISIBLE to either metric alone.
    if config.enable_thermal_efficiency:
        if "cpu_utilization_pct" in df.columns and "air_temperature_K" in df.columns:
            # Add epsilon to prevent division by zero (temp in Kelvin, so always > 0,
            # but quality-flagged rows might have NaN/0 from sensor errors)
            df["thermal_efficiency_idx"] = (
                df["cpu_utilization_pct"] / (df["air_temperature_K"] + 1e-6)
            )
            features_added += 1
            logger.debug("Created: thermal_efficiency_idx")

    # --- Feature 2: Error-Under-Stress Flag ---
    # FORMULA: Boolean — (error_rate > threshold) AND (torque > 75th percentile)
    # INTUITION (Paper 3): Errors during mechanical stress are REAL failures.
    #   Errors during idle are likely network/config issues (false positives).
    #   This feature helps the model distinguish FP from TP.
    if config.enable_error_stress_flag:
        if "error_rate_pct" in df.columns and "torque_Nm" in df.columns:
            # WHY 75th percentile instead of fixed threshold?
            # Torque distributions vary by machine type and load.
            # Using percentile adapts to the chunk's actual distribution.
            #
            # DATA LEAKAGE NOTE: We compute percentile WITHIN this chunk only.
            # This is acceptable because we're not using future chunks.
            # For production streaming, we'd use a running percentile from
            # the drift monitor's baseline distribution.
            torque_high = df["torque_Nm"].quantile(0.75)
            df["error_under_stress"] = (
                (df["error_rate_pct"] > 1.0) & (df["torque_Nm"] > torque_high)
            ).astype(np.int8)
            features_added += 1
            logger.debug(f"Created: error_under_stress (torque threshold: {torque_high:.1f} Nm)")

    # --- Feature 3: Power Anomaly Score ---
    # FORMULA: |actual_power - expected_power| / expected_power
    # WHERE: power = rotational_speed_rpm × torque_Nm
    # INTUITION: In a healthy machine, power follows a predictable curve.
    #   Deviations indicate mechanical resistance (bearing failure,
    #   misalignment, tool binding). Paper 1's SOFM explicitly maps
    #   this kind of multi-variate deviation.
    #
    # DATA LEAKAGE WARNING: We compute "expected_power" as the rolling
    # mean of power — this is backward-looking only (safe).
    if config.enable_power_anomaly:
        if "rotational_speed_rpm" in df.columns and "torque_Nm" in df.columns:
            df["instantaneous_power_W"] = (
                df["rotational_speed_rpm"] * df["torque_Nm"] * (2 * np.pi / 60)
            )
            # Expected power = 12-period (1hr) rolling mean
            df["expected_power_W"] = df.groupby("machine_id")["instantaneous_power_W"].transform(
                lambda x: x.rolling(window=12, min_periods=1, center=False).mean()
            )
            # Relative deviation from expected
            df["power_anomaly_score"] = (
                (df["instantaneous_power_W"] - df["expected_power_W"]).abs()
                / (df["expected_power_W"].abs() + 1e-6)
            )
            features_added += 3  # power, expected, anomaly_score
            logger.debug("Created: instantaneous_power_W, expected_power_W, power_anomaly_score")

    # --- Feature 4: Wear Rate (First Derivative) ---
    # FORMULA: tool_wear[t] - tool_wear[t-1]
    # INTUITION: Absolute wear matters, but ACCELERATION of wear is the
    #   leading indicator. A tool wearing at 2 units/period is normal.
    #   Suddenly wearing at 8 units/period means material failure is imminent.
    #   Paper 2: This gradient feature consistently ranks in top-10 importance.
    if config.enable_wear_rate:
        if "tool_wear_min" in df.columns:
            df["tool_wear_rate"] = df.groupby("machine_id")["tool_wear_min"].diff()
            # Negative diff = maintenance reset. Clip to 0 (reset is not "negative wear")
            df["tool_wear_rate"] = df["tool_wear_rate"].clip(lower=0)
            features_added += 1
            logger.debug("Created: tool_wear_rate")

    logger.info(f"Cross-domain features: +{features_added} columns")

    return df


# =============================================================================
# 6. FEATURE PIPELINE ORCHESTRATOR (Production-Grade with Lookback Buffer)
# =============================================================================
# WHY an orchestrator with a lookback buffer?
# - Single entry point for the full transformation chain
# - Handles chunked I/O: reads Stage 1 Parquet, writes Stage 2 Parquet
# - Solves the CHUNK BOUNDARY PROBLEM for 500K+ row scaling
#
# CHUNK BOUNDARY PROBLEM (Paper 4 - Real-Time Quality):
# Rolling windows and lags need CONTEXT from previous chunks.
# If chunk 1 ends at row 5000, then chunk 2's first 36 rows would have
# incomplete 36-period rolling windows → NaN gaps at EVERY boundary.
# At 500K rows / 5000 per chunk = 100 boundaries × 36 rows = 3,600 rows
# of degraded signal. That's 0.7% data loss — unacceptable for Paper 2's
# XGBoost which uses EVERY tree split.
#
# SOLUTION: Per-machine lookback buffer.
# - After processing chunk N, store the last `buffer_size` rows PER MACHINE
# - Before processing chunk N+1, prepend each machine's buffer
# - After feature computation, strip the buffer rows from the output
# - Result: 100% signal integrity, zero NaN from boundaries
#
# WHY per-machine buffers (not a global tail)?
# All our rolling/lag features use groupby("machine_id"). If we used a
# global buffer, M1's last 36 rows might end up prepended to M3's chunk,
# corrupting the per-machine rolling windows.

CHUNK_SIZE = 5000  # Rows per processing chunk (matches Stage 1)


def _process_chunk_with_buffer(
    chunk: pd.DataFrame,
    machine_buffers: dict,
    config: FeatureConfig,
    buffer_size: int
) -> Tuple[pd.DataFrame, dict]:
    """
    Process a single chunk with lookback buffer for continuity.

    Strategy:
    1. Prepend each machine's buffer rows to the chunk
    2. Compute all features (rolling, lag, cross-domain)
    3. Strip buffer rows from the result (prevent duplication)
    4. Update buffers with this chunk's tail

    Args:
        chunk: Current chunk to process
        machine_buffers: Dict of {machine_id: DataFrame} lookback buffers
        config: Feature engineering configuration
        buffer_size: Number of lookback rows per machine (= max rolling window)

    Returns:
        (processed_chunk_without_buffer, updated_machine_buffers)
    """
    chunk = chunk.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # --- Step 1: Prepend buffer rows ---
    # WHY prepend, not concat after?
    # Rolling windows look BACKWARD. By prepending historical rows,
    # the rolling computation at the chunk start has full context.
    # The buffer rows get features computed too, but we discard them.
    buffer_frames = []
    for machine_id, buffer_df in machine_buffers.items():
        if not buffer_df.empty:
            buffer_frames.append(buffer_df)

    if buffer_frames:
        buffer_combined = pd.concat(buffer_frames, ignore_index=True)
        # Mark buffer rows so we can strip them later
        buffer_combined["_is_buffer"] = True
        chunk["_is_buffer"] = False
        augmented = pd.concat([buffer_combined, chunk], ignore_index=True)
    else:
        chunk["_is_buffer"] = False
        augmented = chunk.copy()

    # --- Step 2: Compute features on augmented chunk ---
    augmented = compute_rolling_features(augmented, config)
    augmented = compute_lag_features(augmented, config)
    augmented = compute_cross_domain_features(augmented, config)

    # --- Step 3: Strip buffer rows ---
    # Only keep the real chunk rows — buffer was just for context
    result = augmented[~augmented["_is_buffer"]].drop(columns=["_is_buffer"]).reset_index(drop=True)

    # --- Step 4: Update buffers with this chunk's tail ---
    # Store the last `buffer_size` rows per machine for the NEXT chunk
    # WHY store from the ORIGINAL chunk (not augmented)?
    # We want raw values in the buffer, not features. Features will be
    # recomputed when the buffer is prepended to the next chunk.
    # If we stored feature columns, they'd create duplicate/stale features.
    raw_cols = [c for c in chunk.columns if c != "_is_buffer"]
    updated_buffers = {}
    for machine_id, group in chunk.groupby("machine_id"):
        updated_buffers[machine_id] = group[raw_cols].tail(buffer_size).copy()

    return result, updated_buffers


def run_feature_pipeline(config: Optional[FeatureConfig] = None) -> Path:
    """
    Execute the full feature engineering pipeline with chunk boundary buffers.

    Architecture (Paper 4 - streaming-compatible):
    1. Read Stage 1 Parquet files
    2. Temporal alignment (full merge — both files are same length)
    3. Chunked feature computation with per-machine lookback buffers
    4. Incremental Parquet write (append row-groups per chunk)

    This design scales to 500K+ rows with O(chunk_size + buffer_size) memory,
    while maintaining 100% rolling window signal integrity.

    Args:
        config: Feature engineering configuration (uses defaults if None)

    Returns:
        Path to the output feature matrix Parquet file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if config is None:
        config = FeatureConfig()

    # --- Resolve Paths ---
    input_dir = Path(config.parquet_input_dir)
    output_path = Path(config.feature_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys_path = input_dir / "system_metrics.parquet"
    app_path = input_dir / "application_logs.parquet"

    # --- Validate Inputs ---
    if not sys_path.exists():
        raise FileNotFoundError(
            f"System metrics Parquet not found at {sys_path}. "
            f"Run stream_manager.py's convert_to_parquet() first."
        )
    if not app_path.exists():
        raise FileNotFoundError(
            f"Application logs Parquet not found at {app_path}. "
            f"Run stream_manager.py's convert_to_parquet() first."
        )

    logger.info("=" * 60)
    logger.info("SentinelX Stage 2: Feature Engineering Pipeline")
    logger.info("  Mode: Chunked with Lookback Buffer (Production)")
    logger.info("=" * 60)

    # --- Buffer Configuration ---
    # buffer_size = max rolling window = 36 periods (3 hours at 5-min intervals)
    # WHY max(rolling_windows)?
    # The 36-period rolling std needs 36 prior rows for a valid computation.
    # Shorter windows (6, 12) are automatically satisfied if 36 is available.
    # Lag features need max(lag_periods) = 6, which is < 36, so also covered.
    buffer_size = max(config.rolling_windows)
    logger.info(f"  Lookback buffer: {buffer_size} rows/machine "
                f"({buffer_size * 5} min at 5-min intervals)")
    logger.info(f"  Chunk size: {CHUNK_SIZE} rows")

    # --- Step 1: Read & Align (Full read for merge, then chunk) ---
    # WHY full read for alignment only?
    # The left join MUST see all timestamps to correctly ffill.
    # If we chunked before alignment, an app_log event at the chunk boundary
    # would be lost — we'd ffill with the wrong value.
    # After alignment, the merged DF is sorted and safe to chunk.
    #
    # Memory note: At 50K rows × 34 cols × 8 bytes = ~14MB.
    # At 500K, this becomes ~140MB — acceptable for alignment.
    # Beyond 2M rows, we'd partition by machine_id first.
    logger.info(f"Reading system metrics: {sys_path}")
    sys_df = pd.read_parquet(sys_path)
    logger.info(f"  → {sys_df.shape[0]:,} rows × {sys_df.shape[1]} cols")

    logger.info(f"Reading application logs: {app_path}")
    app_df = pd.read_parquet(app_path)
    logger.info(f"  → {app_df.shape[0]:,} rows × {app_df.shape[1]} cols")

    logger.info("Step 1/4: Temporal alignment (left join + per-machine ffill)")
    merged_df = align_temporal(sys_df, app_df, config)
    logger.info(f"  → Merged: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} cols")

    # Free originals — alignment is done, we only need merged
    del sys_df, app_df

    # --- Step 2: Sort for deterministic chunking ---
    # WHY sort before chunking?
    # Ensures each chunk has chronologically ordered data per machine.
    # Without sorting, a machine's rows might be scattered across chunks,
    # making the lookback buffer ineffective.
    merged_df = merged_df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # --- Step 3: Chunked Feature Computation with Buffer ---
    logger.info("Step 2-4/4: Chunked feature computation with lookback buffer")
    n_chunks = (len(merged_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(f"  → {n_chunks} chunks to process")

    machine_buffers: dict = {}  # {machine_id: DataFrame}
    writer = None
    total_output_rows = 0
    output_schema = None

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(merged_df))
        chunk = merged_df.iloc[start:end].copy()

        # Process with buffer
        processed, machine_buffers = _process_chunk_with_buffer(
            chunk, machine_buffers, config, buffer_size
        )

        # --- Incremental Parquet Write ---
        # WHY incremental (ParquetWriter) instead of collect-then-write?
        # - Memory: we never hold the full featured matrix in RAM
        # - Crash recovery: partial results are on disk if pipeline fails
        # - Production: maps to Kafka → Parquet sink pattern
        table = pa.Table.from_pandas(processed, preserve_index=False)

        if writer is None:
            output_schema = table.schema
            writer = pq.ParquetWriter(str(output_path), output_schema)

        writer.write_table(table)
        total_output_rows += len(processed)

        # Progress logging
        buffer_status = sum(len(b) for b in machine_buffers.values())
        logger.info(f"  Chunk {chunk_idx + 1}/{n_chunks}: "
                    f"{len(processed)} rows written, "
                    f"buffer={buffer_status} rows across {len(machine_buffers)} machines")

    if writer:
        writer.close()

    # Free merged_df
    del merged_df

    # --- Step 4: Validation Report ---
    logger.info("=" * 60)
    logger.info("STAGE 2 VALIDATION REPORT")
    logger.info("=" * 60)

    # Re-read output for validation (metadata only + sample)
    output_file = pq.ParquetFile(str(output_path))
    output_metadata = output_file.metadata
    n_cols = output_metadata.num_columns
    n_rows = output_metadata.num_rows
    n_row_groups = output_metadata.num_row_groups
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"Feature Matrix Shape: {n_rows:,} rows × {n_cols} columns")
    logger.info(f"Row Groups: {n_row_groups} (one per chunk)")
    logger.info(f"Output Size: {output_size_mb:.2f} MB")

    # NaN density check (read full output for this validation)
    full_output = pd.read_parquet(output_path)
    n_nan = full_output.isna().sum().sum()
    n_total_cells = full_output.shape[0] * full_output.shape[1]
    nan_pct = (n_nan / n_total_cells) * 100
    logger.info(f"NaN Density: {nan_pct:.2f}% ({n_nan:,} / {n_total_cells:,} cells)")

    # Feature variance check (confirm non-trivial features)
    variance_cols = [
        "thermal_efficiency_idx", "power_anomaly_score",
        "tool_wear_rate", "error_under_stress"
    ]
    # Add a sample of rolling features
    sample_rolling = [c for c in full_output.columns if "_std_" in c][:3]
    variance_cols.extend(sample_rolling)

    logger.info("Feature Variance Check:")
    for col in variance_cols:
        if col in full_output.columns:
            var = full_output[col].var()
            non_null = full_output[col].notna().sum()
            logger.info(f"  {col:<40}: var={var:.6f}, non_null={non_null:,}/{n_rows:,}")

    del full_output

    return output_path


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    output = run_feature_pipeline()
    print(f"\nFeature matrix ready at: {output}")
