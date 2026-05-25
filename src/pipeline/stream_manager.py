"""
SentinelX - Stage 1: Data Ingestion & Quality Gate
===================================================
This module handles streaming ingestion of dual data sources
(system_metrics + application_logs) with real-time quality
assessment and drift monitoring.

Architecture References:
- Paper 1 (SOFM+SVM): Multi-modal inputs require normalized, aligned data
- Paper 2 (XGBoost+SMOTE): Imbalanced data → flag anomalies, don't discard
- Paper 3 (54% FP): Dirty data causes most false positives
- Paper 4 (Real-Time Quality): In-stream quality gates, not batch post-processing
"""

from dataclasses import dataclass, field
from typing import Generator, Dict, Tuple, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SentinelX.StreamManager")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# WHY a dataclass?
# - Immutable configuration that travels with the pipeline
# - Type-safe, IDE-friendly, serializable for experiment tracking
# - When we move to AWS, this becomes a SageMaker ProcessingInput config

@dataclass
class StreamConfig:
    """Pipeline configuration for data ingestion."""

    # --- Paths ---
    system_metrics_path: str = "data/raw/system_metrics.csv"
    application_logs_path: str = "data/raw/application_logs.csv"
    parquet_output_dir: str = "data/parquet"

    # --- Chunking ---
    # WHY 5000? At 5-min intervals, 5000 rows = ~17 days per chunk.
    # Large enough for statistical validity in drift detection,
    # small enough for constant RAM (~2MB per chunk in memory).
    chunk_size: int = 5000

    # --- Quality Gate Bounds ---
    # WHY physics-based bounds?
    # Paper 3: 54% of false positives come from data issues.
    # These bounds represent physical impossibilities, not statistical outliers.
    # A reading outside these bounds is DEFINITELY a sensor error, not an anomaly.
    sensor_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "air_temperature_K": (250.0, 350.0),        # Physical: below 250K or above 350K is instrument failure
        "process_temperature_K": (260.0, 380.0),
        "rotational_speed_rpm": (0.0, 5000.0),      # Motor physical limits
        "torque_Nm": (0.0, 150.0),
        "tool_wear_min": (0.0, 300.0),
        "vibration_mm_s": (0.0, 100.0),
        "pressure_psi": (0.0, 200.0),
        "network_latency_ms": (0.0, 5000.0),        # >5s = connection timeout, not latency
        "edge_processing_time_ms": (0.0, 300.0),
        "api_response_latency_ms": (0.0, 10000.0),  # >10s = timeout
        "cpu_utilization_pct": (0.0, 100.0),
        "memory_utilization_pct": (0.0, 100.0),
        "error_rate_pct": (0.0, 100.0),
        "packet_loss_pct": (0.0, 100.0),
    })

    # --- Drift Detection ---
    # WHY PSI > 0.2?
    # Industry standard threshold (Paper 4):
    #   PSI < 0.1  → No significant drift
    #   0.1 - 0.2  → Moderate drift, monitor closely
    #   PSI > 0.2  → Significant drift, retrain signal
    psi_threshold: float = 0.2
    psi_bins: int = 10
    drift_columns: List[str] = field(default_factory=lambda: [
        "air_temperature_K", "torque_Nm", "tool_wear_min",
        "network_latency_ms", "api_response_latency_ms",
        "error_rate_pct", "cpu_utilization_pct"
    ])

    # --- Quality Gate ---
    max_missing_pct: float = 5.0   # Flag chunk if >5% missing in any column
    latency_spike_multiplier: float = 3.0  # Flag if latency > 3x rolling median


# =============================================================================
# 2. DATA STREAM READER (Generator-based)
# =============================================================================
# WHY generators?
# - Memory: 500K rows × 20 cols × 8 bytes = ~80MB if loaded at once
#   With generators: constant ~2MB regardless of file size
# - Backpressure: downstream (quality gate, model) controls the pace
# - Production: maps directly to Kafka/Kinesis consumer pattern
#
# WHY chunk-based, not row-by-row?
# - Row-by-row has too much Python overhead (GIL, function call cost)
# - Chunk = vectorized NumPy/Pandas ops within each chunk = fast
# - Chunk size is the tuning knob between latency and throughput

def stream_csv(filepath: str, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    """
    Generator that yields DataFrame chunks from a CSV file.
    Memory usage: O(chunk_size), not O(file_size).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data source not found: {filepath}")

    reader = pd.read_csv(
        filepath,
        chunksize=chunk_size,
        parse_dates=["timestamp"],
        low_memory=True  # Prevents dtype inference on full file
    )

    for chunk_idx, chunk in enumerate(reader):
        logger.debug(f"Yielding chunk {chunk_idx} from {filepath.name} "
                     f"({len(chunk)} rows, {chunk['timestamp'].iloc[0]} → {chunk['timestamp'].iloc[-1]})")
        yield chunk


def stream_parquet(filepath: str, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    """
    Generator for Parquet files using row-group based reading.

    WHY Parquet row-groups?
    - Parquet stores data in row-groups (typically 128MB each)
    - Each row-group can be read independently → true streaming
    - Column pruning: only read columns we need (no wasted I/O)
    - Predicate pushdown: skip row-groups that don't match filters
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required for Parquet support: pip install pyarrow")

    filepath = Path(filepath)
    parquet_file = pq.ParquetFile(filepath)

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk = batch.to_pandas()
        if "timestamp" in chunk.columns:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        yield chunk


# =============================================================================
# 3. QUALITY GATE
# =============================================================================
# WHY a quality gate BEFORE the model?
# Paper 3: "54% of alerts are false positives" — most caused by data issues.
# Paper 4: Quality assessment must be real-time, in-stream.
#
# Design principle: FLAG, don't DROP.
# Paper 2 rationale: With 5.2% failure rate, extreme values ARE real failures.
# If we drop them, we lose the very events we're trying to detect.
# Instead, we add a quality_flag column and let the model weigh it.

@dataclass
class QualityReport:
    """Report generated by the quality gate for each chunk."""
    chunk_idx: int
    total_rows: int
    missing_counts: Dict[str, int] = field(default_factory=dict)
    out_of_bounds_counts: Dict[str, int] = field(default_factory=dict)
    latency_spikes: int = 0
    quality_score: float = 1.0  # 0.0 = terrible, 1.0 = perfect
    passed: bool = True

    def summary(self) -> str:
        issues = []
        if self.missing_counts:
            total_missing = sum(self.missing_counts.values())
            issues.append(f"missing={total_missing}")
        if self.out_of_bounds_counts:
            total_oob = sum(self.out_of_bounds_counts.values())
            issues.append(f"out_of_bounds={total_oob}")
        if self.latency_spikes:
            issues.append(f"latency_spikes={self.latency_spikes}")
        status = "PASS" if self.passed else "FLAG"
        return f"[{status}] Chunk {self.chunk_idx}: score={self.quality_score:.3f} | {', '.join(issues) if issues else 'clean'}"


class QualityGate:
    """
    Real-time data quality assessment gate.

    Performs three checks per chunk:
    1. Missing values (sensor dropout, network gaps)
    2. Out-of-bounds (physical impossibilities → sensor malfunction)
    3. Latency spikes (sudden jumps → potential system issue vs real anomaly)

    Returns the chunk WITH quality flags — never drops rows.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self._chunk_counter = 0
        # Rolling median for latency spike detection
        self._latency_median: Optional[float] = None
        self._latency_ema_alpha = 0.3  # Exponential moving average weight

    def assess(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Assess a single chunk and return (flagged_chunk, report).

        WHY return both?
        - The chunk gets quality_flag column added (for model awareness)
        - The report goes to monitoring/alerting (for ops team awareness)
        """
        report = QualityReport(
            chunk_idx=self._chunk_counter,
            total_rows=len(chunk)
        )
        self._chunk_counter += 1

        # Initialize quality flag (0 = clean, bitwise flags for issues)
        chunk = chunk.copy()
        chunk["_quality_flag"] = 0

        # --- Check 1: Missing Values ---
        missing = chunk.isnull().sum()
        missing_cols = missing[missing > 0].to_dict()
        if missing_cols:
            report.missing_counts = missing_cols
            # Flag rows with any missing value
            missing_mask = chunk.isnull().any(axis=1)
            chunk.loc[missing_mask, "_quality_flag"] |= 1  # bit 0: missing data
            # Check if any column exceeds threshold
            for col, count in missing_cols.items():
                if (count / len(chunk) * 100) > self.config.max_missing_pct:
                    report.passed = False

        # --- Check 2: Out-of-Bounds (Physics Violations) ---
        oob_counts = {}
        for col, (low, high) in self.config.sensor_bounds.items():
            if col not in chunk.columns:
                continue
            oob_mask = (chunk[col] < low) | (chunk[col] > high)
            oob_count = oob_mask.sum()
            if oob_count > 0:
                oob_counts[col] = oob_count
                chunk.loc[oob_mask, "_quality_flag"] |= 2  # bit 1: out of bounds
        report.out_of_bounds_counts = oob_counts

        # --- Check 3: Latency Spikes ---
        # WHY exponential moving median, not fixed threshold?
        # Paper 4: "Static thresholds cause alert fatigue under concept drift"
        # The system adapts to gradual increases but catches sudden jumps.
        latency_cols = [c for c in chunk.columns if "latency" in c.lower()]
        spike_count = 0
        for col in latency_cols:
            if col not in chunk.columns:
                continue
            current_median = chunk[col].median()
            if self._latency_median is not None:
                threshold = self._latency_median * self.config.latency_spike_multiplier
                spike_mask = chunk[col] > threshold
                spike_count += spike_mask.sum()
                chunk.loc[spike_mask, "_quality_flag"] |= 4  # bit 2: latency spike
            # Update rolling median (EMA)
            if self._latency_median is None:
                self._latency_median = current_median
            else:
                self._latency_median = (self._latency_ema_alpha * current_median +
                                        (1 - self._latency_ema_alpha) * self._latency_median)
        report.latency_spikes = spike_count

        # --- Compute Quality Score ---
        total_issues = (
            sum(report.missing_counts.values()) +
            sum(report.out_of_bounds_counts.values()) +
            report.latency_spikes
        )
        max_possible_issues = len(chunk) * len(chunk.columns)
        report.quality_score = 1.0 - min(total_issues / max_possible_issues, 1.0)

        return chunk, report


# =============================================================================
# 4. DRIFT MONITOR
# =============================================================================
# WHY monitor drift in-stream?
# Paper 4: Models trained on historical data degrade when data distribution shifts.
# Our dataset has 8% temporal drift baked in — the monitor should catch this.
#
# WHY PSI (Population Stability Index)?
# - Lightweight: only stores bin counts, not raw values
# - Interpretable: directly maps to "how different is this from training"
# - Stateless: each chunk compared to a fixed reference, no accumulation
#
# Formula: PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

class DriftMonitor:
    """
    Lightweight distribution shift detector using Population Stability Index.

    The first chunk becomes the reference distribution (baseline).
    Subsequent chunks are compared against it.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self._reference_bins: Dict[str, np.ndarray] = {}
        self._reference_edges: Dict[str, np.ndarray] = {}
        self._is_baseline_set = False
        self._drift_history: List[Dict[str, float]] = []

    def _compute_bins(self, values: np.ndarray, edges: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Bin values into histogram, returning proportions and edges."""
        values = values[~np.isnan(values)]
        if edges is None:
            counts, edges = np.histogram(values, bins=self.config.psi_bins)
        else:
            counts, _ = np.histogram(values, bins=edges)
        # Add small epsilon to avoid log(0) — standard PSI practice
        proportions = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        return proportions, edges

    def set_baseline(self, chunk: pd.DataFrame) -> None:
        """
        Set the reference distribution from the first chunk.

        WHY first chunk as baseline?
        - In production, this would be your training data distribution
        - For streaming, the first window represents "expected" behavior
        - Alternative: load a pre-computed baseline from model training
        """
        for col in self.config.drift_columns:
            if col not in chunk.columns:
                continue
            values = chunk[col].dropna().values
            proportions, edges = self._compute_bins(values)
            self._reference_bins[col] = proportions
            self._reference_edges[col] = edges

        self._is_baseline_set = True
        logger.info(f"Drift baseline set from {len(chunk)} rows across {len(self._reference_bins)} columns")

    def check_drift(self, chunk: pd.DataFrame) -> Dict[str, float]:
        """
        Compute PSI for each monitored column in the current chunk.

        Returns dict of {column: psi_value}.
        PSI > 0.2 = significant drift → retrain signal.
        """
        if not self._is_baseline_set:
            self.set_baseline(chunk)
            return {col: 0.0 for col in self.config.drift_columns if col in chunk.columns}

        psi_scores = {}
        for col in self.config.drift_columns:
            if col not in chunk.columns or col not in self._reference_bins:
                continue

            values = chunk[col].dropna().values
            actual_proportions, _ = self._compute_bins(values, self._reference_edges[col])
            expected_proportions = self._reference_bins[col]

            # PSI formula
            psi = np.sum(
                (actual_proportions - expected_proportions) *
                np.log(actual_proportions / expected_proportions)
            )
            psi_scores[col] = round(float(psi), 6)

        # Log alerts for drifting columns
        drifting = {k: v for k, v in psi_scores.items() if v > self.config.psi_threshold}
        if drifting:
            logger.warning(f"DRIFT DETECTED: {drifting}")

        self._drift_history.append(psi_scores)
        return psi_scores


# =============================================================================
# 5. STREAM PIPELINE (Orchestrator)
# =============================================================================
# WHY an orchestrator?
# - Single entry point: consumer code doesn't know about CSV vs Parquet
# - Composable: add new stages (feature engineering, model inference) later
# - Testable: each component tested in isolation, pipeline tested end-to-end

class StreamPipeline:
    """
    Orchestrates the full ingestion pipeline:
    Reader → Quality Gate → Drift Monitor → Yield clean, annotated chunks.

    Usage:
        config = StreamConfig()
        pipeline = StreamPipeline(config)
        for system_chunk, app_chunk, reports in pipeline.stream():
            # Both chunks are quality-assessed, drift-monitored
            # Ready for feature engineering / model inference
            pass
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.system_quality_gate = QualityGate(config)
        self.app_quality_gate = QualityGate(config)
        self.drift_monitor = DriftMonitor(config)

    def _get_reader(self, filepath: str) -> Generator[pd.DataFrame, None, None]:
        """Select appropriate reader based on file extension."""
        if filepath.endswith(".parquet"):
            return stream_parquet(filepath, self.config.chunk_size)
        return stream_csv(filepath, self.config.chunk_size)

    def stream(self) -> Generator[
        Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None
    ]:
        """
        Main streaming generator. Yields:
            (system_chunk, app_chunk, metadata_dict)

        WHY yield both streams together?
        - Temporal alignment: both chunks cover the same time window
        - The correlation engine (Stage 2) needs synchronized inputs
        - Paper 1: Multi-modal fusion requires aligned timestamps
        """
        system_reader = self._get_reader(self.config.system_metrics_path)
        app_reader = self._get_reader(self.config.application_logs_path)

        for sys_chunk, app_chunk in zip(system_reader, app_reader):
            # --- Quality Assessment ---
            sys_chunk, sys_report = self.system_quality_gate.assess(sys_chunk)
            app_chunk, app_report = self.app_quality_gate.assess(app_chunk)

            # --- Drift Monitoring (on system metrics — primary signal) ---
            drift_scores = self.drift_monitor.check_drift(sys_chunk)

            # --- Metadata for downstream consumers ---
            metadata = {
                "system_quality": sys_report,
                "app_quality": app_report,
                "drift_scores": drift_scores,
                "time_range": (
                    sys_chunk["timestamp"].iloc[0],
                    sys_chunk["timestamp"].iloc[-1]
                ),
                "chunk_size": len(sys_chunk),
            }

            logger.info(sys_report.summary())
            logger.info(app_report.summary())

            yield sys_chunk, app_chunk, metadata


# =============================================================================
# 6. PARQUET CONVERSION UTILITY
# =============================================================================
# WHY convert to Parquet?
# - CSV: 10MB for 50K rows → Parquet: ~3MB (70% compression)
# - Column pruning: read only needed columns (critical for 33-col dataset)
# - Type preservation: no more "is this column a float or string?" issues
# - Row-group reading: native chunking without line-counting overhead
#
# This is a one-time migration step. After conversion, the pipeline
# reads Parquet by default for all subsequent runs.

def convert_to_parquet(config: StreamConfig) -> Tuple[str, str]:
    """
    Convert CSV sources to Parquet format.
    Returns paths to the new Parquet files.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required: pip install pyarrow")

    output_dir = Path(config.parquet_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for csv_path in [config.system_metrics_path, config.application_logs_path]:
        csv_path = Path(csv_path)
        parquet_path = output_dir / csv_path.with_suffix(".parquet").name

        # Read CSV in chunks and write as Parquet row-groups
        # WHY chunked write? Same reason as chunked read — RAM efficiency
        writer = None
        for chunk in stream_csv(str(csv_path), config.chunk_size):
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(str(parquet_path), table.schema)
            writer.write_table(table)
        if writer:
            writer.close()

        csv_size = csv_path.stat().st_size / 1024 / 1024
        pq_size = parquet_path.stat().st_size / 1024 / 1024
        ratio = (1 - pq_size / csv_size) * 100
        logger.info(f"Converted {csv_path.name}: {csv_size:.1f}MB → {pq_size:.1f}MB ({ratio:.0f}% reduction)")
        converted.append(str(parquet_path))

    return tuple(converted)


# =============================================================================
# MAIN — Demo execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SentinelX Stream Manager — Stage 1 Demo")
    print("=" * 60)

    config = StreamConfig()
    pipeline = StreamPipeline(config)

    chunk_count = 0
    total_rows = 0
    all_drift = []

    for sys_chunk, app_chunk, metadata in pipeline.stream():
        chunk_count += 1
        total_rows += metadata["chunk_size"]

        # Collect drift data
        all_drift.append(metadata["drift_scores"])

        # Stop after a few chunks for demo
        if chunk_count >= 3:
            print(f"\n[Demo: processed {chunk_count} chunks, {total_rows:,} rows]")
            print(f"[Stopping early — remove the break to process all {config.chunk_size * 10} rows]")
            break

    # --- Drift Summary ---
    if all_drift:
        print(f"\nDrift scores (last chunk):")
        for col, score in all_drift[-1].items():
            status = "OK" if score < 0.1 else ("MONITOR" if score < 0.2 else "DRIFT!")
            print(f"  {col:<30}: PSI={score:.4f} [{status}]")

    print(f"\n{'='*60}")
    print("Stage 1 complete. Next: Stage 2 (Feature Engineering & Correlation)")
    print("=" * 60)
