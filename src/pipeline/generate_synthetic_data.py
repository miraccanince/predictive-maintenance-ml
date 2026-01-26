"""
SentinelX - Unified Synthetic Dataset Generator (v2)
Generates 50K rows of fully synthetic hardware + software correlated data.
Independent of ai4i2020 - creates its own realistic sensor distributions.
Outputs two separate files: system_metrics.csv and application_logs.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# === CONFIGURATION ===
N_ROWS = 50000
N_MACHINES = 5
INTERVAL_MINUTES = 5
START_TIME = datetime(2024, 1, 1, 0, 0, 0)
FAILURE_RATE = 0.034  # ~3.4% base failure rate

print("=" * 60)
print("SentinelX Synthetic Data Generator v2")
print(f"Target: {N_ROWS:,} rows | {N_MACHINES} machines | {INTERVAL_MINUTES}-min intervals")
print("=" * 60)

# === 1. TEMPORAL & IDENTITY ===
timestamps = [START_TIME + timedelta(minutes=INTERVAL_MINUTES * i) for i in range(N_ROWS)]
machine_ids = [f"M{(i % N_MACHINES) + 1}" for i in range(N_ROWS)]
product_types = np.random.choice(['L', 'M', 'H'], N_ROWS, p=[0.6, 0.3, 0.1])

# === 2. HARDWARE SENSOR DATA (realistic distributions) ===

# Tool wear: cumulative per machine, resets after maintenance cycles
tool_wear = np.zeros(N_ROWS)
wear_per_machine = {f"M{i+1}": 0 for i in range(N_MACHINES)}
wear_max = 250  # max before forced maintenance

for i in range(N_ROWS):
    mid = machine_ids[i]
    # Wear increases 1-5 units per step, varies by product type
    wear_increment = np.random.uniform(0.5, 3.0)
    if product_types[i] == 'H':
        wear_increment *= 1.5
    wear_per_machine[mid] += wear_increment
    # Reset after reaching max (maintenance event)
    if wear_per_machine[mid] > wear_max:
        wear_per_machine[mid] = 0
    tool_wear[i] = wear_per_machine[mid]

# Air temperature: ~295-305 K with daily cycles and gradual seasonal drift
time_hours = np.arange(N_ROWS) * INTERVAL_MINUTES / 60
daily_cycle = 2 * np.sin(2 * np.pi * time_hours / 24)  # ±2K daily
seasonal_drift = 3 * np.sin(2 * np.pi * time_hours / (24 * 180))  # ±3K over 6 months
air_temp = 300 + daily_cycle + seasonal_drift + np.random.normal(0, 0.5, N_ROWS)

# Process temperature: air_temp + 10K offset + machine load
process_temp = air_temp + 10 + np.random.normal(0, 0.8, N_ROWS)

# Rotational speed: bimodal (low power ~1500rpm, high power ~2500rpm)
speed_mode = np.random.choice([0, 1], N_ROWS, p=[0.7, 0.3])
rotational_speed = np.where(speed_mode == 0,
                            np.random.normal(1500, 100, N_ROWS),
                            np.random.normal(2500, 150, N_ROWS))
rotational_speed = np.clip(rotational_speed, 1000, 3000)

# Torque: inversely related to speed (power = torque × speed)
torque_base = 9550 * 7.5 / rotational_speed  # P=7.5kW constant power approximation
torque = torque_base + np.random.normal(0, 5, N_ROWS)
torque = np.clip(torque, 5, 80)

# === 3. FAILURE MECHANICS (physics-based) ===
# Failures are caused by specific conditions, not random
tool_wear_norm = tool_wear / wear_max
speed_norm = (rotational_speed - 1000) / 2000
torque_norm = (torque - 5) / 75

# Tool Wear Failure (TWF): high wear
twf_prob = np.where(tool_wear > 200, 0.15, np.where(tool_wear > 180, 0.05, 0.001))
twf = (np.random.random(N_ROWS) < twf_prob).astype(int)

# Heat Dissipation Failure (HDF): temp difference too low
temp_diff = process_temp - air_temp
hdf_prob = np.where(temp_diff < 8.6, 0.12, 0.001)
hdf = (np.random.random(N_ROWS) < hdf_prob).astype(int)

# Power Failure (PWF): power outside bounds
power = torque * rotational_speed * 2 * np.pi / 60
pwf_prob = np.where((power < 3500) | (power > 9000), 0.08, 0.001)
pwf = (np.random.random(N_ROWS) < pwf_prob).astype(int)

# Overstrain Failure (OSF): torque × wear too high
overstrain = torque * tool_wear
osf_threshold = {'L': 11000, 'M': 12000, 'H': 13000}
osf_prob = np.array([0.1 if overstrain[i] > osf_threshold[product_types[i]] else 0.001
                     for i in range(N_ROWS)])
osf = (np.random.random(N_ROWS) < osf_prob).astype(int)

# Random Failure (RNF): rare random events
rnf = (np.random.random(N_ROWS) < 0.002).astype(int)

# Machine failure: any failure mode triggers it
machine_failure = np.clip(twf + hdf + pwf + osf + rnf, 0, 1)

print(f"\nFailure rate: {machine_failure.mean()*100:.2f}% ({machine_failure.sum()} events)")

# === 4. PRE-FAILURE WINDOWS (degradation ramp) ===
pre_failure_window = np.zeros(N_ROWS)
window_size = 18  # 90 minutes before failure (18 × 5min)
failure_indices = np.where(machine_failure == 1)[0]
for idx in failure_indices:
    start = max(0, idx - window_size)
    for w in range(start, idx):
        progress = (w - start) / (idx - start)
        pre_failure_window[w] = max(pre_failure_window[w], progress)
    pre_failure_window[idx] = 1.0

# === 5. IIoT EDGE METRICS (correlated with hardware) ===

# Vibration: speed × torque interaction + failure spikes
vibration = 15 + 35 * speed_norm * torque_norm + np.random.normal(0, 3, N_ROWS)
vibration += pre_failure_window * np.random.uniform(10, 30, N_ROWS)
vibration = np.clip(vibration, 0, 85)

# Pressure: process dynamics
pressure = 90 + 25 * torque_norm + 10 * (air_temp - 298) / 6
pressure += np.random.normal(0, 2, N_ROWS) + pre_failure_window * 10
pressure = np.clip(pressure, 70, 145)

# Network Latency: baseline + wear drift + failure spikes
net_latency = 8 + 20 * tool_wear_norm ** 1.3 + np.random.exponential(2, N_ROWS)
net_latency += pre_failure_window * np.random.uniform(25, 70, N_ROWS)
net_latency = np.clip(net_latency, 1, 130)

# Edge Processing Time
edge_proc = 3 + 6 * tool_wear_norm + 4 * (air_temp - 295) / 10
edge_proc += np.random.exponential(1.5, N_ROWS) + pre_failure_window * np.random.uniform(5, 18, N_ROWS)
edge_proc = np.clip(edge_proc, 1, 55)

# Fuzzy PID Output (control quality)
fuzzy_pid = 0.88 - 0.18 * tool_wear_norm - 0.25 * pre_failure_window
fuzzy_pid += np.random.normal(0, 0.04, N_ROWS)
fuzzy_pid = np.clip(fuzzy_pid, 0.1, 1.0)

# === 6. SOFTWARE APPLICATION METRICS ===

# API Response Latency
api_latency = 40 + 90 * tool_wear_norm ** 1.5 + np.random.lognormal(0, 0.3, N_ROWS) * 5
api_latency += pre_failure_window * np.random.uniform(120, 350, N_ROWS)
api_latency = np.clip(api_latency, 8, 900)

# Error Rate (%)
error_rate = 0.4 + 2.5 * tool_wear_norm + np.random.exponential(0.3, N_ROWS)
error_rate += pre_failure_window ** 2 * np.random.uniform(10, 25, N_ROWS)
error_rate = np.clip(error_rate, 0, 40)

# HTTP 5xx Count
request_volume = np.random.poisson(160, N_ROWS) + 40
http_5xx = np.clip(np.round(error_rate / 100 * request_volume + np.random.poisson(1, N_ROWS)), 0, 120).astype(int)

# Request Throughput (req/s)
throughput = 190 - 45 * pre_failure_window - 25 * tool_wear_norm + np.random.normal(0, 12, N_ROWS)
throughput = np.clip(throughput, 40, 240)

# Queue Depth
queue_depth = 4 + 12 * tool_wear_norm + np.random.exponential(2, N_ROWS)
queue_depth += pre_failure_window ** 1.5 * np.random.uniform(35, 90, N_ROWS)
queue_depth = np.clip(queue_depth, 0, 160).astype(int)

# CPU Utilization (%)
cpu_util = 32 + 22 * (air_temp - 295) / 10 + 18 * tool_wear_norm
cpu_util += np.random.normal(0, 5, N_ROWS) + pre_failure_window * np.random.uniform(15, 38, N_ROWS)
cpu_util = np.clip(cpu_util, 5, 99)

# Memory Utilization (%)
mem_util = 38 + 28 * tool_wear_norm ** 0.8 + np.random.normal(0, 3, N_ROWS)
mem_util += pre_failure_window * np.random.uniform(12, 28, N_ROWS)
mem_util = np.clip(mem_util, 12, 98)

# Packet Loss Rate (%)
pkt_loss = 0.08 + 1.8 * tool_wear_norm + np.random.exponential(0.2, N_ROWS)
pkt_loss += pre_failure_window * np.random.uniform(2.5, 9, N_ROWS)
pkt_loss = np.clip(pkt_loss, 0, 18)

# Disk I/O Wait (ms)
disk_io = 1.5 + 9 * (queue_depth / 160) + np.random.exponential(1, N_ROWS)
disk_io += pre_failure_window * 6
disk_io = np.clip(disk_io, 0, 35)

# === 7. SOFTWARE-ONLY ANOMALIES (2.5% of normal rows) ===
sw_only_mask = (machine_failure == 0) & (pre_failure_window == 0) & (np.random.random(N_ROWS) < 0.025)
sw_anomaly_count = sw_only_mask.sum()
api_latency[sw_only_mask] *= np.random.uniform(2.5, 5, sw_anomaly_count)
error_rate[sw_only_mask] += np.random.uniform(8, 20, sw_anomaly_count)
cpu_util[sw_only_mask] += np.random.uniform(15, 30, sw_anomaly_count)
pkt_loss[sw_only_mask] += np.random.uniform(3, 8, sw_anomaly_count)

print(f"Software-only anomalies: {sw_anomaly_count} ({sw_anomaly_count/N_ROWS*100:.1f}%)")

# === 8. TEMPORAL DRIFT (sensor aging + seasonal) ===
time_progress = np.linspace(0, 1, N_ROWS)
drift_factor = 1 + 0.08 * time_progress  # 8% drift over 6 months
net_latency *= drift_factor
api_latency *= drift_factor * (1 + 0.015 * np.random.randn(N_ROWS))
edge_proc *= (1 + 0.04 * time_progress)

# === 9. COMPOSITE LABELS ===
# Maintenance Status
conditions = [
    machine_failure == 1,
    pre_failure_window > 0.6,
    pre_failure_window > 0,
]
choices = ['Failure', 'Warning', 'Degrading']
maintenance_status = np.select(conditions, choices, default='Normal')

# Anomaly Score (continuous 0-1)
anomaly_components = np.column_stack([
    pre_failure_window,
    tool_wear_norm * 0.35,
    (error_rate / 40) * 0.3,
    (net_latency / 130) * 0.2,
    sw_only_mask.astype(float) * 0.85
])
anomaly_score = np.clip(anomaly_components.max(axis=1), 0, 1)

# Binary anomaly
is_anomaly = (anomaly_score > 0.55).astype(int)

# === 10. BUILD DATAFRAMES ===

# --- System Metrics (Hardware + Edge) ---
system_metrics = pd.DataFrame({
    'timestamp': timestamps,
    'machine_id': machine_ids,
    'product_type': product_types,
    'air_temperature_K': np.round(air_temp, 2),
    'process_temperature_K': np.round(process_temp, 2),
    'rotational_speed_rpm': np.round(rotational_speed, 1),
    'torque_Nm': np.round(torque, 2),
    'tool_wear_min': np.round(tool_wear, 1),
    'vibration_mm_s': np.round(vibration, 3),
    'pressure_psi': np.round(pressure, 2),
    'network_latency_ms': np.round(net_latency, 3),
    'edge_processing_time_ms': np.round(edge_proc, 3),
    'fuzzy_pid_output': np.round(fuzzy_pid, 4),
    # Labels
    'machine_failure': machine_failure,
    'failure_TWF': twf,
    'failure_HDF': hdf,
    'failure_PWF': pwf,
    'failure_OSF': osf,
    'failure_RNF': rnf,
    'maintenance_status': maintenance_status,
})

# --- Application Logs (Software Metrics) ---
application_logs = pd.DataFrame({
    'timestamp': timestamps,
    'machine_id': machine_ids,
    'api_response_latency_ms': np.round(api_latency, 2),
    'error_rate_pct': np.round(error_rate, 4),
    'http_5xx_count': http_5xx,
    'request_throughput_rps': np.round(throughput, 2),
    'queue_depth': queue_depth,
    'cpu_utilization_pct': np.round(cpu_util, 2),
    'memory_utilization_pct': np.round(mem_util, 2),
    'packet_loss_pct': np.round(pkt_loss, 4),
    'disk_io_wait_ms': np.round(disk_io, 3),
    # Labels (duplicated for independent analysis capability)
    'anomaly_score': np.round(anomaly_score, 4),
    'is_anomaly': is_anomaly,
    'maintenance_status': maintenance_status,
})

# === 11. SAVE ===
system_metrics.to_csv("logs/system_metrics.csv", index=False)
application_logs.to_csv("logs/application_logs.csv", index=False)

print(f"\n{'='*60}")
print("OUTPUT FILES:")
print(f"{'='*60}")
print(f"\n1. logs/system_metrics.csv")
print(f"   Shape: {system_metrics.shape}")
print(f"   Columns: {list(system_metrics.columns)}")
print(f"\n2. logs/application_logs.csv")
print(f"   Shape: {application_logs.shape}")

print(f"\n{'='*60}")
print("DATASET STATISTICS:")
print(f"{'='*60}")
print(f"\nTime range: {timestamps[0]} → {timestamps[-1]}")
days = (timestamps[-1] - timestamps[0]).days
print(f"Duration: {days} days (~{days/30:.1f} months)")
print(f"Machines: {N_MACHINES}")

print(f"\nMaintenance Status Distribution:")
for status in ['Normal', 'Degrading', 'Warning', 'Failure']:
    count = (maintenance_status == status).sum()
    print(f"  {status:>10}: {count:>5} ({count/N_ROWS*100:.1f}%)")

print(f"\nAnomaly: {is_anomaly.sum()} / {N_ROWS} ({is_anomaly.mean()*100:.1f}%)")
print(f"Machine Failures: {machine_failure.sum()} ({machine_failure.mean()*100:.2f}%)")

print(f"\nFailure Type Breakdown:")
print(f"  TWF (Tool Wear):     {twf.sum()}")
print(f"  HDF (Heat):          {hdf.sum()}")
print(f"  PWF (Power):         {pwf.sum()}")
print(f"  OSF (Overstrain):    {osf.sum()}")
print(f"  RNF (Random):        {rnf.sum()}")

print(f"\nCorrelation with Machine Failure:")
corr_data = {
    'network_latency_ms': net_latency,
    'api_response_latency_ms': api_latency,
    'error_rate_pct': error_rate,
    'cpu_utilization_pct': cpu_util,
    'queue_depth': queue_depth.astype(float),
    'packet_loss_pct': pkt_loss,
    'vibration_mm_s': vibration,
}
for name, arr in corr_data.items():
    r = np.corrcoef(arr, machine_failure)[0, 1]
    print(f"  {name:<28}: {r:.3f}")

# Remove old unified file if exists
import os
old_file = "logs/sentinelx_unified_dataset.csv"
if os.path.exists(old_file):
    os.remove(old_file)
    print(f"\nRemoved old file: {old_file}")

print(f"\n{'='*60}")
print("GENERATION COMPLETE")
print(f"{'='*60}")
