import os
import csv
from collections import Counter

LOG_DIR = "logs"

def file_summary(filepath):
    size = os.path.getsize(filepath)
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    s = size
    while s >= 1024 and unit_idx < len(units) - 1:
        s /= 1024
        unit_idx += 1
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Size: {s:.2f} {units[unit_idx]}")

    # Count lines (sample-based estimate for large files)
    line_count = 0
    sample_lines = []
    with open(filepath, "r", errors="replace") as f:
        for i, line in enumerate(f):
            line_count += 1
            if i < 10:
                sample_lines.append(line.rstrip())
            if line_count >= 100000 and size > 100_000_000:
                # Estimate for large files
                avg_line_len = sum(len(l) for l in sample_lines) / len(sample_lines)
                estimated_lines = int(size / avg_line_len)
                print(f"Lines (estimated): ~{estimated_lines:,}")
                break
        else:
            print(f"Lines: {line_count:,}")

    print(f"\nFirst 5 lines:")
    for line in sample_lines[:5]:
        print(f"  {line[:200]}")

    return sample_lines


def analyze_csv(filepath):
    file_summary(filepath)
    with open(filepath, "r", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            print(f"\nColumns ({len(header)}):")
            for col in header:
                print(f"  - {col}")
            # Read a few rows to understand data types
            rows = []
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                rows.append(row)
            if rows:
                print(f"\nSample data (first 3 rows):")
                for row in rows[:3]:
                    print(f"  {row[:10]}")  # first 10 cols max


def analyze_log(filepath):
    sample = file_summary(filepath)
    # Try to detect log format
    if sample:
        print(f"\nFormat analysis:")
        # Check if it looks like Apache/Nginx access log
        if " - - [" in sample[0] or "HTTP/" in sample[0]:
            print("  Detected: Apache/Nginx access log format")
        # Check common fields
        parts = sample[0].split()
        print(f"  Fields per line (first line): {len(parts)}")


print("=" * 60)
print("LOG FILE ANALYSIS")
print("=" * 60)

for fname in sorted(os.listdir(LOG_DIR)):
    filepath = os.path.join(LOG_DIR, fname)
    if not os.path.isfile(filepath):
        continue
    # Skip duplicates (copy files)
    if "copy" in fname.lower():
        print(f"\n[Skipping '{fname}' - duplicate of original]")
        continue

    if fname.endswith(".csv"):
        analyze_csv(filepath)
    elif fname.endswith(".log"):
        analyze_log(filepath)

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print("=" * 60)
