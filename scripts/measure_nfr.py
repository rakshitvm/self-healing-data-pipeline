"""Lightweight, reproducible NFR measurement script.

Not a benchmark framework — just real wall-clock timing of:
1. A baseline (uninstrumented) read of a large CSV file, vs. the full
   Tier 1 CLI `repair` command on the same (healthy) file.
2. End-to-end CLI repair latency against an existing Tier 1 fixture.

Every number printed is a real, freshly-measured value from this
machine, this run — nothing here is invented or hardcoded. Run with:

    PYTHONPATH=src python scripts/measure_nfr.py
"""

from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTHY_FILE = Path("/tmp/nfr_healthy_100mb.csv")
REPAIR_FILE = Path("/tmp/nfr_repair_fixture.csv")
ITERATIONS = 5


def _ensure_healthy_fixture(target_rows_bytes: int = 100 * 1024 * 1024) -> None:
    if HEALTHY_FILE.exists() and HEALTHY_FILE.stat().st_size >= target_rows_bytes:
        return
    print(f"Generating ~100MB healthy CSV fixture at {HEALTHY_FILE} ...")
    with HEALTHY_FILE.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "name", "value", "category"])
        row_id = 0
        written = 0
        while written < target_rows_bytes:
            row_id += 1
            row = [row_id, f"user_{row_id}", f"{row_id * 3.14:.2f}", f"category_{row_id % 50}"]
            writer.writerow(row)
            written = fh.tell()
    print(f"Done: {HEALTHY_FILE.stat().st_size / (1024 * 1024):.1f} MB")


def _time_baseline_read(iterations: int) -> list[float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with HEALTHY_FILE.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            row_count = sum(1 for _ in reader)
        times.append(time.perf_counter() - start)
    print(f"  (baseline read {row_count} rows per iteration)")
    return times


def _time_cli_repair(file_path: Path, iterations: int) -> list[float]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = subprocess.run(  # noqa: S603 - fixed local argv, not user input
            [sys.executable, "-m", "self_healing_pipeline.interfaces.cli.main", "repair", str(file_path)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if result.returncode not in (0, 1):
            print(f"  WARNING: unexpected exit code {result.returncode}")
            print(result.stderr[-2000:])
    return times


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _report(label: str, times: list[float]) -> None:
    print(f"{label}:")
    print(f"  iterations: {len(times)}")
    print(f"  values (s): {[round(t, 4) for t in times]}")
    print(f"  mean:  {statistics.mean(times):.4f}s")
    if len(times) > 1:
        print(f"  stdev: {statistics.stdev(times):.4f}s")
    print(f"  p95:   {_percentile(times, 95):.4f}s")
    print()


def main() -> None:
    print("=== A. Healthy-path overhead (100MB file) ===\n")
    _ensure_healthy_fixture()

    print("Baseline (uninstrumented csv.reader row count):")
    baseline_times = _time_baseline_read(ITERATIONS)
    _report("baseline_read", baseline_times)

    print("Instrumented (full CLI `repair` command, healthy path):")
    cli_healthy_times = _time_cli_repair(HEALTHY_FILE, ITERATIONS)
    _report("cli_healthy_path", cli_healthy_times)

    baseline_mean = statistics.mean(baseline_times)
    cli_mean = statistics.mean(cli_healthy_times)
    overhead_pct = ((cli_mean - baseline_mean) / baseline_mean) * 100
    print(f"Overhead: {overhead_pct:.1f}% (cli_mean={cli_mean:.4f}s vs baseline_mean={baseline_mean:.4f}s)\n")

    print("=== B. Repair latency (existing small fixture, real Groq LLM) ===\n")
    REPAIR_FILE.write_text("id;name;value\n1;alpha;10\n2;beta;20\n3;gamma;30\n", encoding="utf-8")
    repair_times = _time_cli_repair(REPAIR_FILE, ITERATIONS)
    _report("cli_repair_wrong_delimiter", repair_times)
    print("NOTE: this includes real network latency to the configured LLM provider (Groq).")


if __name__ == "__main__":
    main()
