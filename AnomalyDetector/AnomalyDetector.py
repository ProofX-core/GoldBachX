#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GoldbachX Entropy Analytics Anomaly Detector."""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Type Aliases
JSONL = str
Timestamp = str  # ISO 8601 format
ModuleName = str
MetricName = str
AnomalyType = Literal[
    "entropy_spike", "entropy_drought", "failure_burst", "density_collapse", "trend_drift"
]

class TelemetryEvent(TypedDict):
    ts: Timestamp
    module: ModuleName
    metric: MetricName
    value: float
    tags: Dict[str, str]

class Anomaly(TypedDict):
    ts: Timestamp
    module: ModuleName
    metric: MetricName
    anomaly_type: AnomalyType
    value: float
    threshold: float
    z_score: Optional[float]
    window_min: int

class DetectionResult(TypedDict):
    anomalies: List[Anomaly]
    stats: Dict[str, int]
    config: Dict[str, Any]

@dataclass
class DetectionConfig:
    method: str = "zscore"
    alpha: float = 3.0
    window_min: int = 30
    failure_threshold: float = 5.0  # failures per minute
    density_threshold: float = 0.5  # 50% drop
    trend_window: int = 15  # minutes for trend comparison

class AnomalyDetector:
    """Core anomaly detection engine for GoldbachX telemetry streams."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            self._set_random_seed(seed)

    def _set_random_seed(self, seed: int) -> None:
        """Set random seeds for deterministic behavior."""
        import random
        random.seed(seed)
        if NUMPY_AVAILABLE:
            np.random.seed(seed)

    def ingest(
        self,
        dir_telemetry: str = "./proof-data/telemetry",
        window_min: int = 30,
    ) -> List[TelemetryEvent]:
        """Read and filter telemetry events from JSONL files."""
        events = []
        path = Path(dir_telemetry)

        if not path.exists():
            warnings.warn(f"Telemetry directory not found: {dir_telemetry}")
            return []

        cutoff = datetime.now() - timedelta(minutes=window_min)

        for file in path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if not self._validate_event(event):
                            continue
                        event_ts = datetime.fromisoformat(event["ts"])
                        if event_ts >= cutoff:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue

        # Sort by module, metric, then timestamp
        events.sort(key=lambda x: (x["module"], x["metric"], x["ts"]))
        return events

    def _validate_event(self, event: Dict[str, Any]) -> bool:
        """Validate telemetry event structure."""
        required = {"ts", "module", "metric", "value"}
        return all(field in event for field in required)

    def detect(
        self,
        events: List[TelemetryEvent],
        *,
        method: str = "zscore",
        alpha: float = 3.0,
        window_min: int = 30,
    ) -> DetectionResult:
        """Detect anomalies in telemetry events."""
        config = DetectionConfig(
            method=method,
            alpha=alpha,
            window_min=window_min,
        )

        if not events:
            return {
                "anomalies": [],
                "stats": {
                    "events_processed": 0,
                    "anomalies_found": 0,
                },
                "config": config.__dict__,
            }

        # Group events by module and metric
        grouped = defaultdict(list)
        for event in events:
            key = (event["module"], event["metric"])
            grouped[key].append(event)

        anomalies = []
        stats = defaultdict(int)
        stats["events_processed"] = len(events)

        for (module, metric), metric_events in grouped.items():
            values = [e["value"] for e in metric_events]
            timestamps = [e["ts"] for e in metric_events]

            # Detect different types of anomalies
            anomalies.extend(
                self._detect_entropy_anomalies(
                    module, metric, values, timestamps, config
                )
            )
            anomalies.extend(
                self._detect_failure_bursts(
                    module, metric, values, timestamps, config
                )
            )
            anomalies.extend(
                self._detect_density_collapses(
                    module, metric, values, timestamps, config
                )
            )

            if NUMPY_AVAILABLE:
                anomalies.extend(
                    self._detect_trend_drifts(
                        module, metric, values, timestamps, config
                    )
                )

        stats["anomalies_found"] = len(anomalies)
        return {
            "anomalies": anomalies,
            "stats": dict(stats),
            "config": config.__dict__,
        }

    def _detect_entropy_anomalies(
        self,
        module: str,
        metric: str,
        values: List[float],
        timestamps: List[str],
        config: DetectionConfig,
    ) -> List[Anomaly]:
        """Detect entropy spikes and droughts using z-scores."""
        if len(values) < 2:
            return []

        mean_val = mean(values)
        std_val = stdev(values) if len(values) > 1 else 0.0

        anomalies = []
        for val, ts in zip(values, timestamps):
            if std_val == 0:
                continue

            z_score = (val - mean_val) / std_val
            if z_score > config.alpha:
                anomalies.append({
                    "ts": ts,
                    "module": module,
                    "metric": metric,
                    "anomaly_type": "entropy_spike",
                    "value": val,
                    "threshold": config.alpha,
                    "z_score": z_score,
                    "window_min": config.window_min,
                })
            elif z_score < -config.alpha:
                anomalies.append({
                    "ts": ts,
                    "module": module,
                    "metric": metric,
                    "anomaly_type": "entropy_drought",
                    "value": val,
                    "threshold": -config.alpha,
                    "z_score": z_score,
                    "window_min": config.window_min,
                })

        return anomalies

    def _detect_failure_bursts(
        self,
        module: str,
        metric: str,
        values: List[float],
        timestamps: List[str],
        config: DetectionConfig,
    ) -> List[Anomaly]:
        """Detect failure rate bursts."""
        if "failure" not in metric.lower():
            return []

        # Count failures per minute
        time_buckets = defaultdict(int)
        for ts in timestamps:
            dt = datetime.fromisoformat(ts)
            bucket = dt.replace(second=0, microsecond=0)
            time_buckets[bucket] += 1

        anomalies = []
        for bucket, count in time_buckets.items():
            if count > config.failure_threshold:
                anomalies.append({
                    "ts": bucket.isoformat(),
                    "module": module,
                    "metric": metric,
                    "anomaly_type": "failure_burst",
                    "value": count,
                    "threshold": config.failure_threshold,
                    "z_score": None,
                    "window_min": config.window_min,
                })

        return anomalies

    def _detect_density_collapses(
        self,
        module: str,
        metric: str,
        values: List[float],
        timestamps: List[str],
        config: DetectionConfig,
    ) -> List[Anomaly]:
        """Detect significant drops in metric density."""
        if "density" not in metric.lower() and "pair_count" not in metric.lower():
            return []

        if len(values) < 5:  # Need enough data for moving average
            return []

        window_size = max(3, len(values) // 5)
        moving_avg = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            moving_avg.append(mean(window))

        if not moving_avg:
            return []

        max_avg = max(moving_avg)
        anomalies = []
        for i, avg in enumerate(moving_avg):
            if avg < max_avg * (1 - config.density_threshold):
                ts = timestamps[i + window_size - 1]
                anomalies.append({
                    "ts": ts,
                    "module": module,
                    "metric": metric,
                    "anomaly_type": "density_collapse",
                    "value": avg,
                    "threshold": max_avg * (1 - config.density_threshold),
                    "z_score": None,
                    "window_min": config.window_min,
                })

        return anomalies

    def _detect_trend_drifts(
        self,
        module: str,
        metric: str,
        values: List[float],
        timestamps: List[str],
        config: DetectionConfig,
    ) -> List[Anomaly]:
        """Detect significant trend changes using linear regression."""
        if not NUMPY_AVAILABLE or len(values) < 10:
            return []

        # Split into two time windows
        split_idx = len(values) // 2
        window1 = values[:split_idx]
        window2 = values[split_idx:]

        # Calculate slopes
        x = np.arange(len(window1))
        slope1 = np.polyfit(x, window1, 1)[0]
        x = np.arange(len(window2))
        slope2 = np.polyfit(x, window2, 1)[0]

        # Significant change in slope direction
        if slope1 * slope2 < 0 and abs(slope1 - slope2) > config.alpha * 0.1:
            return [{
                "ts": timestamps[split_idx],
                "module": module,
                "metric": metric,
                "anomaly_type": "trend_drift",
                "value": slope2,
                "threshold": slope1,
                "z_score": None,
                "window_min": config.window_min,
            }]
        return []

    def summarize(self, findings: DetectionResult) -> Dict[str, Any]:
        """Generate summary statistics from detection results."""
        if not findings["anomalies"]:
            return {
                "total_anomalies": 0,
                "by_module": {},
                "by_metric": {},
                "by_type": {},
            }

        by_module = defaultdict(int)
        by_metric = defaultdict(int)
        by_type = defaultdict(int)

        for anomaly in findings["anomalies"]:
            by_module[anomaly["module"]] += 1
            by_metric[anomaly["metric"]] += 1
            by_type[anomaly["anomaly_type"]] += 1

        return {
            "total_anomalies": len(findings["anomalies"]),
            "by_module": dict(by_module),
            "by_metric": dict(by_metric),
            "by_type": dict(by_type),
            "events_processed": findings["stats"].get("events_processed", 0),
        }

    @staticmethod
    def export(obj: Dict[str, Any], path: str) -> None:
        """Export results to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def metadata() -> Dict[str, Any]:
        """Return component metadata."""
        return {
            "component": "AnomalyDetector",
            "version": "1.0.0",
            "dependencies": {
                "numpy": NUMPY_AVAILABLE,
                "streamlit": STREAMLIT_AVAILABLE,
            },
        }

    @staticmethod
    def discover() -> Dict[str, str]:
        """Service discovery information."""
        return {"component": "AnomalyDetector"}

def run_cli(args) -> None:
    """Run anomaly detection from command line."""
    start_time = time.time()
    detector = AnomalyDetector(seed=args.seed)

    print(f"Scanning telemetry from {args.telemetry}...", file=sys.stderr)
    events = detector.ingest(args.telemetry, args.window_min)

    if not events:
        print("No valid telemetry events found.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(events)} events...", file=sys.stderr)
    results = detector.detect(
        events,
        method=args.method,
        alpha=args.alpha,
        window_min=args.window_min,
    )
    summary = detector.summarize(results)

    if args.export:
        detector.export(results, args.export)
        print(f"Results exported to {args.export}", file=sys.stderr)

    # Print summary to stdout as JSONL
    print(json.dumps({
        "files_scanned": len(list(Path(args.telemetry).glob("*.jsonl"))),
        "events_ok": len(events),
        "events_bad": 0,  # TODO: Track parsing errors
        "anomalies_found": len(results["anomalies"]),
        "time_ms": int((time.time() - start_time) * 1000),
    }))

    if not results["anomalies"]:
        print("No anomalies detected.", file=sys.stderr)
    else:
        print(f"Detected {len(results['anomalies'])} anomalies:", file=sys.stderr)
        print(json.dumps(summary, indent=2), file=sys.stderr)

def run_ui(args) -> None:
    """Run Streamlit UI if available."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Falling back to CLI mode.", file=sys.stderr)
        run_cli(args)
        return

    import streamlit as st

    detector = AnomalyDetector(seed=args.seed)
    st.set_page_config(page_title="GoldbachX Anomaly Detector", layout="wide")

    # Sidebar controls
    st.sidebar.title("Configuration")
    telemetry_dir = st.sidebar.text_input(
        "Telemetry Directory",
        value=args.telemetry,
    )
    window_min = st.sidebar.slider(
        "Time Window (minutes)",
        min_value=1,
        max_value=120,
        value=args.window_min,
    )
    method = st.sidebar.selectbox(
        "Detection Method",
        ["zscore", "iqr"],
        index=0 if args.method == "zscore" else 1,
    )
    alpha = st.sidebar.slider(
        "Alpha Threshold",
        min_value=1.0,
        max_value=5.0,
        value=args.alpha,
        step=0.1,
    )

    # Main content
    st.title("GoldbachX Telemetry Anomaly Detection")

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Analyzing telemetry..."):
            events = detector.ingest(telemetry_dir, window_min)
            results = detector.detect(
                events,
                method=method,
                alpha=alpha,
                window_min=window_min,
            )
            summary = detector.summarize(results)

            # Display KPIs
            col1, col2, col3 = st.columns(3)
            col1.metric("Events Processed", summary["events_processed"])
            col2.metric("Anomalies Detected", summary["total_anomalies"])
            col3.metric("Analysis Window", f"{window_min} minutes")

            # Display summary tables
            st.subheader("Anomaly Breakdown")
            st.write("By Module:")
            st.dataframe(pd.DataFrame.from_dict(
                summary["by_module"], orient="index", columns=["Count"]
            ))

            st.write("By Metric:")
            st.dataframe(pd.DataFrame.from_dict(
                summary["by_metric"], orient="index", columns=["Count"]
            ))

            st.write("By Anomaly Type:")
            st.dataframe(pd.DataFrame.from_dict(
                summary["by_type"], orient="index", columns=["Count"]
            ))

            # Detailed anomalies table
            if results["anomalies"]:
                st.subheader("Detailed Anomalies")
                df = pd.DataFrame(results["anomalies"])
                st.dataframe(df)

                # Export button
                if st.button("Export Results to JSON"):
                    export_path = os.path.join(os.getcwd(), "anomalies.json")
                    detector.export(results, export_path)
                    st.success(f"Results exported to {export_path}")
            else:
                st.success("No anomalies detected in the selected time window.")

def self_test() -> bool:
    """Run self-tests with synthetic data."""
    print("Running self-tests...", file=sys.stderr)

    # Create synthetic data with known anomalies
    detector = AnomalyDetector(seed=42)
    test_events = []
    base_time = datetime.now()

    # Normal baseline
    for i in range(30):
        test_events.append({
            "ts": (base_time - timedelta(minutes=30 - i)).isoformat(),
            "module": "test",
            "metric": "entropy",
            "value": 10.0 + (i % 3),
            "tags": {},
        })

    # Add a spike
    test_events.append({
        "ts": (base_time - timedelta(minutes=2)).isoformat(),
        "module": "test",
        "metric": "entropy",
        "value": 25.0,
        "tags": {},
    })

    # Add a drought
    test_events.append({
        "ts": (base_time - timedelta(minutes=1)).isoformat(),
        "module": "test",
        "metric": "entropy",
        "value": 2.0,
        "tags": {},
    })

    # Add failure bursts
    for i in range(10):
        test_events.append({
            "ts": (base_time - timedelta(minutes=5, seconds=i)).isoformat(),
            "module": "test",
            "metric": "failure_count",
            "value": 1.0,
            "tags": {},
        })

    # Run detection
    results = detector.detect(test_events, alpha=2.5)
    summary = detector.summarize(results)

    # Verify expected anomalies
    expected = {
        "entropy_spike": 1,
        "entropy_drought": 1,
        "failure_burst": 1,
    }

    passed = True
    for anomaly_type, count in expected.items():
        if summary["by_type"].get(anomaly_type, 0) < count:
            print(f"Test failed: Expected at least {count} {anomaly_type}", file=sys.stderr)
            passed = False

    if passed:
        print("All self-tests passed.", file=sys.stderr)
    return passed

def main() -> None:
    """Entry point for CLI and UI modes."""
    parser = argparse.ArgumentParser(description="GoldbachX Anomaly Detector")
    parser.add_argument(
        "--mode",
        choices=["cli", "ui"],
        default="cli",
        help="Execution mode (cli or ui)",
    )
    parser.add_argument(
        "--telemetry",
        default="./proof-data/telemetry",
        help="Path to telemetry directory",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=30,
        help="Time window in minutes to analyze",
    )
    parser.add_argument(
        "--method",
        choices=["zscore", "iqr"],
        default="zscore",
        help="Detection method",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Threshold for anomaly detection",
    )
    parser.add_argument(
        "--export",
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic behavior",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-tests and exit",
    )

    args = parser.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.mode == "ui":
        run_ui(args)
    else:
        run_cli(args)

if __name__ == "__main__":
    main()
