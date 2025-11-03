#!/usr/bin/env python3
"""EntropyStats: GoldbachX telemetry aggregation for entropy/instability metrics."""

import argparse
import json
import glob
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import random
from typing import TypedDict

# Optional imports with feature gating
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Type aliases
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
Event = Dict[str, JSONValue]
Aggregate = Dict[str, Dict[str, Dict[str, float]]]
Summary = Dict[str, Dict[str, Union[float, int, List[Dict[str, float]]]]]]

class TelemetryStats(TypedDict):
    files_scanned: int
    events_ok: int
    events_bad: int
    metrics_seen: int
    spikes: int
    droughts: int
    time_ms: int

def _log_telemetry(stats: TelemetryStats) -> None:
    """Output telemetry as JSONL to stdout."""
    print(json.dumps(stats))

def _warn_empty(message: str) -> None:
    """Standardized warning for empty inputs."""
    warnings.warn(f"EntropyStats: {message}", RuntimeWarning, stacklevel=2)

def _parse_timestamp(ts: Union[str, float]) -> Optional[datetime]:
    """Parse timestamp from event data."""
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    elif isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except (ValueError, OSError):
            return None
    return None

def _is_in_window(event: Event, window_min: int, now: Optional[datetime] = None) -> bool:
    """Check if event is within the time window."""
    if now is None:
        now = datetime.now()

    ts = event.get("timestamp")
    if ts is None:
        return False

    event_time = _parse_timestamp(ts)
    if event_time is None:
        return False

    return event_time >= (now - timedelta(minutes=window_min))

def _validate_event(event: Event) -> bool:
    """Basic validation of event structure."""
    return (
        isinstance(event, dict) and
        "module" in event and
        isinstance(event["module"], str) and
        "timestamp" in event and
        any(metric in event for metric in [
            "entropy", "frontier_size", "rule_diversity",
            "backtrack_rate", "density"
        ])
    )

def _calculate_trend(values: List[float]) -> float:
    """Calculate trend slope using least squares if numpy available, else simple diff."""
    if not values or len(values) < 2:
        return 0.0

    if NUMPY_AVAILABLE:
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    else:
        return (values[-1] - values[0]) / len(values)

def ingest(dir_telemetry: str = "./proof-data/telemetry", window_min: int = 15) -> List[Event]:
    """Read and filter telemetry events from JSONL files.

    Args:
        dir_telemetry: Directory containing JSONL telemetry files
        window_min: Time window in minutes to consider events

    Returns:
        List of valid events within the time window
    """
    start_time = time.time()
    stats: TelemetryStats = {
        "files_scanned": 0,
        "events_ok": 0,
        "events_bad": 0,
        "metrics_seen": 0,
        "spikes": 0,
        "droughts": 0,
        "time_ms": 0
    }

    if not os.path.isdir(dir_telemetry):
        _warn_empty(f"Telemetry directory not found: {dir_telemetry}")
        return []

    pattern = os.path.join(dir_telemetry, "*.jsonl")
    files = glob.glob(pattern)
    if not files:
        _warn_empty(f"No JSONL files found in {dir_telemetry}")
        return []

    events: List[Event] = []
    metrics_seen = set()
    now = datetime.now()

    for filepath in files:
        stats["files_scanned"] += 1
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if not _validate_event(event):
                            stats["events_bad"] += 1
                            continue

                        if not _is_in_window(event, window_min, now):
                            continue

                        # Track unique metrics seen
                        for metric in [
                            "entropy", "frontier_size", "rule_diversity",
                            "backtrack_rate", "density"
                        ]:
                            if metric in event:
                                metrics_seen.add(metric)

                        events.append(event)
                        stats["events_ok"] += 1
                    except json.JSONDecodeError:
                        stats["events_bad"] += 1
        except IOError:
            continue

    stats["metrics_seen"] = len(metrics_seen)
    stats["time_ms"] = int((time.time() - start_time) * 1000)
    _log_telemetry(stats)

    return events

def aggregate(events: List[Event]) -> Dict[str, Any]:
    """Aggregate metrics from events into statistical summaries.

    Args:
        events: List of validated telemetry events

    Returns:
        Dictionary with "by_module", "by_metric", and "kpis" keys
    """
    if not events:
        _warn_empty("No events provided to aggregate")
        return {"by_module": {}, "by_metric": {}, "kpis": {}}

    # Initialize data structures
    by_module: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_metric: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    kpis: Dict[str, int] = {
        "total_events": len(events),
        "unique_modules": 0,
        "unique_metrics": 0
    }

    # Collect all values
    for event in events:
        module = event["module"]
        for metric in [
            "entropy", "frontier_size", "rule_diversity",
            "backtrack_rate", "density"
        ]:
            if metric in event and isinstance(event[metric], (int, float)):
                value = float(event[metric])
                by_module[module][metric].append(value)
                by_metric[metric][module].append(value)

    # Calculate statistics
    agg_by_module: Dict[str, Dict[str, Dict[str, float]]] = {}
    for module, metrics in by_module.items():
        agg_by_module[module] = {}
        for metric, values in metrics.items():
            if not values:
                continue

            mean = sum(values) / len(values)
            squared_diffs = [(x - mean) ** 2 for x in values]
            std = (sum(squared_diffs) / len(values)) ** 0.5
            sorted_values = sorted(values)
            p95 = sorted_values[int(0.95 * len(sorted_values))]

            # Spike/drought detection
            spikes = 0
            droughts = 0
            if std > 0:  # Avoid division by zero
                for x in values:
                    z = (x - mean) / std
                    if z > 2:
                        spikes += 1
                    elif z < -2:
                        droughts += 1

            agg_by_module[module][metric] = {
                "mean": mean,
                "std": std,
                "p95": p95,
                "spike_count": spikes,
                "drought_count": droughts,
                "trend": _calculate_trend(values)
            }

    # Aggregate by metric across all modules
    agg_by_metric: Dict[str, Dict[str, float]] = {}
    for metric, modules in by_metric.items():
        all_values = [v for module_values in modules.values() for v in module_values]
        if not all_values:
            continue

        mean = sum(all_values) / len(all_values)
        squared_diffs = [(x - mean) ** 2 for x in all_values]
        std = (sum(squared_diffs) / len(all_values)) ** 0.5
        sorted_values = sorted(all_values)
        p95 = sorted_values[int(0.95 * len(sorted_values))]

        spikes = 0
        droughts = 0
        if std > 0:
            for x in all_values:
                z = (x - mean) / std
                if z > 2:
                    spikes += 1
                elif z < -2:
                    droughts += 1

        agg_by_metric[metric] = {
            "mean": mean,
            "std": std,
            "p95": p95,
            "spike_count": spikes,
            "drought_count": droughts,
            "trend": _calculate_trend(all_values)
        }

    kpis.update({
        "unique_modules": len(by_module),
        "unique_metrics": len(by_metric),
        "total_spikes": sum(m["spike_count"] for m in agg_by_metric.values()),
        "total_droughts": sum(m["drought_count"] for m in agg_by_metric.values())
    })

    return {
        "by_module": agg_by_module,
        "by_metric": agg_by_metric,
        "kpis": kpis
    }

def summarize(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Create compact summary statistics from aggregation.

    Args:
        agg: Aggregation dictionary from aggregate()

    Returns:
        Compact summary with spikes, droughts, and key statistics
    """
    if not agg or not agg.get("by_module"):
        _warn_empty("No aggregation data to summarize")
        return {}

    summary: Dict[str, Any] = {
        "overview": {
            "modules": len(agg["by_module"]),
            "metrics": len(agg["by_metric"]),
            "total_events": agg["kpis"]["total_events"],
            "total_spikes": agg["kpis"]["total_spikes"],
            "total_droughts": agg["kpis"]["total_droughts"]
        },
        "modules": [],
        "metrics": []
    }

    # Top modules by spike count
    module_spikes = []
    for module, metrics in agg["by_module"].items():
        total_spikes = sum(m["spike_count"] for m in metrics.values())
        module_spikes.append({
            "module": module,
            "spikes": total_spikes,
            "metrics": len(metrics)
        })
    summary["modules"] = sorted(
        module_spikes,
        key=lambda x: x["spikes"],
        reverse=True
    )[:10]  # Top 10 only

    # Metric trends
    metric_trends = []
    for metric, stats in agg["by_metric"].items():
        metric_trends.append({
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "trend": stats["trend"],
            "spikes": stats["spike_count"],
            "droughts": stats["drought_count"]
        })
    summary["metrics"] = sorted(
        metric_trends,
        key=lambda x: abs(x["trend"]),
        reverse=True
    )

    return summary

def export(summary: Dict[str, Any], path: str) -> None:
    """Export summary to JSON file.

    Args:
        summary: Summary dictionary from summarize()
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

def metadata() -> Dict[str, Any]:
    """Return metadata about this component."""
    return {
        "version": "1.0.0",
        "created": "2023-11-15",
        "dependencies": {
            "numpy": NUMPY_AVAILABLE,
            "pandas": PANDAS_AVAILABLE,
            "streamlit": STREAMLIT_AVAILABLE
        },
        "metrics_supported": [
            "entropy", "frontier_size", "rule_diversity",
            "backtrack_rate", "density"
        ]
    }

def discover() -> Dict[str, str]:
    """Return basic discovery information."""
    return {"component": "EntropyStats"}

def _generate_test_events(seed: int = 42) -> List[Event]:
    """Generate synthetic test events for self-testing."""
    random.seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)

    modules = ["module_a", "module_b", "module_c"]
    metrics = ["entropy", "frontier_size", "rule_diversity"]
    events = []
    now = datetime.now()

    for i in range(100):
        module = random.choice(modules)
        event: Event = {
            "module": module,
            "timestamp": (now - timedelta(minutes=i)).isoformat()
        }

        for metric in metrics:
            # Base value with some spikes and droughts
            base = random.uniform(0, 100)
            if random.random() < 0.05:  # 5% spike
                value = base * 3
            elif random.random() < 0.05:  # 5% drought
                value = base / 3
            else:
                value = base + random.uniform(-10, 10)

            event[metric] = value

        events.append(event)

    return events

def self_test() -> bool:
    """Run self-tests and return success status."""
    print("Running EntropyStats self-test...")
    test_events = _generate_test_events()

    # Test ingest filtering
    filtered = ingest(dir_telemetry=":memory:", window_min=15)
    assert len(filtered) == 0, "Empty ingest should return empty list"

    # Test aggregation
    agg = aggregate(test_events)
    assert agg["kpis"]["total_events"] == 100, "Should process all test events"
    assert len(agg["by_module"]) == 3, "Should have 3 test modules"
    assert len(agg["by_metric"]) == 3, "Should have 3 test metrics"

    # Test spike/drought detection
    total_spikes = agg["kpis"]["total_spikes"]
    total_droughts = agg["kpis"]["total_droughts"]
    print(f"Detected {total_spikes} spikes and {total_droughts} droughts in test data")
    assert total_spikes > 0 and total_droughts > 0, "Should detect some spikes/droughts"

    # Test summary
    summary = summarize(agg)
    assert summary["overview"]["total_spikes"] == total_spikes
    assert len(summary["modules"]) == 3, "Should summarize all 3 modules"
    assert len(summary["metrics"]) == 3, "Should summarize all 3 metrics"

    # Test determinism
    agg2 = aggregate(_generate_test_events(42))
    assert agg == agg2, "Results should be deterministic with same seed"

    print("All self-tests passed!")
    return True

def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="EntropyStats: GoldbachX telemetry analyzer")
    parser.add_argument(
        "--mode",
        choices=["cli", "self-test"],
        default="cli",
        help="Execution mode"
    )
    parser.add_argument(
        "--telemetry",
        default="./proof-data/telemetry",
        help="Telemetry directory path"
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=30,
        help="Time window in minutes"
    )
    parser.add_argument(
        "--export",
        default="./proof-data/aggregates/entropy_stats.json",
        help="Output file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic output"
    )

    args = parser.parse_args()
    random.seed(args.seed)
    if NUMPY_AVAILABLE:
        np.random.seed(args.seed)

    if args.mode == "self-test":
        success = self_test()
        sys.exit(0 if success else 1)

    # Normal CLI operation
    events = ingest(args.telemetry, args.window_min)
    agg = aggregate(events)
    summary = summarize(agg)
    export(summary, args.export)

    # Print readable output if pandas not available
    if not PANDAS_AVAILABLE:
        print("\nEntropyStats Summary:")
        print(f"Modules: {summary['overview']['modules']}")
        print(f"Metrics: {summary['overview']['metrics']}")
        print(f"Events: {summary['overview']['total_events']}")
        print(f"Spikes: {summary['overview']['total_spikes']}")
        print(f"Droughts: {summary['overview']['total_droughts']}")

        print("\nTop Modules by Spikes:")
        for module in summary["modules"]:
            print(f"  {module['module']}: {module['spikes']} spikes")

        print("\nMetric Trends:")
        for metric in summary["metrics"]:
            trend_dir = "↑" if metric["trend"] > 0 else "↓"
            print(f"  {metric['metric']}: {metric['trend']:.2f} {trend_dir}")

if __name__ == "__main__":
    main()
