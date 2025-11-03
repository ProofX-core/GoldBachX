#!/usr/bin/env python3.11
"""
GoldbachHeatmapPanel - Live and historical heatmaps of Goldbach activity.
Operates as both Streamlit UI and CLI snapshotter.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import datetime as dt

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Constants
DEFAULT_WINDOW_MIN = 15
ALERT_ENTROPY_THRESH = 0.85
ALERT_FAILURE_THRESH = 0.15
COLORBLIND_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

class TelemetryLogger:
    """Handles logging of operational metrics to stdout in JSONL format."""

    @staticmethod
    def log(event: Dict[str, Any]) -> None:
        """Log a telemetry event to stdout."""
        event['timestamp'] = dt.datetime.utcnow().isoformat() + 'Z'
        print(json.dumps(event), file=sys.stdout)

class DataIngester:
    """Handles data loading and preprocessing from proof-data directories."""

    def __init__(self):
        self.last_scan_time = None

    def ingest(self, dir_results: str = "./proof-data",
               dir_telemetry: str = "./proof-data/telemetry",
               window_min: int = DEFAULT_WINDOW_MIN) -> List[Dict]:
        """
        Load and filter events from JSON/JSONL files.
        Returns chronological list of events within time window.
        """
        start_time = time.time()
        files_scanned = 0
        lines_ok = 0
        lines_bad = 0
        events = []

        cutoff_time = dt.datetime.utcnow() - dt.timedelta(minutes=window_min)

        # Scan both directories
        for data_dir in [dir_results, dir_telemetry]:
            if not os.path.exists(data_dir):
                continue

            for filepath in Path(data_dir).rglob('*'):
                if filepath.suffix.lower() not in ('.json', '.jsonl'):
                    continue

                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            try:
                                event = json.loads(line.strip())
                                if self._is_valid_event(event):
                                    event_time = dt.datetime.fromisoformat(event['timestamp'].replace('Z', ''))
                                    if event_time > cutoff_time:
                                        events.append(event)
                                    lines_ok += 1
                                else:
                                    lines_bad += 1
                            except json.JSONDecodeError:
                                lines_bad += 1
                    files_scanned += 1
                except Exception as e:
                    lines_bad += 1

        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])

        # Log telemetry
        TelemetryLogger.log({
            "component": "GoldbachHeatmapPanel",
            "event": "data_ingest",
            "files_scanned": files_scanned,
            "lines_ok": lines_ok,
            "lines_bad": lines_bad,
            "refresh_ms": int((time.time() - start_time) * 1000)
        })

        return events

    def _is_valid_event(self, event: Dict) -> bool:
        """Validate event structure."""
        required_fields = {'timestamp', 'module', 'metric', 'value'}
        return all(field in event for field in required_fields)

class HeatmapGenerator:
    """Generates heatmap matrices and KPIs from event data."""

    @staticmethod
    def compute_matrices(events: List[Dict]) -> Dict[str, Any]:
        """
        Compute heatmap matrices for pair counts, entropy, and failures.
        Returns dict with matrices and metadata.
        """
        if not events:
            return {
                "pair_count": [],
                "entropy": [],
                "failures": [],
                "modules": [],
                "time_buckets": []
            }

        # Get unique modules and create time buckets
        modules = sorted(list({e['module'] for e in events}))
        time_buckets = HeatmapGenerator._create_time_buckets(events)

        # Initialize matrices
        pair_count = [[0 for _ in modules] for _ in time_buckets]
        entropy = [[0.0 for _ in modules] for _ in time_buckets]
        failures = [[0 for _ in modules] for _ in time_buckets]
        event_counts = [[0 for _ in modules] for _ in time_buckets]

        # Populate matrices
        for event in events:
            try:
                module_idx = modules.index(event['module'])
                event_time = dt.datetime.fromisoformat(event['timestamp'].replace('Z', ''))
                time_idx = HeatmapGenerator._find_time_bucket(event_time, time_buckets)

                if time_idx is not None:
                    if event['metric'] == 'pair_count':
                        pair_count[time_idx][module_idx] += event['value']
                    elif event['metric'] == 'entropy':
                        entropy[time_idx][module_idx] = max(entropy[time_idx][module_idx], event['value'])
                    elif event['metric'] == 'failure':
                        failures[time_idx][module_idx] += event['value']

                    event_counts[time_idx][module_idx] += 1
            except (ValueError, KeyError):
                continue

        # Normalize entropy and compute failure rates
        for i in range(len(time_buckets)):
            for j in range(len(modules)):
                if event_counts[i][j] > 0:
                    failures[i][j] = failures[i][j] / event_counts[i][j]

        return {
            "pair_count": pair_count,
            "entropy": entropy,
            "failures": failures,
            "modules": modules,
            "time_buckets": [tb.strftime('%H:%M') for tb in time_buckets]
        }

    @staticmethod
    def _create_time_buckets(events: List[Dict], num_buckets: int = 10) -> List[dt.datetime]:
        """Create time buckets covering the event time range."""
        if not events:
            return []

        timestamps = [dt.datetime.fromisoformat(e['timestamp'].replace('Z', '')) for e in events]
        min_time = min(timestamps)
        max_time = max(timestamps)

        bucket_size = (max_time - min_time) / num_buckets
        return [min_time + i * bucket_size for i in range(num_buckets)]

    @staticmethod
    def _find_time_bucket(event_time: dt.datetime, buckets: List[dt.datetime]) -> Optional[int]:
        """Find which time bucket an event belongs to."""
        for i, bucket_time in enumerate(buckets):
            if event_time <= bucket_time:
                return i
        return None

    @staticmethod
    def build_snapshot(events: List[Dict]) -> Dict[str, Any]:
        """Generate a snapshot with KPIs, matrices, and alerts."""
        matrices = HeatmapGenerator.compute_matrices(events)
        alerts = []

        # Compute KPIs
        total_events = len(events)
        window_min = DEFAULT_WINDOW_MIN
        events_per_min = total_events / window_min if window_min > 0 else 0
        active_runs = len({e['module'] for e in events})

        # Check for alerts
        if matrices['entropy'] and matrices['failures']:
            max_entropy = max(max(row) for row in matrices['entropy'])
            max_failure = max(max(row) for row in matrices['failures'])

            if max_entropy > ALERT_ENTROPY_THRESH and max_failure > ALERT_FAILURE_THRESH:
                alerts.append({
                    "type": "entropy_failure_spike",
                    "entropy": max_entropy,
                    "failure_rate": max_failure,
                    "thresholds": {
                        "entropy": ALERT_ENTROPY_THRESH,
                        "failure": ALERT_FAILURE_THRESH
                    }
                })

        return {
            "timestamp": dt.datetime.utcnow().isoformat() + 'Z',
            "kpis": {
                "events_per_min": events_per_min,
                "active_runs": active_runs,
                "error_rate": sum(1 for e in events if e.get('metric') == 'failure') / total_events if total_events > 0 else 0
            },
            "matrices": matrices,
            "alerts": alerts
        }

class StreamlitUI:
    """Streamlit-based user interface for the heatmap panel."""

    def __init__(self):
        self.ingester = DataIngester()
        self.last_refresh = 0
        st.set_page_config(layout="wide")

    def run(self, refresh_sec: int = 60):
        """Run the Streamlit UI loop."""
        st.title("Goldbach Activity Heatmap")

        # Sidebar controls
        with st.sidebar:
            st.header("Configuration")
            dir_results = st.text_input("Results Directory", "./proof-data")
            dir_telemetry = st.text_input("Telemetry Directory", "./proof-data/telemetry")
            window_min = st.slider("Time Window (minutes)", 1, 120, DEFAULT_WINDOW_MIN)
            refresh_sec = st.slider("Refresh Rate (seconds)", 5, 300, 60)
            metrics = st.multiselect(
                "Metrics to Display",
                ["pair_count", "entropy", "failures"],
                ["pair_count", "entropy", "failures"]
            )

        # Main display
        if time.time() - self.last_refresh > refresh_sec:
            events = self.ingester.ingest(dir_results, dir_telemetry, window_min)
            snapshot = HeatmapGenerator.build_snapshot(events)
            self.last_refresh = time.time()
        else:
            # Use cached data if not time to refresh yet
            if 'snapshot' not in st.session_state:
                events = self.ingester.ingest(dir_results, dir_telemetry, window_min)
                snapshot = HeatmapGenerator.build_snapshot(events)
                st.session_state.snapshot = snapshot
            else:
                snapshot = st.session_state.snapshot

        # Display KPIs
        self._show_kpis(snapshot['kpis'])

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Heatmap", "Time Series", "Runs"])

        with tab1:
            self._show_heatmaps(snapshot['matrices'], metrics)

        with tab2:
            self._show_time_series(snapshot['matrices'])

        with tab3:
            self._show_runs(snapshot['matrices'])

        # Show alerts if any
        if snapshot['alerts']:
            for alert in snapshot['alerts']:
                st.error(
                    f"ALERT: {alert['type']} detected! "
                    f"Entropy: {alert['entropy']:.2f} (threshold: {alert['thresholds']['entropy']}), "
                    f"Failure rate: {alert['failure_rate']:.2f} (threshold: {alert['thresholds']['failure']})"
                )

    def _show_kpis(self, kpis: Dict[str, Any]):
        """Display KPI cards."""
        cols = st.columns(3)
        with cols[0]:
            st.metric("Events/Min", f"{kpis['events_per_min']:.1f}")
        with cols[1]:
            st.metric("Active Runs", kpis['active_runs'])
        with cols[2]:
            st.metric("Error Rate", f"{kpis['error_rate']:.1%}")

    def _show_heatmaps(self, matrices: Dict[str, Any], metrics: List[str]):
        """Display heatmap visualizations."""
        if not matrices['modules']:
            st.warning("No data available for heatmaps")
            return

        for metric in metrics:
            if metric not in matrices:
                continue

            st.subheader(metric.replace('_', ' ').title())
            df = pd.DataFrame(
                matrices[metric],
                columns=matrices['modules'],
                index=matrices['time_buckets']
            )

            fig = px.imshow(
                df,
                labels=dict(x="Module", y="Time", color=metric),
                color_continuous_scale=COLORBLIND_PALETTE,
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _show_time_series(self, matrices: Dict[str, Any]):
        """Display time series visualizations."""
        if not matrices['modules']:
            st.warning("No data available for time series")
            return

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

        for i, metric in enumerate(['pair_count', 'entropy', 'failures']):
            df = pd.DataFrame(
                matrices[metric],
                columns=matrices['modules'],
                index=matrices['time_buckets']
            )

            for module in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[module],
                        name=f"{module} ({metric})",
                        mode='lines+markers'
                    ),
                    row=i+1,
                    col=1
                )

        fig.update_layout(height=900, title_text="Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)

    def _show_runs(self, matrices: Dict[str, Any]):
        """Display run-specific visualizations."""
        if not matrices['modules']:
            st.warning("No data available for runs analysis")
            return

        # Show module-level statistics
        st.subheader("Module Performance")

        # Create a summary DataFrame
        data = []
        for i, module in enumerate(matrices['modules']):
            pair_counts = [row[i] for row in matrices['pair_count']]
            entropies = [row[i] for row in matrices['entropy']]
            failures = [row[i] for row in matrices['failures']]

            data.append({
                "module": module,
                "total_pairs": sum(pair_counts),
                "max_entropy": max(entropies) if entropies else 0,
                "failure_rate": sum(failures) / len(failures) if failures else 0
            })

        df = pd.DataFrame(data)
        st.dataframe(df.style.background_gradient(cmap='Blues'))

class CLIHandler:
    """Command-line interface handler for snapshot generation."""

    @staticmethod
    def run(args):
        """Execute CLI command."""
        start_time = time.time()

        ingester = DataIngester()
        events = ingester.ingest(args.dir, args.telemetry, args.window_min)
        snapshot = HeatmapGenerator.build_snapshot(events)

        if args.export:
            with open(args.export, 'w') as f:
                json.dump(snapshot, f, indent=2)
            print(f"Snapshot saved to {args.export}")
        else:
            print(json.dumps(snapshot, indent=2))

        TelemetryLogger.log({
            "component": "GoldbachHeatmapPanel",
            "event": "cli_snapshot",
            "duration_ms": int((time.time() - start_time) * 1000),
            "events_processed": len(events)
        })

def self_test():
    """Run self-test of core functionality."""
    test_events = []
    now = dt.datetime.utcnow()
    modules = ["module_a", "module_b", "module_c"]
    metrics = ["pair_count", "entropy", "failure"]

    # Generate test data
    for i in range(100):
        module = modules[i % len(modules)]
        metric = metrics[i % len(metrics)]
        value = i % 10 if metric == "pair_count" else i / 100 if metric == "entropy" else i % 2
        test_events.append({
            "timestamp": (now - dt.timedelta(minutes=i)).isoformat() + 'Z',
            "module": module,
            "metric": metric,
            "value": value
        })

    # Test ingestion and processing
    ingester = DataIngester()
    processed_events = ingester.ingest(window_min=60)  # Should ignore dirs for test

    # Test matrix computation
    matrices = HeatmapGenerator.compute_matrices(test_events)
    assert len(matrices['modules']) == 3, "Module count mismatch"
    assert len(matrices['time_buckets']) == 10, "Time bucket count mismatch"
    assert len(matrices['pair_count'][0]) == 3, "Matrix dimension mismatch"

    # Test snapshot generation
    snapshot = HeatmapGenerator.build_snapshot(test_events)
    assert 'kpis' in snapshot, "KPIs missing from snapshot"
    assert 'matrices' in snapshot, "Matrices missing from snapshot"

    print("Self-test passed successfully!")
    return True

def discover() -> Dict[str, str]:
    """Return component metadata."""
    return {"component": "GoldbachHeatmapPanel"}

def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Goldbach Activity Heatmap Panel")
    parser.add_argument('--mode', choices=['ui', 'cli', 'self-test'], default='ui',
                      help='Operation mode (default: ui)')
    parser.add_argument('--dir', default='./proof-data',
                      help='Directory containing proof data')
    parser.add_argument('--telemetry', default='./proof-data/telemetry',
                      help='Directory containing telemetry data')
    parser.add_argument('--window-min', type=int, default=DEFAULT_WINDOW_MIN,
                      help='Time window in minutes for analysis')
    parser.add_argument('--once', type=int, default=0,
                      help='Run once and exit (for CLI mode)')
    parser.add_argument('--export',
                      help='Export snapshot to file (CLI mode)')
    parser.add_argument('--self-test', action='store_true',
                      help='Run self-test and exit')

    args = parser.parse_args()

    if args.self_test or args.mode == 'self-test':
        return self_test()

    if args.mode == 'cli':
        CLIHandler.run(args)
    else:
        if not PLOTLY_AVAILABLE:
            print("Error: Streamlit and Plotly are required for UI mode", file=sys.stderr)
            sys.exit(1)
        ui = StreamlitUI()
        ui.run()

if __name__ == '__main__':
    main()
