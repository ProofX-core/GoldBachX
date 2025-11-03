#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PartitionExplorer - Explore Goldbach partition results with filtering and visualization.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
from collections import defaultdict
import sys

# Optional dependencies (feature-gated)
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

VERSION = "1.0.0"
COMPONENT_NAME = "PartitionExplorer"

def discover() -> dict:
    """Identify this component's metadata."""
    return {
        "component": COMPONENT_NAME,
        "version": VERSION,
        "plotly_available": PLOTLY_AVAILABLE
    }

def load_runs(dir: str = "./proof-data") -> List[dict]:
    """Load all JSON result files from directory."""
    runs = []
    dir_path = Path(dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir}")

    for json_file in dir_path.glob("*.json"):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and 'n' in data:  # Basic validation
                    runs.append(data)
                    _log_telemetry("runs_loaded", {"file": str(json_file), "n": data['n']})
            except json.JSONDecodeError:
                continue
    return sorted(runs, key=lambda x: x['n'])

def normalize(run: dict) -> dict:
    """Convert a run into a normalized wide-row format."""
    normalized = {
        "n": run.get('n'),
        "pairs_count": len(run.get('pairs', [])) if 'pairs' in run else run.get('pairs_count', 0),
        "largest_prime": max(p for pair in run.get('pairs', []) for p in pair) if run.get('pairs') else 0,
        "time_ms": run.get('time_ms', run.get('duration_ms', 0)),
        "notes": run.get('notes', ''),
        "module": run.get('module', 'unknown'),
        "timestamp": run.get('timestamp', '')
    }
    return normalized

def compare(runs: List[dict]) -> dict:
    """Compare multiple normalized runs and compute deltas."""
    if len(runs) < 2:
        return {}

    base = runs[0]
    comparison = {
        "base_n": base['n'],
        "compared_to": [r['n'] for r in runs[1:]],
        "deltas": []
    }

    for other in runs[1:]:
        delta = {
            "n": other['n'],
            "pairs_count_diff": other['pairs_count'] - base['pairs_count'],
            "largest_prime_diff": other['largest_prime'] - base['largest_prime'],
            "time_ms_diff": other['time_ms'] - base['time_ms'],
            "relative_speed": base['time_ms'] / other['time_ms'] if other['time_ms'] else float('inf')
        }
        comparison['deltas'].append(delta)

    return comparison

def export_table(rows: List[dict], path: str) -> None:
    """Export normalized rows to CSV or JSON based on extension."""
    path = Path(path)
    if path.suffix == '.csv':
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    elif path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(rows, f, indent=2)
    else:
        raise ValueError("Unsupported export format. Use .csv or .json")

    _log_telemetry("export_count", {"rows": len(rows), "path": str(path)})

def _log_telemetry(event: str, data: dict) -> None:
    """Log telemetry events to JSONL file."""
    log_entry = {
        "timestamp": time.time(),
        "component": COMPONENT_NAME,
        "event": event,
        "data": data
    }
    with open("partition_explorer_telemetry.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def _apply_filters(runs: List[dict], filters: dict) -> List[dict]:
    """Apply filters to normalized runs."""
    filtered = []
    for run in runs:
        if filters.get('min_n') and run['n'] < filters['min_n']:
            continue
        if filters.get('max_n') and run['n'] > filters['max_n']:
            continue
        if filters.get('min_pairs') and run['pairs_count'] < filters['min_pairs']:
            continue
        if filters.get('max_pairs') and run['pairs_count'] > filters['max_pairs']:
            continue
        filtered.append(run)

    _log_telemetry("filters_applied", {"original_count": len(runs), "filtered_count": len(filtered)})
    return filtered

def _parse_filter_string(filter_str: str) -> dict:
    """Parse CLI filter string into filter dict."""
    filters = {}
    if not filter_str:
        return filters

    for condition in filter_str.split(','):
        if '>=' in condition:
            key, val = condition.split('>=')
            filters[f'min_{key.strip()}'] = int(val.strip())
        elif '<=' in condition:
            key, val = condition.split('<=')
            filters[f'max_{key.strip()}'] = int(val.strip())
        elif '>' in condition:
            key, val = condition.split('>')
            filters[f'min_{key.strip()}'] = int(val.strip()) + 1
        elif '<' in condition:
            key, val = condition.split('<')
            filters[f'max_{key.strip()}'] = int(val.strip()) - 1
        elif '=' in condition:
            key, val = condition.split('=')
            filters[f'min_{key.strip()}'] = filters[f'max_{key.strip()}'] = int(val.strip())

    return filters

def _render_ui(runs: List[dict]) -> None:
    """Render Streamlit UI components."""
    st.title("Goldbach Partition Explorer")
    st.sidebar.header("Filters")

    # Normalize all runs first
    normalized_runs = [normalize(run) for run in runs]

    # Sidebar filters
    min_n = st.sidebar.number_input("Minimum n", min_value=0, value=0)
    max_n = st.sidebar.number_input("Maximum n", min_value=0, value=10000)
    min_pairs = st.sidebar.number_input("Minimum pairs", min_value=0, value=0)
    max_pairs = st.sidebar.number_input("Maximum pairs", min_value=0, value=1000)
    include_p_eq_q = st.sidebar.checkbox("Include p = q pairs", value=True)
    exclude_twins = st.sidebar.checkbox("Exclude twin primes", value=False)

    # Apply filters
    filtered_runs = _apply_filters(normalized_runs, {
        'min_n': min_n,
        'max_n': max_n,
        'min_pairs': min_pairs,
        'max_pairs': max_pairs
    })

    # Main display
    st.header("Runs Table")
    if PLOTLY_AVAILABLE:
        df = pd.DataFrame(filtered_runs)
        st.dataframe(df)
    else:
        st.json(filtered_runs)

    # Run details
    if filtered_runs:
        selected_idx = st.selectbox("Select run to inspect", range(len(filtered_runs)), format_func=lambda i: f"n={filtered_runs[i]['n']}")
        selected_run = runs[[r['n'] for r in normalized_runs].index(filtered_runs[selected_idx]['n'])]

        st.header("Details")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Metadata")
            st.json({k: v for k, v in runs[selected_run].items() if k != 'pairs'})
        with col2:
            st.subheader("Partitions")
            st.write(f"Found {len(runs[selected_run].get('pairs', []))} pairs")

            if PLOTLY_AVAILABLE and 'pairs' in runs[selected_run]:
                pairs = runs[selected_run]['pairs']
                df_pairs = pd.DataFrame(pairs, columns=['p', 'q'])

                st.subheader("Visualizations")
                tab1, tab2, tab3 = st.tabs(["Histogram", "Scatter", "Distribution"])

                with tab1:
                    fig = px.histogram(df_pairs, x=['p', 'q'], barmode='overlay')
                    st.plotly_chart(fig)

                with tab2:
                    fig = px.scatter(df_pairs, x='p', y='q')
                    st.plotly_chart(fig)

                with tab3:
                    fig = px.box(df_pairs, y=['p', 'q'])
                    st.plotly_chart(fig)

    # Comparison
    st.header("Compare Runs")
    if len(filtered_runs) >= 2:
        compare_count = st.slider("Number of runs to compare", 2, min(4, len(filtered_runs)), 2)
        selected_for_compare = st.multiselect(
            "Select runs to compare",
            [f"n={r['n']}" for r in filtered_runs],
            [f"n={r['n']}" for r in filtered_runs[:compare_count]]
        )

        selected_ns = [int(s.split('=')[1]) for s in selected_for_compare]
        compare_runs = [r for r in normalized_runs if r['n'] in selected_ns]

        if len(compare_runs) >= 2:
            comparison = compare(compare_runs)
            st.subheader("Comparison Results")
            st.json(comparison)

            if PLOTLY_AVAILABLE:
                df_compare = pd.DataFrame(comparison['deltas'])
                st.dataframe(df_compare)

def _self_test() -> bool:
    """Run self-test with synthetic data."""
    test_runs = [
        {"n": 10, "pairs": [(3,7), (5,5), (7,3)], "time_ms": 5, "module": "test"},
        {"n": 12, "pairs": [(5,7), (7,5)], "time_ms": 3, "module": "test"},
        {"n": 14, "pairs": [(3,11), (7,7), (11,3)], "time_ms": 7, "module": "test"}
    ]

    # Test normalization
    normalized = [normalize(run) for run in test_runs]
    assert len(normalized) == 3, "Normalization failed"
    assert normalized[0]['pairs_count'] == 3, "Pairs count incorrect"
    assert normalized[1]['largest_prime'] == 7, "Largest prime incorrect"

    # Test comparison
    comp = compare(normalized[:2])
    assert comp['deltas'][0]['pairs_count_diff'] == -1, "Comparison diff incorrect"

    # Test export round-trip
    test_path = "test_export.json"
    export_table(normalized, test_path)
    assert Path(test_path).exists(), "Export failed"

    # Cleanup
    Path(test_path).unlink()
    return True

def main():
    parser = argparse.ArgumentParser(description="Goldbach Partition Explorer")
    parser.add_argument("--mode", choices=["cli", "ui", "self-test"], default="ui", help="Operation mode")
    parser.add_argument("--dir", default="./proof-data", help="Directory with JSON results")
    parser.add_argument("--filter", default="", help="Filter conditions, e.g. 'n>=1000,pairs_count>=2'")
    parser.add_argument("--export", help="Export path (CSV or JSON)")
    args = parser.parse_args()

    if args.mode == "self-test":
        if _self_test():
            print("✅ Self-test passed")
            sys.exit(0)
        else:
            print("❌ Self-test failed")
            sys.exit(1)

    try:
        runs = load_runs(args.dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.mode == "cli":
        filters = _parse_filter_string(args.filter)
        filtered_runs = _apply_filters([normalize(run) for run in runs], filters)

        if args.export:
            export_table(filtered_runs, args.export)
            print(f"Exported {len(filtered_runs)} rows to {args.export}")
        else:
            print(f"Found {len(filtered_runs)} runs matching filters")
            for run in filtered_runs:
                print(f"n={run['n']}: {run['pairs_count']} pairs (max p={run['largest_prime']})")

    elif args.mode == "ui":
        if not PLOTLY_AVAILABLE:
            print("Streamlit, pandas, and plotly are required for UI mode")
            sys.exit(1)
        _render_ui(runs)

if __name__ == "__main__":
    main()
