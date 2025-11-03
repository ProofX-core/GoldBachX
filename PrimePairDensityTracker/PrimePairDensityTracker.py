#!/usr/bin/env python3
"""
GoldbachX Prime Pair Density Tracker
Analyzes distribution patterns of Goldbach prime pairs across ranges of n.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def metadata() -> Dict[str, Any]:
    """Return component metadata."""
    return {
        "component": "PrimePairDensityTracker",
        "version": "1.0.0",
        "dependencies": {
            "numpy": HAS_NUMPY,
            "pandas": HAS_PANDAS,
            "streamlit": HAS_STREAMLIT,
        },
    }


def discover() -> Dict[str, str]:
    """Discovery endpoint for system integration."""
    return {"component": "PrimePairDensityTracker"}


def ingest_pairs(source: Optional[str] = "./proof-data") -> List[Dict[str, Any]]:
    """
    Load Goldbach pair counts from JSON files in source directory.

    Args:
        source: Directory containing JSON files with {'n': int, 'count': int}

    Returns:
        List of records sorted by n, with duplicates removed
    """
    records = []
    source_path = Path(source or ".")

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    for file in sorted(source_path.glob("*.json")):
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    records.append(data)
                elif isinstance(data, list):
                    records.extend(data)
            except json.JSONDecodeError:
                continue

    # Deduplicate and sort
    unique_records = {r['n']: r for r in records}.values()
    return sorted(unique_records, key=lambda x: x['n'])


def _validate_inputs(ns: List[int], counts: List[int]) -> Tuple[List[int], List[int]]:
    """Ensure inputs are aligned and valid."""
    if len(ns) != len(counts):
        raise ValueError("Length of ns and counts must match")

    # Ensure ns is strictly increasing
    for i in range(1, len(ns)):
        if ns[i] <= ns[i-1]:
            raise ValueError("ns must be strictly increasing")

    return ns, counts


def compute_density(ns: List[int], counts: List[int], *, window: int = 1000) -> Dict[str, Any]:
    """
    Compute moving statistics for prime pair density.

    Args:
        ns: List of n values (must be strictly increasing)
        counts: Corresponding prime pair counts
        window: Rolling window size in elements (not n units)

    Returns:
        {
            'n': List[int],
            'counts': List[int],
            'density': List[float],  # counts[i]/ns[i]
            'rolling_mean': List[float],
            'rolling_std': List[float],
            'z_scores': List[float]
        }
    """
    ns, counts = _validate_inputs(ns, counts)

    if window <= 0:
        raise ValueError("Window must be positive")

    density = [c/n if n > 0 else 0.0 for c, n in zip(counts, ns)]

    results = {
        'n': ns,
        'counts': counts,
        'density': density,
        'rolling_mean': [],
        'rolling_std': [],
        'z_scores': []
    }

    if HAS_NUMPY:
        density_arr = np.array(density)
        pad = window // 2

        # Use numpy for efficient rolling stats
        rolling_mean = np.convolve(
            density_arr,
            np.ones(window)/window,
            mode='same'
        )
        rolling_std = np.sqrt(np.convolve(
            density_arr**2,
            np.ones(window)/window,
            mode='same'
        ) - rolling_mean**2)

        # Handle edge effects
        for i in range(pad):
            rolling_mean[i] = np.mean(density_arr[:i+pad+1])
            rolling_mean[-i-1] = np.mean(density_arr[-i-pad-1:])
            rolling_std[i] = np.std(density_arr[:i+pad+1])
            rolling_std[-i-1] = np.std(density_arr[-i-pad-1:])

        z_scores = (density_arr - rolling_mean) / (rolling_std + 1e-10)

        results['rolling_mean'] = rolling_mean.tolist()
        results['rolling_std'] = rolling_std.tolist()
        results['z_scores'] = z_scores.tolist()
    else:
        # Fallback to simple Python implementation
        for i in range(len(density)):
            start = max(0, i - window//2)
            end = min(len(density), i + window//2 + 1)
            window_data = density[start:end]
            mean = sum(window_data) / len(window_data)
            std = (sum((x - mean)**2 for x in window_data) / len(window_data))**0.5
            results['rolling_mean'].append(mean)
            results['rolling_std'].append(std)
            results['z_scores'].append((density[i] - mean) / (std + 1e-10))

    return results


def detect_regimes(series: Dict[str, Any], *, dip_z: float = -2.0, surge_z: float = 2.0) -> Dict[str, Any]:
    """
    Identify density regimes based on z-scores.

    Args:
        series: Output from compute_density
        dip_z: Z-score threshold for density dips
        surge_z: Z-score threshold for density surges

    Returns:
        {
            'regimes': List[str],  # 'DIP', 'SURGE', 'STEADY'
            'transitions': List[int],  # Indices where regime changes
            'stats': {
                'dip_count': int,
                'surge_count': int,
                'steady_count': int
            }
        }
    """
    if not series or 'z_scores' not in series:
        raise ValueError("Invalid input series - must contain z_scores")

    regimes = []
    current_regime = None
    transitions = []
    stats = {
        'dip_count': 0,
        'surge_count': 0,
        'steady_count': 0
    }

    for i, z in enumerate(series['z_scores']):
        if z <= dip_z:
            regime = 'DIP'
        elif z >= surge_z:
            regime = 'SURGE'
        else:
            regime = 'STEADY'

        if regime != current_regime:
            transitions.append(i)
            current_regime = regime

        regimes.append(regime)
        stats[f"{regime.lower()}_count"] += 1

    return {
        'regimes': regimes,
        'transitions': transitions,
        'stats': stats
    }


def export(obj: Dict[str, Any], path: str) -> None:
    """Export results to JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _run_self_tests() -> bool:
    """Run verification tests on synthetic data."""
    print("Running self-tests...")

    # Create test data with known patterns
    test_ns = list(range(1000, 2000, 2))
    test_counts = [50 + int(10 * np.sin(i/50)) for i in range(len(test_ns))]

    # Plant some obvious dips and surges
    test_counts[300:320] = [20] * 20  # Dip
    test_counts[600:620] = [80] * 20  # Surge

    try:
        # Test basic functionality
        density_data = compute_density(test_ns, test_counts, window=100)
        regimes = detect_regimes(density_data, dip_z=-1.5, surge_z=1.5)

        # Verify we detected the planted patterns
        dip_detected = any(r == 'DIP' for r in regimes['regimes'][300:320])
        surge_detected = any(r == 'SURGE' for r in regimes['regimes'][600:620])

        if not (dip_detected and surge_detected):
            print("FAILED: Did not detect planted regimes")
            return False

        # Test determinism
        data1 = compute_density(test_ns, test_counts, window=100)
        data2 = compute_density(test_ns, test_counts, window=100)
        if data1['rolling_mean'] != data2['rolling_mean']:
            print("FAILED: Non-deterministic results")
            return False

        print("All self-tests passed")
        return True
    except Exception as e:
        print(f"Self-test failed: {str(e)}")
        return False


def _console_report(results: Dict[str, Any]) -> None:
    """Print human-readable summary to console."""
    if not results:
        print("No results to display")
        return

    regimes = results.get('regime_analysis', {})
    stats = regimes.get('stats', {})

    print("\nPrime Pair Density Analysis")
    print("=" * 40)
    print(f"Range: {results['n'][0]} to {results['n'][-1]}")
    print(f"Data points: {len(results['n'])}")
    print(f"\nRegime Statistics:")
    print(f"DIP zones:    {stats.get('dip_count', 0)}")
    print(f"SURGE zones:  {stats.get('surge_count', 0)}")
    print(f"STEADY zones: {stats.get('steady_count', 0)}")

    if 'transitions' in regimes and regimes['transitions']:
        print("\nKey Transitions:")
        for idx in regimes['transitions'][:5]:  # Show first 5 transitions
            print(f"n={results['n'][idx]}: {regimes['regimes'][idx]}")


def _run_streamlit_ui() -> None:
    """Launch interactive Streamlit dashboard."""
    if not HAS_STREAMLIT:
        print("Streamlit not available - falling back to CLI mode")
        return

    st.set_page_config(page_title="Goldbach Prime Pair Density", layout="wide")
    st.title("Goldbach Prime Pair Density Tracker")

    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Parameters")
        source_dir = st.text_input("Data Directory", "./proof-data")
        window_size = st.slider("Rolling Window Size", 100, 5000, 1000, 100)
        dip_threshold = st.slider("Dip Z-Threshold", -5.0, 0.0, -2.0, 0.1)
        surge_threshold = st.slider("Surge Z-Threshold", 0.0, 5.0, 2.0, 0.1)
        smoothing = st.checkbox("Apply Smoothing", True)

    try:
        # Load and process data
        records = ingest_pairs(source_dir)
        if not records:
            st.error("No valid data found in the specified directory")
            return

        ns = [r['n'] for r in records]
        counts = [r['count'] for r in records]

        with st.spinner("Computing density statistics..."):
            density_data = compute_density(ns, counts, window=window_size)
            regime_data = detect_regimes(
                density_data,
                dip_z=dip_threshold,
                surge_z=surge_threshold
            )

        # Combine results for display
        results = {**density_data, 'regime_analysis': regime_data}

        # Main display
        tab1, tab2 = st.tabs(["Density Analysis", "Regime Detection"])

        with tab1:
            st.subheader("Prime Pair Density")
            if HAS_PANDAS:
                df = pd.DataFrame({
                    'n': results['n'],
                    'density': results['density'],
                    'rolling_mean': results['rolling_mean'],
                    'rolling_std': results['rolling_std']
                })
                st.line_chart(df.set_index('n'))
            else:
                st.write("Install pandas for better charting")
                st.json(results)

        with tab2:
            st.subheader("Density Regimes")
            if HAS_PANDAS:
                df = pd.DataFrame({
                    'n': results['n'],
                    'z_score': results['z_scores'],
                    'regime': results['regime_analysis']['regimes']
                })

                # Create a color-coded plot
                chart = st.line_chart(df.set_index('n')['z_score'])

                # Add regime highlights
                for i, regime in enumerate(results['regime_analysis']['regimes']):
                    if regime == 'DIP':
                        chart.add_rows(pd.DataFrame({
                            'z_score': [results['z_scores'][i]],
                            'regime': [regime]
                        }, index=[results['n'][i]]))

                st.write("Regime Legend:")
                st.markdown("- DIP: Red\n- SURGE: Green\n- STEADY: Gray")
            else:
                st.write(results['regime_analysis'])

        # Export option
        if st.button("Export Results"):
            export_path = f"density_results_{int(time.time())}.json"
            export(results, export_path)
            st.success(f"Exported to {export_path}")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Goldbach Prime Pair Density Analyzer")
    parser.add_argument("--mode", choices=["cli", "ui", "self-test"], default="cli")
    parser.add_argument("--source", default="./proof-data")
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--dip-z", type=float, default=-2.0)
    parser.add_argument("--surge-z", type=float, default=2.0)
    parser.add_argument("--export")

    args = parser.parse_args()

    if args.mode == "self-test":
        success = _run_self_tests()
        sys.exit(0 if success else 1)
    elif args.mode == "ui":
        if not HAS_STREAMLIT:
            print("Streamlit not available - falling back to CLI mode")
        else:
            _run_streamlit_ui()
            return

    # CLI mode processing
    try:
        records = ingest_pairs(args.source)
        if not records:
            print(f"No valid data found in {args.source}")
            sys.exit(1)

        ns = [r['n'] for r in records]
        counts = [r['count'] for r in records]

        start_time = time.time()
        density_data = compute_density(ns, counts, window=args.window)
        regime_data = detect_regimes(
            density_data,
            dip_z=args.dip_z,
            surge_z=args.surge_z
        )
        elapsed_ms = int((time.time() - start_time) * 1000)

        results = {
            **density_data,
            'regime_analysis': regime_data,
            'telemetry': {
                'pairs_loaded': len(records),
                'window': args.window,
                'regimes': regime_data['stats'],
                'time_ms': elapsed_ms
            }
        }

        _console_report(results)

        if args.export:
            export(results, args.export)
            print(f"Results exported to {args.export}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
