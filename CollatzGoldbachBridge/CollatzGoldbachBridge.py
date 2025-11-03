#!/usr/bin/env python3
"""
Collatz-Goldbach Bridge: Experimental analytics between Collatz dynamics and Goldbach partitions.
"""

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

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


# --- Core Collatz Functions ---
@lru_cache(maxsize=2**20)
def collatz_sequence(n: int) -> List[int]:
    """Generate Collatz sequence for a given n."""
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence


def collatz_features(n: int) -> Dict[str, Union[int, float, str]]:
    """Compute comprehensive Collatz features for a given n.

    Returns:
        dict: Contains steps, max_spike, stopping_time, parity_profile, tail_length
    """
    sequence = collatz_sequence(n)
    steps = len(sequence) - 1
    max_spike = max(sequence)

    # Stopping time: steps until first value < n
    stopping_time = next((i for i, x in enumerate(sequence) if x < n), steps)

    # Parity profile: first 8 parity bits as string
    parity_profile = ''.join(['1' if x % 2 else '0' for x in sequence[:8]])
    if len(parity_profile) < 8:
        parity_profile = parity_profile.ljust(8, '0')

    # Tail length: number of consecutive halving steps at end
    tail_length = 0
    for x in reversed(sequence):
        if x % 2 == 0:
            tail_length += 1
        else:
            break

    return {
        'n': n,
        'steps': steps,
        'max_spike': max_spike,
        'stopping_time': stopping_time,
        'parity_profile': parity_profile,
        'tail_length': tail_length
    }


def batch_collatz(ns: List[int]) -> List[Dict[str, Union[int, float, str]]]:
    """Batch compute Collatz features for multiple numbers."""
    return [collatz_features(n) for n in ns]


# --- Correlation & Bridge Functions ---
def correlate(ns: List[int], pairs_counts: List[int]) -> Dict[str, Union[float, str]]:
    """Compute correlations between Collatz features and Goldbach pair counts.

    Requires numpy for advanced statistics.
    """
    if len(ns) != len(pairs_counts):
        raise ValueError("Length mismatch between ns and pairs_counts")

    features = batch_collatz(ns)
    result = {
        'n_samples': len(ns),
        'simple_correlations': {}
    }

    if not NUMPY_AVAILABLE:
        return result

    # Extract all features for correlation analysis
    feature_names = ['steps', 'max_spike', 'stopping_time', 'tail_length']
    feature_arrays = {
        name: np.array([f[name] for f in features])
        for name in feature_names
    }
    pair_array = np.array(pairs_counts)

    # Compute correlations
    for name, values in feature_arrays.items():
        corr = np.corrcoef(values, pair_array)[0, 1]
        result['simple_correlations'][name] = float(corr)

    # Generate insights
    strongest = max(result['simple_correlations'].items(), key=lambda x: abs(x[1]), None)
    if strongest:
        feature, corr = strongest
        direction = "positive" if corr > 0 else "negative"
        result['insight'] = (
            f"Strongest correlation: {feature} shows {direction} relationship with pair counts (ρ = {corr:.3f})"
        )

    return result


def bridge_table(ns: List[int], pairs_counts: List[int]) -> List[Dict[str, Union[int, float, str]]]:
    """Create a joined table of Collatz features and Goldbach pair counts."""
    if len(ns) != len(pairs_counts):
        raise ValueError("Length mismatch between ns and pairs_counts")

    features = batch_collatz(ns)
    return [
        {**feat, 'pairs_count': count}
        for feat, count in zip(features, pairs_counts)
    ]


# --- I/O Utilities ---
def export(obj: Union[dict, list], path: str) -> None:
    """Export data to JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_pairs_counts(path: str) -> Dict[int, int]:
    """Load Goldbach pairs counts from CSV file."""
    try:
        with open(path) as f:
            return {
                int(line.split(',')[0]): int(line.split(',')[1])
                for line in f if line.strip()
            }
    except Exception as e:
        raise ValueError(f"Failed to load pairs counts: {e}")


# --- Metadata ---
def metadata() -> Dict[str, str]:
    """Return module metadata."""
    return {
        "version": "1.0.0",
        "author": "GoldbachX Team",
        "description": "Collatz-Goldbach Bridge Analytics",
        "dependencies": {
            "numpy": NUMPY_AVAILABLE,
            "streamlit": STREAMLIT_AVAILABLE
        }
    }


def discover() -> Dict[str, str]:
    """Discovery endpoint for plugin system."""
    return {"component": "CollatzGoldbachBridge"}


# --- CLI Interface ---
def parse_args():
    parser = argparse.ArgumentParser(description="Collatz-Goldbach Bridge Analytics")
    parser.add_argument('--mode', choices=['cli', 'ui', 'self-test'], default='cli')
    parser.add_argument('--start', type=int, default=4)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--pairs', type=str, help="Path to pairs counts CSV")
    parser.add_argument('--export', type=str, help="Export path for results")
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility")
    return parser.parse_args()


def run_cli(start: int, end: int, pairs_path: Optional[str], export_path: Optional[str]):
    """Run the CLI workflow."""
    telemetry = {
        'start_time': time.time(),
        'features_computed': 0,
        'bridge_rows': 0,
        'corr_ready': False
    }

    ns = list(range(start, end + 1))
    telemetry['features_computed'] = len(ns)

    if pairs_path:
        pairs_data = load_pairs_counts(pairs_path)
        pairs_counts = [pairs_data.get(n, 0) for n in ns]
        corr_result = correlate(ns, pairs_counts)
        bridge_data = bridge_table(ns, pairs_counts)
        telemetry['bridge_rows'] = len(bridge_data)
        telemetry['corr_ready'] = True

        print(f"\nAnalysis Results (n={start}-{end}):")
        print(f"  - Samples processed: {len(ns)}")
        if NUMPY_AVAILABLE and 'insight' in corr_result:
            print(f"  - {corr_result['insight']}")

        if export_path:
            export(bridge_data, export_path)
            print(f"  - Exported bridge data to {export_path}")
    else:
        features = batch_collatz(ns)
        if export_path:
            export(features, export_path)
            print(f"Exported Collatz features to {export_path}")

    telemetry['time_ms'] = int((time.time() - telemetry['start_time']) * 1000)
    return telemetry


# --- Streamlit UI ---
def run_ui():
    """Run the Streamlit UI (if available)."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: pip install streamlit")
        return

    st.set_page_config(page_title="Collatz-Goldbach Bridge", layout="wide")
    st.title("Collatz-Goldbach Bridge Analytics")

    with st.sidebar:
        st.header("Parameters")
        start = st.number_input("Start", min_value=2, value=4)
        end = st.number_input("End", min_value=start+1, value=1000)
        pairs_file = st.file_uploader("Goldbach Pairs CSV", type=['csv'])
        features = st.multiselect(
            "Features to Analyze",
            ['steps', 'max_spike', 'stopping_time', 'tail_length'],
            default=['stopping_time', 'tail_length']
        )

    if st.sidebar.button("Run Analysis"):
        ns = list(range(start, end + 1))
        pairs_counts = [0] * len(ns)

        if pairs_file:
            pairs_data = load_pairs_counts(pairs_file.name)
            pairs_counts = [pairs_data.get(n, 0) for n in ns]

        bridge_data = bridge_table(ns, pairs_counts)
        corr_result = correlate(ns, pairs_counts)

        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Correlation Summary")
            if NUMPY_AVAILABLE and corr_result.get('simple_correlations'):
                st.table(pd.DataFrame.from_dict(
                    corr_result['simple_correlations'],
                    orient='index',
                    columns=['Correlation']
                ).sort_values('Correlation', key=abs, ascending=False))

            if 'insight' in corr_result:
                st.info(corr_result['insight'])

        with col2:
            st.write("### Sample Data")
            st.dataframe(pd.DataFrame(bridge_data).head(20))

        if features and NUMPY_AVAILABLE:
            st.subheader("Feature Relationships")
            for feature in features:
                fig = px.scatter(
                    x=[d[feature] for d in bridge_data],
                    y=[d['pairs_count'] for d in bridge_data],
                    labels={'x': feature, 'y': 'Pairs Count'},
                    title=f"{feature} vs. Pairs Count"
                )
                st.plotly_chart(fig, use_container_width=True)


# --- Self-Tests ---
def self_test():
    """Run self-tests to verify functionality."""
    tests = {
        'collatz_sequence': [
            (1, [1]),
            (2, [2, 1]),
            (3, [3, 10, 5, 16, 8, 4, 2, 1]),
            (4, [4, 2, 1])
        ],
        'collatz_features': [
            (1, {'steps': 0, 'max_spike': 1, 'stopping_time': 0}),
            (3, {'steps': 7, 'max_spike': 16, 'stopping_time': 6})
        ]
    }

    failures = 0
    for func, cases in tests.items():
        for n, expected in cases:
            result = globals()[func](n)
            if func == 'collatz_features':
                # Only check specified fields
                match = all(result[k] == expected[k] for k in expected)
            else:
                match = result == expected

            if not match:
                print(f"FAIL: {func}({n})")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
                failures += 1

    # Test memoization
    collatz_sequence.cache_clear()
    _ = collatz_sequence(27)  # Should compute
    hits_before = collatz_sequence.cache_info().hits
    _ = collatz_sequence(27)  # Should hit cache
    hits_after = collatz_sequence.cache_info().hits
    if hits_after - hits_before != 1:
        print("FAIL: Memoization not working for collatz_sequence")
        failures += 1

    # Test correlation (deterministic)
    if NUMPY_AVAILABLE:
        test_ns = [10, 20, 30, 40, 50]
        test_counts = [2, 2, 3, 3, 4]
        corr = correlate(test_ns, test_counts)
        if not isinstance(corr['simple_correlations']['steps'], float):
            print("FAIL: Correlation computation failed")
            failures += 1

    if failures == 0:
        print("All self-tests passed")
    return failures == 0


# --- Main ---
def main():
    args = parse_args()

    if args.seed is not None:
        if NUMPY_AVAILABLE:
            np.random.seed(args.seed)

    if args.mode == 'cli':
        telemetry = run_cli(args.start, args.end, args.pairs, args.export)
        print(f"\nCompleted in {telemetry['time_ms']}ms")
    elif args.mode == 'ui':
        run_ui()
    elif args.mode == 'self-test':
        success = self_test()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
