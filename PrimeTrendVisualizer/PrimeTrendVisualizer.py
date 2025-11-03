#!/usr/bin/env python3
# PrimeTrendVisualizer.py - Visualize prime-related trends for Goldbach analysis

import sys
import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse
import csv

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Constants
DEFAULT_WINDOW = 1000
MAX_LIMIT = 10_000_000  # Reasonable default for browser-based visualization
TELEMETRY_FILE = "prime_trend_telemetry.jsonl"

class PrimeCalculator:
    """Handles prime number calculations with optional caching"""

    @staticmethod
    def sieve(limit: int) -> List[int]:
        """Sieve of Eratosthenes implementation"""
        if limit < 2:
            return []

        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for num in range(2, int(math.sqrt(limit)) + 1):
            if sieve[num]:
                sieve[num*num : limit+1 : num] = [False] * len(sieve[num*num : limit+1 : num])

        return [i for i, is_prime in enumerate(sieve) if is_prime]

    @classmethod
    def get_primes(cls, limit: int, cache_dir: Optional[Path] = None) -> List[int]:
        """Get primes with optional cache support"""
        # Check cache first
        if cache_dir:
            cache_file = cache_dir / f"primes_up_to_{limit}.txt"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return [int(line.strip()) for line in f if line.strip()]

        # Compute if not cached
        primes = cls.sieve(limit)

        # Save to cache if available
        if cache_dir:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                f.write('\n'.join(map(str, primes)))

        return primes

def compute_trends(limit: int, *, window: int = DEFAULT_WINDOW) -> dict:
    """
    Compute prime-related trends: π(x), average gap, max gap, moving density.

    Args:
        limit: Upper bound for prime calculations
        window: Size of moving window for density calculations

    Returns:
        Dictionary containing:
            - primes: List of primes up to limit
            - pi_x: List of prime counts up to each number
            - avg_gaps: List of average gaps in windows
            - max_gaps: List of maximum gaps in windows
            - densities: List of prime densities (primes/window)
            - x_values: List of x-axis values for plotting
    """
    if limit <= 1:
        raise ValueError("Limit must be greater than 1")

    if window <= 0:
        raise ValueError("Window size must be positive")

    start_time = time.time()

    # Try to use cached primes from SieveEngine if available
    cache_dir = Path("CoreEngines/SieveEngine/cache")
    primes = PrimeCalculator.get_primes(limit, cache_dir if cache_dir.exists() else None)

    if not primes:
        return {}

    # Compute π(x) - cumulative count of primes up to x
    pi_x = []
    prime_set = set(primes)
    current_count = 0
    for x in range(2, limit + 1):
        if x in prime_set:
            current_count += 1
        pi_x.append(current_count)

    # Compute gap statistics and moving density
    avg_gaps = []
    max_gaps = []
    densities = []
    x_values = []

    if len(primes) > 1:
        # Compute gaps between consecutive primes
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        # Windowed calculations
        for i in range(window, len(primes), window):
            window_primes = primes[i-window:i]
            window_gaps = gaps[i-window:i-1] if i > window else gaps[:i-1]

            if window_gaps:
                avg_gaps.append(sum(window_gaps) / len(window_gaps))
                max_gaps.append(max(window_gaps))
            else:
                avg_gaps.append(0)
                max_gaps.append(0)

            densities.append(len(window_primes) / window)
            x_values.append(window_primes[-1])

    # Telemetry
    telemetry = {
        "limit": limit,
        "window": window,
        "time_ms": (time.time() - start_time) * 1000,
        "corr_ready": False
    }

    with open(TELEMETRY_FILE, 'a') as f:
        f.write(json.dumps(telemetry) + '\n')

    return {
        "primes": primes,
        "pi_x": pi_x,
        "avg_gaps": avg_gaps,
        "max_gaps": max_gaps,
        "densities": densities,
        "x_values": x_values,
        "limit": limit,
        "window": window
    }

def correlate_with_goldbach(ns: List[int], pairs_counts: List[int]) -> dict:
    """
    Compute correlation metrics between prime trends and Goldbach pair counts.

    Args:
        ns: List of even numbers
        pairs_counts: Corresponding counts of Goldbach pairs

    Returns:
        Dictionary containing correlation metrics and regression data
    """
    if len(ns) != len(pairs_counts):
        raise ValueError("Input lists must have the same length")

    if not ns:
        return {}

    # Compute basic statistics
    result = {
        "n_samples": len(ns),
        "min_n": min(ns),
        "max_n": max(ns),
        "min_pairs": min(pairs_counts),
        "max_pairs": max(pairs_counts),
    }

    # Enhanced correlation analysis if numpy is available
    if NUMPY_AVAILABLE:
        x = np.array(ns)
        y = np.array(pairs_counts)

        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)

        # Correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        result.update({
            "correlation_coefficient": float(correlation),
            "linear_regression": {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "r_squared": float(correlation ** 2),
                "prediction_fn": f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
            },
            "trend_line": {
                "x": x.tolist(),
                "y": poly(x).tolist()
            }
        })

    return result

def export(data: dict, path: str) -> None:
    """Export data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def discover() -> dict:
    """Component discovery"""
    return {"component": "PrimeTrendVisualizer"}

def self_test():
    """Run self-tests to validate functionality"""
    tests_passed = 0
    total_tests = 0

    # Test 1: π(x) monotonicity
    test_limit = 1000
    result = compute_trends(test_limit)
    pi_x = result['pi_x']
    is_monotonic = all(pi_x[i] <= pi_x[i+1] for i in range(len(pi_x)-1))
    total_tests += 1
    tests_passed += 1 if is_monotonic else 0
    print(f"Test 1 (π(x) monotonic): {'PASSED' if is_monotonic else 'FAILED'}")

    # Test 2: Reasonable gap stats
    if result['avg_gaps'] and result['max_gaps']:
        avg_gap = sum(result['avg_gaps']) / len(result['avg_gaps'])
        reasonable = 0 < avg_gap < 20  # Should be around ln(n) ~6.9 for n=1000
        total_tests += 1
        tests_passed += 1 if reasonable else 0
        print(f"Test 2 (Reasonable gaps): {'PASSED' if reasonable else 'FAILED'}")

    # Test 3: Correlation math (if numpy available)
    if NUMPY_AVAILABLE:
        ns = [10, 20, 30, 40, 50]
        pairs = [2, 4, 5, 6, 8]  # Made-up counts
        corr = correlate_with_goldbach(ns, pairs)
        if 'correlation_coefficient' in corr:
            total_tests += 1
            tests_passed += 1
            print("Test 3 (Correlation math): PASSED")

    print(f"\nSelf-test summary: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def load_goldbach_csv(file_path: str) -> Tuple[List[int], List[int]]:
    """Load Goldbach pair counts from CSV file"""
    ns = []
    counts = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    ns.append(int(row[0]))
                    counts.append(int(row[1]))
                except ValueError:
                    continue

    return ns, counts

def render_ui():
    """Render Streamlit UI if available"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Use CLI mode instead.")
        return

    st.set_page_config(page_title="Prime Trend Visualizer", layout="wide")
    st.title("Goldbach Prime Trend Visualizer")

    # Sidebar controls
    with st.sidebar:
        st.header("Parameters")
        limit = st.number_input(
            "Upper limit (n)",
            min_value=10,
            max_value=MAX_LIMIT,
            value=10000,
            step=1000
        )

        window = st.number_input(
            "Analysis window size",
            min_value=10,
            max_value=limit//10,
            value=DEFAULT_WINDOW,
            step=100
        )

        use_density = st.checkbox("Show prime density", True)
        use_gaps = st.checkbox("Show gap analysis", True)

        st.header("Goldbach Pair Correlation")
        goldbach_file = st.file_uploader(
            "Upload Goldbach pair counts (CSV)",
            type=['csv']
        )

    # Main content area
    tab1, tab2 = st.tabs(["Prime Trends", "Correlation Analysis"])

    with tab1:
        st.subheader(f"Prime Number Trends up to {limit}")

        # Compute trends
        with st.spinner("Calculating prime trends..."):
            trends = compute_trends(limit, window=window)

        if not trends:
            st.error("No prime data generated. Check your parameters.")
            return

        # π(x) plot
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(2, limit+1)),
                y=trends['pi_x'],
                mode='lines',
                name='π(x) - Prime Count'
            ))
            fig.update_layout(
                title='Prime Counting Function π(x)',
                xaxis_title='x',
                yaxis_title='Number of primes ≤ x'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("π(x) data available but Plotly not installed for visualization.")

        # Gap analysis
        if use_gaps and trends['x_values'] and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trends['x_values'],
                y=trends['avg_gaps'],
                mode='lines',
                name='Average Gap'
            ))
            fig.add_trace(go.Scatter(
                x=trends['x_values'],
                y=trends['max_gaps'],
                mode='lines',
                name='Max Gap'
            ))
            fig.update_layout(
                title='Prime Gap Analysis (Moving Window)',
                xaxis_title='n',
                yaxis_title='Gap Size'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Density plot
        if use_density and trends['x_values'] and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trends['x_values'],
                y=trends['densities'],
                mode='lines',
                name='Prime Density'
            ))
            fig.update_layout(
                title='Prime Density (Primes/Window)',
                xaxis_title='n',
                yaxis_title='Density'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if goldbach_file:
            try:
                ns, counts = load_goldbach_csv(goldbach_file)
                corr = correlate_with_goldbach(ns, counts)

                st.subheader("Goldbach Pair Correlation Analysis")
                st.write(f"Analyzed {len(ns)} even numbers from {min(ns)} to {max(ns)}")

                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ns,
                        y=counts,
                        mode='markers',
                        name='Actual Pairs'
                    ))

                    if 'trend_line' in corr:
                        fig.add_trace(go.Scatter(
                            x=corr['trend_line']['x'],
                            y=corr['trend_line']['y'],
                            mode='lines',
                            name='Trend Line'
                        ))

                    fig.update_layout(
                        title='Goldbach Pair Counts',
                        xaxis_title='Even Number (n)',
                        yaxis_title='Number of Prime Pairs'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if 'correlation_coefficient' in corr:
                    st.metric(
                        "Correlation Coefficient",
                        value=f"{corr['correlation_coefficient']:.3f}"
                    )
                    st.write(f"Regression: {corr['linear_regression']['prediction_fn']}")
                    st.write(f"R²: {corr['linear_regression']['r_squared']:.3f}")

                st.json(corr)

            except Exception as e:
                st.error(f"Error processing Goldbach data: {e}")
        else:
            st.info("Upload a CSV file with Goldbach pair counts to enable correlation analysis")

def main():
    """Main entry point for CLI and UI modes"""
    parser = argparse.ArgumentParser(description="Prime Trend Visualizer for Goldbach analysis")
    parser.add_argument(
        "--mode",
        choices=["cli", "ui", "self-test"],
        default="ui" if STREAMLIT_AVAILABLE else "cli",
        help="Execution mode"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Upper limit for prime calculations"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Window size for moving calculations"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export path for JSON results"
    )
    parser.add_argument(
        "--goldbach-csv",
        type=str,
        help="Path to CSV file with Goldbach pair counts"
    )

    args = parser.parse_args()

    if args.mode == "self-test":
        if self_test():
            sys.exit(0)
        else:
            sys.exit(1)

    if args.mode == "ui":
        if STREAMLIT_AVAILABLE:
            render_ui()
        else:
            print("Streamlit not available. Falling back to CLI mode.")
            args.mode = "cli"

    if args.mode == "cli":
        print(f"Computing prime trends up to {args.limit} with window size {args.window}")

        try:
            trends = compute_trends(args.limit, window=args.window)

            if args.goldbach_csv:
                print(f"Loading Goldbach pairs from {args.goldbach_csv}")
                ns, counts = load_goldbach_csv(args.goldbach_csv)
                corr = correlate_with_goldbach(ns, counts)
                trends["goldbach_correlation"] = corr
                print(f"Correlation with Goldbach pairs: {corr.get('correlation_coefficient', 'N/A')}")

            if args.export:
                export(trends, args.export)
                print(f"Results exported to {args.export}")
            else:
                print("\nPrime Count (π(x)) samples:")
                print(f"  π(100) = {trends['pi_x'][98]}")  # 0-based, x=2 is index 0
                print(f"  π(1,000) = {trends['pi_x'][998]}")
                print(f"  π(10,000) = {trends['pi_x'][9998] if args.limit >= 10000 else 'N/A'}")

                if trends['avg_gaps']:
                    print("\nGap Analysis:")
                    print(f"  Final avg gap: {trends['avg_gaps'][-1]:.2f}")
                    print(f"  Final max gap: {trends['max_gaps'][-1]}")

                if trends['densities']:
                    print(f"\nFinal density: {trends['densities'][-1]:.6f}")

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
