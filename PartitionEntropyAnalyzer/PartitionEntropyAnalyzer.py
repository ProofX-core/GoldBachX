"""
GoldbachX Partition Entropy Analyzer - Computes entropy statistics for prime-pair partitions.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Constants
DEFAULT_BUCKET_SIZE = 1000
KL_EPSILON = 1e-10  # For numerical stability in KL divergence
TELEMETRY_SCHEMA = {"runs_loaded": int, "metrics_computed": int,
                   "correlations_done": int, "time_ms": float}

class PartitionEntropyAnalyzer:
    """Main analyzer class containing all business logic."""

    def __init__(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self.telemetry: dict[str, Any] = defaultdict(int)

    def load_runs(self, dir_results: str = "./proof-data") -> list[dict]:
        """Load GoldbachX run data from directory."""
        start_time = time.time()
        runs = []
        dir_path = Path(dir_results)

        if not dir_path.exists():
            raise FileNotFoundError(f"Results directory not found: {dir_path}")

        for file in dir_path.glob("*.json"):
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                    if self._validate_run(data):
                        runs.append(data)
                except json.JSONDecodeError:
                    continue

        self.telemetry["runs_loaded"] = len(runs)
        self.telemetry["time_ms"] += (time.time() - start_time) * 1000
        return runs

    def _validate_run(self, run: dict) -> bool:
        """Validate run schema has required fields."""
        required = ["n", "module", "pairs"]
        return all(key in run for key in required)

    def partition_stats(self, run: dict, ref_dist: Optional[str] = None) -> dict:
        """Compute statistics for a single run's partition."""
        start_time = time.time()
        stats: dict[str, Any] = {"n": run["n"], "module": run["module"]}

        pairs = np.array(run["pairs"])
        p_values = pairs[:, 0]
        q_values = pairs[:, 1]
        pair_count = len(p_values)

        # Basic counts
        stats["pair_count"] = pair_count
        stats["p_mean"] = float(np.mean(p_values))
        stats["q_mean"] = float(np.mean(q_values))

        # Histograms (10 bins between min and max)
        stats["p_hist"] = np.histogram(p_values, bins=10)[0].tolist()
        stats["q_hist"] = np.histogram(q_values, bins=10)[0].tolist()

        # Entropy metrics
        p_normalized = p_values / np.sum(p_values)
        stats["entropy_p"] = float(-np.sum(p_normalized * np.log(p_normalized + KL_EPSILON)))

        # Gini coefficient
        stats["gini_p"] = self._gini_coefficient(p_values)

        # Dispersion indices
        stats["cv_p"] = float(np.std(p_values) / np.mean(p_values))  # Coefficient of variation

        # KL divergence if reference distribution is provided
        if ref_dist == "uniform":
            uniform_ref = np.ones_like(p_values) / len(p_values)
            stats["kl_divergence"] = float(np.sum(p_normalized * np.log(p_normalized / uniform_ref + KL_EPSILON)))

        self.telemetry["metrics_computed"] += 1
        self.telemetry["time_ms"] += (time.time() - start_time) * 1000
        return stats

    def _gini_coefficient(self, x: NDArray[np.float64]) -> float:
        """Compute Gini coefficient for an array of values."""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

    def batch_analyze(self, runs: list[dict], bucket_size: int = DEFAULT_BUCKET_SIZE) -> dict:
        """Aggregate statistics across runs with bucketing by n."""
        start_time = time.time()
        results = defaultdict(list)

        for run in runs:
            stats = self.partition_stats(run)
            bucket = (run["n"] // bucket_size) * bucket_size
            results[bucket].append(stats)

        # Aggregate by bucket
        aggregated = {}
        for bucket, stats_list in results.items():
            df = pd.DataFrame(stats_list)
            agg_stats = {
                "n_start": bucket,
                "n_end": bucket + bucket_size - 1,
                "mean_pair_count": float(df["pair_count"].mean()),
                "mean_entropy": float(df["entropy_p"].mean()),
                "mean_gini": float(df["gini_p"].mean()),
                "module_counts": df["module"].value_counts().to_dict()
            }
            aggregated[bucket] = agg_stats

        self.telemetry["time_ms"] += (time.time() - start_time) * 1000
        return aggregated

    def correlate_with_failures(self, runs: list[dict],
                              telemetry: Optional[list[dict]] = None) -> dict:
        """Compute correlations between partition stats and failure rates."""
        start_time = time.time()
        if not telemetry:
            telemetry = []

        # Create DataFrame of run stats
        stats_df = pd.DataFrame([self.partition_stats(run) for run in runs])

        # If telemetry available, merge with failure data
        if telemetry:
            telemetry_df = pd.DataFrame(telemetry)
            merged = pd.merge(stats_df, telemetry_df, on=["n", "module"], how="left")
            merged["failed"] = merged.get("failed", 0)

            # Compute correlations
            correlations = {
                "pair_count_failure": np.corrcoef(merged["pair_count"], merged["failed"])[0, 1],
                "entropy_failure": np.corrcoef(merged["entropy_p"], merged["failed"])[0, 1],
                "gini_failure": np.corrcoef(merged["gini_p"], merged["failed"])[0, 1]
            }
        else:
            correlations = {"warning": "No telemetry data provided"}

        self.telemetry["correlations_done"] += 1
        self.telemetry["time_ms"] += (time.time() - start_time) * 1000
        return correlations

    def export(self, obj: dict, path: str) -> None:
        """Export results to JSON file."""
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    def metadata(self) -> dict:
        """Return metadata about this analyzer."""
        return {
            "version": "1.0.0",
            "metrics_provided": ["entropy_p", "gini_p", "kl_divergence", "cv_p"],
            "dependencies": ["numpy", "pandas"]
        }

    def discover(self) -> dict:
        """Service discovery information."""
        return {"component": "PartitionEntropyAnalyzer"}

    def run_self_tests(self) -> bool:
        """Run self-tests with controlled distributions."""
        test_cases = [
            {"n": 100, "module": "test", "pairs": [[3, 97], [11, 89], [17, 83]]},  # Low entropy
            {"n": 102, "module": "test", "pairs": [[5, 97], [13, 89], [19, 83], [23, 79]]},  # Medium entropy
            {"n": 104, "module": "test", "pairs": [[3, 101], [7, 97], [13, 91], [19, 85], [31, 73]]}  # High entropy
        ]

        results = [self.partition_stats(tc) for tc in test_cases]
        entropies = [r["entropy_p"] for r in results]
        ginis = [r["gini_p"] for r in results]

        # Verify monotonicity of entropy and Gini
        entropy_increasing = all(x < y for x, y in zip(entropies, entropies[1:]))
        gini_decreasing = all(x > y for x, y in zip(ginis, ginis[1:]))

        return entropy_increasing and gini_decreasing

def cli() -> None:
    """Command line interface for the analyzer."""
    parser = argparse.ArgumentParser(description="GoldbachX Partition Entropy Analyzer")
    parser.add_argument("--mode", choices=["cli", "ui", "self-test"], default="cli")
    parser.add_argument("--results", default="./proof-data", help="Directory with proof results")
    parser.add_argument("--telemetry", help="Path to telemetry data")
    parser.add_argument("--export", help="Path to export analysis results")
    parser.add_argument("--bucket-size", type=int, default=DEFAULT_BUCKET_SIZE,
                       help="Bucket size for aggregating by n")
    args = parser.parse_args()

    analyzer = PartitionEntropyAnalyzer()

    if args.mode == "self-test":
        print("Running self-tests...")
        success = analyzer.run_self_tests()
        print(f"Self-tests {'passed' if success else 'failed'}")
        return

    # Load data
    runs = analyzer.load_runs(args.results)
    telemetry = []
    if args.telemetry and os.path.exists(args.telemetry):
        with open(args.telemetry, 'r') as f:
            telemetry = [json.loads(line) for line in f]

    if not runs:
        print("No valid runs found in results directory")
        return

    # Analyze
    batch_results = analyzer.batch_analyze(runs, args.bucket_size)
    correlations = analyzer.correlate_with_failures(runs, telemetry)

    # Export if requested
    if args.export:
        output = {
            "batch_results": batch_results,
            "correlations": correlations,
            "telemetry": dict(analyzer.telemetry)
        }
        analyzer.export(output, args.export)
        print(f"Results exported to {args.export}")

    # Print summary
    print(f"Analyzed {len(runs)} runs")
    print(f"Correlations with failures: {correlations}")

def ui() -> None:
    """Streamlit-based user interface."""
    if not HAS_STREAMLIT:
        st.error("Streamlit is required for the UI mode. Please install with: pip install streamlit")
        return

    st.title("GoldbachX Partition Entropy Analyzer")
    analyzer = PartitionEntropyAnalyzer()

    with st.sidebar:
        st.header("Configuration")
        results_dir = st.text_input("Results Directory", "./proof-data")
        telemetry_path = st.text_input("Telemetry Path", "./proof-data/telemetry.jsonl")
        bucket_size = st.number_input("Bucket Size", min_value=100, value=DEFAULT_BUCKET_SIZE, step=100)
        metrics = st.multiselect(
            "Metrics to Display",
            ["pair_count", "entropy_p", "gini_p", "kl_divergence"],
            default=["entropy_p", "gini_p"]
        )

    try:
        runs = analyzer.load_runs(results_dir)
        telemetry = []
        if os.path.exists(telemetry_path):
            with open(telemetry_path, 'r') as f:
                telemetry = [json.loads(line) for line in f]

        if not runs:
            st.warning("No valid runs found in results directory")
            return

        # Batch analysis
        batch_results = analyzer.batch_analyze(runs, bucket_size)
        correlations = analyzer.correlate_with_failures(runs, telemetry)

        # Display results
        st.header("Aggregate Statistics by N-Bucket")
        st.dataframe(pd.DataFrame(batch_results.values()))

        st.header("Correlations with Failures")
        st.write(correlations)

        # Charts
        st.header("Visualizations")
        df = pd.DataFrame([analyzer.partition_stats(run) for run in runs])

        if "pair_count" in metrics:
            st.subheader("Pair Count vs N")
            st.line_chart(df.set_index("n")["pair_count"])

        if "entropy_p" in metrics:
            st.subheader("Entropy vs N")
            st.line_chart(df.set_index("n")["entropy_p"])

        if "gini_p" in metrics and telemetry:
            st.subheader("Gini Coefficient vs Failure Rate")
            merged = pd.merge(df, pd.DataFrame(telemetry), on=["n", "module"], how="left")
            st.scatter_chart(merged[["gini_p", "failed"]])

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "ui", "self-test"], default="cli")
    args, _ = parser.parse_known_args()

    if args.mode == "ui":
        if HAS_STREAMLIT:
            ui()
        else:
            print("Streamlit is required for UI mode. Install with: pip install streamlit")
    elif args.mode == "self-test":
        analyzer = PartitionEntropyAnalyzer()
        success = analyzer.run_self_tests()
        print(f"Self-tests {'passed' if success else 'failed'}")
    else:
        cli()
