#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatisticalHeatmapGenerator - Produces module×metric or n-bucket×metric heatmaps from proof results.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Type aliases
MatrixType = Dict[str, Dict[str, Dict[str, float]]]
PathLike = Union[str, Path]

class StatisticalHeatmapGenerator:
    """Core heatmap generation logic with deterministic processing."""

    def __init__(self) -> None:
        self.telemetry: List[Dict[str, Any]] = []
        self.metrics_whitelist = {
            'pair_count', 'failure_rate', 'entropy_mean',
            'entropy_p95', 'execution_time_mean'
        }

    def _scan_files(self, results_dir: PathLike, telemetry_dir: PathLike) -> Tuple[List[Dict], List[Dict]]:
        """Scan directories for JSON result files and telemetry."""
        results = []
        telemetry = []

        for path in Path(results_dir).glob('**/*.json'):
            if path.is_file() and not path.name.startswith('.'):
                with open(path, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            results.append(data)
                    except json.JSONDecodeError:
                        continue

        telemetry_path = Path(telemetry_dir) / 'telemetry.jsonl'
        if telemetry_path.exists():
            with open(telemetry_path, 'r') as f:
                for line in f:
                    try:
                        telemetry.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        self._record_telemetry(
            files_scanned=len(results) + len(telemetry),
            files_parsed=len(results)
        )
        return results, telemetry

    def _record_telemetry(self, **kwargs: Any) -> None:
        """Record operation metrics."""
        self.telemetry.append({
            'timestamp': time.time(),
            **kwargs
        })

    def _process_results(self, results: List[Dict], bucket_size: int) -> MatrixType:
        """Convert raw results into matrix structure."""
        matrix: MatrixType = defaultdict(lambda: defaultdict(dict))
        bucket_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for result in results:
            module = result.get('module', 'unknown')
            n = result.get('n', 0)
            bucket = (n // bucket_size) * bucket_size if bucket_size > 0 else 0

            for metric in self.metrics_whitelist:
                if metric in result:
                    value = result[metric]
                    if isinstance(value, (int, float)):
                        matrix[module][metric]['value'] = value
                        bucket_metrics[bucket][metric].append(value)

        # Add bucket aggregates
        for bucket, metrics in bucket_metrics.items():
            bucket_key = f"n_{bucket}-{bucket + bucket_size - 1}"
            for metric, values in metrics.items():
                if values:
                    matrix[bucket_key][metric] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'score': self._normalize(values[-1], min(values), max(values))
                    }

        self._record_telemetry(
            cells_built=sum(len(mod) for mod in matrix.values()),
            matrix_shape=(len(matrix), len(self.metrics_whitelist))
        )
        return matrix

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range with protection against division by zero."""
        if max_val == min_val:
            return 0.5  # Midpoint when all values are equal
        return (value - min_val) / (max_val - min_val)

    def build_matrix(self, results_dir: str = "./proof-data",
                    telemetry_dir: str = "./proof-data/telemetry",
                    *, bucket: int = 1000) -> dict:
        """
        Build statistical matrix from results and telemetry.

        Args:
            results_dir: Directory containing proof result JSON files
            telemetry_dir: Directory containing telemetry data
            bucket: Size of n-value buckets for bucket×metric mode

        Returns:
            Dictionary with matrix data and metadata
        """
        start_time = time.time()
        results, telemetry = self._scan_files(results_dir, telemetry_dir)
        matrix = self._process_results(results, bucket)

        return {
            'matrix': matrix,
            'metadata': {
                'generated_at': time.time(),
                'time_ms': int((time.time() - start_time) * 1000),
                'bucket_size': bucket,
                'metrics': sorted(self.metrics_whitelist),
                'modules': sorted(matrix.keys())
            }
        }

    def to_dataframe(self, matrix: dict) -> Union["pd.DataFrame", List[List]]:
        """Convert matrix to pandas DataFrame or 2D list."""
        matrix_data = matrix.get('matrix', {})
        metrics = matrix.get('metadata', {}).get('metrics', [])

        if PANDAS_AVAILABLE:
            df_data = []
            for module, mod_metrics in matrix_data.items():
                row = {'module': module}
                for metric in metrics:
                    if metric in mod_metrics:
                        row[metric] = mod_metrics[metric].get('mean', mod_metrics[metric].get('value', None))
                    else:
                        row[metric] = None
                df_data.append(row)
            return pd.DataFrame(df_data).set_index('module')
        else:
            # Fallback to list of lists
            headers = ['module'] + metrics
            data = [headers]
            for module, mod_metrics in matrix_data.items():
                row = [module]
                for metric in metrics:
                    if metric in mod_metrics:
                        row.append(mod_metrics[metric].get('mean', mod_metrics[metric].get('value', None)))
                    else:
                        row.append(None)
                data.append(row)
            return data

    @staticmethod
    def export(obj: dict, path: str) -> None:
        """Export matrix or visualization to file."""
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def metadata() -> dict:
        """Return component metadata."""
        return {
            'version': '1.0.0',
            'dependencies': {
                'pandas': PANDAS_AVAILABLE,
                'plotly': PLOTLY_AVAILABLE,
                'streamlit': STREAMLIT_AVAILABLE
            }
        }

    @staticmethod
    def discover() -> dict:
        """Return component discovery information."""
        return {'component': 'StatisticalHeatmapGenerator'}

def run_cli(args: argparse.Namespace) -> None:
    """Command line interface entry point."""
    generator = StatisticalHeatmapGenerator()

    print(f"Building matrix from {args.results} (bucket={args.bucket})...")
    matrix = generator.build_matrix(
        results_dir=args.results,
        telemetry_dir=args.telemetry,
        bucket=args.bucket
    )

    if args.export:
        generator.export(matrix, args.export)
        print(f"Exported matrix to {args.export}")

    if PANDAS_AVAILABLE:
        df = generator.to_dataframe(matrix)
        print("\nMatrix Preview:")
        print(df.head())
    else:
        print("\nMatrix Stats:")
        print(f"Modules: {len(matrix['matrix'])}")
        print(f"Metrics: {len(matrix['metadata']['metrics'])}")

    print("\nDone.")

def run_ui() -> None:
    """Streamlit UI entry point."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is required for UI mode", file=sys.stderr)
        sys.exit(1)

    generator = StatisticalHeatmapGenerator()
    st.set_page_config(page_title="Statistical Heatmap", layout="wide")

    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("View Mode", ["Modules × Metrics", "Buckets × Metrics"])
    refresh_sec = st.sidebar.number_input("Refresh Seconds", min_value=1, max_value=300, value=10)
    bucket_size = st.sidebar.number_input("Bucket Size", min_value=100, max_value=10000, value=1000)

    # Main display
    st.title("GoldbachX Statistical Heatmap")

    placeholder = st.empty()
    last_update = time.time()

    while True:
        if time.time() - last_update > refresh_sec:
            matrix = generator.build_matrix(bucket=bucket_size)
            df = generator.to_dataframe(matrix)

            with placeholder.container():
                if PANDAS_AVAILABLE:
                    st.write("### Matrix View")
                    st.dataframe(df)

                    if PLOTLY_AVAILABLE:
                        st.write("### Heatmap Visualization")
                        fig = px.imshow(
                            df,
                            labels=dict(x="Metric", y="Module", color="Value"),
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Pandas is required for full visualization")

            last_update = time.time()

        time.sleep(1)

def self_test() -> bool:
    """Run self-test with synthetic data."""
    import tempfile
    import shutil

    print("Running self-test...")
    test_dir = Path(tempfile.mkdtemp())
    results_dir = test_dir / "proof-data"
    telemetry_dir = results_dir / "telemetry"
    os.makedirs(telemetry_dir)

    # Create test data
    test_data = [
        {'module': 'test1', 'n': 1000, 'pair_count': 42, 'failure_rate': 0.1},
        {'module': 'test2', 'n': 2000, 'pair_count': 84, 'failure_rate': 0.05},
    ]

    for i, data in enumerate(test_data):
        with open(results_dir / f"result_{i}.json", 'w') as f:
            json.dump(data, f)

    # Run tests
    generator = StatisticalHeatmapGenerator()
    matrix = generator.build_matrix(str(results_dir), str(telemetry_dir), bucket=500)

    # Validate
    tests_passed = 0
    try:
        assert 'matrix' in matrix
        assert 'test1' in matrix['matrix']
        assert 'pair_count' in matrix['matrix']['test1']
        assert 0 <= matrix['matrix']['test1']['pair_count']['score'] <= 1
        tests_passed += 1

        if PANDAS_AVAILABLE:
            df = generator.to_dataframe(matrix)
            assert isinstance(df, pd.DataFrame)
            tests_passed += 1

        print(f"Self-test passed ({tests_passed}/2 checks)")
        return True
    except AssertionError as e:
        print(f"Self-test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir)

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Statistical Heatmap Generator")
    parser.add_argument('--mode', choices=['cli', 'ui', 'self-test'], default='cli')
    parser.add_argument('--results', default='./proof-data')
    parser.add_argument('--telemetry', default='./proof-data/telemetry')
    parser.add_argument('--bucket', type=int, default=1000)
    parser.add_argument('--export')

    args = parser.parse_args()

    if args.mode == 'cli':
        run_cli(args)
    elif args.mode == 'ui':
        run_ui()
    elif args.mode == 'self-test':
        self_test()

if __name__ == '__main__':
    main()
