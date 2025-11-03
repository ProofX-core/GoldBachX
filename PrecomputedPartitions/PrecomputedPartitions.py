#!/usr/bin/env python3
"""
GoldbachX Precomputed Partitions Manager
Handles building, caching, and querying prime-pair decompositions for even numbers.
"""

import argparse
import json
import lzma
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import math
from datetime import datetime

# Constants
DEFAULT_ROOT = "./datasets/precomputed_partitions"
VERSION = "1.0"
COMPRESS_EXT = ".xz"

# Type aliases
PrimePair = Tuple[int, int]
Dataset = Dict[str, Union[Dict, List]]


def _log_event(event: str, **kwargs) -> None:
    """Log telemetry events as JSONL to stdout."""
    payload = {"event": event, "timestamp": datetime.utcnow().isoformat()}
    payload.update(kwargs)
    print(json.dumps(payload), file=sys.stdout)


def _validate_even_range(start: int, end: int) -> None:
    """Ensure range boundaries are even and valid."""
    if start < 4:
        raise ValueError("Start must be at least 4 (first even prime pair)")
    if end <= start:
        raise ValueError("End must be greater than start")
    if start % 2 != 0 or end % 2 != 0:
        raise ValueError("Range boundaries must be even numbers")


def _sieve(limit: int) -> List[bool]:
    """Sieve of Eratosthenes to find primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for num in range(2, int(math.sqrt(limit)) + 1):
        if sieve[num]:
            sieve[num*num : limit+1 : num] = [False] * len(sieve[num*num : limit+1 : num])
    return sieve


def _generate_primes_up_to(n: int) -> List[int]:
    """Generate all primes up to n using Sieve of Eratosthenes."""
    sieve = _sieve(n)
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def _find_prime_pairs(n: int, primes: List[int], unique: bool = True) -> List[PrimePair]:
    """Find all prime pairs (p, q) such that p + q = n."""
    pairs = []
    prime_set = set(primes)
    for p in primes:
        if p > n // 2 and unique:
            break
        q = n - p
        if q in prime_set:
            if unique and p > q:
                continue
            pairs.append((p, q))
    return pairs


def build_range(
    start: int,
    end: int,
    *,
    algo: str = "eratosthenes",
    chunk: int = 10000,
    seed: Optional[int] = None,
) -> Dataset:
    """
    Build prime-pair decompositions for even numbers in [start, end].

    Args:
        start: First even number (inclusive)
        end: Last even number (inclusive)
        algo: Prime generation algorithm (only 'eratosthenes' supported)
        chunk: Processing chunk size for memory management
        seed: Optional random seed for deterministic output

    Returns:
        Dataset dictionary with meta, index, and data
    """
    _validate_even_range(start, end)

    if seed is not None:
        random.seed(seed)

    _log_event("build_started", start=start, end=end, algo=algo, chunk=chunk)

    # Generate all primes up to end (we'll need them all anyway)
    primes = _generate_primes_up_to(end)
    prime_set = set(primes)

    dataset = {
        "meta": {
            "version": VERSION,
            "built_at": datetime.utcnow().isoformat(),
            "algo": algo,
            "start": start,
            "end": end,
            "seed": seed,
        },
        "index": {},
        "data": [],
    }

    current_chunk = []
    chunk_start = time.time()

    for n in range(start, end + 1, 2):
        pairs = _find_prime_pairs(n, primes)
        offset = len(dataset["data"])
        dataset["index"][str(n)] = (offset, len(pairs))
        dataset["data"].extend(pairs)

        # Chunk processing telemetry
        if n % chunk == 0 or n == end:
            chunk_time = time.time() - chunk_start
            _log_event(
                "chunk_done",
                current=n,
                pairs=len(pairs),
                chunk_time=chunk_time,
                total_pairs=len(dataset["data"]),
            )
            chunk_start = time.time()

    _log_event("build_complete", total_evens=len(dataset["index"]), total_pairs=len(dataset["data"]))
    return dataset


def save(dataset: Dataset, path: str) -> None:
    """
    Save dataset to file with optional compression.

    Args:
        dataset: Dataset dictionary to save
        path: Destination path (compression inferred from extension)
    """
    tmp_path = f"{path}.tmp"
    try:
        if path.endswith(COMPRESS_EXT):
            with lzma.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(dataset, f)
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f)

        # Atomic write via rename
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise IOError(f"Failed to save dataset: {str(e)}") from e


def load(path: str) -> Dataset:
    """
    Load dataset from file with optional decompression.

    Args:
        path: Source path (compression inferred from extension)

    Returns:
        Loaded dataset dictionary
    """
    try:
        if path.endswith(COMPRESS_EXT):
            with lzma.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load dataset: {str(e)}") from e


def query(dataset: Dataset, n: int, *, unique: bool = True) -> List[PrimePair]:
    """
    Query prime pairs for a specific even number.

    Args:
        dataset: Loaded dataset dictionary
        n: Even number to query
        unique: If True, return only pairs with p <= q

    Returns:
        List of prime pairs (p, q) where p + q = n
    """
    if n % 2 != 0:
        raise ValueError("Query number must be even")

    n_str = str(n)
    if n_str not in dataset["index"]:
        raise ValueError(f"Number {n} not found in dataset (range {dataset['meta']['start']}-{dataset['meta']['end']})")

    offset, length = dataset["index"][n_str]
    pairs = dataset["data"][offset : offset + length]

    if unique:
        # Ensure p <= q (should already be true from build)
        pairs = [(p, q) for p, q in pairs if p <= q]

    _log_event("query_ok", n=n, pairs=len(pairs))
    return pairs


def stats(dataset: Dataset) -> Dict:
    """
    Compute statistics about the dataset.

    Args:
        dataset: Loaded dataset dictionary

    Returns:
        Dictionary of statistics including counts, ranges, and averages
    """
    index = dataset["index"]
    data = dataset["data"]
    meta = dataset["meta"]

    # Calculate average pairs per even without scanning all data
    total_pairs = len(data)
    total_evens = len(index)

    stats = {
        "version": meta["version"],
        "start": meta["start"],
        "end": meta["end"],
        "total_evens": total_evens,
        "total_pairs": total_pairs,
        "avg_pairs": total_pairs / total_evens if total_evens else 0,
        "built_at": meta["built_at"],
        "algo": meta["algo"],
    }

    _log_event("stats_ok", **stats)
    return stats


def metadata() -> Dict:
    """Return component metadata."""
    return {
        "component": "PrecomputedPartitions",
        "version": VERSION,
        "description": "Manages precomputed Goldbach partitions for even numbers",
    }


def discover() -> Dict:
    """Discover component information (alias for metadata)."""
    return metadata()


def _run_self_test() -> bool:
    """Run self-tests and return True if all passed."""
    test_ok = True

    try:
        # Test small range build
        test_ds = build_range(4, 200, algo="eratosthenes", seed=42)

        # Test known queries
        assert len(query(test_ds, 10)) == 2  # (3,7), (5,5)
        assert len(query(test_ds, 16)) == 2  # (3,13), (5,11)

        # Test save/load roundtrip
        test_path = "test_dataset.json"
        save(test_ds, test_path)
        loaded_ds = load(test_path)
        assert loaded_ds["meta"]["start"] == test_ds["meta"]["start"]
        assert len(loaded_ds["data"]) == len(test_ds["data"])
        os.unlink(test_path)

        # Test deterministic output with seed
        ds1 = build_range(4, 100, seed=123)
        ds2 = build_range(4, 100, seed=123)
        assert ds1["data"] == ds2["data"]

    except AssertionError as e:
        print(f"Self-test failed: {str(e)}", file=sys.stderr)
        test_ok = False

    _log_event("self_test_complete", passed=test_ok)
    return test_ok


def _parse_cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GoldbachX Precomputed Partitions Manager")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build mode
    build_parser = subparsers.add_parser("build", help="Build a new dataset")
    build_parser.add_argument("--range", nargs=2, type=int, required=True, metavar=("START", "END"), help="Even number range")
    build_parser.add_argument("--algo", default="eratosthenes", choices=["eratosthenes"], help="Prime generation algorithm")
    build_parser.add_argument("--chunk", type=int, default=10000, help="Processing chunk size")
    build_parser.add_argument("--seed", type=int, default=None, help="Random seed for determinism")
    build_parser.add_argument("--out", required=True, help="Output file path")

    # Query mode
    query_parser = subparsers.add_parser("query", help="Query a dataset")
    query_parser.add_argument("--in", dest="in_file", required=True, help="Input dataset path")
    query_parser.add_argument("--n", type=int, required=True, help="Even number to query")

    # Stats mode
    stats_parser = subparsers.add_parser("stats", help="Get dataset statistics")
    stats_parser.add_argument("--in", dest="in_file", required=True, help="Input dataset path")

    # Self-test mode
    subparsers.add_parser("self-test", help="Run self-tests")

    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = _parse_cli()

    if args.mode == "self-test":
        if not _run_self_test():
            sys.exit(1)
        return

    if args.mode == "build":
        try:
            dataset = build_range(
                start=args.range[0],
                end=args.range[1],
                algo=args.algo,
                chunk=args.chunk,
                seed=args.seed,
            )
            save(dataset, args.out)
            print(f"Successfully built dataset to {args.out}", file=sys.stderr)
        except Exception as e:
            print(f"Build failed: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "query":
        try:
            dataset = load(args.in_file)
            pairs = query(dataset, args.n)
            print(json.dumps({"n": args.n, "pairs": pairs}))
        except Exception as e:
            print(f"Query failed: {str(e)}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "stats":
        try:
            dataset = load(args.in_file)
            print(json.dumps(stats(dataset), indent=2))
        except Exception as e:
            print(f"Stats failed: {str(e)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
