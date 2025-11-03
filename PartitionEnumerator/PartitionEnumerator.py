#!/usr/bin/env python3
"""
GoldbachX Partition Enumerator - Finds prime pairs summing to n.
"""

import json
import sys
import time
from typing import Optional, Tuple, Dict, List
import argparse
import numpy as np  # type: ignore

# Feature gate for numpy
HAS_NUMPY = True
try:
    import numpy as np  # type: ignore
except ImportError:
    HAS_NUMPY = False


def discover() -> Dict[str, str]:
    """Component discovery."""
    return {"component": "PartitionEnumerator"}


def metadata() -> Dict[str, str]:
    """Return metadata about this component."""
    return {
        "version": "1.0.0",
        "author": "GoldbachX Team",
        "description": "Enumerates prime pairs summing to n",
        "dependencies": {"numpy": "optional"},
    }


def filters_signature() -> Dict[str, Tuple[str, type, object]]:
    """Return supported filters and their metadata."""
    return {
        "allow_equal": ("Allow p=q pairs", bool, True),
        "exclude_twins": ("Exclude twin primes", bool, False),
        "unique": ("Return unique pairs only", bool, True),
    }


def _validate_input(n: int, primes: List[int]) -> None:
    """Validate input parameters."""
    if n < 4 or n % 2 != 0:
        raise ValueError("n must be an even integer ≥ 4")
    if not primes:
        raise ValueError("Primes list cannot be empty")
    if primes[0] < 2:
        raise ValueError("Primes must be ≥ 2")
    if primes[-1] > n:
        raise ValueError("All primes must be ≤ n")
    if not all(p < q for p, q in zip(primes, primes[1:])):
        raise ValueError("Primes must be sorted and unique")


def _is_twin_prime(p: int, q: int, primes_set: set) -> bool:
    """Check if (p,q) are twin primes."""
    return abs(p - q) == 2 and (p + 2 in primes_set or q + 2 in primes_set)


def enumerate_partitions(
    n: int,
    primes: List[int],
    *,
    allow_equal: bool = True,
    exclude_twins: bool = False,
    unique: bool = True,
) -> List[Tuple[int, int]]:
    """
    Enumerate all prime pairs (p,q) with p + q = n.

    Args:
        n: Target even number ≥4
        primes: Sorted list of primes ≤n
        allow_equal: Allow p=q pairs (default True)
        exclude_twins: Exclude twin prime pairs (default False)
        unique: Return only pairs with p≤q (default True)

    Returns:
        List of prime pairs meeting criteria, ordered by p then q
    """
    _validate_input(n, primes)

    metrics = {
        "comparisons": 0,
        "lookups": 0,
        "candidates_checked": 0,
    }

    primes_set = set(primes)
    result = []
    seen = set()

    # Use numpy if available for faster operations
    if HAS_NUMPY:
        primes_arr = np.array(primes)
        complements = n - primes_arr
        valid_mask = np.isin(complements, primes_arr)
        candidates = primes_arr[valid_mask]
    else:
        candidates = [p for p in primes if (n - p) in primes_set]

    metrics["candidates_checked"] = len(candidates)

    for p in candidates:
        q = n - p
        metrics["comparisons"] += 1

        if not allow_equal and p == q:
            continue

        if exclude_twins and _is_twin_prime(p, q, primes_set):
            continue

        if unique:
            pair = (p, q) if p <= q else (q, p)
        else:
            pair = (p, q)

        if unique and pair in seen:
            continue

        seen.add(pair)
        result.append(pair)

    metrics["lookups"] = len(primes_set) * 2

    # Sort by p then q
    result.sort()
    return result


def count_partitions(n: int, primes: List[int], **filters) -> int:
    """
    Count prime pairs summing to n with optional filters.

    Args:
        n: Target even number ≥4
        primes: Sorted list of primes ≤n
        **filters: Same as enumerate_partitions

    Returns:
        Count of prime pairs meeting criteria
    """
    return len(enumerate_partitions(n, primes, **filters))


def _cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Goldbach Partition Enumerator")
    parser.add_argument("--n", type=int, required=True, help="Target even number ≥4")
    parser.add_argument(
        "--allow-equal",
        type=int,
        choices=[0, 1],
        default=1,
        help="Allow p=q pairs (default 1)",
    )
    parser.add_argument(
        "--exclude-twins",
        type=int,
        choices=[0, 1],
        default=0,
        help="Exclude twin primes (default 0)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Generate primes up to n (simplified for CLI)
    if args.n < 2:
        primes = []
    else:
        sieve = [True] * (args.n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(args.n**0.5) + 1):
            if sieve[i]:
                sieve[i*i :: i] = [False] * len(sieve[i*i :: i])
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    try:
        pairs = enumerate_partitions(
            args.n,
            primes,
            allow_equal=bool(args.allow_equal),
            exclude_twins=bool(args.exclude_twins),
            unique=True,
        )
        output = {
            "n": args.n,
            "pairs": pairs,
            "count": len(pairs),
            "metrics": {
                "primes_count": len(primes),
                "elapsed_ms": int((time.time() - start_time) * 1000),
            },
        }
        print(json.dumps(output))
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
