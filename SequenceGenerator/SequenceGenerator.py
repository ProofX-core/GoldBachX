"""
GoldbachX Sequence Generator - Produces even number sequences for conjecture verification.
"""

import argparse
import json
import math
import random
import sys
import time
from typing import Optional, Union, List, Dict
from numpy import isin  # type: ignore

# Feature gate for numpy
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def metadata() -> Dict:
    """Return component metadata."""
    return {
        "component": "SequenceGenerator",
        "version": "1.0",
        "author": "GoldbachX Team",
        "dependencies": {"numpy": HAS_NUMPY},
    }


def describe_modes() -> Dict:
    """Documentation for all supported sequence generation modes."""
    return {
        "even": "All even numbers in [start, end] range (inclusive)",
        "random-sample": "k distinct even numbers sampled uniformly from [start, end]",
        "twin-adjacent": "Even numbers n where at least one Goldbach partition uses twin primes",
        "large-gap": "Even numbers near known large prime gaps (heuristic based on density drops)",
    }


def generate_sequence(
    mode: str = "even",
    *,
    start: int = 4,
    end: int = 100,
    k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Generate a sequence of even numbers based on the specified mode.

    Args:
        mode: Generation mode (see describe_modes())
        start: First even number in range (inclusive, must be ≥4 and even)
        end: Last even number in range (inclusive, must be ≥start and even)
        k: Sample size for 'random-sample' mode
        seed: Optional random seed for reproducibility

    Returns:
        Sorted list of unique even integers meeting the criteria
    """
    # Validate common parameters
    if start < 4:
        raise ValueError("Start must be ≥4")
    if end < start:
        raise ValueError("End must be ≥ start")
    if mode == "random-sample" and k is None:
        raise ValueError("k must be specified for random-sample mode")
    if mode == "random-sample" and k <= 0:
        raise ValueError("k must be positive")

    # Ensure even alignment for modes that require it
    if mode in ["even", "random-sample"]:
        if start % 2 != 0:
            start += 1
            print(f"Warning: Adjusted start to next even number: {start}", file=sys.stderr)
        if end % 2 != 0:
            end -= 1
            print(f"Warning: Adjusted end to previous even number: {end}", file=sys.stderr)

    # Initialize random seed if provided
    if seed is not None:
        random.seed(seed)
        if HAS_NUMPY:
            np.random.seed(seed)

    # Record start time for telemetry
    start_time = time.time()

    # Dispatch to appropriate generator
    if mode == "even":
        sequence = _generate_even(start, end)
    elif mode == "random-sample":
        sequence = _generate_random_sample(start, end, k)
    elif mode == "twin-adjacent":
        sequence = _generate_twin_adjacent(start, end)
    elif mode == "large-gap":
        sequence = _generate_large_gap(start, end)
    else:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(describe_modes().keys())}")

    # Ensure sorted and unique (some generators might produce duplicates)
    sequence = sorted(list(set(sequence)))

    # Record telemetry
    duration_ms = int((time.time() - start_time) * 1000)
    _log_telemetry(mode, len(sequence), duration_ms)

    return sequence


def _generate_even(start: int, end: int) -> List[int]:
    """Generate all even numbers in [start, end] range."""
    return list(range(start, end + 1, 2))


def _generate_random_sample(start: int, end: int, k: int) -> List[int]:
    """Generate k distinct random even numbers from [start, end]."""
    if k > (end - start) // 2 + 1:
        raise ValueError(f"Cannot sample {k} unique evens from range {start}-{end}")

    population = list(range(start, end + 1, 2))
    if HAS_NUMPY:
        sample = np.random.choice(population, size=k, replace=False)
        return sorted([int(x) for x in sample])
    else:
        return sorted(random.sample(population, k))


def _generate_twin_adjacent(start: int, end: int) -> List[int]:
    """
    Generate even numbers where at least one Goldbach partition uses twin primes.
    Uses a simple sieve to find primes up to end, then checks for twin prime pairs.
    """
    if end < 6:  # No twin primes below 6
        return []

    # Sieve of Eratosthenes up to end
    sieve = [True] * (end + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(end)) + 1):
        if sieve[i]:
            sieve[i*i :: i] = [False] * len(sieve[i*i :: i])

    primes = [i for i, is_prime in enumerate(sieve) if is_prime]

    # Identify twin primes (p, p+2 both prime)
    twin_primes = set()
    for i in range(len(primes) - 1):
        if primes[i+1] == primes[i] + 2:
            twin_primes.add(primes[i])
            twin_primes.add(primes[i+1])

    # Generate even numbers that can be expressed as sum of twin primes
    result = set()
    for p in twin_primes:
        for q in twin_primes:
            if p + q >= start and p + q <= end and (p + q) % 2 == 0:
                result.add(p + q)

    return sorted(result)


def _generate_large_gap(start: int, end: int) -> List[int]:
    """
    Generate even numbers near known large prime gaps (heuristic).
    Uses a windowed approach to find regions of lower prime density.
    """
    if end < 100:  # Not meaningful for small ranges
        return _generate_even(start, end)

    # Simple heuristic: look for regions with fewer primes in a window
    window_size = max(100, end // 100)
    sieve = [True] * (end + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(end)) + 1):
        if sieve[i]:
            sieve[i*i :: i] = [False] * len(sieve[i*i :: i])

    # Calculate prime counts in sliding windows
    prime_counts = []
    for i in range(start, end - window_size + 1):
        count = sum(sieve[i:i+window_size])
        prime_counts.append((i, count))

    # Find windows with lowest prime density
    prime_counts.sort(key=lambda x: x[1])
    candidate_regions = [x[0] + window_size // 2 for x in prime_counts[:10]]

    # Generate even numbers around these regions
    result = set()
    for center in candidate_regions:
        lower = max(start, center - window_size)
        upper = min(end, center + window_size)
        result.update(range(lower, upper + 1, 2))

    return sorted(result)


def _log_telemetry(mode: str, count: int, time_ms: int) -> None:
    """Log generation telemetry in JSONL format."""
    log_entry = {
        "event": "sequence_generated",
        "timestamp": time.time(),
        "mode": mode,
        "count": count,
        "time_ms": time_ms,
    }
    print(json.dumps(log_entry), file=sys.stderr)


def discover() -> Dict:
    """Discovery function for component registration."""
    return {"component": "SequenceGenerator"}


def main() -> None:
    """Command line interface for sequence generation."""
    parser = argparse.ArgumentParser(
        description="GoldbachX Sequence Generator - Produce even number sequences for conjecture verification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="even",
        choices=describe_modes().keys(),
        help="Sequence generation mode",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=4,
        help="First even number in range (inclusive, must be ≥4)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=100,
        help="Last even number in range (inclusive, must be ≥start)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Sample size for random-sample mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available generation modes and exit",
    )

    args = parser.parse_args()

    if args.list_modes:
        print("Available generation modes:")
        for mode, desc in describe_modes().items():
            print(f"  {mode}: {desc}")
        return

    try:
        sequence = generate_sequence(
            mode=args.mode,
            start=args.start,
            end=args.end,
            k=args.k,
            seed=args.seed,
        )

        # Pretty print results
        print(f"Generated {len(sequence)} numbers in {args.mode} mode:")
        if len(sequence) <= 20:
            print(sequence)
        else:
            print(f"First 10: {sequence[:10]}")
            print(f"Last 10: {sequence[-10:]}")
            print(f"Full range: {sequence[0]} to {sequence[-1]}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
