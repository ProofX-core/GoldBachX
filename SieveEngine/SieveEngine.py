"""
GoldbachX SieveEngine - Multiple prime sieve strategies with caching.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any, List

# Constants
CACHE_DIR = Path("./.cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
GROUND_TRUTH_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71
]
TELEMETRY_LOG = Path("./sieve_telemetry.jsonl")

def _log_telemetry(event: str, data: Dict[str, Any]) -> None:
    """Log telemetry data to JSONL file."""
    record = {"event": event, "timestamp": time.time(), **data}
    with TELEMETRY_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def _validate_limit(limit: int) -> None:
    """Validate the limit parameter."""
    if limit < 2:
        raise ValueError(f"Limit must be ≥ 2, got {limit}")

def _validate_primes(primes: List[int]) -> None:
    """Validate that primes list is strictly increasing and contains only primes."""
    if not primes:
        raise ValueError("Empty primes list")

    if primes[0] != 2:
        raise ValueError("First prime must be 2")

    for i in range(len(primes) - 1):
        if primes[i] >= primes[i+1]:
            raise ValueError(f"Primes not strictly increasing at index {i}")

    # Check against ground truth for small primes
    for i, p in enumerate(primes):
        if i >= len(GROUND_TRUTH_PRIMES):
            break
        if p != GROUND_TRUTH_PRIMES[i]:
            raise ValueError(f"Prime mismatch at index {i}: expected {GROUND_TRUTH_PRIMES[i]}, got {p}")

def _get_cache_filename(algo: str, limit: int) -> Path:
    """Get the cache filename for given algorithm and limit."""
    return CACHE_DIR / f"primes_{algo}_{limit}.json"

def _cache_is_valid(cache_file: Path, primes: Optional[List[int]] = None) -> bool:
    """Check if cache file is valid."""
    if not cache_file.exists():
        return False

    try:
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cached_primes = data["primes"]
        if not isinstance(cached_primes, list) or not all(isinstance(p, int) for p in cached_primes):
            return False

        # Check metadata
        if data["count"] != len(cached_primes):
            return False

        if cached_primes and (cached_primes[0] != 2 or cached_primes[-1] > data["limit"]):
            return False

        # If primes are provided, check hash
        if primes is not None:
            primes_hash = hashlib.sha256(str(primes).encode()).hexdigest()
            if data["hash"] != primes_hash:
                return False

        return True
    except (json.JSONDecodeError, KeyError, ValueError):
        return False

def _write_cache(cache_file: Path, primes: List[int], limit: int) -> None:
    """Write primes to cache file with validation metadata."""
    data = {
        "primes": primes,
        "count": len(primes),
        "limit": limit,
        "hash": hashlib.sha256(str(primes).encode()).hexdigest(),
        "created": time.time()
    }

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    _log_telemetry("cache_write", {
        "algo": cache_file.stem.split("_")[1],
        "limit": limit,
        "count": len(primes)
    })

def _read_cache(cache_file: Path) -> List[int]:
    """Read primes from cache file."""
    with cache_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["primes"]

def eratosthenes(limit: int) -> List[int]:
    """
    Sieve of Eratosthenes algorithm for finding primes up to limit.

    Args:
        limit: Upper bound for primes (inclusive)

    Returns:
        List of primes ≤ limit
    """
    _validate_limit(limit)
    if limit == 2:
        return [2]

    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"

    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = b"\x00" * ((limit - i*i) // i + 1)

    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    _validate_primes(primes)
    return primes

def atkin(limit: int) -> List[int]:
    """
    Sieve of Atkin algorithm for finding primes up to limit.

    Args:
        limit: Upper bound for primes (inclusive)

    Returns:
        List of primes ≤ limit
    """
    _validate_limit(limit)
    if limit == 2:
        return [2]
    if limit == 3:
        return [2, 3]

    sieve = bytearray([0]) * (limit + 1)

    # Preliminary work
    sqrt_limit = int(math.sqrt(limit)) + 1
    for x in range(1, sqrt_limit):
        for y in range(1, sqrt_limit):
            n = 4*x*x + y*y
            if n <= limit and n % 12 in (1, 5):
                sieve[n] ^= 1

            n = 3*x*x + y*y
            if n <= limit and n % 12 == 7:
                sieve[n] ^= 1

            n = 3*x*x - y*y
            if x > y and n <= limit and n % 12 == 11:
                sieve[n] ^= 1

    # Mark all multiples of squares as non-prime
    for n in range(5, sqrt_limit):
        if sieve[n]:
            sieve[n*n::n*n] = b"\x00" * ((limit - n*n) // (n*n) + 1)

    # Compile results
    primes = [2, 3]
    primes.extend(i for i in range(5, limit+1) if sieve[i])

    _validate_primes(primes)
    return primes

def wheel(limit: int, *, wheel: Tuple[int, ...] = (2, 3, 5)) -> List[int]:
    """
    Wheel sieve algorithm for finding primes up to limit.

    Args:
        limit: Upper bound for primes (inclusive)
        wheel: Wheel basis (default: (2, 3, 5))

    Returns:
        List of primes ≤ limit
    """
    _validate_limit(limit)
    if not wheel:
        raise ValueError("Wheel must not be empty")

    # Handle small limits
    if limit == 2:
        return [2]
    if limit == 3:
        return [2, 3] if 3 in wheel else [2]

    # Generate increments from wheel
    basis = list(sorted(wheel))
    circumference = 1
    for p in basis:
        circumference *= p

    # Generate possible increments
    increments = []
    for i in range(1, circumference + 1):
        if all(i % p != 0 for p in basis):
            increments.append(i)

    # Initialize sieve
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"

    # Sieve using wheel
    for p in basis:
        if p > limit:
            continue
        sieve[p*p::p] = b"\x00" * ((limit - p*p) // p + 1)

    # Sieve remaining numbers using wheel pattern
    sqrt_limit = int(math.sqrt(limit)) + 1
    for n in range(basis[-1] + 1, sqrt_limit):
        if sieve[n]:
            step = next(i for i in increments if (n + i) % circumference != 0)
            sieve[n*n::n] = b"\x00" * ((limit - n*n) // n + 1)

    # Compile results
    primes = [p for p in basis if p <= limit]
    primes.extend(i for i in range(basis[-1] + 1, limit + 1) if sieve[i])

    _validate_primes(primes)
    return primes

def get_primes(limit: int, *, algo: str = "eratosthenes", use_cache: bool = True) -> List[int]:
    """
    Get primes up to limit using specified algorithm, with optional caching.

    Args:
        limit: Upper bound for primes (inclusive)
        algo: Algorithm to use ('eratosthenes', 'atkin', or 'wheel')
        use_cache: Whether to use cache if available

    Returns:
        List of primes ≤ limit
    """
    _validate_limit(limit)

    algo_map = {
        "eratosthenes": eratosthenes,
        "atkin": atkin,
        "wheel": wheel
    }

    if algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(algo_map.keys())}")

    sieve_func = algo_map[algo]
    cache_file = _get_cache_filename(algo, limit)

    primes: List[int] = []
    cache_used = False

    # Try cache first
    if use_cache and cache_file.exists() and _cache_is_valid(cache_file):
        try:
            primes = _read_cache(cache_file)
            cache_used = True
            _log_telemetry("cache_hit", {
                "algo": algo,
                "limit": limit,
                "count": len(primes)
            })
        except (IOError, json.JSONDecodeError, KeyError):
            pass  # Fall through to recompute

    # Compute if cache not available or invalid
    if not primes:
        start_time = time.perf_counter()
        primes = sieve_func(limit) if algo != "wheel" else wheel(limit)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        _log_telemetry("sieve_built", {
            "algo": algo,
            "limit": limit,
            "count": len(primes),
            "time_ms": elapsed_ms
        })

        if use_cache:
            _write_cache(cache_file, primes, limit)

    return primes

def metadata() -> Dict[str, Any]:
    """Return metadata about this engine."""
    return {
        "version": "1.0.0",
        "algorithms": ["eratosthenes", "atkin", "wheel"],
        "cache_location": str(CACHE_DIR.absolute()),
        "max_tested_limit": 10**7
    }

def discover() -> Dict[str, str]:
    """Discovery function for component registration."""
    return {"component": "SieveEngine"}

def _cli() -> None:
    """Command line interface for SieveEngine."""
    parser = argparse.ArgumentParser(description="Prime number sieve generator")
    parser.add_argument("--limit", type=int, required=True, help="Upper limit for primes")
    parser.add_argument("--algo", type=str, default="eratosthenes",
                       choices=["eratosthenes", "atkin", "wheel"],
                       help="Sieve algorithm to use")
    parser.add_argument("--use-cache", type=int, default=1, choices=[0, 1],
                       help="Whether to use cache (1) or not (0)")

    args = parser.parse_args()

    try:
        start_time = time.perf_counter()
        primes = get_primes(args.limit, algo=args.algo, use_cache=bool(args.use_cache))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Prepare summary
        summary = {
            "limit": args.limit,
            "algorithm": args.algo,
            "count": len(primes),
            "first_10": primes[:10],
            "last_10": primes[-10:] if len(primes) > 10 else primes,
            "time_ms": elapsed_ms,
            "cache_used": bool(args.use_cache) and _get_cache_filename(args.algo, args.limit).exists()
        }

        print(json.dumps(summary, indent=2))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import sys
    _cli()
