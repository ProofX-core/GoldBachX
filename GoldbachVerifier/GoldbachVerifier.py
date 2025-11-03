#!/usr/bin/env python3
"""Goldbach's conjecture verifier for even integers n ≥ 4."""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__VERSION__ = "1.0.0"

# Module-level constants
INTERNAL_SIEVES = ["eratosthenes", "atkin", "wheel"]
DEFAULT_SIEVE = "eratosthenes"
MAX_CACHE_SIZE = 10_000_000  # Prevent memory exhaustion
_PRIME_CACHE: Dict[int, List[int]] = {}  # {limit: primes}


def metadata() -> Dict[str, Union[str, bool, List[str]]]:
    """Return module metadata and capabilities."""
    return {
        "component": "GoldbachVerifier",
        "version": __VERSION__,
        "numpy_available": HAS_NUMPY,
        "internal_sieves": INTERNAL_SIEVES,
        "parallel_capable": True,
    }


def _log_event(event_type: str, data: Optional[Dict] = None) -> None:
    """Log telemetry event as JSONL to stdout."""
    event = {"event": event_type, "timestamp": time.time()}
    if data:
        event.update(data)
    print(json.dumps(event), file=sys.stdout)


def _validate_even(n: int, name: str) -> None:
    """Validate that n is even and ≥ 4."""
    if n < 4:
        raise ValueError(f"{name} must be ≥ 4, got {n}")
    if n % 2 != 0:
        raise ValueError(f"{name} must be even, got {n}")


def _internal_sieve(limit: int, algo: str = DEFAULT_SIEVE) -> List[int]:
    """Generate primes up to limit using specified algorithm."""
    if limit < 2:
        return []

    if algo not in INTERNAL_SIEVES:
        raise ValueError(f"Unknown sieve algorithm: {algo}. Choose from {INTERNAL_SIEVES}")

    if algo == "eratosthenes":
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i :: i] = [False] * len(sieve[i*i :: i])
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    elif algo == "atkin":
        # Sieve of Atkin implementation
        sieve = [False] * (limit + 1)
        for x in range(1, int(math.sqrt(limit)) + 1):
            for y in range(1, int(math.sqrt(limit)) + 1):
                n = 4*x*x + y*y
                if n <= limit and n % 12 in (1, 5):
                    sieve[n] ^= True
                n = 3*x*x + y*y
                if n <= limit and n % 12 == 7:
                    sieve[n] ^= True
                n = 3*x*x - y*y
                if x > y and n <= limit and n % 12 == 11:
                    sieve[n] ^= True
        for r in range(5, int(math.sqrt(limit)) + 1):
            if sieve[r]:
                sieve[r*r :: r*r] = [False] * len(sieve[r*r :: r*r])
        primes = [2, 3] + [i for i, is_prime in enumerate(sieve) if is_prime and i >= 5]
    else:  # wheel factorization
        if limit < 2:
            return []
        primes = [2]
        # Start with 3 and use a 2-3 wheel
        sieve = [True] * (limit + 1)
        for i in range(3, int(math.sqrt(limit)) + 1, 2):
            if sieve[i]:
                primes.append(i)
                sieve[i*i :: 2*i] = [False] * len(sieve[i*i :: 2*i])
        for i in range(int(math.sqrt(limit)) + 1, limit + 1, 2):
            if sieve[i]:
                primes.append(i)
    return primes


def _get_primes_up_to(limit: int, sieve: Optional[Callable[[int], List[int]]] = None) -> List[int]:
    """Get primes up to limit, using cache or provided sieve."""
    if limit in _PRIME_CACHE:
        _log_event("cache_hit", {"limit": limit, "cache_size": len(_PRIME_CACHE)})
        return _PRIME_CACHE[limit]

    if sieve is not None:
        primes = sieve(limit)
    else:
        primes = _internal_sieve(limit)

    if limit <= MAX_CACHE_SIZE:
        _PRIME_CACHE[limit] = primes

    return primes


def verify(n: int, sieve: Optional[Callable[[int], List[int]]] = None, *, dedup: bool = True) -> Dict:
    """
    Verify Goldbach's conjecture for a given even integer n ≥ 4.

    Args:
        n: Even integer to verify (≥ 4)
        sieve: Optional sieve function (limit -> list of primes)
        dedup: If True, ensure p ≤ q and remove duplicates

    Returns:
        Dictionary with verification results and metrics
    """
    _validate_even(n, "n")
    start_time = time.perf_counter()
    ops = 0
    bitlen_max = 0

    primes = _get_primes_up_to(n, sieve)
    primes_set = set(primes)
    pairs = []

    for p in primes:
        ops += 1
        q = n - p
        if q in primes_set:
            if dedup and p > q:
                continue
            pairs.append([p, q])
            current_bitlen = max(p.bit_length(), q.bit_length())
            if current_bitlen > bitlen_max:
                bitlen_max = current_bitlen

    # Ensure unique pairs when dedup is True
    if dedup:
        unique_pairs = []
        seen = set()
        for pair in pairs:
            pair_tuple = tuple(pair)
            if pair_tuple not in seen:
                seen.add(pair_tuple)
                unique_pairs.append(pair)
        pairs = unique_pairs

    largest_prime = max(pairs[-1]) if pairs else 0
    time_ms = (time.perf_counter() - start_time) * 1000

    result = {
        "n": n,
        "pairs": pairs,
        "pairs_count": len(pairs),
        "largest_prime": largest_prime,
        "time_ms": time_ms,
        "ops": ops,
        "bitlen_max": bitlen_max,
        "notes": "Verified" if pairs else "No pairs found",
    }

    _log_event("verified_single", {"n": n, "time_ms": time_ms, "pairs_count": len(pairs)})
    return result


def verify_range(
    start: int,
    end: int,
    sieve: Optional[Callable[[int], List[int]]] = None,
    *,
    step: int = 2,
    parallel: bool = False,
    max_workers: int = 0,
) -> List[Dict]:
    """
    Verify Goldbach's conjecture for a range of even integers.

    Args:
        start: First even integer (≥ 4)
        end: Last even integer (≥ start)
        sieve: Optional sieve function (limit -> list of primes)
        step: Step between numbers (must be even)
        parallel: If True, use parallel processing
        max_workers: Number of workers (0 = auto)

    Returns:
        List of verification results for each number in range
    """
    _validate_even(start, "start")
    _validate_even(end, "end")
    if start > end:
        raise ValueError(f"start ({start}) must be ≤ end ({end})")
    if step % 2 != 0:
        raise ValueError(f"step must be even, got {step}")

    numbers = range(start, end + 1, step)
    results = []
    start_time = time.perf_counter()

    if parallel:
        workers = max_workers if max_workers > 0 else None
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(verify, n, sieve): n for n in numbers}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    n = futures[future]
                    results.append({
                        "n": n,
                        "pairs": [],
                        "pairs_count": 0,
                        "largest_prime": 0,
                        "time_ms": 0,
                        "ops": 0,
                        "bitlen_max": 0,
                        "notes": f"Error: {str(e)}",
                    })
    else:
        for n in numbers:
            results.append(verify(n, sieve))

    total_time_ms = (time.perf_counter() - start_time) * 1000
    _log_event("verified_range", {
        "start": start,
        "end": end,
        "count": len(results),
        "total_time_ms": total_time_ms,
        "parallel": parallel,
        "max_workers": max_workers if parallel else 0,
    })

    return results


def export_results(results: List[Dict], path: str) -> None:
    """Export verification results to JSON file."""
    start_time = time.perf_counter()
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    time_ms = (time.perf_counter() - start_time) * 1000
    _log_event("export_done", {"path": path, "time_ms": time_ms, "count": len(results)})


def _self_test() -> bool:
    """Run self-tests and return True if all pass."""
    test_passed = True

    # Test basic verification
    test_cases = [4, 6, 8, 10]
    for n in test_cases:
        result = verify(n)
        if not result["pairs"]:
            print(f"Self-test failed for n={n}: no pairs found", file=sys.stderr)
            test_passed = False

    # Test dedup
    result = verify(10, dedup=True)
    pairs = result["pairs"]
    if len(pairs) != 2 or [3, 7] not in pairs or [5, 5] not in pairs:
        print("Self-test failed for dedup verification", file=sys.stderr)
        test_passed = False

    # Test range verification
    results = verify_range(4, 10, step=2, parallel=False)
    if len(results) != 4 or any(not r["pairs"] for r in results):
        print("Self-test failed for range verification", file=sys.stderr)
        test_passed = False

    # Test sieve injection
    def test_sieve(limit: int) -> List[int]:
        return [2, 3, 5, 7]
    result = verify(10, sieve=test_sieve)
    if len(result["pairs"]) != 2:
        print("Self-test failed for sieve injection", file=sys.stderr)
        test_passed = False

    return test_passed


def main():
    """Command-line interface for GoldbachVerifier."""
    parser = argparse.ArgumentParser(description="Verify Goldbach's conjecture for even integers.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n", type=int, help="Single even integer to verify (≥4)")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                      help="Range of even integers to verify")
    group.add_argument("--self-test", action="store_true", help="Run self-tests")

    parser.add_argument("--export", type=str, help="Path to export results as JSON")
    parser.add_argument("--algo", choices=INTERNAL_SIEVES, default=DEFAULT_SIEVE,
                      help=f"Sieve algorithm (default: {DEFAULT_SIEVE})")
    parser.add_argument("--parallel", type=int, choices=[0, 1], default=0,
                      help="Use parallel processing (0=off, 1=on)")
    parser.add_argument("--workers", type=int, default=0,
                      help="Number of parallel workers (0=auto)")
    parser.add_argument("--step", type=int, default=2,
                      help="Step between numbers in range (must be even)")
    parser.add_argument("--seed", type=int, help="Random seed (for deterministic results)")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        if HAS_NUMPY:
            np.random.seed(args.seed)

    if args.self_test:
        if _self_test():
            print("All self-tests passed", file=sys.stderr)
            sys.exit(0)
        else:
            print("Some self-tests failed", file=sys.stderr)
            sys.exit(1)

    sieve_func = partial(_internal_sieve, algo=args.algo) if args.algo else None

    if args.n is not None:
        _log_event("start", {"mode": "single", "n": args.n})
        results = [verify(args.n, sieve_func)]
    else:
        start, end = args.range
        _log_event("start", {
            "mode": "range",
            "start": start,
            "end": end,
            "step": args.step,
            "parallel": bool(args.parallel),
            "workers": args.workers if args.parallel else 0,
        })
        results = verify_range(
            start, end, sieve_func,
            step=args.step,
            parallel=bool(args.parallel),
            max_workers=args.workers,
        )

    if args.export:
        export_results(results, args.export)

    # Print results to stdout (non-JSONL)
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
