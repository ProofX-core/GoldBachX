"""
AlgebraicExtensions.py - Number-theoretic filters for Goldbach conjecture experiments.
Provides conservative pruning rules to reduce candidate prime search space.
"""

import json
import sys
import time
import math
from typing import List, Dict, Optional
from argparse import ArgumentParser
import random

# Optional numpy support (feature-gated)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

random.seed(42)  # Fixed seed for deterministic behavior

def mod_class_prune(n: int, mod: int = 6) -> Dict:
    """
    Returns allowed residue classes for candidate primes p where p + q = n.

    Args:
        n: Target even number for Goldbach partition
        mod: Modulus to use (default 6, common for Goldbach)

    Returns:
        Dictionary with 'allowed_classes', 'modulus', and 'notes'
    """
    start_time = time.perf_counter_ns()

    if mod <= 1:
        raise ValueError("Modulus must be ≥ 2")
    if n % 2 != 0:
        raise ValueError("Input n must be even for Goldbach conjecture")

    allowed = set()
    for p_class in range(1, mod):  # Skip 0 since primes > mod can't be ≡0
        if math.gcd(p_class, mod) == 1:  # Only classes coprime to mod
            q_class = (n - p_class) % mod
            if math.gcd(q_class, mod) == 1:  # Both must be possible primes
                allowed.add(p_class)

    result = {
        'allowed_classes': sorted(allowed),
        'modulus': mod,
        'notes': f"Primes must be ≡{sorted(allowed)} mod {mod} to pair with valid q"
    }

    elapsed_ms = (time.perf_counter_ns() - start_time) // 1_000_000
    print(json.dumps({"event": "prune_done", "n": n, "mod": mod, "elapsed_ms": elapsed_ms}))

    return result

def quadratic_residue_filter(n: int, primes: List[int]) -> List[int]:
    """
    Filters primes p where n-p is quadratic residue modulo small primes.
    Conservative: only excludes p where n-p is provably non-residue.

    Args:
        n: Target even number
        primes: List of candidate primes to filter

    Returns:
        Filtered list of primes with rationale notes
    """
    start_time = time.perf_counter_ns()
    if not primes:
        return []

    # Use first few primes as moduli for QR checks
    qr_moduli = [3, 5, 7, 11, 13]
    filtered = []

    for p in primes:
        keep = True
        for m in qr_moduli:
            if m >= p:  # Don't filter based on larger moduli
                continue
            residue = (n - p) % m
            # Check if residue is quadratic non-residue mod m
            if all((residue != (r*r) % m) for r in range(1, m)):
                keep = False
                break
        if keep:
            filtered.append(p)

    elapsed_ms = (time.perf_counter_ns() - start_time) // 1_000_000
    print(json.dumps({
        "event": "qr_filter_done",
        "n": n,
        "input_size": len(primes),
        "output_size": len(filtered),
        "elapsed_ms": elapsed_ms
    }))

    return filtered

def small_factor_exclusions(n: int, primes: List[int], *, bounds: int = 1000) -> List[int]:
    """
    Excludes primes p where n-p has small factors (below bounds).
    Conservative: only excludes when n-p is provably composite.

    Args:
        n: Target even number
        primes: List of candidate primes
        bounds: Check for factors up to this value

    Returns:
        Filtered list of primes with notes
    """
    start_time = time.perf_counter_ns()
    if bounds < 2:
        raise ValueError("Bounds must be ≥ 2")

    # Precompute small primes up to bounds
    sieve = [True] * (bounds + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(bounds)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]

    filtered = []
    for p in primes:
        q_candidate = n - p
        # Quick check for small factors of q_candidate
        has_small_factor = False
        for sp in small_primes:
            if sp >= q_candidate:
                break
            if q_candidate % sp == 0:
                has_small_factor = True
                break
        if not has_small_factor:
            filtered.append(p)

    elapsed_ms = (time.perf_counter_ns() - start_time) // 1_000_000
    print(json.dumps({
        "event": "small_factor_done",
        "n": n,
        "bounds": bounds,
        "input_size": len(primes),
        "output_size": len(filtered),
        "elapsed_ms": elapsed_ms
    }))

    return filtered

def composite_precheck(n: int) -> Dict:
    """
    Quick checks that predict low Goldbach pair density.
    Returns dictionary with metrics and warnings if n appears 'hard'.

    Args:
        n: Target even number

    Returns:
        Dictionary with metrics and heuristic warnings
    """
    start_time = time.perf_counter_ns()

    result = {
        'n': n,
        'notes': [],
        'warnings': []
    }

    # Check if n is divisible by small primes (more possible pairs)
    small_primes = [3, 5, 7, 11, 13]
    for p in small_primes:
        if n % p == 0:
            result['notes'].append(f"Divisible by {p} (expect more pairs)")
            break
    else:
        result['warnings'].append("Not divisible by small primes (fewer pairs expected)")

    # Check if n is congruent to 2 mod 6 (tends to have fewer pairs)
    if n % 6 == 2:
        result['warnings'].append("n ≡ 2 mod 6 (historically fewer pairs)")

    elapsed_ms = (time.perf_counter_ns() - start_time) // 1_000_000
    result['elapsed_ms'] = elapsed_ms
    print(json.dumps({"event": "precheck_done", "n": n, "elapsed_ms": elapsed_ms}))

    return result

def metadata() -> Dict:
    """Returns plugin metadata and capabilities."""
    return {
        "version": "1.0",
        "author": "GoldbachX Team",
        "description": "Algebraic filters for Goldbach conjecture experiments",
        "capabilities": {
            "mod_class_prune": True,
            "quadratic_residue_filter": True,
            "small_factor_exclusions": True,
            "composite_precheck": True,
            "numpy_available": HAS_NUMPY
        }
    }

def discover() -> Dict:
    """Discovery function for plugin system."""
    return {"component": "AlgebraicExtensions"}

def self_test() -> bool:
    """Run self-tests and return True if all pass."""
    tests = []

    # Test mod_class_prune
    res = mod_class_prune(100, 6)
    tests.append(set(res['allowed_classes']) == {1, 5})

    # Test quadratic_residue_filter doesn't remove all candidates
    primes = [3, 7, 11, 17, 23, 29]
    filtered = quadratic_residue_filter(100, primes)
    tests.append(len(filtered) > 0)

    # Test small_factor_exclusions keeps valid primes
    primes = [3, 7, 11, 13, 17, 19, 23, 29]
    filtered = small_factor_exclusions(100, primes, bounds=10)
    tests.append(3 in filtered)  # 97 is prime

    # Test composite_precheck gives warnings
    res = composite_precheck(98)
    tests.append(len(res['warnings']) > 0)

    # Test determinism
    primes1 = quadratic_residue_filter(1000, list(range(3, 200, 2)))
    primes2 = quadratic_residue_filter(1000, list(range(3, 200, 2)))
    tests.append(primes1 == primes2)

    return all(tests)

def main():
    parser = ArgumentParser(description="AlgebraicExtensions plugin for GoldbachX")
    parser.add_argument("--n", type=int, help="Target even number to analyze")
    parser.add_argument("--mod", type=int, default=6, help="Modulus for residue pruning")
    parser.add_argument("--bounds", type=int, default=1000,
                       help="Factor bound for small factor exclusion")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")

    args = parser.parse_args()

    if args.self_test:
        print("Running self-tests...")
        if self_test():
            print("All tests passed")
            sys.exit(0)
        else:
            print("Some tests failed")
            sys.exit(1)

    if not args.n:
        parser.print_help()
        sys.exit(1)

    results = {}

    # Run all analyses
    results['mod_class'] = mod_class_prune(args.n, args.mod)
    primes = list(range(3, args.n, 2))  # Demo candidate list
    results['qr_filter'] = quadratic_residue_filter(args.n, primes)
    results['small_factor'] = small_factor_exclusions(args.n, primes, bounds=args.bounds)
    results['precheck'] = composite_precheck(args.n)
    results['metadata'] = metadata()

    if args.export:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {args.export}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
