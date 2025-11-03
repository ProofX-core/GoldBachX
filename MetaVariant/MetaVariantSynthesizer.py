"""
MetaVariantSynthesizer - Generates and evaluates Goldbach conjecture variants.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union
import itertools
import numpy as np

# Optional Streamlit support (feature-gated)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Constants
DEFAULT_RANGE_START = 4
DEFAULT_RANGE_END = 100_000
DEFAULT_BUDGET_MS = 2000
PRIMES_UP_TO = 1_000_000  # Precompute primes up to this limit

# Precomputed primes (lazy-loaded)
_PRIMES: List[int] = []
_PRIME_SET: Set[int] = set()

class DSLSyntaxError(Exception):
    """Custom exception for DSL parsing errors."""
    pass

@dataclass
class VariantTemplate:
    """Represents a parameterized Goldbach variant template."""
    name: str
    dsl_pattern: str
    parameters: Dict[str, Tuple[type, List[Union[int, str]]]]
    description: str

# Core templates
TEMPLATES = [
    VariantTemplate(
        name="mod_constraint",
        dsl_pattern="Every even n ≥ {N0} is sum of two primes with p mod {m} ∈ {S}",
        parameters={
            "N0": (int, [4, 6, 8, 10]),
            "m": (int, [3, 4, 5, 6, 7, 8]),
            "S": (str, ["{1}", "{1,2}", "{1,3,5}", "{0,1}"])
        },
        description="Goldbach with modular constraints on one prime"
    ),
    VariantTemplate(
        name="k_decompositions",
        dsl_pattern="At least {k} distinct decompositions for evens in [{a}, {b}]",
        parameters={
            "k": (int, [1, 2, 3, 5, 10]),
            "a": (int, [4, 100, 1000, 10_000]),
            "b": (int, [100, 1000, 10_000, 100_000])
        },
        description="Minimum number of Goldbach pairs required"
    ),
    VariantTemplate(
        name="interval_constraint",
        dsl_pattern="One prime in each pair lies in [{c}n^{d}, {C}n^{D}]",
        parameters={
            "c": (float, [0.1, 0.5, 1.0, 2.0]),
            "d": (float, [0.8, 0.9, 1.0, 1.1]),
            "C": (float, [0.5, 1.0, 2.0, 3.0]),
            "D": (float, [0.9, 1.0, 1.1])
        },
        description="Prime must lie in scaled interval"
    )
]

def _ensure_primes() -> None:
    """Lazy-load primes using Sieve of Eratosthenes."""
    global _PRIMES, _PRIME_SET
    if not _PRIMES:
        sieve = [True] * (PRIMES_UP_TO + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(PRIMES_UP_TO)) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        _PRIMES = [i for i, is_prime in enumerate(sieve) if is_prime]
        _PRIME_SET = set(_PRIMES)

@lru_cache(maxsize=100_000)
def is_prime(n: int) -> bool:
    """Check if a number is prime (cached)."""
    if n > PRIMES_UP_TO:
        raise ValueError(f"Primes only precomputed up to {PRIMES_UP_TO}")
    _ensure_primes()
    return n in _PRIME_SET

def goldbach_pairs(n: int) -> List[Tuple[int, int]]:
    """Return all Goldbach pairs for even n >= 4."""
    if n % 2 != 0 or n < 4:
        return []

    _ensure_primes()
    pairs = []
    for p in _PRIMES:
        if p > n // 2:
            break
        q = n - p
        if is_prime(q):
            pairs.append((p, q))
    return pairs

def generate_templates(seed: Optional[int] = None) -> List[str]:
    """Return all available template DSL strings."""
    return [t.dsl_pattern for t in TEMPLATES]

def instantiate(tmpl: str, params: Dict[str, Union[int, str, float]]) -> Dict:
    """
    Instantiate a template with parameters.

    Returns:
        dict: {"statement": str, "params": dict}
    """
    try:
        template = next(t for t in TEMPLATES if t.dsl_pattern == tmpl)
    except StopIteration:
        raise DSLSyntaxError(f"Unknown template: {tmpl}")

    # Validate parameters
    for param_name, (param_type, allowed_values) in template.parameters.items():
        if param_name not in params:
            raise DSLSyntaxError(f"Missing parameter: {param_name}")
        if not isinstance(params[param_name], param_type):
            raise DSLSyntaxError(f"Parameter {param_name} must be {param_type}")
        if allowed_values and params[param_name] not in allowed_values:
            raise DSLSyntaxError(f"Invalid value for {param_name}. Allowed: {allowed_values}")

    # Format the statement
    statement = tmpl.format(**params)
    return {"statement": statement, "params": params}

def evaluate(
    stmt: str,
    params: Dict,
    *,
    start: int = DEFAULT_RANGE_START,
    end: int = DEFAULT_RANGE_END,
    budget_ms: int = DEFAULT_BUDGET_MS
) -> Dict:
    """
    Evaluate a variant statement against empirical data.

    Returns:
        dict: {
            "support": float,
            "simplicity": float,
            "novelty": float,
            "rationale": str,
            "tested_up_to": int,
            "time_ms": float
        }
    """
    start_time = time.time()
    template = next(t for t in TEMPLATES if t.dsl_pattern in stmt)
    tested = 0
    satisfied = 0
    total = 0

    # Evaluation logic per template type
    if template.name == "mod_constraint":
        N0 = params["N0"]
        m = params["m"]
        S = eval(params["S"])  # Safe because we control the DSL
        S = set(S)

        for n in range(start, end + 1, 2):
            if n < N0:
                continue
            if time.time() - start_time > budget_ms / 1000:
                break
            pairs = goldbach_pairs(n)
            if not pairs:
                continue
            total += 1
            has_valid_pair = any(p % m in S for p, _ in pairs) or any(q % m in S for _, q in pairs)
            if has_valid_pair:
                satisfied += 1
            tested += 1

    elif template.name == "k_decompositions":
        k = params["k"]
        a = params["a"]
        b = params["b"]

        for n in range(a, b + 1, 2):
            if time.time() - start_time > budget_ms / 1000:
                break
            pairs = goldbach_pairs(n)
            total += 1
            if len(pairs) >= k:
                satisfied += 1
            tested += 1

    elif template.name == "interval_constraint":
        c = params["c"]
        d = params["d"]
        C = params["C"]
        D = params["D"]

        for n in range(start, end + 1, 2):
            if time.time() - start_time > budget_ms / 1000:
                break
            pairs = goldbach_pairs(n)
            if not pairs:
                continue
            total += 1
            lower = c * (n ** d)
            upper = C * (n ** D)
            has_valid_pair = any(lower <= p <= upper or lower <= q <= upper for p, q in pairs)
            if has_valid_pair:
                satisfied += 1
            tested += 1

    # Calculate scores
    support = satisfied / max(1, total)
    simplicity = 1 / (1 + sum(len(str(v)) for v in params.values()))  # MDL-like
    novelty = 1 - 0.5 * (template.name == "mod_constraint")  # Simple novelty proxy

    return {
        "support": support,
        "simplicity": simplicity,
        "novelty": novelty,
        "rationale": f"Tested {tested} evens: {satisfied} satisfied ({support:.1%})",
        "tested_up_to": tested,
        "time_ms": (time.time() - start_time) * 1000
    }

def synthesize(budget: int = 500, seed: Optional[int] = None) -> Dict:
    """
    Explore template×param space and return Pareto-optimal variants.

    Returns:
        dict: {
            "pareto_front": List[dict],
            "templates_tried": int,
            "candidates_tested": int,
            "time_ms": float
        }
    """
    start_time = time.time()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pareto_front = []
    templates_tried = 0
    candidates_tested = 0

    for _ in range(budget):
        # Random template and parameters
        template = random.choice(TEMPLATES)
        params = {}
        for param, (param_type, allowed_values) in template.parameters.items():
            if param_type == str:
                params[param] = random.choice(allowed_values)
            else:
                if len(allowed_values) > 1:
                    if param_type == int:
                        params[param] = random.randint(allowed_values[0], allowed_values[-1])
                    else:  # float
                        params[param] = random.uniform(allowed_values[0], allowed_values[-1])
                else:
                    params[param] = allowed_values[0]

        try:
            instantiated = instantiate(template.dsl_pattern, params)
            result = evaluate(instantiated["statement"], params)
            candidates_tested += 1

            # Check Pareto optimality
            dominated = False
            new_front = []
            for candidate in pareto_front:
                if (result["support"] >= candidate["support"] and
                    result["simplicity"] >= candidate["simplicity"] and
                    result["novelty"] >= candidate["novelty"]):
                    dominated = True
                if not (result["support"] > candidate["support"] and
                        result["simplicity"] > candidate["simplicity"] and
                        result["novelty"] > candidate["novelty"]):
                    new_front.append(candidate)

            if not dominated:
                new_front.append({
                    "statement": instantiated["statement"],
                    "params": params,
                    **result
                })
                pareto_front = new_front

        except (DSLSyntaxError, ValueError):
            continue

    return {
        "pareto_front": sorted(
            pareto_front,
            key=lambda x: (-x["support"], -x["simplicity"], -x["novelty"])
        ),
        "templates_tried": templates_tried,
        "candidates_tested": candidates_tested,
        "time_ms": (time.time() - start_time) * 1000
    }

def export(obj: Dict, path: str) -> None:
    """Export results to JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def metadata() -> Dict:
    """Return metadata about this component."""
    return {
        "version": "1.0",
        "templates": [t.name for t in TEMPLATES],
        "max_prime": PRIMES_UP_TO
    }

def discover() -> Dict:
    """Return discovery information."""
    return {"component": "MetaVariantSynthesizer"}

def run_self_tests() -> bool:
    """Run self-tests and return True if all pass."""
    # Test 1: DSL round-trip
    try:
        tmpl = TEMPLATES[0].dsl_pattern
        params = {k: v[1][0] for k, v in TEMPLATES[0].parameters.items()}
        instantiated = instantiate(tmpl, params)
        assert tmpl in instantiated["statement"], "DSL round-trip failed"
    except Exception as e:
        print(f"Self-test failed (DSL): {e}")
        return False

    # Test 2: Scoring monotonicity
    try:
        r1 = evaluate("Every even n ≥ 4 is sum of two primes with p mod 3 ∈ {1}", {"N0": 4, "m": 3, "S": "{1}"})
        r2 = evaluate("Every even n ≥ 4 is sum of two primes with p mod 3 ∈ {1,2}", {"N0": 4, "m": 3, "S": "{1,2}"})
        assert r2["support"] >= r1["support"], "Support should be monotonic"
    except Exception as e:
        print(f"Self-test failed (scoring): {e}")
        return False

    # Test 3: Seed stability
    try:
        r1 = synthesize(budget=10, seed=42)
        r2 = synthesize(budget=10, seed=42)
        assert r1["pareto_front"] == r2["pareto_front"], "Results should be seed-stable"
    except Exception as e:
        print(f"Self-test failed (seed): {e}")
        return False

    return True

def main_cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Goldbach Variant Synthesizer")
    parser.add_argument("--mode", choices=["cli", "self-test"], default="cli")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "self-test":
        success = run_self_tests()
        print("Self-tests passed!" if success else "Self-tests failed!")
        exit(0 if success else 1)

    print("Synthesizing variants...")
    results = synthesize(budget=args.budget, seed=args.seed)

    print(f"\nPareto frontier ({len(results['pareto_front'])} variants):")
    for i, variant in enumerate(results["pareto_front"], 1):
        print(f"\n#{i}: {variant['statement']}")
        print(f"  Support: {variant['support']:.1%}")
        print(f"  Simplicity: {variant['simplicity']:.2f}")
        print(f"  Novelty: {variant['novelty']:.2f}")
        print(f"  Rationale: {variant['rationale']}")

    if args.export:
        export(results, args.export)
        print(f"\nResults exported to {args.export}")

def main_streamlit():
    """Streamlit web interface."""
    if not HAS_STREAMLIT:
        st.error("Streamlit not available")
        return

    st.title("Goldbach Variant Synthesizer")
    st.sidebar.header("Configuration")

    budget = st.sidebar.slider("Search budget", 10, 1000, 200)
    seed = st.sidebar.number_input("Random seed", value=42)

    if st.sidebar.button("Run Synthesis"):
        with st.spinner("Synthesizing variants..."):
            results = synthesize(budget=budget, seed=seed if seed != 0 else None)

        st.success(f"Found {len(results['pareto_front'])} Pareto-optimal variants")

        for i, variant in enumerate(results["pareto_front"], 1):
            with st.expander(f"Variant #{i}: {variant['statement']}"):
                st.metric("Support", f"{variant['support']:.1%}")
                st.metric("Simplicity", f"{variant['simplicity']:.2f}")
                st.metric("Novelty", f"{variant['novelty']:.2f}")
                st.caption(variant["rationale"])

                if st.button(f"Export Variant #{i}", key=f"export_{i}"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(variant, indent=2),
                        file_name=f"goldbach_variant_{i}.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    if HAS_STREAMLIT and "streamlit" in __import__("sys").argv[0]:
        main_streamlit()
    else:
        main_cli()
