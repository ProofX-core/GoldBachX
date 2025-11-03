#!/usr/bin/env python3
"""
HypotheticalAxiomInjector - Safe injection of heuristic rules for Goldbach conjecture verification.
"""

import json
import sys
import time
from typing import TypedDict, Optional, List, Tuple, Dict, Literal
from argparse import ArgumentParser, Namespace
import random
from dataclasses import dataclass
import hashlib

# Optional Pydantic support (feature-gated)
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Type Definitions
AxiomLevel = Literal["strong", "weak", "hint"]
AxiomConfig = TypedDict('AxiomConfig', {
    'name': str,
    'level': AxiomLevel,
    'params': Dict[str, object],
    'enabled': bool,
    'notes': str
})

class ApplyResult(TypedDict):
    n: int
    filters_applied: List[str]
    hints: Dict[str, object]
    notes: str

@dataclass
class TelemetryEvent:
    event_type: str
    data: Dict[str, object]
    timestamp: float

# Core Implementation
class HypotheticalAxiomInjector:
    _PRESETS = {
        "ResidueClassBias": {
            "name": "ResidueClassBias",
            "level": "weak",
            "params": {"mod": 3, "prefer": 1},
            "enabled": True,
            "notes": "Prefers primes congruent to 1 mod 3"
        },
        "TwinPrimeFavor": {
            "name": "TwinPrimeFavor",
            "level": "hint",
            "params": {"radius": 2},
            "enabled": True,
            "notes": "Prioritizes primes with twin in ±2 range"
        },
        "GapAwareSearch": {
            "name": "GapAwareSearch",
            "params": {"max_gap": 100},
            "level": "strong",
            "enabled": True,
            "notes": "Excludes primes in large gap regions"
        }
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._telemetry: List[TelemetryEvent] = []

    def _log_telemetry(self, event_type: str, data: Dict[str, object]) -> None:
        self._telemetry.append(TelemetryEvent(
            event_type=event_type,
            data=data,
            timestamp=time.time()
        ))
        print(json.dumps({
            "event": event_type,
            "data": data,
            "ts": time.time()
        }))

    def presets(self) -> List[AxiomConfig]:
        """Return all available preset configurations."""
        return list(self._PRESETS.values())

    def validate_config(self, cfg: Dict) -> Tuple[bool, str]:
        """Validate a configuration dictionary."""
        required_keys = {'name', 'level', 'params', 'enabled', 'notes'}
        if not all(k in cfg for k in required_keys):
            return False, "Missing required fields"

        if cfg['level'] not in ["strong", "weak", "hint"]:
            return False, "Invalid level"

        if PYDANTIC_AVAILABLE:
            try:
                _ = AxiomConfigPydantic(**cfg)
            except ValidationError as e:
                return False, str(e)

        return True, "Valid"

    def _apply_residue_bias(self, n: int, primes: List[int], params: Dict) -> List[int]:
        mod = params.get('mod', 3)
        prefer = params.get('prefer', 1)
        return [p for p in primes if p % mod == prefer]

    def _apply_twin_favor(self, n: int, primes: List[int], params: Dict) -> List[int]:
        radius = params.get('radius', 2)
        def has_twin(p: int) -> bool:
            return (p + radius in primes) or (p - radius in primes)
        return sorted(primes, key=lambda p: -1 if has_twin(p) else 0)

    def _apply_gap_aware(self, n: int, primes: List[int], params: Dict) -> List[int]:
        max_gap = params.get('max_gap', 100)
        filtered = []
        for i in range(1, len(primes)):
            if primes[i] - primes[i-1] <= max_gap:
                filtered.append(primes[i-1])
        if primes:
            filtered.append(primes[-1])  # Always include last prime
        return filtered

    def apply(self, n: int, pairs: Optional[List[Tuple[int, int]]], cfgs: List[Dict]) -> ApplyResult:
        """Apply configurations to modify search behavior."""
        start_time = time.time()
        primes = list({p for pair in pairs or [] for p in pair}) if pairs else []
        original_count = len(primes) if primes else 0

        applied_filters = []
        hints = {"priority_primes": [], "algo": "default"}

        for cfg in sorted(cfgs, key=lambda x: x.get('level', 'hint')):
            if not cfg.get('enabled', True):
                continue

            name = cfg['name']
            params = cfg.get('params', {})
            level = cfg.get('level', 'hint')

            try:
                if name == "ResidueClassBias":
                    filtered = self._apply_residue_bias(n, primes, params)
                elif name == "TwinPrimeFavor":
                    filtered = self._apply_twin_favor(n, primes, params)
                elif name == "GapAwareSearch":
                    filtered = self._apply_gap_aware(n, primes, params)
                else:
                    continue

                # Safety check: don't eliminate all primes
                if level in ["strong", "weak"] and filtered and len(filtered) < original_count:
                    if not filtered:  # Would eliminate all primes
                        hints["priority_primes"] = filtered  # Downgrade to hint
                        applied_filters.append(f"{name}(downgraded)")
                    else:
                        primes = filtered
                        applied_filters.append(name)
                elif level == "hint":
                    hints["priority_primes"] = filtered
                    hints["algo"] = f"priority_{name.lower()}"

            except Exception as e:
                self._log_telemetry("apply_error", {
                    "cfg": name,
                    "error": str(e)
                })

        self._log_telemetry("apply_done", {
            "n": n,
            "original_count": original_count,
            "filtered_count": len(primes),
            "time_ms": (time.time() - start_time) * 1000
        })

        return {
            "n": n,
            "filters_applied": applied_filters,
            "hints": hints,
            "notes": f"Applied {len(applied_filters)} filters"
        }

    def export(self, cfgs: List[Dict], path: str) -> None:
        """Export configurations to JSON file."""
        start_time = time.time()
        with open(path, 'w') as f:
            json.dump(cfgs, f, indent=2)
        self._log_telemetry("export_done", {
            "path": path,
            "count": len(cfgs),
            "time_ms": (time.time() - start_time) * 1000
        })

    def metadata(self) -> Dict:
        """Return component metadata."""
        return {
            "version": "1.0",
            "pydantic": PYDANTIC_AVAILABLE,
            "presets": list(self._PRESETS.keys())
        }

    def discover(self) -> Dict:
        """Discovery information for the component."""
        return {
            "component": "HypotheticalAxiomInjector",
            "capabilities": ["filter", "hint_generation"]
        }

# Pydantic Models (if available)
if PYDANTIC_AVAILABLE:
    class AxiomConfigPydantic(BaseModel):
        name: str
        level: AxiomLevel
        params: Dict[str, object]
        enabled: bool
        notes: str

# CLI Interface
def parse_args() -> Namespace:
    parser = ArgumentParser(description="Hypothetical Axiom Injector")
    parser.add_argument("--n", type=int, help="Target integer for Goldbach verification")
    parser.add_argument("--preset", type=str, help="Preset configuration name")
    parser.add_argument("--cfg", type=str, help="Path to configuration JSON file")
    parser.add_argument("--export", type=str, help="Path to export configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")
    return parser.parse_args()

def self_test() -> bool:
    """Run basic validation tests."""
    injector = HypotheticalAxiomInjector(seed=42)

    # Test presets
    presets = injector.presets()
    for preset in presets:
        valid, msg = injector.validate_config(preset)
        if not valid:
            print(f"Self-test failed: {preset['name']} - {msg}")
            return False

    # Test composition
    result = injector.apply(100, [(3, 97), (11, 89)], presets)
    if not result['hints']['priority_primes']:
        print("Self-test failed: empty hints")
        return False

    # Test determinism
    r1 = injector.apply(100, [(3, 97), (11, 89)], presets[:1])
    injector2 = HypotheticalAxiomInjector(seed=42)
    r2 = injector2.apply(100, [(3, 97), (11, 89)], presets[:1])
    if r1 != r2:
        print("Self-test failed: non-deterministic results")
        return False

    return True

def main() -> None:
    args = parse_args()
    injector = HypotheticalAxiomInjector(seed=args.seed)

    if args.self_test:
        if self_test():
            print("Self-tests passed")
            sys.exit(0)
        else:
            print("Self-tests failed")
            sys.exit(1)

    cfgs = []
    if args.preset:
        preset = next((p for p in injector.presets() if p['name'] == args.preset), None)
        if preset:
            cfgs.append(preset)
        else:
            print(f"Unknown preset: {args.preset}", file=sys.stderr)
            sys.exit(1)

    if args.cfg:
        try:
            with open(args.cfg) as f:
                cfgs.extend(json.load(f))
        except Exception as e:
            print(f"Config load error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.export:
        injector.export(cfgs, args.export)

    if args.n:
        # Generate some example pairs for demonstration
        pairs = None
        if args.n == 100:  # Demo case
            pairs = [(3, 97), (11, 89), (17, 83)]
        result = injector.apply(args.n, pairs, cfgs)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
