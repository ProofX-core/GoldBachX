#!/usr/bin/env python3.11
"""
GoldbachX Conjecture Lab Interface - Interactive workbench for Goldbach experiments.
Runs as Streamlit UI (preferred) or headless CLI with identical validation logic.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from enum import Enum
from typing import Literal, Optional, Tuple, TypedDict, Union

try:
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated  # type: ignore

# Optional dependencies with graceful degradation
HAS_STREAMLIT = False
HAS_PYDANTIC = False
HAS_RICH = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    pass

try:
    from pydantic import BaseModel, Field, ValidationError, validator
    HAS_PYDANTIC = True
except ImportError:
    pass

try:
    from rich import print as rprint
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    pass

# Constants
MAX_RANGE = 10_000_000  # Safety limit for batch runs
TELEMETRY_EVENTS = {"ui_submit", "cli_submit", "validate_ok", "health_ok", "validate_error"}
PRESETS = {
    "SmallRange": {"mode": "range", "start": 4, "end": 100, "algo": "eratosthenes"},
    "StressRange": {"mode": "range", "start": 1_000_000, "end": 1_000_100, "algo": "atkin"},
    "TwinAdjacent": {"mode": "range", "start": 100, "end": 1000, "algo": "wheel", "exclude_twins": True}
}

# Type Definitions
Algorithm = Literal["eratosthenes", "atkin", "wheel"]
ExperimentMode = Literal["single", "range"]

if HAS_PYDANTIC:
    class ExperimentPayload(BaseModel):
        mode: ExperimentMode
        n: Optional[int] = Field(None, ge=4, le=MAX_RANGE)
        start_end: Optional[Tuple[int, int]] = Field(None, min_items=2, max_items=2)
        algo: Algorithm = "eratosthenes"
        allow_equal: bool = True
        exclude_twins: bool = False
        seed: Optional[int] = None
        notes: Optional[str] = None
        ts: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

        @validator("start_end")
        def validate_range(cls, v, values):
            if v is None:
                return v
            start, end = v
            if start >= end:
                raise ValueError("start must be < end")
            if start < 4:
                raise ValueError("start must be ≥4")
            if end > MAX_RANGE:
                raise ValueError(f"end must be ≤{MAX_RANGE}")
            if values.get("mode") == "range" and start % 2 != 0:
                raise ValueError("range start must be even for Goldbach validation")
            return v

        class Config:
            schema_extra = {
                "example": {
                    "mode": "single",
                    "n": 100,
                    "algo": "eratosthenes",
                    "allow_equal": True,
                    "exclude_twins": False,
                    "seed": 42,
                    "notes": "Test run"
                }
            }
else:
    class ExperimentPayload(TypedDict, total=False):
        mode: ExperimentMode
        n: Optional[int]
        start_end: Optional[Tuple[int, int]]
        algo: Algorithm
        allow_equal: bool
        exclude_twins: bool
        seed: Optional[int]
        notes: Optional[str]
        ts: str

class OrchestratorClient:
    """Pluggable backend client with in-memory mock or HTTP transport."""

    def __init__(self):
        self.base_url = os.getenv("GOLDBACHX_ORCH_URL")
        self.session = None
        if self.base_url:
            try:
                import requests
                self.session = requests.Session()
            except ImportError:
                print("[!] GOLDBACHX_ORCH_URL set but requests not installed", file=sys.stderr)

    def submit(self, payload: dict) -> dict:
        """Submit experiment payload to backend."""
        start_time = time.time()

        if self.session:
            try:
                resp = self.session.post(f"{self.base_url}/submit", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                result["notes"] = "queued"
                _log_telemetry("health_ok", start_time)
                return result
            except Exception as e:
                return {"accepted": False, "error": str(e)}

        # Mock response
        mock_id = f"mock-{random.getrandbits(64):016x}"
        _log_telemetry("health_ok", start_time)
        return {
            "accepted": True,
            "id": mock_id,
            "echo": payload,
            "notes": "mock"
        }

    def health(self) -> dict:
        """Check backend health."""
        if self.session:
            try:
                resp = self.session.get(f"{self.base_url}/health", timeout=5)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"healthy": False, "error": str(e)}
        return {"healthy": True, "component": "mock"}

def validate(payload: dict) -> dict:
    """Validate experiment payload structure and parameters."""
    start_time = time.time()
    errors = []

    # Structural checks
    required = {"mode", "algo"}
    missing = required - set(payload.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")

    # Parameter validation
    try:
        mode = payload["mode"]
        if mode not in ("single", "range"):
            errors.append("mode must be 'single' or 'range'")

        if mode == "single":
            n = payload.get("n")
            if not isinstance(n, int) or n < 4:
                errors.append("n must be integer ≥4")
            if n % 2 != 0:
                errors.append("n must be even for Goldbach validation")
        else:
            start_end = payload.get("start_end")
            if not isinstance(start_end, (list, tuple)) or len(start_end) != 2:
                errors.append("range requires [start,end] pair")
            else:
                start, end = start_end
                if start >= end:
                    errors.append("start must be < end")
                if start < 4:
                    errors.append("start must be ≥4")
                if end > MAX_RANGE:
                    errors.append(f"end must be ≤{MAX_RANGE}")
                if start % 2 != 0:
                    errors.append("range start must be even")

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    if errors:
        _log_telemetry("validate_error", start_time)
        return {"valid": False, "errors": errors}

    _log_telemetry("validate_ok", start_time)
    return {"valid": True}

def discover() -> dict:
    """Component discovery endpoint."""
    return {"component": "ConjectureLabInterface"}

def _log_telemetry(event: str, start_time: float):
    """Emit JSON-line telemetry."""
    if event not in TELEMETRY_EVENTS:
        return

    elapsed_ms = int((time.time() - start_time) * 1000)
    print(json.dumps({
        "event": event,
        "ts": datetime.datetime.utcnow().isoformat(),
        "latency_ms": elapsed_ms
    }), file=sys.stdout)

# CLI Implementation
def parse_cli_args():
    parser = argparse.ArgumentParser(description="GoldbachX Experiment Interface")
    parser.add_argument("--mode", choices=["single", "range"], required=True)
    parser.add_argument("--n", type=int, help="Single number to test")
    parser.add_argument("--start", type=int, help="Range start")
    parser.add_argument("--end", type=int, help="Range end")
    parser.add_argument("--algo", choices=["eratosthenes", "atkin", "wheel"], default="eratosthenes")
    parser.add_argument("--allow-equal", type=int, choices=[0, 1], default=1)
    parser.add_argument("--exclude-twins", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--notes", type=str, help="Experiment notes")
    parser.add_argument("--export", type=str, help="Export payload to JSON file")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")
    return parser.parse_args()

def run_self_tests():
    """Validate core functionality."""
    tests = [
        ("Valid single", {"mode": "single", "n": 100, "algo": "eratosthenes"}, True),
        ("Valid range", {"mode": "range", "start_end": [4, 100], "algo": "atkin"}, True),
        ("Invalid n", {"mode": "single", "n": 3, "algo": "wheel"}, False),
        ("Missing algo", {"mode": "single", "n": 10}, False),
        ("Odd start", {"mode": "range", "start_end": [5, 100], "algo": "eratosthenes"}, False)
    ]

    client = OrchestratorClient()
    passed = 0

    for name, payload, should_pass in tests:
        result = validate(payload)
        if result["valid"] == should_pass:
            passed += 1
            status = "PASS" if should_pass else "PASS (caught invalid)"
        else:
            status = "FAIL"

        if HAS_RICH:
            rprint(Panel.fit(f"{name}: {status}", title="Test"))
        else:
            print(f"{name}: {status}")

    # Test submission
    test_payload = {"mode": "single", "n": 42, "algo": "wheel", "seed": 123}
    submit_result = client.submit(test_payload)
    if submit_result.get("accepted"):
        passed += 1
        if HAS_RICH:
            rprint(Panel.fit("Submit test: PASS", title="Test"))
        else:
            print("Submit test: PASS")
    else:
        if HAS_RICH:
            rprint(Panel.fit(f"Submit test: FAIL - {submit_result.get('error')}", title="Test"))
        else:
            print(f"Submit test: FAIL - {submit_result.get('error')}")

    print(f"\nSelf-test complete: {passed}/{len(tests)+1} passed")
    sys.exit(0 if passed == len(tests)+1 else 1)

def build_cli_payload(args) -> dict:
    """Construct payload from CLI args."""
    payload = {
        "mode": args.mode,
        "algo": args.algo,
        "allow_equal": bool(args.allow_equal),
        "exclude_twins": bool(args.exclude_twins),
        "seed": args.seed,
        "notes": args.notes
    }

    if args.mode == "single":
        payload["n"] = args.n
    else:
        payload["start_end"] = (args.start, args.end)

    return payload

# Streamlit UI Implementation
def render_ui():
    """Streamlit dashboard layout."""
    st.set_page_config(page_title="GoldbachX Lab", layout="wide")
    st.title("Goldbach Conjecture Experiment Lab")

    tab1, tab2, tab3, tab4 = st.tabs(["Compose", "Presets", "Validation", "Submission Log"])

    with tab1:
        with st.form("experiment_form"):
            mode = st.radio("Mode", ["single", "range"], horizontal=True)

            if mode == "single":
                n = st.number_input("n (even ≥4)", min_value=4, step=2, value=100)
            else:
                col1, col2 = st.columns(2)
                start = col1.number_input("Start (even ≥4)", min_value=4, step=2, value=4)
                end = col2.number_input("End", min_value=start+2, value=100)

            algo = st.selectbox("Algorithm", ["eratosthenes", "atkin", "wheel"], index=0)
            col1, col2 = st.columns(2)
            allow_equal = col1.checkbox("Allow equal primes", value=True)
            exclude_twins = col2.checkbox("Exclude twin primes", value=False)

            seed = st.number_input("Random seed (optional)", min_value=0, value=None)
            notes = st.text_area("Notes")

            submitted = st.form_submit_button("Validate & Submit")

            if submitted:
                payload = {
                    "mode": mode,
                    "algo": algo,
                    "allow_equal": allow_equal,
                    "exclude_twins": exclude_twins,
                    "seed": seed,
                    "notes": notes
                }

                if mode == "single":
                    payload["n"] = n
                else:
                    payload["start_end"] = (start, end)

                validation = validate(payload)
                if validation["valid"]:
                    client = OrchestratorClient()
                    submit_result = client.submit(payload)
                    st.session_state.setdefault("submissions", []).append({
                        "payload": payload,
                        "result": submit_result,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    st.success("Experiment submitted!")
                else:
                    for error in validation["errors"]:
                        st.error(error)

    with tab2:
        st.header("Quick-run Presets")
        selected_preset = st.selectbox("Choose preset", list(PRESETS.keys()))
        preset = PRESETS[selected_preset]

        with st.expander("Preset Details"):
            st.json(preset)

        if st.button("Load Preset"):
            st.session_state.preset = preset
            st.experimental_rerun()

    with tab3:
        if "preset" in st.session_state:
            payload = st.session_state.preset
            st.json(payload)
            validation = validate(payload)

            if validation["valid"]:
                st.success("Payload is valid")
            else:
                st.error("Validation errors:")
                for err in validation["errors"]:
                    st.error(err)

    with tab4:
        if "submissions" not in st.session_state:
            st.info("No submissions yet")
        else:
            for i, sub in enumerate(reversed(st.session_state.submissions[-50:])):
                with st.expander(f"Submission #{len(st.session_state.submissions)-i}"):
                    st.json(sub["payload"])
                    if sub["result"]["accepted"]:
                        st.success(f"Accepted: {sub['result']['id']}")
                    else:
                        st.error(f"Rejected: {sub['result'].get('error')}")

# Main Entry Point
def main():
    args = parse_cli_args()

    if args.self_test:
        run_self_tests()
        return

    if not HAS_STREAMLIT and not any(vars(args).values()):
        print("No CLI args provided and Streamlit not available", file=sys.stderr)
        sys.exit(1)

    if HAS_STREAMLIT and not any(vars(args).values()):
        render_ui()
    else:
        payload = build_cli_payload(args)
        validation = validate(payload)

        if not validation["valid"]:
            print("Validation errors:", file=sys.stderr)
            for err in validation["errors"]:
                print(f"- {err}", file=sys.stderr)
            sys.exit(1)

        if args.export:
            with open(args.export, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Payload saved to {args.export}")

        client = OrchestratorClient()
        result = client.submit(payload)

        if HAS_RICH:
            rprint(Panel.fit(json.dumps(result, indent=2), title="Submission Result")
        else:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
