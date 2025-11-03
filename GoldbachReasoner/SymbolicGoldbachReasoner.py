#!/usr/bin/env python3
"""
Symbolic reasoning engine for Goldbach-style conjectures.
Heuristic rule-based system with explanation tracing.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class Status(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNDECIDED = "undecided"


@dataclass
class Rule:
    premises: List[str]
    conclusion: str
    weight: float = 1.0
    name: Optional[str] = None

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Rule weight must be positive")


class SymbolicGoldbachReasoner:
    def __init__(self, seed: Optional[int] = None):
        self.rules: List[Rule] = []
        self._rule_index: Dict[str, List[Rule]] = defaultdict(list)
        self._seed = seed
        self._rng = random.Random(seed)
        self._builtin_rules_added = False

        self._add_builtin_rules()

    def _add_builtin_rules(self) -> None:
        if self._builtin_rules_added:
            return

        builtin_rules = [
            Rule(
                premises=["even(n)", "n >= 4"],
                conclusion="exists_prime_pair(n)",
                weight=0.8,
                name="goldbach_base_case"
            ),
            Rule(
                premises=["n % 6 == 0"],
                conclusion="exists_prime_pair(n)",
                weight=0.6,
                name="residue_class_6"
            ),
            Rule(
                premises=["n % 10 == 0", "n > 100"],
                conclusion="exists_prime_pair(n)",
                weight=0.7,
                name="residue_class_10"
            ),
            Rule(
                premises=["is_prime(p)", "is_prime(p+2)", "n == p + (p+2)"],
                conclusion="exists_prime_pair(n)",
                weight=1.0,
                name="twin_prime_pair"
            ),
            Rule(
                premises=["n > 1e6"],
                conclusion="exists_prime_pair(n)",
                weight=0.3,
                name="large_number_heuristic"
            ),
            Rule(
                premises=["n < 4"],
                conclusion="not exists_prime_pair(n)",
                weight=1.0,
                name="small_number_refutation"
            ),
            Rule(
                premises=["odd(n)"],
                conclusion="not exists_prime_pair(n)",
                weight=1.0,
                name="odd_number_refutation"
            )
        ]

        for rule in builtin_rules:
            self.add_rule(rule)

        self._builtin_rules_added = True

    def add_rule(self, rule: Union[Rule, Dict[str, Any]]) -> None:
        if isinstance(rule, dict):
            if PYDANTIC_AVAILABLE:
                rule = Rule(**rule)
            else:
                rule = Rule(
                    premises=rule["premises"],
                    conclusion=rule["conclusion"],
                    weight=rule.get("weight", 1.0),
                    name=rule.get("name")
                )

        self.rules.append(rule)
        for premise in rule.premises:
            self._rule_index[premise].append(rule)

    def list_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "premises": rule.premises,
                "conclusion": rule.conclusion,
                "weight": rule.weight,
                "name": rule.name
            }
            for rule in self.rules
        ]

    def prove(
        self,
        statement: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        start_time = time.time()
        context = context or {}

        # Log start event
        self._log_event("prove_started", {
            "statement": statement,
            "context": context,
            "seed": self._seed
        })

        try:
            result = self._prove_internal(statement, context)
        except Exception as e:
            result = {
                "status": Status.UNDECIDED,
                "score": 0.0,
                "explanation": [f"Error during proof: {str(e)}"],
                "used_rules": []
            }

        # Log finish event
        self._log_event("prove_finished", {
            "statement": statement,
            "result": result,
            "time_ms": (time.time() - start_time) * 1000
        })

        return result

    def _prove_internal(
        self,
        statement: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        working_memory = set(context.get("known_facts", []))
        goal = self._parse_statement(statement)
        explanation = []
        used_rules = []
        visited = set()
        agenda = deque([goal])

        while agenda:
            current_goal = agenda.popleft()

            if current_goal in working_memory:
                continue

            if current_goal in visited:
                continue

            visited.add(current_goal)

            # Try to find rules that can conclude this goal
            applicable_rules = []
            for rule in self.rules:
                if rule.conclusion == current_goal:
                    applicable_rules.append(rule)

            if not applicable_rules:
                # No rules can conclude this goal
                explanation.append(f"No rules found to prove '{current_goal}'")
                continue

            # Sort rules by weight (descending) for deterministic order
            applicable_rules.sort(key=lambda r: (-r.weight, r.name or ""))

            rule_applied = False
            for rule in applicable_rules:
                all_premises_satisfied = True
                premise_explanations = []

                for premise in rule.premises:
                    if premise not in working_memory:
                        # Try to prove the premise
                        premise_result = self._prove_internal(premise, context)
                        if premise_result["status"] != Status.SUPPORTED:
                            all_premises_satisfied = False
                            break
                        premise_explanations.extend(premise_result["explanation"])
                        used_rules.extend(premise_result["used_rules"])

                if all_premises_satisfied:
                    working_memory.add(current_goal)
                    rule_applied = True
                    used_rules.append({
                        "name": rule.name,
                        "premises": rule.premises,
                        "conclusion": rule.conclusion,
                        "weight": rule.weight
                    })
                    explanation.append(f"Applied rule '{rule.name or 'anonymous'}': {', '.join(rule.premises)} => {rule.conclusion}")
                    explanation.extend(premise_explanations)
                    break

            if not rule_applied:
                agenda.extend(rule.premises for rule in applicable_rules)

        # Determine final status
        status = Status.UNDECIDED
        score = 0.0

        if goal in working_memory:
            status = Status.SUPPORTED
            # Calculate score as average weight of used rules
            if used_rules:
                score = sum(r["weight"] for r in used_rules) / len(used_rules)
        elif any(f"not {goal}" == fact for fact in working_memory):
            status = Status.REFUTED
            score = 1.0  # Refutations are treated as certain

        return {
            "status": status,
            "score": score,
            "explanation": explanation,
            "used_rules": used_rules
        }

    def _parse_statement(self, statement: str) -> str:
        """Convert natural language statements to internal form."""
        statement = statement.lower().strip()

        if statement.startswith("every even n in"):
            return "exists_prime_pair(n)"
        elif statement.startswith("no odd n can be"):
            return "not exists_prime_pair(n)"
        elif "prime pair" in statement:
            return "exists_prime_pair(n)"

        return statement

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log telemetry events in JSONL format."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        }
        print(json.dumps(event), file=sys.stderr)

    def export(self, obj: Dict[str, Any], path: str) -> None:
        """Export result to JSON file."""
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the reasoner."""
        return {
            "rule_count": len(self.rules),
            "builtin_rules": self._builtin_rules_added,
            "seed": self._seed
        }

    def discover(self) -> Dict[str, str]:
        """Discovery information for component system."""
        return {"component": "SymbolicGoldbachReasoner"}


def run_cli():
    parser = argparse.ArgumentParser(description="Symbolic Goldbach Reasoner")
    parser.add_argument("--statement", type=str, help="Statement to prove")
    parser.add_argument("--context", type=str, help="JSON context file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--export", type=str, help="Export result to JSON file")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    reasoner = SymbolicGoldbachReasoner(seed=args.seed)

    if args.self_test:
        run_self_tests(reasoner)
        return

    if not args.statement:
        parser.print_help()
        return

    context = {}
    if args.context:
        with open(args.context) as f:
            context = json.load(f)

    result = reasoner.prove(args.statement, context=context, seed=args.seed)
    print(json.dumps(result, indent=2))

    if args.export:
        reasoner.export(result, args.export)


def run_self_tests(reasoner: SymbolicGoldbachReasoner) -> None:
    """Run basic sanity checks on the reasoner."""
    print("Running self-tests...")

    # Test 1: Basic Goldbach statement
    result = reasoner.prove("Every even n in [4,100] has ≥1 prime pair")
    assert result["status"] in [Status.SUPPORTED, Status.UNDECIDED], "Test 1 failed"
    print("✓ Test 1 passed")

    # Test 2: Odd number refutation
    result = reasoner.prove("Every odd n has prime pair")
    assert result["status"] == Status.REFUTED, "Test 2 failed"
    print("✓ Test 2 passed")

    # Test 3: Add and test custom rule
    custom_rule = {
        "premises": ["n == 10"],
        "conclusion": "exists_prime_pair(n)",
        "weight": 1.0,
        "name": "custom_10_test"
    }
    reasoner.add_rule(custom_rule)
    result = reasoner.prove("exists_prime_pair(10)")
    assert result["status"] == Status.SUPPORTED, "Test 3 failed"
    print("✓ Test 3 passed")

    # Test 4: Determinism with seed
    result1 = reasoner.prove("Every even n in [4,1e6] has ≥1 prime pair", seed=42)
    result2 = reasoner.prove("Every even n in [4,1e6] has ≥1 prime pair", seed=42)
    assert result1["score"] == result2["score"], "Test 4 failed"
    print("✓ Test 4 passed")

    print("All self-tests passed!")


if STREAMLIT_AVAILABLE:
    def run_streamlit_ui():
        """Run interactive Streamlit UI."""
        st.title("Symbolic Goldbach Reasoner")

        reasoner = SymbolicGoldbachReasoner()

        with st.sidebar:
            st.header("Configuration")
            seed = st.number_input("Random seed", value=42)
            reasoner._rng = random.Random(seed)

        st.subheader("Statement Analysis")
        statement = st.text_area(
            "Enter statement to analyze:",
            "Every even n in [4,100] has ≥1 prime pair"
        )

        if st.button("Analyze"):
            with st.spinner("Reasoning..."):
                result = reasoner.prove(statement, seed=seed)

            st.subheader("Result")
            st.json(result)

            st.subheader("Explanation")
            for step in result["explanation"]:
                st.text(step)

        st.subheader("Rule Browser")
        if st.button("Refresh Rules"):
            pass  # Just triggers a rerun

        for rule in reasoner.list_rules():
            with st.expander(rule.get("name", "Anonymous rule")):
                st.markdown(f"**Premises:** {', '.join(rule['premises'])}")
                st.markdown(f"**Conclusion:** {rule['conclusion']}")
                st.markdown(f"**Weight:** {rule['weight']}")


def main():
    if STREAMLIT_AVAILABLE and len(sys.argv) == 1:
        run_streamlit_ui()
    else:
        run_cli()


if __name__ == "__main__":
    main()
