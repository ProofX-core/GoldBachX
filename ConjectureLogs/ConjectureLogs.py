#!/usr/bin/env python3.11
"""
GoldbachX Conjecture Logs Manager

Canonicalizes, stores, and queries experiment logs for GoldbachX runs.
Handles JSON/JSONL ingestion with schema normalization and lightweight indexing.

Schema:
- id: str (UUID)
- ts: str (ISO8601 UTC)
- module: str
- n|range: int|str
- pairs_count|attempts: int
- time_ms: int
- status: str
- notes: str
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Constants
DEFAULT_LOG_DIR = "./proof-data/logs"
SCHEMA = {
    "id": str,
    "ts": str,
    "module": str,
    "n": Optional[int],
    "range": Optional[str],
    "pairs_count": Optional[int],
    "attempts": Optional[int],
    "time_ms": int,
    "status": str,
    "notes": str,
}
DEFAULTS = {
    "notes": "",
    "time_ms": 0,
    "status": "unknown",
}

def _emit_telemetry(event: str, **kwargs) -> None:
    """Emit JSONL telemetry to stdout."""
    data = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    print(json.dumps(data), file=sys.stdout)

def normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce stable schema on log records.

    Args:
        record: Input log record (partial fields allowed)

    Returns:
        Normalized record with all schema fields

    Raises:
        ValueError: If record is missing required fields
    """
    start = time.perf_counter()
    normalized = DEFAULTS.copy()

    # Handle ID and timestamp
    normalized["id"] = record.get("id", str(uuid.uuid4()))
    normalized["ts"] = record.get("ts", datetime.now(timezone.utc).isoformat())

    # Required fields
    if "module" not in record:
        raise ValueError("Missing required field: module")
    normalized["module"] = record["module"]

    # Numeric fields
    for field in ["n", "pairs_count", "attempts", "time_ms"]:
        if field in record:
            try:
                normalized[field] = int(record[field])
            except (ValueError, TypeError):
                pass

    # Range field
    if "range" in record:
        normalized["range"] = str(record["range"])
    elif "n" not in normalized:
        raise ValueError("Must provide either 'n' or 'range'")

    # Status and notes
    normalized["status"] = str(record.get("status", DEFAULTS["status"]))
    normalized["notes"] = str(record.get("notes", DEFAULTS["notes"]))

    _emit_telemetry("normalize_ok", latency_ms=(time.perf_counter()-start)*1000)
    return normalized

def append_log(path: str, record: Dict[str, Any]) -> None:
    """
    Append normalized record to JSONL file.

    Args:
        path: Path to JSONL file
        record: Log record to append
    """
    start = time.perf_counter()
    try:
        normalized = normalize(record)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(normalized) + "\n")

        _emit_telemetry("append_ok", path=path, latency_ms=(time.perf_counter()-start)*1000)
    except Exception as e:
        _emit_telemetry("append_error", error=str(e), path=path)
        raise

def load_logs(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and normalize logs from JSONL file.

    Args:
        path: Path to JSONL file
        limit: Maximum number of records to load

    Returns:
        List of normalized log records
    """
    start = time.perf_counter()
    records = []
    error_count = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break

                try:
                    record = json.loads(line.strip())
                    normalized = normalize(record)
                    records.append(normalized)
                except (json.JSONDecodeError, ValueError) as e:
                    error_count += 1
                    continue

        _emit_telemetry("load_done",
                       path=path,
                       count=len(records),
                       errors=error_count,
                       latency_ms=(time.perf_counter()-start)*1000)
        return records
    except Exception as e:
        _emit_telemetry("load_error", error=str(e), path=path)
        raise

def index_logs(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build in-memory inverted index from log records.

    Args:
        records: List of normalized log records

    Returns:
        Index mapping {field: {value: [ids]}}
    """
    start = time.perf_counter()
    index = {
        "module": {},
        "status": {},
        "n": {},
        "date": {},
    }

    for record in records:
        # Index by module
        module = record["module"]
        index["module"].setdefault(module, []).append(record["id"])

        # Index by status
        status = record["status"]
        index["status"].setdefault(status, []).append(record["id"])

        # Index by n if present
        if "n" in record:
            n = record["n"]
            index["n"].setdefault(n, []).append(record["id"])

        # Index by date (YYYY-MM-DD)
        date = record["ts"][:10]
        index["date"].setdefault(date, []).append(record["id"])

    _emit_telemetry("index_built",
                   records=len(records),
                   latency_ms=(time.perf_counter()-start)*1000)
    return index

def _parse_filter_term(term: str) -> tuple[str, str, Union[str, int, float]]:
    """Parse a single filter term into (field, operator, value)."""
    operators = [">=", "<=", "=", ">", "<"]
    for op in operators:
        if op in term:
            field, value = term.split(op, 1)
            return field.strip(), op, value.strip()
    raise ValueError(f"Invalid filter term: {term}")

def query(index: Dict[str, Dict[str, List[str]]], **filters) -> List[str]:
    """
    Query index with field filters.

    Args:
        index: Inverted index from index_logs()
        **filters: Filter terms (e.g., module="GoldbachVerifier", n>=1000)

    Returns:
        List of matching record IDs
    """
    start = time.perf_counter()
    result_sets = []

    for filter_str in filters.get("filter", "").split(","):
        filter_str = filter_str.strip()
        if not filter_str:
            continue

        field, op, value = _parse_filter_term(filter_str)

        if field not in index:
            continue

        # Handle numeric comparisons
        if field in ["n", "time_ms"]:
            try:
                value_num = int(value)
                matching_keys = [
                    k for k in index[field].keys()
                    if (op == "=" and k == value_num) or
                       (op == ">=" and k >= value_num) or
                       (op == "<=" and k <= value_num) or
                       (op == ">" and k > value_num) or
                       (op == "<" and k < value_num)
                ]
            except ValueError:
                continue
        else:
            # String equality match
            if op != "=":
                continue
            matching_keys = [value] if value in index[field] else []

        # Collect IDs for matching keys
        for key in matching_keys:
            result_sets.append(set(index[field][key]))

    # Intersect all result sets
    if not result_sets:
        matching_ids = []
    else:
        matching_ids = list(set.intersection(*result_sets)) if len(result_sets) > 1 else list(result_sets[0])

    _emit_telemetry("query_ok",
                   filters=filters,
                   matches=len(matching_ids),
                   latency_ms=(time.perf_counter()-start)*1000)
    return matching_ids

def export(records: List[Dict[str, Any]], path: str) -> None:
    """
    Export records to JSON file.

    Args:
        records: List of records to export
        path: Destination path
    """
    start = time.perf_counter()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        _emit_telemetry("export_done",
                       path=path,
                       count=len(records),
                       latency_ms=(time.perf_counter()-start)*1000)
    except Exception as e:
        _emit_telemetry("export_error", error=str(e), path=path)
        raise

def metadata() -> Dict[str, Any]:
    """Return module metadata and capabilities."""
    return {
        "schema": list(SCHEMA.keys()),
        "capabilities": {
            "pandas": HAS_PANDAS,
            "streamlit": HAS_STREAMLIT,
        }
    }

def discover() -> Dict[str, str]:
    """Return component identification."""
    return {"component": "ConjectureLogs"}

def _self_test() -> bool:
    """Run self-tests and return success status."""
    test_dir = os.path.join(DEFAULT_LOG_DIR, "self-test")
    test_path = os.path.join(test_dir, "test.jsonl")

    # Create test records
    test_records = [
        {"module": "Test", "n": 100, "status": "ok", "time_ms": 50},
        {"module": "Test", "n": 200, "status": "fail", "time_ms": 100},
        {"module": "Verify", "range": "100-200", "status": "ok", "attempts": 5},
        {"module": "Verify", "n": 300, "status": "ok", "pairs_count": 10},
        {"module": "Generate", "n": 400, "status": "partial", "notes": "test"},
    ]

    try:
        # Test append and load
        for record in test_records:
            append_log(test_path, record)

        loaded = load_logs(test_path)
        if len(loaded) != 5:
            raise ValueError(f"Expected 5 records, got {len(loaded)}")

        # Test index and query
        idx = index_logs(loaded)
        if len(query(idx, filter="module=Test")) != 2:
            raise ValueError("Module filter failed")
        if len(query(idx, filter="status=ok")) != 3:
            raise ValueError("Status filter failed")
        if len(query(idx, filter="n>=200")) != 3:
            raise ValueError("Numeric filter failed")

        # Test export
        export_path = os.path.join(test_dir, "export.json")
        export(loaded, export_path)
        if not os.path.exists(export_path):
            raise ValueError("Export failed")

        # Clean up
        os.remove(test_path)
        os.remove(export_path)

        _emit_telemetry("self_test_pass")
        return True
    except Exception as e:
        _emit_telemetry("self_test_fail", error=str(e))
        return False

def main() -> None:
    """Handle CLI commands."""
    parser = argparse.ArgumentParser(description="GoldbachX Conjecture Logs Manager")
    parser.add_argument("--mode", choices=["append", "query", "self-test"], required=True)
    parser.add_argument("--file", help="Input JSON file for append mode")
    parser.add_argument("--src", help="Source JSONL file for query mode")
    parser.add_argument("--dst", help="Destination path", default=os.path.join(DEFAULT_LOG_DIR, "gbx.jsonl"))
    parser.add_argument("--filter", help="Query filter string")
    parser.add_argument("--export", help="Export path for query results")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)

    if args.mode == "append":
        if not args.file:
            print("Error: --file required for append mode", file=sys.stderr)
            sys.exit(1)

        with open(args.file, "r", encoding="utf-8") as f:
            record = json.load(f)
        append_log(args.dst, record)
        print(f"Appended record to {args.dst}")

    elif args.mode == "query":
        if not args.src:
            print("Error: --src required for query mode", file=sys.stderr)
            sys.exit(1)

        records = load_logs(args.src)
        idx = index_logs(records)
        matching_ids = query(idx, filter=args.filter)

        # Get full records for matching IDs
        id_map = {r["id"]: r for r in records}
        results = [id_map[id] for id in matching_ids if id in id_map]

        if args.export:
            export(results, args.export)
            print(f"Exported {len(results)} records to {args.export}")
        else:
            print(json.dumps(results, indent=2))

    elif args.mode == "self-test":
        success = _self_test()
        print("Self-test passed" if success else "Self-test failed")
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
