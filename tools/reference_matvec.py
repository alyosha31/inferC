#!/usr/bin/env python3
"""Generate deterministic matrix-vector reference data for InferC.

This is a development oracle, not production inference code.  The C test
should consume the emitted JSON and independently compute the same result.
"""

import argparse
import json
from pathlib import Path


def matvec(weights: list[list[float]], vector: list[float]) -> list[float]:
    """Return weights @ vector for a row-major matrix."""
    rows = len(weights)
    cols = len(vector)
    if rows == 0 or cols == 0:
        raise ValueError("matrix and vector must be non-empty")
    if any(len(row) != cols for row in weights):
        raise ValueError("matrix rows must all have the vector length")

    return [sum(row[col] * vector[col] for col in range(cols)) for row in weights]


def build_fixture() -> dict[str, object]:
    weights = [
        [1.0, -2.0, 0.5],
        [4.0, 3.0, -1.0],
    ]
    vector = [10.0, 20.0, 30.0]
    return {
        "operation": "matvec",
        "rows": len(weights),
        "cols": len(vector),
        "weights": weights,
        "vector": vector,
        "expected": matvec(weights, vector),
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/tmp/inferc-matvec-reference.json"),
        help="output JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"expected: {fixture['expected']}")


if __name__ == "__main__":
    main()
