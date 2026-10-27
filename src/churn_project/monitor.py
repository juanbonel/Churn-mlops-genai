"""Simple input-feature drift monitor using Population Stability Index (PSI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn_project.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, make_churn_dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "data" / "reference.csv"
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_THRESHOLD = 0.20
EPSILON = 1e-6


def _normalized_counts(series: pd.Series, categories: list) -> np.ndarray:
    values = series.astype("object").where(series.notna(), "<MISSING>")
    counts = pd.Series(values.astype(str)).value_counts(normalize=True)
    probabilities = np.array([float(counts.get(str(category), 0.0)) for category in categories])
    probabilities = np.clip(probabilities, EPSILON, None)
    return probabilities / probabilities.sum()


def calculate_numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI with quantile bins derived only from the reference sample."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([ref.min() - 1, ref.max() + 1], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    ref_binned = pd.cut(ref, bins=edges, include_lowest=True)
    cur_binned = pd.cut(cur, bins=edges, include_lowest=True)
    categories = list(ref_binned.cat.categories)
    expected = _normalized_counts(ref_binned, categories)
    actual = _normalized_counts(cur_binned, categories)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def calculate_categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype("object").where(reference.notna(), "<MISSING>").astype(str)
    cur = current.astype("object").where(current.notna(), "<MISSING>").astype(str)
    categories = sorted(set(ref.unique()).union(set(cur.unique())))
    expected = _normalized_counts(ref, categories)
    actual = _normalized_counts(cur, categories)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    missing_ref = [column for column in FEATURES if column not in reference.columns]
    missing_cur = [column for column in FEATURES if column not in current.columns]
    if missing_ref or missing_cur:
        raise ValueError(
            f"Required feature columns missing; reference={missing_ref}, current={missing_cur}"
        )

    rows = []
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            psi = calculate_numeric_psi(reference[feature], current[feature])
        elif feature in CATEGORICAL_FEATURES:
            psi = calculate_categorical_psi(reference[feature], current[feature])
        else:
            continue
        rows.append(
            {
                "feature": feature,
                "psi": float(psi),
                "threshold": float(threshold),
                "drift_detected": bool(np.isfinite(psi) and psi >= threshold),
            }
        )

    drifted = [row["feature"] for row in rows if row["drift_detected"]]
    return {
        "status": "DRIFT_DETECTED" if drifted else "NO_MAJOR_DRIFT_DETECTED",
        "method": "Population Stability Index (PSI)",
        "threshold": float(threshold),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_checked": len(rows),
        "drifted_features": drifted,
        "feature_results": rows,
        "limitations": [
            "PSI checks feature-distribution changes, not label/concept drift.",
            "The 0.20 threshold is a configurable heuristic, not a universal standard.",
            "The current cohort is compared with a synthetic reference dataset.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current", type=Path, help="CSV with the same features as reference.csv")
    parser.add_argument("--simulate-shift", action="store_true", help="Generate a shifted synthetic cohort")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 when drift is detected")
    args = parser.parse_args()

    if not args.reference.exists():
        raise SystemExit("Reference file not found. Run `python -m churn_project.train` first.")
    reference = pd.read_csv(args.reference)
    if args.current:
        current = pd.read_csv(args.current)
    elif args.simulate_shift:
        current = make_churn_dataset(n_samples=1000, random_state=2026, shift=True)[FEATURES]
    else:
        # Baseline smoke test: same generating process, independent random sample.
        current = make_churn_dataset(n_samples=1000, random_state=2025)[FEATURES]

    report = build_drift_report(reference, current[FEATURES], threshold=args.threshold)
    DEFAULT_REPORTS.mkdir(exist_ok=True)
    (DEFAULT_REPORTS / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["feature_results"]).to_csv(DEFAULT_REPORTS / "drift_report.csv", index=False)

    print(f"Drift status: {report['status']}")
    print(f"Rows: reference={report['reference_rows']}, current={report['current_rows']}")
    print(pd.DataFrame(report["feature_results"]).sort_values("psi", ascending=False).to_string(index=False))
    if args.fail_on_drift and report["drifted_features"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
