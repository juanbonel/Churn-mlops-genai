"""Shared inference helpers used by the REST API, assistant and MCP tools."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from churn_project.data import FEATURES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m churn_project.train` first."
        )
    try:
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Run `python -m pip install -e .`.") from exc
    return mlflow.sklearn.load_model(str(model_path))


def score_customer(features: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in FEATURES if key not in features]
    if missing:
        raise ValueError(f"Missing feature(s): {', '.join(missing)}")
    row = pd.DataFrame([{key: features[key] for key in FEATURES}], columns=FEATURES)
    model = load_model()
    label = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    metadata = {}
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}
    return {
        "churn_prediction": label,
        "churn_probability": round(probability, 6),
        "model_name": metadata.get("registered_model_name", "subscription_churn_model"),
        "model_type": metadata.get("best_model", "unknown"),
        "note": "Synthetic portfolio model; not validated for real customer decisions.",
    }
