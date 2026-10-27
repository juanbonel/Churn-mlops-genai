"""Train candidate churn models and track/register the winner with MLflow."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_project.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    make_churn_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EXPERIMENT_NAME = "churn-mlops-genai"
REGISTERED_MODEL_NAME = "subscription_churn_model"
SEED = 42


def make_pipeline(classifier) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = make_churn_dataset(n_samples=3000, random_state=SEED)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # The reference sample is the baseline for feature-distribution monitoring.
    X_train.sample(n=min(1200, len(X_train)), random_state=SEED).to_csv(
        DATA_DIR / "reference.csv", index=False
    )
    X_test.assign(churn=y_test.to_numpy()).to_csv(DATA_DIR / "test.csv", index=False)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", tracking_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        fitted_models[name] = pipeline
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "model_type": name,
                    "random_state": SEED,
                    "test_size": 0.2,
                    "dataset_rows": len(df),
                    "feature_count": len(FEATURES),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset", "synthetic_subscription_churn_v1")
            mlflow.set_tag("purpose", "portfolio_demo_not_for_business_use")
            results.append(
                {"model_type": name, "run_id": run.info.run_id, **metrics}
            )
        print(f"{name}: ROC-AUC={metrics['roc_auc']:.3f}, F1={metrics['f1']:.3f}")

    # Select on ROC-AUC; this is a portfolio baseline, not a production threshold.
    winner = max(results, key=lambda result: result["roc_auc"])
    best_name = winner["model_type"]
    best_model = fitted_models[best_name]
    signature = infer_signature(X_test, best_model.predict(X_test))
    input_example = X_train.head(5)

    model_dir = MODELS_DIR / "churn_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(
        sk_model=best_model,
        path=str(model_dir),
        signature=signature,
        input_example=input_example,
    )

    with mlflow.start_run(run_name="best_model_registered") as best_run:
        mlflow.log_params(
            {
                "selected_model": best_name,
                "selection_metric": "roc_auc",
                "selection_policy": "highest_test_roc_auc",
                "random_state": SEED,
            }
        )
        mlflow.log_metrics({key: float(winner[key]) for key in [
            "roc_auc", "accuracy", "f1", "precision", "recall"
        ]})
        mlflow.set_tag("model_role", "selected_candidate")
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        best_run_id = best_run.info.run_id

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": tracking_uri,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "best_model": best_name,
        "best_run_id": best_run_id,
        "local_model_path": str(model_dir.relative_to(ROOT)),
        "metrics": {key: float(winner[key]) for key in [
            "roc_auc", "accuracy", "f1", "precision", "recall"
        ]},
        "candidates": results,
        "notes": "Synthetic portfolio data; metrics do not imply production readiness.",
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Selected model: {best_name}")
    print(f"MLflow run: {best_run_id}")
    print(f"Local model: {model_dir}")
    print(f"Registered model name: {REGISTERED_MODEL_NAME}")
    print(f"Tracked model URI: {model_info.model_uri}")
    print("Next: start the API with `uvicorn churn_project.api:app --reload`.")


if __name__ == "__main__":
    main()
