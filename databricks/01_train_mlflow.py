# Databricks notebook source
# MAGIC %md
# MAGIC # Churn MLOps: entrenamiento y MLflow en Databricks
# MAGIC
# MAGIC Ejecutar en Databricks Free Edition usando un notebook Python con compute serverless.
# MAGIC Este notebook es autocontenido y crea datos sintéticos. Registra parámetros, métricas y el artefacto del modelo en el experimento MLflow asociado al notebook.
# MAGIC
# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# COMMAND ----------

SEED = 42
NUMERIC = [
    "tenure_months", "monthly_charges", "support_tickets_90d", "usage_gb",
    "payment_failures_90d", "autopay", "satisfaction_score",
]
CATEGORICAL = ["contract_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def make_data(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    charges = np.clip(rng.normal(72, 20, n), 18, 150).round(2)
    tickets = rng.poisson(1.25, n)
    usage = np.clip(rng.normal(95, 34, n), 1, 250).round(2)
    failures = rng.poisson(0.28, n)
    satisfaction = rng.integers(1, 6, n)
    contract = rng.choice(["monthly", "annual", "two_year"], n, p=[0.55, 0.30, 0.15])
    autopay = rng.binomial(1, 0.58, n)
    logit = (-1.25 + 0.032 * (charges - 70) - 0.035 * (tenure - 24)
             + 0.48 * tickets + 0.85 * failures + 0.78 * (contract == "monthly")
             - 0.42 * (contract == "annual") - 0.60 * autopay
             - 0.42 * (satisfaction - 3) - 0.004 * (usage - 90))
    probability = 1 / (1 + np.exp(-np.clip(logit, -15, 15)))
    churn = rng.binomial(1, probability, n)
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": charges,
        "support_tickets_90d": tickets, "usage_gb": usage,
        "payment_failures_90d": failures, "autopay": autopay,
        "satisfaction_score": satisfaction, "contract_type": contract,
        "churn": churn,
    })


df = make_data()
display(df.head())
print(f"Rows={len(df):,} | churn rate={df[TARGET].mean():.1%}")

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED, stratify=df[TARGET]
)


def make_pipeline(classifier):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=180, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    ),
}

runs = []
models = {}
for model_name, classifier in candidates.items():
    model = make_pipeline(classifier)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    models[model_name] = model
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": model_name, "random_state": SEED, "test_size": 0.2})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("dataset", "synthetic_subscription_churn_v1")
        mlflow.set_tag("purpose", "portfolio_demo_not_for_business_use")
    runs.append({"model_type": model_name, **metrics})

results_df = pd.DataFrame(runs).sort_values("roc_auc", ascending=False)
display(results_df)
best = results_df.iloc[0]
best_name = str(best["model_type"])
best_model = models[best_name]

# COMMAND ----------

# Log the selected model with a signature and example for reproducible inference.
signature = infer_signature(X_test, best_model.predict(X_test))
with mlflow.start_run(run_name="best_model_selected") as run:
    mlflow.log_params({"selected_model": best_name, "selection_metric": "roc_auc", "random_state": SEED})
    mlflow.log_metrics({key: float(best[key]) for key in ["roc_auc", "accuracy", "f1"]})
    mlflow.set_tag("model_role", "selected_candidate")
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
    )
    print("Selected model:", best_name)
    print("Run ID:", run.info.run_id)
    print("Model URI:", model_info.model_uri)

# COMMAND ----------

# Optional registry step:
# 1. Open the selected run in the MLflow UI and choose Register model.
# 2. If using Unity Catalog, set registry URI to "databricks-uc" and use a full name
#    like "<catalog>.<schema>.subscription_churn_model" with permissions to create it.
# Don't assume a catalog/schema name; use one available in your workspace.

# COMMAND ----------

# A basic feature reference sample for future drift monitoring.
reference_sample = X_train.sample(n=min(1200, len(X_train)), random_state=SEED)
display(reference_sample.describe(include="all").T)
