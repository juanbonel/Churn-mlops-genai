# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Churn MLOps: entrenamiento y MLflow en Databricks
# MAGIC
# MAGIC Ejecutar en Databricks Free Edition usando un notebook Python con compute serverless.
# MAGIC Este notebook es autocontenido y crea datos sintéticos. Registra parámetros, métricas y el artefacto del modelo en el experimento MLflow asociado al notebook.

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

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

model_to_find = "subscription_churn_model"

for registry_uri in ["databricks-uc", "databricks"]:
    print(f"\n--- Revisando registry: {registry_uri} ---")

    try:
        mlflow.set_registry_uri(registry_uri)
        client = MlflowClient()

        models = client.search_registered_models()

        matches = [
            model for model in models
            if model.name == model_to_find
            or model.name.endswith("." + model_to_find)
        ]

        if not matches:
            print("No se encontró el modelo con este nombre.")
            continue

        for model in matches:
            print("MODELO REGISTRADO:", model.name)

            for version in model.latest_versions:
                print("  Versión:", version.version)
                print("  Estado:", version.status)
                print("  Run ID:", version.run_id)

    except Exception as e:
        print(f"No se pudo consultar este registry: {e}")

# COMMAND ----------

# Consultar el catálogo y esquema actuales
spark.sql("""
    SELECT
        current_catalog() AS catalog,
        current_schema() AS schema
""").show(truncate=False)

# Mostrar los catálogos disponibles
spark.sql("SHOW CATALOGS").show(truncate=False)

# COMMAND ----------

SHOW SCHEMAS IN workspace;

# COMMAND ----------

# MAGIC %md
# MAGIC cargar el modelo registrado y predecir

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# 1. Configurar el registro de Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "workspace.default.subscription_churn_model"
client = MlflowClient(registry_uri="databricks-uc")

# 2. Consultar las versiones registradas
versions = client.search_model_versions(
    f"name='{model_name}'"
)

print("Versiones encontradas:")
for v in versions:
    print(f"Versión {v.version} | Estado: {v.status}")

# 3. Seleccionar la última versión lista
ready_versions = [
    v for v in versions if v.status == "READY"
]

if not ready_versions:
    raise RuntimeError(
        "No se encontró ninguna versión READY del modelo."
    )

version = max(ready_versions, key=lambda v: int(v.version))
model_uri = f"models:/{model_name}/{version.version}"

print("\nCargando:", model_uri)

# 4. Cargar el modelo registrado
model = mlflow.pyfunc.load_model(model_uri)

# 5. Crear un cliente sintético de prueba
sample = pd.DataFrame([{
    "tenure_months": 12,
    "monthly_charges": 85.5,
    "support_tickets_90d": 2,
    "usage_gb": 150.0,
    "payment_failures_90d": 1,
    "autopay": 0,
    "satisfaction_score": 3,
    "contract_type": "monthly"
}])

# 6. Realizar la predicción
prediction = model.predict(sample)

print("\nModelo utilizado:", model_uri)
print("Predicción de churn:", int(prediction[0]))
print("0 = No abandono predicho | 1 = Abandono predicho")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md Este monitor compara una población de referencia con otra población sintética en la que introducimos cambios deliberados. Calcula PSI por variable y guarda el reporte en workspace.default.churn_drift_reports, utilizando el formato de tabla de tres niveles de Unity Catalog.

# COMMAND ----------

# DBTITLE 1,Drift monitor: guardar reporte PSI en Delta
# Tabla Delta destino en Unity Catalog
table_name = "workspace.default.churn_drift_reports"

# 1. Crear explícitamente la tabla si todavía no existe
spark.sql("""
CREATE TABLE IF NOT EXISTS workspace.default.churn_drift_reports (
    feature STRING,
    psi DOUBLE,
    threshold DOUBLE,
    drift_detected BOOLEAN,
    reference_rows BIGINT,
    current_rows BIGINT,
    monitor_run_ts STRING
)
USING DELTA
""")

print("Tabla creada o ya existente:", table_name)

# 2. Población actual sintética con cambios deliberados vs. la referencia (celda 7)
from datetime import datetime, timezone

current_sample = df[FEATURES].sample(n=len(reference_sample), random_state=SEED).copy()
current_sample["tenure_months"] = np.maximum(current_sample["tenure_months"] - 8, 1)
current_sample["monthly_charges"] = (current_sample["monthly_charges"] * 1.20).clip(18, 150)
current_sample["support_tickets_90d"] = current_sample["support_tickets_90d"] + 1

# 3. Calcular PSI por variable (bins por cuantiles de la referencia)
PSI_BINS = 10
PSI_THRESHOLD = 0.2


def compute_psi(expected, actual):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.dtype.kind in "OUS":  # variable categórica
        categories = np.union1d(np.unique(expected), np.unique(actual))
        p = np.array([(expected == c).mean() for c in categories])
        q = np.array([(actual == c).mean() for c in categories])
    else:  # variable numérica
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, PSI_BINS + 1)))
        inner = edges[1:-1]
        p = np.bincount(np.searchsorted(inner, expected, side="right"), minlength=len(inner) + 1) / len(expected)
        q = np.bincount(np.searchsorted(inner, actual, side="right"), minlength=len(inner) + 1) / len(actual)
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


monitor_run_ts = datetime.now(timezone.utc).isoformat()
rows = []
for feature in FEATURES:
    psi_value = compute_psi(reference_sample[feature], current_sample[feature])
    rows.append({
        "feature": feature,
        "psi": round(psi_value, 4),
        "threshold": PSI_THRESHOLD,
        "drift_detected": psi_value > PSI_THRESHOLD,
        "reference_rows": len(reference_sample),
        "current_rows": len(current_sample),
        "monitor_run_ts": monitor_run_ts,
    })
report = pd.DataFrame(rows)
display(report)

# 4. Guardar el reporte en la tabla
spark_report = spark.createDataFrame(report)

(
    spark_report.write
    .format("delta")
    .mode("append")
    .saveAsTable(table_name)
)

print("Reporte guardado correctamente.")

# 3. Leer y mostrar los registros guardados
display(
    spark.table(table_name)
    .orderBy("monitor_run_ts", ascending=False)
)

# COMMAND ----------

display(
    spark.table("workspace.default.churn_drift_reports")
    .orderBy("monitor_run_ts", ascending=False)
)