# Churn MLOps + GenAI: reproducible end-to-end portfolio project

**Stack:** Python · scikit-learn · MLflow · FastAPI · PSI drift monitoring · RAG · tool-using assistant · MCP (optional) · Databricks Free Edition · GitHub Actions

> **Portfolio disclaimer:** the dataset is synthetic and the model is educational. This repository demonstrates engineering patterns; it is not a production-certified churn model and must not be used to make decisions about real customers.

## Why this project?

The goal is to demonstrate a complete, reproducible ML lifecycle rather than a notebook containing only model training:

1. Generate deterministic data and split it reproducibly.
2. Train and compare Logistic Regression and Random Forest pipelines.
3. Track experiment parameters, metrics, model artifacts and tags with MLflow; register the selected model in the local Model Registry.
4. Expose inference as a REST API with validation and health checks.
5. Compare incoming feature distributions with a reference cohort using PSI and emit a JSON/CSV report.
6. Add a small RAG assistant that retrieves project documentation and can call tools to score a customer or read the drift report. A real LLM is optional; the fallback works without a paid API key.
7. Expose two of the same tools through an optional MCP server.
8. Run tests automatically with GitHub Actions.

## Architecture

```text
                deterministic synthetic data
                           |
                           v
                  feature preprocessing
                           |
             +-------------+-------------+
             |                           |
      Logistic Regression          Random Forest
             |                           |
             +------- evaluate ----------+
                           |
                   choose by ROC-AUC
                           |
             +-------------+-------------+
             |                           |
       MLflow Tracking             MLflow Registry
       params/metrics/runs          model versions
             |                           |
             +-------------+-------------+
                           v
                     saved pipeline
                           |
                           v
                  FastAPI REST endpoint
                   /predict  /health
                           |
                prediction inputs / cohorts
                           |
                           v
               PSI feature-drift monitor
                  JSON + CSV report

   Markdown knowledge ---> TF-IDF retrieval ---> RAG assistant
                                                  |
                                +-----------------+----------------+
                                |                                  |
                         score_customer                     get_drift_status
                                |                                  |
                           churn model                       drift report
                                |
                        optional MCP tools
```

## Repository layout

```text
src/churn_project/
  data.py          # deterministic synthetic data generator
  train.py         # training, evaluation, MLflow tracking/registration
  scoring.py       # model loading and common inference function
  api.py           # FastAPI: /health, /features, /predict, /assistant
  monitor.py       # feature drift using PSI
  rag.py           # local TF-IDF retriever over knowledge/*.md
  agent.py         # optional OpenAI-compatible LLM + tools; fallback mode
  mcp_server.py    # optional MCP stdio server
knowledge/         # documents indexed by the RAG retriever
 databricks/       # Databricks notebook source
sql/               # example SQL template for monitoring events
 tests/             # deterministic checks
.github/workflows/  # CI: train + tests + simulated drift check
```

## 1. Prerequisites

- Python **3.11** (recommended) and Git.
- Windows PowerShell, macOS or Linux.
- An account for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) only for the cloud extension; the local project does not require it.

The local setup is designed to use free, open-source Python libraries. An API key is **not** needed to train, track, serve predictions, run drift monitoring, use local retrieval, or run the deterministic agent fallback.

## 2. Run locally (Windows PowerShell)

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,genai]"
```

If PowerShell blocks activation, see Python's virtual environment instructions or activate the environment from your IDE. On macOS/Linux, use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

### Train, compare and track

```powershell
python -m churn_project.train
```

The command creates the synthetic dataset and artifacts under `data/`, registers runs in the local SQLite-backed MLflow Tracking/Model Registry, and writes the selected model under `models/churn_model/`. Expected terminal output includes ROC-AUC/F1 for both candidates, the selected model, run ID and model URI. Exact metrics may vary if dependency versions change.

Open a **second terminal** in the activated environment to view MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Inspect the `churn-mlops-genai` experiment. Runs contain model parameters, metrics and tags. The selected run logs the model to the registry as `subscription_churn_model`; each training execution can create a new model version.

### Start the REST endpoint

In another terminal:

```powershell
uvicorn churn_project.api:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Available routes:

- `GET /health`: checks whether the locally trained model can be loaded.
- `GET /features`: lists expected input fields.
- `POST /predict`: validates one customer's synthetic feature values and returns the prediction/probability.
- `POST /assistant`: retrieves relevant documentation and optionally calls the scoring/drift tools.

Example request in PowerShell:

```powershell
$body = @{
  tenure_months = 8
  monthly_charges = 95.0
  support_tickets_90d = 3
  usage_gb = 70.0
  payment_failures_90d = 1
  autopay = 0
  satisfaction_score = 2
  contract_type = "monthly"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body $body
```

The result includes `churn_prediction` (0/1), `churn_probability`, model metadata and a warning that the model uses synthetic data. The `POST /predict` endpoint is a local REST inference endpoint, not a public cloud deployment.

### Run drift monitoring

First, compare a new sample from the same synthetic data-generating process to the reference sample:

```powershell
python -m churn_project.monitor
```

Next, simulate a shifted population:

```powershell
python -m churn_project.monitor --simulate-shift
```

The monitor writes `reports/drift_report.json` and `reports/drift_report.csv`. The report includes PSI per feature, a configurable threshold (default `0.20`), the row counts and a list of features that crossed the threshold. To make an automated job fail when it detects drift:

```powershell
python -m churn_project.monitor --simulate-shift --fail-on-drift
```

**Interpretation matters:** PSI is a lightweight distribution-shift check. The default threshold is a heuristic, not a universal rule. It does not detect all forms of drift and does not establish that predictive performance has degraded. In a real system you would also monitor prediction volume, latency, errors, data quality and (when labels arrive) model performance, and define a governed retraining process. This demo generates a report; it does not automatically retrain or replace the model.

### Tests

```powershell
pytest
```

Tests cover reproducible data, feature-shift detection, feature schema validation and API input validation. GitHub Actions repeats model training, test execution and the shifted-drift smoke test on pushes and pull requests.

## 3. GenAI: RAG + tools + agent

The local retriever indexes the Markdown documents in `knowledge/` using TF-IDF. The assistant can:

- retrieve supporting project documentation and return source filenames;
- call `score_customer` to obtain an inference from the trained model;
- call `get_drift_status` to inspect the latest drift report;
- optionally pass retrieved context and tool definitions to an OpenAI-compatible LLM.

With no API key, `POST /assistant` runs in deterministic demo mode. To enable generated responses and model-selected tool calls, install the optional SDK and configure credentials locally. **API usage may incur charges depending on the provider and account; check the provider's pricing before using it. Never put secrets in source code or commit `.env`.**

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
$env:LLM_MODEL = "gpt-4o-mini"
# Optional for an OpenAI-compatible endpoint:
# $env:OPENAI_BASE_URL = "https://your-provider.example/v1"
```

Request example:

```json
{
  "question": "¿Qué probabilidad de churn tiene este cliente y qué factores conviene revisar?",
  "customer": {
    "tenure_months": 8,
    "monthly_charges": 95.0,
    "support_tickets_90d": 3,
    "usage_gb": 70.0,
    "payment_failures_90d": 1,
    "autopay": 0,
    "satisfaction_score": 2,
    "contract_type": "monthly"
  }
}
```

The assistant must treat the result as an educational score, not a factual explanation of causation. The score is correlational and based on simulated data.

## 4. Optional MCP server

The MCP server exposes two tools:

- `predict_customer_churn`: runs model inference with the required feature fields.
- `read_latest_drift_report`: returns the most recent saved drift report.

Install and run in the same virtual environment:

```powershell
python -m pip install -e ".[mcp]"
python -m churn_project.mcp_server
```

It uses the standard input/output (stdio) transport and is intended to be launched by an MCP-compatible client, rather than visited as a web page. Client configuration differs by application; configure the command as `python -m churn_project.mcp_server` and set its working directory to this repository. The model must have been trained first.

## 5. Try Databricks Free Edition

Databricks retired the legacy Community Edition in 2025 and offers [Free Edition](https://www.databricks.com/learn/free-edition) for personal learning and prototyping. Free Edition uses serverless compute with usage quotas and has feature limitations; it is not a production environment.

1. Create a Free Edition workspace and connect this GitHub repository through a Git folder, or upload `databricks/01_train_mlflow.py` as a Python notebook.
2. Open `databricks/01_train_mlflow.py` as a notebook and run the cells from top to bottom on available serverless compute. The notebook creates the same type of synthetic dataset, compares two models, logs metrics/parameters and stores the selected model as an MLflow artifact.
3. Open the notebook's MLflow experiment and inspect candidate runs and the `best_model_selected` run.
4. For model versioning in the Databricks environment, use the UI's **Register model** action for the selected logged model. If registering in Unity Catalog, use a catalog/schema that actually exists in your workspace and for which you have permissions; do not blindly copy a catalog name from a tutorial.
5. As an optional cloud-serving extension, open the registered model in the Databricks UI and check whether **Create serving endpoint** is available for that model/workspace. Use the UI's current access and resource limits, then send a test request to its REST API. Serving availability and limits can vary by workspace/edition; stop/remove any resources you do not need.

The notebook intentionally demonstrates MLflow experiment tracking without assuming a specific catalog/schema name. The local FastAPI endpoint is the reproducible serving path included in this repository. The Databricks notebook is a separate cloud adaptation, not a claim that the local SQLite registry transfers directly into Databricks.

## 6. Optional Docker

Train first so `models/churn_model/` and `models/model_metadata.json` exist, then from the repository root:

```powershell
docker build -t churn-mlops-genai .
docker run --rm -p 8000:8000 churn-mlops-genai
```

Test `http://127.0.0.1:8000/docs`. Do not commit model files just to make them available in a container; in a more mature workflow, CI builds a versioned artifact/image and the deployment process retrieves a model by its registered version.

## 7. Push to GitHub

Create an empty GitHub repository and run:

```powershell
git init
git add .
git commit -m "Build reproducible churn MLOps and GenAI demo"
git branch -M main
git remote add origin https://github.com/YOUR_USER/churn-mlops-genai.git
git push -u origin main
```

Check before pushing that `.env`, local databases, generated reports and trained models are ignored by `.gitignore`. The repository includes a CI workflow at `.github/workflows/ci.yml`.

## 8. Suggested development sequence

| Iteration | Deliverable | Evidence for a technical interview |
|---|---|---|
| 1 | Train and evaluate candidates | Metrics, split strategy, data generator and feature pipeline |
| 2 | MLflow tracking + registry | Experiment comparison, run ID and model version |
| 3 | REST serving | API docs, request validation, health route and test |
| 4 | Drift monitor | Reference/current comparison, configurable threshold and report |
| 5 | GitHub Actions | Green tests and repeatable smoke test |
| 6 | RAG + tool-using assistant | Retrieved source documents plus observable tool calls |
| 7 | MCP | Two discoverable tools for compatible clients |
| 8 | Databricks serving (optional) | Screenshot/endpoint invocation and clear notes about quota/limits |

## 9. How to describe it honestly

A suitable CV/interview description after you have run and understood the project:

> Built a reproducible end-to-end MLOps portfolio project using Python, scikit-learn and MLflow: compared classification pipelines, tracked experiments and registered a model, exposed inference through FastAPI, added PSI-based feature-drift monitoring, automated tests in GitHub Actions, and prototyped a RAG assistant with tools and optional MCP integration. Practiced a Databricks Free Edition workflow. The data is synthetic; this was a personal project, not a production deployment.

Only say you used Databricks, deployed a cloud serving endpoint, or integrated an LLM/MCP client after you actually complete and verify those respective optional steps.

## References

- [Databricks Free Edition](https://www.databricks.com/learn/free-edition)
- [Databricks Free Edition limitations](https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition-limitations)
- [Databricks Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [MLflow local model serving](https://mlflow.org/docs/latest/ml/deployment/deploy-model-locally/)
