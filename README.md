# Churn MLOps + GenAI Portfolio Project

**Stack:** Python · scikit-learn · MLflow · FastAPI · Databricks Free Edition · Unity Catalog · Model Serving · PSI drift monitoring · GitHub Actions · RAG · tool-using agent · MCP

> **Scope and responsible use:** this is a personal learning/portfolio project built with synthetic data. It demonstrates engineering workflows, not a production-validated churn model. Predictions must not be used to make decisions about real customers.

## Project overview

This repository demonstrates an end-to-end machine-learning lifecycle for a synthetic customer-churn classification problem:

1. Generate reproducible synthetic data and train classification pipelines.
2. Compare Logistic Regression and Random Forest using ROC-AUC, F1 and accuracy.
3. Track runs, parameters, metrics and model artifacts with MLflow.
4. Register and version the selected model locally and, separately, in Databricks Unity Catalog.
5. Serve predictions through a local FastAPI REST API and a Databricks Model Serving endpoint.
6. Calculate feature-distribution drift with Population Stability Index (PSI), persist reports and schedule monitoring in Databricks.
7. Evaluate retraining candidates against an explicit acceptance gate before registering a new version.
8. Prototype a RAG assistant and tools, and expose local inference/reporting tools through an MCP server.
9. Run automated checks with GitHub Actions.

## Architecture

```text
Synthetic data
   |
   v
Feature preprocessing + train/test split
   |
   +--> Logistic Regression ----+
   |                            +--> evaluate --> choose candidate
   +--> Random Forest ----------+
                                  |
                                  v
                         MLflow experiment tracking
                                  |
                   +--------------+---------------+
                   |                              |
            Local model registry          Databricks Unity Catalog
                   |                              |
            FastAPI REST API               Model Serving endpoint
                   |                              |
                   +--------------+---------------+
                                  |
                                  v
                      PSI feature-drift monitoring
                         |                    |
                  Local JSON/CSV        Databricks Delta table
                                              |
                                      scheduled Databricks Job

Markdown/project docs --> TF-IDF retrieval --> RAG/agent
                                              |       |
                                        model tool   drift tool
                                              |
                                      optional local MCP server
```

## Repository layout

```text
src/churn_project/
  data.py          # reproducible synthetic data
  train.py         # training, evaluation and local MLflow tracking/registry
  scoring.py       # model loading and inference helpers
  api.py           # FastAPI endpoints
  monitor.py       # PSI drift monitor (local)
  rag.py           # TF-IDF document retrieval
  agent.py         # assistant with tool use and optional LLM
  mcp_server.py    # MCP stdio server
knowledge/         # Markdown documents indexed by local RAG
databricks/
  01_train_mlflow.py
  workspace_exports/
    01_train_mlflow.py
    02_monitor_drift.py
    03_retraining_gate.py
    04_genai_rag_agent.py
tests/             # automated tests
.github/workflows/ # GitHub Actions CI
```

The `databricks/workspace_exports/` files are source exports of the notebooks run in the Databricks workspace. The original local project files remain the reproducible local path; notebook exports document the cloud adaptation.

## Verified implementation status

The following steps were run and observed during development:

- **Local training:** the pipeline trained Logistic Regression and Random Forest, compared their metrics and selected the best candidate by ROC-AUC. One observed run reported ROC-AUC 0.805 / F1 0.643 for Logistic Regression and ROC-AUC 0.772 / F1 0.634 for Random Forest. Metrics depend on the synthetic sample and dependencies.
- **Local MLflow:** the run and model artifact were logged and a local registry version was created.
- **Local API:** FastAPI started successfully; a request to `/predict` returned a prediction and estimated churn probability.
- **Local drift monitor:** a simulated shift detected five of eight features above the PSI threshold of 0.20 in one observed run.
- **Tests:** `pytest` completed with **9 passed**. Two deprecation warnings were emitted by dependencies; they did not fail the tests.
- **GitHub Actions:** the workflow completed successfully after the project and notebook exports were pushed.
- **Databricks MLflow:** notebook runs were visible in Experiment Runs; the selected model was registered as `workspace.default.subscription_churn_model`, version 1, state `READY`.
- **Databricks Model Serving:** `churn-serving-endpoint` reached `Ready`; a test request returned `{"predictions": [1]}`.
- **Databricks drift monitoring:** a scheduled Job completed with `Succeeded`; reports were appended to `workspace.default.churn_drift_reports` (eight feature rows per run).
- **Controlled retraining gate:** a candidate was rejected because it did not meet the configured ROC-AUC gain and F1 non-regression criteria. No new registered version was created and the existing serving endpoint was not changed.
- **Local MCP:** MCP Inspector successfully invoked `predict_customer_churn` and `read_latest_drift_report` against the local model and local drift report.
- **GenAI/RAG notebook:** the notebook prototype was executed using a model accessible from the workspace. Availability of `ai_query` model names depends on workspace, region and permissions.

This list describes the observed portfolio workflow; it is not a claim of production operation, real-customer evaluation or continuous monitoring of live endpoint traffic. The local MCP tools read local artifacts; they are not the same as the Databricks Delta table or managed serving endpoint.

## 1. Run locally (Windows PowerShell)

Prerequisites: Python 3.11 recommended, Git, and PowerShell. Cloud execution is optional for the local workflow.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,genai]"
```

If PowerShell blocks activation, in that PowerShell session run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Train and track with MLflow

```powershell
python -m churn_project.train
```

This generates synthetic data, trains the candidate pipelines, logs metrics and artifacts, and registers the selected local model. Inspect experiments in a second terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://127.0.0.1:5000>.

### Start the REST API

```powershell
uvicorn churn_project.api:app --reload
```

Open <http://127.0.0.1:8000/docs>. Use `/health` to check model loading and `POST /predict` to send a synthetic customer row. The local API and Databricks Model Serving are separate deployment paths.

### Run local drift monitoring

```powershell
python -m churn_project.monitor
python -m churn_project.monitor --simulate-shift
```

The monitor writes JSON/CSV reports under `reports/`. PSI is a heuristic measure of feature-distribution change. It does not establish that model quality degraded; that requires labels and appropriate performance monitoring.

### Run tests

```powershell
pytest
```

GitHub Actions runs the automated checks on pushes and pull requests. Check the **Actions** tab for the actual status of each run.

## 2. Databricks Free Edition workflow

The cloud notebooks were exported under `databricks/workspace_exports/`:

1. `01_train_mlflow.py` — trains candidates and records experiment runs in MLflow.
2. `02_monitor_drift.py` — calculates PSI for a synthetic reference/current cohort and appends the report to Delta table `workspace.default.churn_drift_reports`.
3. `03_retraining_gate.py` — compares synthetic candidates against the registered incumbent and only registers a new model version if configured acceptance criteria pass.
4. `04_genai_rag_agent.py` — prototype RAG retrieval, an LLM-based planner/answerer and tools to read drift/model information and query the endpoint.

The registered model observed in the workspace is:

```text
workspace.default.subscription_churn_model
```

Version 1 was observed in `READY` state. The managed endpoint was named `churn-serving-endpoint`; it reached `Ready`, and a test invocation returned:

```json
{"predictions": [1]}
```

To reproduce this extension, import the notebook source files to your own Databricks Free Edition workspace, select the available Serverless compute, and execute the cells in order. Catalog/schema availability, model access, serving capacity and `ai_query` model availability depend on the workspace and permissions. Use existing catalog/schema values from your workspace instead of assuming they exist elsewhere.

**Monitoring scope:** the Databricks PSI Job currently operates on synthetic populations. It does not continuously consume production endpoint traffic, and the drift report does not measure model accuracy. Inference tables/telemetry would need to be configured separately for a real traffic-monitoring implementation.

**Retraining scope:** the gate is a portfolio demonstration based on explicit heuristic thresholds. Candidate metrics are evaluated on a shared synthetic test set for that run, but not on labeled real-world customer data. The endpoint is not automatically updated when a new candidate is registered.

## 3. GenAI, RAG and tools

The local project includes a TF-IDF retriever for Markdown project documentation and a deterministic fallback assistant. A real LLM can be configured for generated responses and tool selection where a compatible provider is available. The Databricks notebook prototype uses a workspace-accessible model via `ai_query`; exact model names and permissions vary by workspace.

Do not hard-code credentials. Configure provider keys as environment variables and check the provider's pricing before sending requests to paid APIs.

The prototype's tools include:

- Reading the latest drift summary.
- Looking up registered model versions.
- Querying the Databricks serving endpoint with a fixed synthetic sample.
- Retrieving project documentation.

The RAG implementation uses TF-IDF retrieval; it is not a production vector database or a fully evaluated retrieval system.

## 4. MCP server

The local MCP server exposes two tools:

- `predict_customer_churn` — inference using the local saved model.
- `read_latest_drift_report` — reads the latest local drift report.

Install the optional dependency and run the server from the activated virtual environment:

```powershell
python -m pip install -e ".[mcp]"
python -m churn_project.mcp_server
```

The server uses stdio transport and is intended to be launched by a compatible MCP client or MCP Inspector, not opened directly in a browser. The local tools read local artifacts; they do not query Databricks unless explicitly extended to do so.

## 5. Controlled retraining logic

The retraining gate compares a candidate model with the currently registered version. The example criteria were:

- Candidate ROC-AUC must improve by at least `0.005`.
- Candidate F1 must not fall by more than `0.02` relative to the incumbent.

An observed candidate was rejected: ROC-AUC improved from approximately `0.8017` to `0.8050`, but F1 decreased from approximately `0.8061` to `0.7720`. Both gate checks therefore failed, so no new model version was registered. This is intended behavior: a candidate should not be promoted simply because one metric improved slightly.

Thresholds are illustrative and should be selected based on business costs, validation design and model requirements in a real system.

## 6. Security and limitations

- All training/evaluation cohorts in this portfolio are synthetic.
- No claims are made about deployment for real customers or production reliability.
- PSI is a distribution-shift signal, not proof of concept drift or model degradation.
- The Databricks monitor is scheduled, but its current source is synthetic data, not live endpoint traffic.
- The endpoint was successfully queried in the workspace, but no public unauthenticated endpoint is provided by this repository.
- Do not commit `.env`, API keys, access tokens, local databases, generated models or personal data.
- Free Edition has resource quotas and feature limitations; it is a learning environment, not a production workspace.

## 7. How to describe this project honestly

> Built a reproducible personal MLOps and GenAI portfolio project using Python, scikit-learn, MLflow, FastAPI and Databricks Free Edition. Compared classification models, tracked experiments, registered a versioned model in Unity Catalog, deployed and queried a Databricks Model Serving endpoint, implemented PSI-based drift monitoring with scheduled Delta reporting, added a retraining acceptance gate, and prototyped RAG/tool use plus a local MCP server. The project uses synthetic data and is not a production implementation for real customers.

Only describe components as completed if you can explain the code, reproduce the execution and show the supporting evidence.

## References

- [Databricks Free Edition](https://www.databricks.com/learn/free-edition)
- [Databricks Model Serving](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/)
- [Databricks Unity Catalog models](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
