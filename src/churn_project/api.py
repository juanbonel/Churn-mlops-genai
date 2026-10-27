"""FastAPI prediction and assistant endpoint for the portfolio project."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_project.data import FEATURES
from churn_project.scoring import load_model, score_customer

app = FastAPI(
    title="Subscription Churn MLOps API",
    version="0.1.0",
    description="Educational endpoint for the churn-mlops-genai portfolio project.",
)


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=1, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_tickets_90d: int = Field(ge=0, le=100)
    usage_gb: float = Field(ge=0, le=10000)
    payment_failures_90d: int = Field(ge=0, le=100)
    autopay: int = Field(ge=0, le=1)
    satisfaction_score: int = Field(ge=1, le=5)
    contract_type: Literal["monthly", "annual", "two_year"]


class AssistantRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    customer: CustomerFeatures | None = None


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        model_loaded = True
    except (FileNotFoundError, OSError, ImportError, RuntimeError):
        model_loaded = False
    return {"status": "ok" if model_loaded else "model_not_trained", "model_loaded": model_loaded}


@app.get("/features")
def features() -> dict:
    return {"required_features": FEATURES, "target": "churn", "classes": {"0": "no churn", "1": "churn"}}


@app.post("/predict")
def predict(customer: CustomerFeatures) -> dict:
    try:
        return score_customer(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/assistant")
def assistant(request: AssistantRequest) -> dict:
    try:
        from churn_project.agent import run_assistant

        customer = request.customer.model_dump() if request.customer else None
        return run_assistant(request.question, customer=customer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}") from exc
