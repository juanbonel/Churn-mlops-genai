"""Generate a deterministic, synthetic subscription-churn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "usage_gb",
    "payment_failures_90d",
    "autopay",
    "satisfaction_score",
]
CATEGORICAL_FEATURES = ["contract_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def make_churn_dataset(
    n_samples: int = 3000,
    random_state: int = 42,
    shift: bool = False,
) -> pd.DataFrame:
    """Return a reproducible synthetic dataset.

    ``shift=True`` simulates a production population whose feature distributions
    differ from the training reference. The function intentionally contains no
    real customer data and is for education/testing, not business decisions.
    """
    if n_samples < 50:
        raise ValueError("n_samples must be at least 50")

    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, size=n_samples)
    charges = np.clip(rng.normal(72, 20, size=n_samples), 18, 150).round(2)
    tickets = rng.poisson(1.25, size=n_samples)
    usage = np.clip(rng.normal(95, 34, size=n_samples), 1, 250).round(2)
    payment_failures = rng.poisson(0.28, size=n_samples)
    satisfaction = rng.integers(1, 6, size=n_samples)

    if shift:
        # A deliberately shifted cohort to exercise the drift monitor.
        tenure = np.clip(tenure - rng.integers(3, 15, size=n_samples), 1, 72)
        charges = np.clip(charges + rng.normal(15, 6, size=n_samples), 18, 170).round(2)
        tickets = tickets + rng.poisson(1.7, size=n_samples)
        usage = np.clip(usage * rng.normal(0.72, 0.12, size=n_samples), 1, 250).round(2)
        payment_failures = payment_failures + rng.binomial(1, 0.35, size=n_samples)
        satisfaction = np.clip(satisfaction - rng.binomial(1, 0.55, size=n_samples), 1, 5)
        contract_probs = [0.78, 0.17, 0.05]
        autopay_prob = 0.35
    else:
        contract_probs = [0.55, 0.30, 0.15]
        autopay_prob = 0.58

    contract_type = rng.choice(
        ["monthly", "annual", "two_year"], size=n_samples, p=contract_probs
    )
    autopay = rng.binomial(1, autopay_prob, size=n_samples)

    # Hidden synthetic relationship used only to create a plausible label.
    logit = (
        -1.25
        + 0.032 * (charges - 70)
        - 0.035 * (tenure - 24)
        + 0.48 * tickets
        + 0.85 * payment_failures
        + 0.78 * (contract_type == "monthly")
        - 0.42 * (contract_type == "annual")
        - 0.60 * autopay
        - 0.42 * (satisfaction - 3)
        - 0.004 * (usage - 90)
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -15, 15)))
    churn = rng.binomial(1, probability, size=n_samples)

    return pd.DataFrame(
        {
            "tenure_months": tenure.astype(int),
            "monthly_charges": charges.astype(float),
            "support_tickets_90d": tickets.astype(int),
            "usage_gb": usage.astype(float),
            "payment_failures_90d": payment_failures.astype(int),
            "autopay": autopay.astype(int),
            "satisfaction_score": satisfaction.astype(int),
            "contract_type": contract_type,
            "churn": churn.astype(int),
        }
    )
