"""Optional MCP server exposing the local ML model and drift report as tools.

Install with: pip install -e ".[mcp]"
Run with: python -m churn_project.mcp_server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from churn_project.agent import get_drift_status
from churn_project.scoring import score_customer

mcp = FastMCP("churn-mlops-genai")


@mcp.tool()
def predict_customer_churn(
    tenure_months: int,
    monthly_charges: float,
    support_tickets_90d: int,
    usage_gb: float,
    payment_failures_90d: int,
    autopay: int,
    satisfaction_score: int,
    contract_type: str,
) -> dict[str, Any]:
    """Predict churn probability for synthetic-demo customer features."""
    return score_customer(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_tickets_90d": support_tickets_90d,
            "usage_gb": usage_gb,
            "payment_failures_90d": payment_failures_90d,
            "autopay": autopay,
            "satisfaction_score": satisfaction_score,
            "contract_type": contract_type,
        }
    )


@mcp.tool()
def read_latest_drift_report() -> dict[str, Any]:
    """Read the latest saved PSI-based feature-drift report."""
    return get_drift_status()


if __name__ == "__main__":
    mcp.run(transport="stdio")
