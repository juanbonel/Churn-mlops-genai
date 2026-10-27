"""Small tool-using assistant backed by RAG and the trained churn model.

With OPENAI_API_KEY, it uses an OpenAI-compatible Chat Completions API with tools.
Without a key it remains useful in a deterministic retrieval/tool-demo mode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from churn_project.rag import retrieve_context
from churn_project.scoring import score_customer

ROOT = Path(__file__).resolve().parents[2]
DRIFT_REPORT = ROOT / "reports" / "drift_report.json"

SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_customer",
        "description": "Predict churn probability for one customer using the project's trained model.",
        "parameters": {
            "type": "object",
            "properties": {
                "tenure_months": {"type": "integer"},
                "monthly_charges": {"type": "number"},
                "support_tickets_90d": {"type": "integer"},
                "usage_gb": {"type": "number"},
                "payment_failures_90d": {"type": "integer"},
                "autopay": {"type": "integer", "enum": [0, 1]},
                "satisfaction_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "contract_type": {"type": "string", "enum": ["monthly", "annual", "two_year"]},
            },
            "required": [
                "tenure_months", "monthly_charges", "support_tickets_90d", "usage_gb",
                "payment_failures_90d", "autopay", "satisfaction_score", "contract_type",
            ],
            "additionalProperties": False,
        },
    },
}
DRIFT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_drift_status",
        "description": "Read the latest saved input-feature drift report.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
TOOLS = [SCORE_TOOL, DRIFT_TOOL]


def get_drift_status() -> dict[str, Any]:
    if not DRIFT_REPORT.exists():
        return {
            "status": "REPORT_NOT_FOUND",
            "message": "Run `python -m churn_project.monitor --simulate-shift` first.",
        }
    return json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))


def _dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "score_customer":
        return score_customer(arguments)
    if name == "get_drift_status":
        return get_drift_status()
    return {"error": f"Tool not found: {name}"}


def _fallback_answer(question: str, customer: dict[str, Any] | None, context: list[dict]) -> dict[str, Any]:
    q = question.lower()
    tool_results = []
    if customer is not None:
        result = _dispatch_tool("score_customer", customer)
        tool_results.append({"tool": "score_customer", "result": result})
    elif any(term in q for term in ["drift", "deriva", "monitoreo", "monitor"]):
        result = get_drift_status()
        tool_results.append({"tool": "get_drift_status", "result": result})

    if tool_results and tool_results[0]["tool"] == "score_customer":
        prediction = tool_results[0]["result"]
        answer = (
            f"La predicción del modelo indica una probabilidad estimada de churn de "
            f"{prediction['churn_probability']:.1%} (clase {prediction['churn_prediction']}). "
            "Es una demostración sobre datos sintéticos, no una recomendación para decisiones reales."
        )
    elif tool_results and tool_results[0]["tool"] == "get_drift_status":
        result = tool_results[0]["result"]
        answer = (
            f"Estado del último reporte: {result.get('status', 'desconocido')}. "
            f"Variables con drift: {', '.join(result.get('drifted_features', [])) or 'ninguna reportada'}. "
            "El PSI detecta cambios de distribución en features; no demuestra concept drift ni degradación del modelo."
        )
    elif context:
        answer = "Modo demo sin LLM: recuperé los fragmentos de documentación más relacionados. " \
            "Para generar una respuesta redactada por un LLM, configura OPENAI_API_KEY.\n\n" + "\n\n".join(
                f"[{item['source']}] {item['text']}" for item in context
            )
    else:
        answer = "No encontré contexto suficiente en la documentación local."

    return {
        "answer": answer,
        "mode": "deterministic_rag_and_tools",
        "sources": [{"source": item["source"], "score": item["score"]} for item in context],
        "tools_used": [item["tool"] for item in tool_results],
        "tool_results": tool_results,
    }


def run_assistant(question: str, customer: dict[str, Any] | None = None) -> dict[str, Any]:
    context = retrieve_context(question, top_k=3)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback_answer(question, customer, context)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install the optional dependency with `pip install -e ".[genai]"`.') from exc

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    retrieved_text = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}" for item in context
    ) or "No local documents were retrieved."
    system_prompt = (
        "You are a careful assistant for a portfolio MLOps demo. Answer in Spanish unless asked otherwise. "
        "Use retrieved documents as evidence, identify uncertainty, and do not describe the synthetic model as production-ready. "
        "Use score_customer only when all required customer features are available. Use get_drift_status for drift questions. "
        "Explain that PSI monitors feature distribution, not concept drift.\n\n"
        f"Retrieved project context:\n{retrieved_text}"
    )
    user_prompt = question
    if customer is not None:
        user_prompt += "\n\nCustomer features available for the score_customer tool:\n" + json.dumps(customer, ensure_ascii=False)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tools_used: list[str] = []
    tool_results: list[dict[str, Any]] = []

    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return {
                "answer": message.content or "El modelo no devolvió texto.",
                "mode": "llm_rag_tool_agent",
                "llm_model": model,
                "sources": [{"source": item["source"], "score": item["score"]} for item in context],
                "tools_used": tools_used,
                "tool_results": tool_results,
            }
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                result = _dispatch_tool(call.function.name, arguments)
            except Exception as exc:  # Surface tool errors to the model rather than crashing the API.
                result = {"error": str(exc)}
            tools_used.append(call.function.name)
            tool_results.append({"tool": call.function.name, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
    # A final call without tools encourages a natural-language synthesis after tool execution.
    final_response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2
    )
    return {
        "answer": final_response.choices[0].message.content or "El modelo no devolvió texto.",
        "mode": "llm_rag_tool_agent",
        "llm_model": model,
        "sources": [{"source": item["source"], "score": item["score"]} for item in context],
        "tools_used": tools_used,
        "tool_results": tool_results,
    }
