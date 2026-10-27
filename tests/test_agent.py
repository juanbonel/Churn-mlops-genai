from churn_project.agent import run_assistant


def test_rag_fallback_returns_source_documents(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_assistant("¿Qué monitorea el drift y qué limitaciones tiene PSI?")
    assert result["mode"] == "deterministic_rag_and_tools"
    assert isinstance(result["sources"], list)
    assert result["answer"]
