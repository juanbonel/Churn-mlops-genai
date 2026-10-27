"""Minimal local retrieval-augmented generation (RAG) component."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "knowledge"


def _load_chunks() -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for part in content.split("\n\n"):
            part = part.strip()
            if len(part) > 35:
                chunks.append({"source": path.name, "text": part})
    return chunks


def retrieve_context(question: str, top_k: int = 3) -> list[dict[str, str | float]]:
    """Retrieve the most relevant Markdown chunks using local TF-IDF similarity."""
    chunks = _load_chunks()
    if not chunks:
        return []
    corpus = [item["text"] for item in chunks]
    vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [question])
    similarities = (matrix[-1] @ matrix[:-1].T).toarray().ravel()
    order = np.argsort(similarities)[::-1][: min(top_k, len(chunks))]
    return [
        {"source": chunks[int(i)]["source"], "text": chunks[int(i)]["text"], "score": round(float(similarities[int(i)]), 4)}
        for i in order
        if similarities[int(i)] > 0
    ]
