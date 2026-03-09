from __future__ import annotations

from typing import List, Optional


def generate_quick_questions(*, document_name: Optional[str], text: Optional[str], limit: int = 3) -> List[str]:
    name = (document_name or "this document").strip() or "this document"

    # Lean, non-LLM heuristics.
    candidates = [
        f"What is {name} about?",
        f"What are the key points in {name}?",
        f"What actions or next steps are mentioned in {name}?",
        f"What dates, names, or numbers should I pay attention to in {name}?",
    ]

    # If we have text, add a summarization question at the front.
    if text and text.strip():
        candidates.insert(0, f"Summarize {name}")

    # Deduplicate while preserving order.
    out: List[str] = []
    for q in candidates:
        if q not in out:
            out.append(q)
        if len(out) >= limit:
            break

    return out
