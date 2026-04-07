from __future__ import annotations

import re
from typing import List, Optional


def _clean_title_words(value: str, max_words: int = 5) -> str:
    words = [word for word in re.split(r"\s+", value.strip()) if word]
    return " ".join(words[:max_words])[:80].strip()


def _humanize_document_name(document_name: Optional[str]) -> str:
    raw_name = (document_name or "").strip()
    if not raw_name:
        return "Document Summary"

    filename = raw_name.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    tokens = re.split(r"[_\-\s]+", stem)

    cleaned_tokens: List[str] = []
    for token in tokens:
        value = token.strip()
        if not value:
            continue
        if len(value) >= 12 and re.fullmatch(r"[0-9a-f]+", value, re.IGNORECASE):
            continue
        if len(value) >= 16 and sum(ch.isdigit() for ch in value) >= max(4, len(value) // 3):
            continue
        if len(value) >= 14 and not re.search(r"[aeiou]", value, re.IGNORECASE):
            continue
        cleaned_tokens.append(value)

    if not cleaned_tokens:
        return "Document Summary"

    title = _clean_title_words(" ".join(cleaned_tokens))
    return title.title() if title else "Document Summary"


def build_suggestion_title(*, document_name: Optional[str], text: Optional[str] = None) -> str:
    snippet = (text or "").strip()
    if snippet:
        for line in snippet.splitlines():
            candidate = re.sub(r"\s+", " ", line).strip(" -:#\t")
            if len(candidate) < 6:
                continue
            if len(candidate.split()) > 10:
                continue
            if not re.search(r"[A-Za-z]", candidate):
                continue
            title = _clean_title_words(candidate)
            if title:
                return title

    return _humanize_document_name(document_name)


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
