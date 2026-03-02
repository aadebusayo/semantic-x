from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from models.infosearch_api import Citation


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_title(user_message: str, max_words: int = 6) -> str:
    words = [w for w in user_message.strip().split() if w]
    if not words:
        return "New chat"
    title = " ".join(words[:max_words])
    return title[:80]


def make_extractive_answer(user_message: str, citations: List[Citation]) -> str:
    if not citations:
        return "I couldn't find relevant content in your indexed documents for that question."

    top = citations[0]
    parts: List[str] = []
    if top.excerpt:
        parts.append(top.excerpt.strip())

    # If we have more evidence, add one more snippet.
    if len(citations) > 1 and citations[1].excerpt:
        parts.append(citations[1].excerpt.strip())

    combined = " ".join([p for p in parts if p])
    if not combined:
        combined = "Relevant content was found, but no excerpt was available to summarize."

    # Keep it short.
    if len(combined) > 500:
        combined = combined[:497].rstrip() + "..."

    return combined
