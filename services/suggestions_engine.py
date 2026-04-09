from __future__ import annotations

import re
from typing import List, Optional


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "if", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "will", "with", "your", "you", "about", "document", "summary", "what", "which", "who",
}

_NOISY_TOKENS = {
    "am", "pm", "rev", "jpg", "jpeg", "png", "pdf", "http", "https", "www", "drive",
    "copy", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "iso", "nca", "los",
}


def _clean_title_words(value: str, max_words: int = 5) -> str:
    words = [word for word in re.split(r"\s+", value.strip()) if word]
    return " ".join(words[:max_words])[:80].strip()


def is_opaque_document_name(document_name: Optional[str]) -> bool:
    raw_name = (document_name or "").strip()
    if not raw_name:
        return True

    filename = raw_name.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    if len(stem) >= 24 and re.fullmatch(r"[0-9a-f]+", stem, re.IGNORECASE):
        return True

    tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]
    if not tokens:
        return True

    opaque_tokens = 0
    for token in tokens:
        digits = sum(ch.isdigit() for ch in token)
        if len(token) >= 16 and digits >= max(4, len(token) // 3):
            opaque_tokens += 1
            continue
        if len(token) >= 14 and not re.search(r"[aeiou]", token, re.IGNORECASE):
            opaque_tokens += 1
    return opaque_tokens == len(tokens)


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


def build_suggestion_subject(*, document_name: Optional[str], text: Optional[str] = None, title: Optional[str] = None) -> str:
    if not is_opaque_document_name(document_name):
        return (document_name or "this document").strip() or "this document"

    candidate = (title or "").strip()
    if candidate:
        return candidate

    fallback = build_suggestion_title(document_name=document_name, text=text)
    return fallback if fallback != "Document Summary" else "this document"


def _normalize_source_text(text: Optional[str]) -> str:
    value = (text or "")
    return (
        value.replace("\x00", " ")
        .replace("ﬀ", "ff")
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
    )


def _extract_key_phrases(text: Optional[str], *, max_phrases: int = 3) -> List[str]:
    normalized = _normalize_source_text(text)
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", normalized)
        if token.lower() not in _STOPWORDS
    ]
    if not tokens:
        return []

    phrases: List[str] = []
    seen = set()
    for size in (3, 2):
        for idx in range(len(tokens) - size + 1):
            phrase_tokens = tokens[idx : idx + size]
            if any(token in _STOPWORDS for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            if phrase in seen:
                continue
            phrase_tokens = phrase.split()
            if any(token in _NOISY_TOKENS for token in phrase_tokens):
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def _document_type_questions(*, subject: str, title: Optional[str], text: Optional[str]) -> List[str]:
    source = f"{title or ''} {_normalize_source_text(text)}".lower()
    questions: List[str] = []

    def add(question: str) -> None:
        if question not in questions:
            questions.append(question)

    if any(word in source for word in ("handbook", "policy", "guideline", "manual")):
        add(f"What rules or expectations does {subject} define?")
        add(f"What responsibilities are outlined in {subject}?")

    if any(word in source for word in ("requirement", "specification", "prd", "scope", "objective")):
        add(f"What requirements are outlined in {subject}?")
        add(f"What objectives or scope does {subject} define?")

    if any(word in source for word in ("collection", "inventory", "asset", "register", "list")):
        add(f"What items or records are listed in {subject}?")
        add(f"What ownership or status details appear in {subject}?")

    if any(word in source for word in ("assessment", "exam", "test", "paper", "question")):
        add(f"What topics or questions are covered in {subject}?")
        add(f"How is {subject} structured or organized?")

    if any(word in source for word in ("cv", "resume", "experience", "employment", "profile")):
        add(f"What experience or qualifications are highlighted in {subject}?")
        add(f"What skills or roles are mentioned in {subject}?")

    if any(word in source for word in ("request", "approval", "letter", "application", "sign")):
        add(f"What action or approval is being requested in {subject}?")
        add(f"Who is involved and what decision is needed in {subject}?")

    if any(word in source for word in ("report", "analysis", "impact", "study", "findings")):
        add(f"What findings or conclusions are presented in {subject}?")
        add(f"What evidence or examples support {subject}?")

    if any(word in source for word in ("invoice", "payment", "receipt", "bill", "amount")):
        add(f"What amounts, dates, or payment details appear in {subject}?")
        add(f"What transaction details are recorded in {subject}?")

    return questions


def _content_aware_questions(*, subject: str, text: Optional[str]) -> List[str]:
    normalized = _normalize_source_text(text).lower()
    questions: List[str] = []

    def add(question: str) -> None:
        if question not in questions:
            questions.append(question)

    add(f"What is the main message of {subject}?")

    title_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", subject)
        if token.lower() not in _STOPWORDS and token.lower() not in _NOISY_TOKENS
    }

    for question in _document_type_questions(subject=subject, title=subject, text=text):
        add(question)

    phrases = _extract_key_phrases(text)
    for phrase in phrases:
        phrase_tokens = set(phrase.split())
        if phrase_tokens and phrase_tokens.issubset(title_tokens):
            continue
        add(f"What does {subject} say about {phrase}?")

    if "positive" in normalized and "negative" in normalized:
        add(f"What positive and negative effects are discussed in {subject}?")
    if any(word in normalized for word in ("statistic", "data", "metric", "figure")):
        add(f"What data or evidence does {subject} cite?")
    if any(word in normalized for word in ("solution", "recommend", "actionable", "improve", "management")):
        add(f"What actions or recommendations are proposed in {subject}?")
    if any(word in normalized for word in ("policy", "rule", "compliance", "requirement", "responsibil")):
        add(f"What requirements or responsibilities does {subject} define?")
    if any(word in normalized for word in ("process", "procedure", "step", "workflow")):
        add(f"What process or steps are described in {subject}?")
    if any(word in normalized for word in ("risk", "issue", "challenge", "problem")):
        add(f"What risks or challenges are highlighted in {subject}?")

    return questions


def generate_quick_questions(*, document_name: Optional[str], text: Optional[str], limit: int = 3) -> List[str]:
    name = build_suggestion_subject(document_name=document_name, text=text)

    candidates = _content_aware_questions(subject=name, text=text)
    if not candidates:
        candidates = [
            f"What is the main message of {name}?",
            f"What are the key points in {name}?",
            f"What actions or next steps are mentioned in {name}?",
            f"What dates, names, or numbers should I pay attention to in {name}?",
        ]

    # Deduplicate while preserving order.
    out: List[str] = []
    for q in candidates:
        if q not in out:
            out.append(q)
        if len(out) >= limit:
            break

    return out
