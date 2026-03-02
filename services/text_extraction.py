from __future__ import annotations

import io
from typing import Optional

from pypdf import PdfReader


def _is_text_blob(name: str) -> bool:
    lower = (name or "").lower()
    return lower.endswith(".txt") or lower.endswith(".md") or lower.endswith(".csv")


def _is_pdf(name: str) -> bool:
    return (name or "").lower().endswith(".pdf")


def extract_text(*, blob_name: str, content: bytes) -> Optional[str]:
    if not content:
        return None

    if _is_text_blob(blob_name):
        # Best-effort decode.
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(enc, errors="ignore")
            except Exception:
                continue
        return None

    if _is_pdf(blob_name):
        reader = PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        joined = "\n\n".join(texts).strip()
        return joined or None

    # Unsupported file types are intentionally skipped to keep the service lean.
    return None
