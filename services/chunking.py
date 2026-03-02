from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class TextChunk:
    index: int
    start: int
    end: int
    text: str


def chunk_text(*, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[TextChunk] = []
    i = 0
    pos = 0
    step = chunk_size - overlap

    while pos < len(cleaned):
        end = min(pos + chunk_size, len(cleaned))
        piece = cleaned[pos:end].strip()
        if piece:
            chunks.append(TextChunk(index=i, start=pos, end=end, text=piece))
            i += 1
        pos += step

    return chunks
