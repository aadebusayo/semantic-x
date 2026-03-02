from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ReactionRequest(BaseModel):
    reaction: Literal["up", "down", "none"]


class DocumentRef(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class Citation(BaseModel):
    document_name: Optional[str] = None
    document_id: Optional[str] = None
    rank: int = Field(..., ge=1)
    score: Optional[float] = None
    excerpt: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    document: Optional[DocumentRef] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class ChatMessageResponse(BaseModel):
    chat_id: str
    title: str
    answer: str
    citations: List[Citation]
    created_at: datetime


class ChatHistoryItem(BaseModel):
    chat_id: str
    title: str
    updated_at: datetime


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatTranscript(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    turns: List[ChatTurn]


class QuickQuestionsCreateRequest(BaseModel):
    document: DocumentRef
    text: Optional[str] = Field(
        default=None,
        description="Optional document text. If omitted, questions will be generic.",
    )


class QuickQuestionsResponse(BaseModel):
    document: DocumentRef
    questions: List[str]
    created_at: datetime


class RecentQuickQuestionsItem(BaseModel):
    document: DocumentRef
    questions: List[str]
    created_at: datetime
