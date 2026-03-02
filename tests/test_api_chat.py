from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_chat_repo,
    get_llm_service,
    get_search_service,
    get_signalr_service,
    get_suggestions_repo,
)

os.environ["ENABLE_INGESTION_WORKER"] = "false"
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = ""
os.environ["AZURE_STORAGE_CONTAINER"] = ""
os.environ["COSMOS_ENDPOINT"] = ""
os.environ["COSMOS_KEY"] = ""

from main import app
from models.infosearch_api import Citation


class DummySearch:
    async def search(self, *, query: str, document=None, top_k: int):
        return [
            Citation(
                document_name="doc-a.pdf",
                document_id="a",
                rank=1,
                score=0.99,
                excerpt="This is the most relevant excerpt.",
            )
        ]

    async def close(self):
        return None


class DummyLLM:
    async def answer(self, *, question: str, citations):
        return "Dummy LLM answer"

    async def generate_title(self, *, user_message: str):
        return "Dummy title"

    async def stream_answer(self, *, question: str, citations):
        for token in ["Dummy", " ", "LLM", " ", "answer"]:
            yield token


class DummyChatRepo:
    def __init__(self):
        self._items = {}

    async def open(self):
        return None

    async def close(self):
        return None

    async def get_chat(self, *, chat_id: str, user_id=None):
        return self._items.get(chat_id)

    async def append_turns(self, *, chat_id: str, user_id=None, title: str, user_message: str, assistant_message: str):
        now = datetime.now(timezone.utc).isoformat()
        existing = self._items.get(chat_id)
        if existing is None:
            existing = {
                "id": chat_id,
                "chatId": chat_id,
                "userId": user_id or "test-user",
                "title": title,
                "createdAt": now,
                "updatedAt": now,
                "turns": [],
            }

        existing["title"] = title
        existing["updatedAt"] = now
        existing["turns"].append({"role": "user", "content": user_message, "createdAt": now})
        existing["turns"].append({"role": "assistant", "content": assistant_message, "createdAt": now})
        self._items[chat_id] = existing
        return existing

    async def list_chats(self, *, user_id=None, limit: int = 20):
        items = sorted(self._items.values(), key=lambda item: item["updatedAt"], reverse=True)
        return items[:limit]


class DummySuggestionsRepo:
    def __init__(self):
        self._items = []

    async def open(self):
        return None

    async def close(self):
        return None

    async def upsert_questions(self, *, document_id=None, document_name=None, questions):
        item = {
            "documentId": document_id,
            "documentName": document_name,
            "questions": list(questions),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self._items.insert(0, item)
        return item

    async def list_recent(self, *, limit: int):
        return self._items[:limit]


class DummySignalR:
    async def open(self):
        return None

    async def close(self):
        return None

    async def publish(self, *, user_id=None, event: str, payload):
        return None


@pytest.fixture(autouse=True)
def override_search_service():
    chat_repo = DummyChatRepo()
    suggestions_repo = DummySuggestionsRepo()
    signalr = DummySignalR()

    def _override(_request=None):
        return DummySearch()

    def _override_llm(_request=None):
        return DummyLLM()

    def _override_chat_repo(_request=None):
        return chat_repo

    def _override_suggestions_repo(_request=None):
        return suggestions_repo

    def _override_signalr(_request=None):
        return signalr

    app.dependency_overrides[get_search_service] = _override
    app.dependency_overrides[get_llm_service] = _override_llm
    app.dependency_overrides[get_chat_repo] = _override_chat_repo
    app.dependency_overrides[get_suggestions_repo] = _override_suggestions_repo
    app.dependency_overrides[get_signalr_service] = _override_signalr
    yield
    app.dependency_overrides.clear()


def test_chat_roundtrip_and_history():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/messages",
            json={"message": "Summarize the document"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chat_id"]
        assert body["title"]
        assert "excerpt" in body["citations"][0]
        assert body["answer"]

        chat_id = body["chat_id"]

        hist = client.get("/api/v1/chats", params={"limit": 10})
        assert hist.status_code == 200
        items = hist.json()
        assert any(i["chat_id"] == chat_id for i in items)

        transcript = client.get(f"/api/v1/chats/{chat_id}")
        assert transcript.status_code == 200
        t = transcript.json()
        assert t["chat_id"] == chat_id
        assert len(t["turns"]) == 2
        assert t["turns"][0]["role"] == "user"
        assert t["turns"][1]["role"] == "assistant"


def test_document_suggestions_and_recent():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/documents/suggestions",
            json={"document": {"name": "lease.pdf"}, "text": "Lease agreement terms..."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["document"]["name"] == "lease.pdf"
        assert len(body["questions"]) == 3

        recent = client.get("/api/v1/suggestions/recent", params={"limit": 3})
        assert recent.status_code == 200
        items = recent.json()
        assert len(items) >= 1
