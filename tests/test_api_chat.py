from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_blob_source,
    get_chat_repo,
    get_document_catalog,
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
from services.document_catalog import DocumentCatalogService


class DummySearch:
    async def search(self, *, query: str, document=None, top_k: int):
        return [
            Citation(
                document_name=(document.name if document and document.name else "doc-a.pdf"),
                document_id=(document.id if document and document.id else "aHR0cHM6Ly9leGFtcGxlL2RvYy1hLnBkZg=="),
                rank=1,
                score=0.99,
                excerpt="This is the most relevant excerpt.",
            )
        ]

    async def preview_document(self, *, document, top_k: int):
        return [
            Citation(
                document_name=document.name or "manual.pdf",
                document_id=document.id or "folder/manual.pdf",
                rank=1,
                score=0.75,
                excerpt="Preview excerpt from the indexed document.",
            )
        ]

    async def close(self):
        return None


class DummyLLM:
    async def answer(self, *, question: str, citations, history=None):
        return "Dummy LLM answer"

    async def generate_title(self, *, user_message: str):
        return "Dummy title"

    async def stream_answer(self, *, question: str, citations, history=None):
        for token in ["Dummy", " ", "LLM", " ", "answer"]:
            yield token

    async def generate_suggestions(self, *, document_name=None, text=None, limit=3):
        base = document_name or "document"
        return [f"Question {idx} about {base}" for idx in range(1, limit + 1)]


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
        self.deleted = []

    async def open(self):
        return None

    async def close(self):
        return None

    async def upsert_questions(self, *, document_id=None, document_name=None, questions, blob_name=None):
        item = {
            "documentId": document_id,
            "documentName": document_name,
            "blobName": blob_name,
            "questions": list(questions),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self._items.insert(0, item)
        return item

    async def list_recent(self, *, limit: int):
        return self._items[:limit]

    async def delete_questions(self, *, document_id=None, document_name=None):
        self.deleted.append({"documentId": document_id, "documentName": document_name})
        self._items = [
            item for item in self._items
            if not (item.get("documentId") == document_id and item.get("documentName") == document_name)
        ]


class DummyBlobSource:
    def __init__(self):
        self._existing = set()

    def set_existing(self, *blob_names):
        self._existing = set(blob_names)

    async def exists(self, *, blob_name: str):
        return blob_name in self._existing


class DummyIngestionRepo:
    def __init__(self):
        self._items = [
            {
                "document_id": "doc-a",
                "document_name": "doc-a.pdf",
                "blobName": "doc-a.pdf",
                "source": "blob",
                "status": "processed",
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            {
                "document_id": "manual",
                "document_name": "Manual.pdf",
                "blobName": "folder/manual.pdf",
                "source": "blob",
                "status": "processed",
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            {
                "document_id": "skipped",
                "document_name": "skipped.pdf",
                "blobName": "folder/skipped.pdf",
                "source": "blob",
                "status": "skipped",
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        ]

    async def open(self):
        return None

    async def close(self):
        return None

    async def list_documents(self):
        return list(self._items)


class DummyEntityRepo:
    async def get_document(self, *, entity_id: str):
        if entity_id == "doc-a":
            return {"id": "doc-a", "Name": "doc-a", "DocExtension": ".pdf"}
        if entity_id == "manual":
            return {"id": "manual", "Name": "Manual", "DocExtension": ".pdf"}
        return None

    async def find_document_by_filename(self, *, filename: str):
        if filename == "doc-a.pdf":
            return {"id": "doc-a", "Name": "doc-a", "DocExtension": ".pdf"}
        if filename == "Manual.pdf":
            return {"id": "manual", "Name": "Manual", "DocExtension": ".pdf"}
        if filename == "manual.pdf":
            return {"id": "manual", "Name": "Manual", "DocExtension": ".pdf"}
        return None


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
    blob_source = DummyBlobSource()
    ingestion_repo = DummyIngestionRepo()
    blob_source.set_existing("doc-a.pdf", "folder/manual.pdf")

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

    def _override_blob_source(_request=None):
        return blob_source

    def _override_document_catalog(_request=None):
        return DocumentCatalogService(
            ingestion_repo=ingestion_repo,
            entity_repo=DummyEntityRepo(),
            blob_source=blob_source,
        )

    app.dependency_overrides[get_search_service] = _override
    app.dependency_overrides[get_llm_service] = _override_llm
    app.dependency_overrides[get_chat_repo] = _override_chat_repo
    app.dependency_overrides[get_suggestions_repo] = _override_suggestions_repo
    app.dependency_overrides[get_signalr_service] = _override_signalr
    app.dependency_overrides[get_blob_source] = _override_blob_source
    app.dependency_overrides[get_document_catalog] = _override_document_catalog
    app.state.blob_source = blob_source
    app.state.ingestion_repo = ingestion_repo
    app.state._test_blob_source = blob_source
    app.state._test_suggestions_repo = suggestions_repo
    yield
    app.dependency_overrides.clear()
    app.state.blob_source = None
    app.state.ingestion_repo = None
    app.state._test_blob_source = None
    app.state._test_suggestions_repo = None


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
        assert body["citations"][0]["document_id"] == "doc-a"

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


def test_list_documents_returns_storage_id_and_name():
    with TestClient(app) as client:
        app.state._test_blob_source.set_existing("folder/manual.pdf")

        resp = client.get("/api/v1/documents", params={"limit": 10})
        assert resp.status_code == 200

        body = resp.json()
        assert len(body) == 1
        assert body[0]["id"] == "manual"
        assert body[0]["name"] == "Manual.pdf"
        assert body[0]["status"] == "processed"


def test_document_preview_returns_citations_without_creating_chat():
    with TestClient(app) as client:
        preview = client.post(
            "/api/v1/documents/preview",
            json={"document": {"id": "folder/manual.pdf", "name": "manual.pdf"}},
        )
        assert preview.status_code == 200

        body = preview.json()
        assert body["document"]["id"] == "manual"
        assert body["document"]["name"] == "Manual.pdf"
        assert body["citations"][0]["document_id"] == "manual"
        assert body["citations"][0]["excerpt"] == "Preview excerpt from the indexed document."

        chats = client.get("/api/v1/chats", params={"limit": 10})
        assert chats.status_code == 200
        assert chats.json() == []


def test_empty_chat_message_is_rejected():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/messages",
            json={"message": "", "document": {"id": "folder/manual.pdf", "name": "manual.pdf"}},
        )
        assert resp.status_code == 400
        assert "documents/preview" in resp.json()["detail"]


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


def test_recent_suggestions_excludes_deleted_blob_documents():
    with TestClient(app) as client:
        suggestions_repo = app.state._test_suggestions_repo
        blob_source = app.state._test_blob_source

        blob_source.set_existing("live/doc.pdf")
        asyncio.run(
            suggestions_repo.upsert_questions(
                document_id="stale-doc",
                document_name="stale.pdf",
                blob_name="stale/doc.pdf",
                questions=["Old question"],
            )
        )
        asyncio.run(
            suggestions_repo.upsert_questions(
                document_id="live-doc",
                document_name="live.pdf",
                blob_name="live/doc.pdf",
                questions=["Live question"],
            )
        )

        recent = client.get("/api/v1/suggestions/recent", params={"limit": 3})
        assert recent.status_code == 200
        items = recent.json()
        assert len(items) == 1
        assert items[0]["document"]["id"] == "live-doc"
        assert suggestions_repo.deleted == [{"documentId": "stale-doc", "documentName": "stale.pdf"}]
