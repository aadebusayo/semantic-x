from __future__ import annotations

import asyncio
from dataclasses import dataclass

from models.infosearch_api import DocumentRef
from services.azure_ai_search import AzureAISearchService
from services.ingestion_worker import IngestionWorker


class _AsyncPageable:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _SearchClientForPreferredDoc:
    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        filter_expr = kwargs.get("filter")
        search_text = kwargs.get("search_text")

        if filter_expr and search_text == "policy terms":
            return _AsyncPageable([])
        if filter_expr and search_text == "*":
            return _AsyncPageable(
                [
                    {
                        "id": "chunk-target-0",
                        "documentId": "target-doc",
                        "documentName": "target.pdf",
                        "content": "Target fallback excerpt.",
                        "@search.score": 0.15,
                    }
                ]
            )

        return _AsyncPageable(
            [
                {
                    "id": "chunk-other-0",
                    "documentId": "other-doc",
                    "documentName": "other.pdf",
                    "content": "Other document excerpt.",
                    "@search.score": 0.99,
                },
                {
                    "id": "chunk-target-0",
                    "documentId": "target-doc",
                    "documentName": "target.pdf",
                    "content": "Target fallback excerpt.",
                    "@search.score": 0.45,
                },
            ]
        )


class _SearchClientForChunkSync:
    def __init__(self):
        self.merge_batches = []
        self.delete_batches = []
        self.search_calls = []

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return _AsyncPageable(
            [
                {"id": "chunk-doc-0"},
                {"id": "chunk-doc-1"},
                {"id": "chunk-doc-2"},
            ]
        )

    async def merge_or_upload_documents(self, *, documents):
        self.merge_batches.append(documents)

    async def delete_documents(self, *, documents):
        self.delete_batches.append(documents)


def _make_search_service(client) -> AzureAISearchService:
    service = AzureAISearchService(
        endpoint="https://example.search.windows.net",
        index_name="docs",
        api_key="key",
        semantic_config="default",
        content_field="content",
        vector_field="contentVector",
        doc_id_field="documentId",
        doc_name_field="documentName",
        filter_doc_id_field="documentId",
        filter_doc_name_field="documentName",
        max_excerpt_chars=280,
        min_citation_score=0.0,
    )
    service._client = client
    service._vector_query_enabled = False
    return service


def test_search_prioritizes_direct_document_as_first_citation():
    async def _run():
        service = _make_search_service(_SearchClientForPreferredDoc())

        citations = await service.search(
            query="policy terms",
            document=DocumentRef(id="target-doc", name="target.pdf"),
            top_k=2,
        )

        assert [citation.document_id for citation in citations] == ["target-doc", "other-doc"]
        assert [citation.rank for citation in citations] == [1, 2]
        assert citations[0].excerpt == "Target fallback excerpt."

    asyncio.run(_run())


def test_upsert_chunks_deletes_stale_search_chunks_after_update():
    async def _run():
        client = _SearchClientForChunkSync()
        service = _make_search_service(client)

        await service.upsert_chunks(
            document_id="folder/doc.pdf",
            document_name="doc.pdf",
            chunks=[
                {
                    "id": "chunk-doc-0",
                    "content": "chunk 0",
                    "chunkIndex": 0,
                    "chunkStart": 0,
                    "chunkEnd": 7,
                },
                {
                    "id": "chunk-doc-1",
                    "content": "chunk 1",
                    "chunkIndex": 1,
                    "chunkStart": 8,
                    "chunkEnd": 15,
                },
            ],
        )

        assert len(client.merge_batches) == 1
        assert [doc["id"] for doc in client.merge_batches[0]] == ["chunk-doc-0", "chunk-doc-1"]
        assert client.delete_batches == [[{"id": "chunk-doc-2"}]]

    asyncio.run(_run())


@dataclass(frozen=True)
class _BlobInfo:
    name: str
    etag: str
    last_modified_iso: str
    size: int


class _BlobSource:
    def __init__(self, blobs, content: bytes):
        self._blobs = blobs
        self._content = content

    async def open(self):
        return None

    async def close(self):
        return None

    async def list_blobs(self, *, limit: int):
        count = 0
        for blob in self._blobs:
            if count >= limit:
                break
            yield blob
            count += 1

    async def download(self, *, blob_name: str):
        return self._content


class _SearchRecorder:
    def __init__(self):
        self.calls = []

    async def upsert_chunks(self, *, document_id: str, document_name: str, chunks):
        self.calls.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "chunks": list(chunks),
            }
        )


class _IngestionRepo:
    def __init__(self, prior):
        self._prior = prior
        self.statuses = []

    async def get(self, *, blob_name: str):
        return self._prior

    async def upsert_status(self, **kwargs):
        self.statuses.append(kwargs)
        return kwargs


class _SuggestionsRepo:
    def __init__(self):
        self.calls = []

    async def upsert_questions(self, *, document_id=None, document_name=None, questions=None):
        self.calls.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "questions": list(questions or []),
            }
        )
        return self.calls[-1]


class _Settings:
    azure_storage_connection_string = "UseDevelopmentStorage=true"
    azure_storage_container = "docs"
    chunk_size_chars = 50
    chunk_overlap_chars = 10
    ingestion_poll_seconds = 60


def test_ingestion_worker_reprocesses_changed_pdf(monkeypatch):
    async def _run():
        blob = _BlobInfo(
            name="folder/manual.pdf",
            etag="new-etag",
            last_modified_iso="2026-03-26T12:00:00+00:00",
            size=128,
        )
        blob_source = _BlobSource([blob], b"updated-pdf-content")
        search = _SearchRecorder()
        ingestion_repo = _IngestionRepo(prior={"etag": "old-etag", "status": "processed"})
        suggestions_repo = _SuggestionsRepo()

        monkeypatch.setattr(
            "services.ingestion_worker.extract_text",
            lambda *, blob_name, content: "Updated PDF text for indexing.",
        )

        worker = IngestionWorker(
            settings=_Settings(),
            blob_source=blob_source,
            search_service=search,
            ingestion_repo=ingestion_repo,
            suggestions_repo=suggestions_repo,
        )

        processed = await worker.run_once(limit=10)

        assert processed == 1
        assert len(search.calls) == 1
        assert search.calls[0]["document_id"] == "folder/manual.pdf"
        assert search.calls[0]["document_name"] == "manual.pdf"
        assert ingestion_repo.statuses[-1]["status"] == "processed"
        assert ingestion_repo.statuses[-1]["etag"] == "new-etag"

    asyncio.run(_run())