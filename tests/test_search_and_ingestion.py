from __future__ import annotations

import asyncio
from dataclasses import dataclass

from models.infosearch_api import DocumentRef
from services.azure_ai_search import AzureAISearchService
from services.document_catalog import DocumentCatalogService
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

    async def list_blobs(self, *, limit: int | None):
        count = 0
        for blob in self._blobs:
            if limit is not None and count >= limit:
                break
            yield blob
            count += 1

    async def download(self, *, blob_name: str):
        return self._content


class _SearchRecorder:
    def __init__(self):
        self.calls = []
        self.deleted = []

    async def upsert_chunks(self, *, document_id: str, document_name: str, chunks):
        self.calls.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "chunks": list(chunks),
            }
        )

    async def delete_document_chunks(self, *, document_id: str):
        self.deleted.append(document_id)


class _IngestionRepo:
    def __init__(self, prior):
        self._prior = prior
        self.statuses = []
        self.documents = [] if prior is None else [prior]
        self.deleted = []

    async def get(self, *, blob_name: str):
        return self._prior

    async def upsert_status(self, **kwargs):
        self.statuses.append(kwargs)
        return kwargs

    async def list_documents(self):
        return list(self.documents)

    async def delete(self, *, blob_name: str):
        self.deleted.append(blob_name)


class _SuggestionsRepo:
    def __init__(self):
        self.calls = []
        self.deleted = []

    async def upsert_questions(self, *, document_id=None, document_name=None, questions=None, blob_name=None):
        self.calls.append(
            {
                "document_id": document_id,
                "document_name": document_name,
                "blob_name": blob_name,
                "questions": list(questions or []),
            }
        )
        return self.calls[-1]

    async def delete_questions(self, *, document_id=None, document_name=None):
        self.deleted.append({"document_id": document_id, "document_name": document_name})


class _EntityRepo:
    async def get_document(self, *, entity_id: str):
        if entity_id == "manual":
            return {"id": "manual", "Name": "Manual", "DocExtension": ".pdf"}
        if entity_id == "target-doc":
            return {"id": "target-doc", "Name": "target", "DocExtension": ".pdf"}
        return None

    async def find_document_by_filename(self, *, filename: str):
        if filename in {"Manual.pdf", "manual.pdf"}:
            return {"id": "manual", "Name": "Manual", "DocExtension": ".pdf"}
        if filename == "target.pdf":
            return {"id": "target-doc", "Name": "target", "DocExtension": ".pdf"}
        return None


class _Settings:
    azure_storage_connection_string = "UseDevelopmentStorage=true"
    azure_storage_container = "docs"
    chunk_size_chars = 50
    chunk_overlap_chars = 10
    ingestion_poll_seconds = 60
    ingestion_reconcile_deletions = True
    ingestion_reconcile_every_polls = 10


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
            document_catalog=DocumentCatalogService(
                ingestion_repo=ingestion_repo,
                entity_repo=_EntityRepo(),
                blob_source=blob_source,
            ),
        )

        processed = await worker.run_once(limit=10)

        assert processed == 1
        assert len(search.calls) == 1
        assert search.calls[0]["document_id"] == "manual"
        assert search.calls[0]["document_name"] == "Manual.pdf"
        assert ingestion_repo.statuses[-1]["status"] == "processed"
        assert ingestion_repo.statuses[-1]["etag"] == "new-etag"
        assert ingestion_repo.statuses[-1]["document_id"] == "manual"
        assert ingestion_repo.statuses[-1]["blob_name"] == "folder/manual.pdf"
        assert suggestions_repo.calls[-1]["document_id"] == "manual"
        assert suggestions_repo.calls[-1]["document_name"] == "Manual.pdf"
        assert suggestions_repo.calls[-1]["blob_name"] == "folder/manual.pdf"

    asyncio.run(_run())


def test_ingestion_worker_reconciles_deleted_documents():
    async def _run():
        live_blob = _BlobInfo(
            name="folder/live.pdf",
            etag="etag-live",
            last_modified_iso="2026-03-26T12:00:00+00:00",
            size=128,
        )
        blob_source = _BlobSource([live_blob], b"live-content")
        search = _SearchRecorder()
        ingestion_repo = _IngestionRepo(prior=None)
        ingestion_repo.documents = [
            {"document_id": "live", "document_name": "live.pdf", "blobName": "folder/live.pdf", "source": "blob", "status": "processed"},
            {"document_id": "deleted", "document_name": "deleted.pdf", "blobName": "folder/deleted.pdf", "source": "blob", "status": "processed"},
        ]
        suggestions_repo = _SuggestionsRepo()

        worker = IngestionWorker(
            settings=_Settings(),
            blob_source=blob_source,
            search_service=search,
            ingestion_repo=ingestion_repo,
            suggestions_repo=suggestions_repo,
            document_catalog=DocumentCatalogService(
                ingestion_repo=ingestion_repo,
                entity_repo=_EntityRepo(),
                blob_source=blob_source,
            ),
        )

        removed = await worker.reconcile_deleted_documents()

        assert removed == 1
        assert search.deleted == ["deleted"]
        assert suggestions_repo.deleted == [
            {"document_id": "deleted", "document_name": "deleted.pdf"},
        ]
        assert ingestion_repo.deleted == ["folder/deleted.pdf"]

    asyncio.run(_run())


def test_preview_document_is_limited_to_selected_document():
    async def _run():
        service = _make_search_service(_SearchClientForPreferredDoc())

        citations = await service.preview_document(
            document=DocumentRef(id="target-doc", name="target.pdf"),
            top_k=2,
        )

        assert [citation.document_id for citation in citations] == ["target-doc"]
        assert citations[0].excerpt == "Target fallback excerpt."

    asyncio.run(_run())