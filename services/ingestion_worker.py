from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from config import Settings
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.chunking import chunk_text
from services.cosmos_repositories import CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService, strip_file_extension
from services.suggestions_engine import build_suggestion_title, generate_quick_questions
from services.text_extraction import extract_text

logger = logging.getLogger(__name__)


def _safe_chunk_id(document_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"chunk_{digest}_{chunk_index}"


class IngestionWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        blob_source: BlobStorageSource,
        search_service: AzureAISearchService,
        ingestion_repo: CosmosIngestionRepository,
        suggestions_repo: CosmosSuggestionsRepository,
        document_catalog: DocumentCatalogService,
    ):
        self._settings = settings
        self._blob_source = blob_source
        self._search = search_service
        self._ingestion_repo = ingestion_repo
        self._suggestions_repo = suggestions_repo
        self._document_catalog = document_catalog

        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._poll_count = 0

    async def open(self) -> None:
        await self._blob_source.open()

    async def close(self) -> None:
        await self._blob_source.close()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None

    async def run_forever(self) -> None:
        poll = self._settings.ingestion_poll_seconds
        while not self._stopping.is_set():
            try:
                processed = await self.run_once(limit=500)
                if processed:
                    logger.info("Ingestion processed %s blobs", processed)

                self._poll_count += 1
                if (
                    self._settings.ingestion_reconcile_deletions
                    and self._poll_count % self._settings.ingestion_reconcile_every_polls == 0
                ):
                    removed = await self.reconcile_deleted_documents()
                    if removed:
                        logger.info("Ingestion reconciled %s deleted blobs", removed)
            except Exception as e:
                logger.error("Ingestion loop error: %s", e, exc_info=True)

            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=poll)
            except asyncio.TimeoutError:
                continue

    async def run_once(self, *, limit: int | None, force: bool = False) -> int:
        if not self._settings.azure_storage_connection_string or not self._settings.azure_storage_container:
            return 0

        count = 0
        async for blob in self._blob_source.list_blobs(limit=limit):
            if self._stopping.is_set():
                break

            # Skip folders
            if blob.name.endswith("/"):
                continue

            prior = await self._ingestion_repo.get(blob_name=blob.name)
            if (not force) and prior and prior.get("etag") == blob.etag and prior.get("status") == "processed":
                continue

            try:
                content = await self._blob_source.download(blob_name=blob.name)
                text = extract_text(blob_name=blob.name, content=content)
                if not text:
                    await self._ingestion_repo.upsert_status(
                        document_id=strip_file_extension(blob.name) or blob.name,
                        document_name=blob.name.split("/")[-1],
                        blob_name=blob.name,
                        storage_container=self._settings.azure_storage_container,
                        etag=blob.etag,
                        last_modified_iso=blob.last_modified_iso,
                        status="skipped",
                        message="unsupported_or_empty",
                    )
                    continue

                chunks = chunk_text(
                    text=text,
                    chunk_size=self._settings.chunk_size_chars,
                    overlap=self._settings.chunk_overlap_chars,
                )

                metadata = await self._document_catalog.resolve_metadata(raw_document_id=blob.name, blob_name=blob.name)
                document_id = metadata.id or blob.name
                document_name = metadata.name or blob.name.split("/")[-1]

                chunk_docs = []
                for ch in chunks:
                    chunk_docs.append(
                        {
                            "id": _safe_chunk_id(document_id, ch.index),
                            "content": ch.text,
                            "chunkIndex": ch.index,
                            "chunkStart": ch.start,
                            "chunkEnd": ch.end,
                        }
                    )

                await self._search.upsert_chunks(
                    document_id=document_id,
                    document_name=document_name,
                    chunks=chunk_docs,
                )

                # Quick questions stored in Cosmos for recency prefill.
                questions = generate_quick_questions(document_name=document_name, text=text[:5000], limit=3)
                suggestion_title = build_suggestion_title(document_name=document_name, text=text[:5000])
                try:
                    await self._suggestions_repo.upsert_questions(
                        document_id=document_id,
                        document_name=document_name,
                        title=suggestion_title,
                        blob_name=blob.name,
                        questions=questions,
                    )
                except TypeError:
                    await self._suggestions_repo.upsert_questions(
                        document_id=document_id,
                        document_name=document_name,
                        blob_name=blob.name,
                        questions=questions,
                    )

                await self._ingestion_repo.upsert_status(
                    document_id=document_id,
                    document_name=document_name,
                    blob_name=blob.name,
                    storage_container=self._settings.azure_storage_container,
                    etag=blob.etag,
                    last_modified_iso=blob.last_modified_iso,
                    status="processed",
                )
                self._document_catalog.invalidate()

                count += 1
            except Exception as e:
                await self._ingestion_repo.upsert_status(
                    document_id=strip_file_extension(blob.name) or blob.name,
                    document_name=blob.name.split("/")[-1],
                    blob_name=blob.name,
                    storage_container=self._settings.azure_storage_container or "",
                    etag=blob.etag,
                    last_modified_iso=blob.last_modified_iso,
                    status="error",
                    message=str(e)[:500],
                )

        return count

    async def reconcile_deleted_documents(self) -> int:
        if not self._settings.azure_storage_connection_string or not self._settings.azure_storage_container:
            return 0

        active_blob_names = set()
        async for blob in self._blob_source.list_blobs(limit=None):
            if blob.name.endswith("/"):
                continue
            active_blob_names.add(blob.name)

        tracked_documents = await self._ingestion_repo.list_documents()
        stale_documents = []
        for item in tracked_documents:
            blob_name = item.get("blobName") or item.get("blob_name") or item.get("document_id")
            if not blob_name or item.get("source") != "blob":
                continue
            if blob_name not in active_blob_names:
                stale_documents.append(item)

        removed = 0
        for item in stale_documents:
            blob_name = item.get("blobName") or item.get("blob_name") or item.get("document_id")
            metadata = await self._document_catalog.resolve_metadata(
                raw_document_id=item.get("documentId") or item.get("document_id"),
                raw_document_name=item.get("documentName") or item.get("document_name"),
                blob_name=blob_name,
            )
            canonical_id = metadata.id or item.get("documentId") or item.get("document_id") or blob_name
            await self._search.delete_document_chunks(document_id=canonical_id)
            await self._suggestions_repo.delete_questions(document_id=canonical_id, document_name=metadata.name)
            await self._ingestion_repo.delete(blob_name=blob_name)
            self._document_catalog.invalidate()
            removed += 1

        return removed
