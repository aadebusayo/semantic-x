from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from config import Settings
from models.infosearch_api import DocumentRef
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.chunking import chunk_text
from services.cosmos_repositories import CosmosIngestionRepository, CosmosSuggestionsRepository
from services.suggestions_engine import generate_quick_questions
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
    ):
        self._settings = settings
        self._blob_source = blob_source
        self._search = search_service
        self._ingestion_repo = ingestion_repo
        self._suggestions_repo = suggestions_repo

        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

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
            except Exception as e:
                logger.error("Ingestion loop error: %s", e, exc_info=True)

            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=poll)
            except asyncio.TimeoutError:
                continue

    async def run_once(self, *, limit: int) -> int:
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
            if prior and prior.get("etag") == blob.etag and prior.get("status") == "processed":
                continue

            try:
                content = await self._blob_source.download(blob_name=blob.name)
                text = extract_text(blob_name=blob.name, content=content)
                if not text:
                    await self._ingestion_repo.upsert_status(
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

                # Use blob name as stable document id for now.
                document_id = blob.name
                document_name = blob.name.split("/")[-1]

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
                await self._suggestions_repo.upsert_questions(
                    document_id=document_id,
                    document_name=document_name,
                    questions=questions,
                )

                await self._ingestion_repo.upsert_status(
                    blob_name=blob.name,
                    storage_container=self._settings.azure_storage_container,
                    etag=blob.etag,
                    last_modified_iso=blob.last_modified_iso,
                    status="processed",
                )

                count += 1
            except Exception as e:
                await self._ingestion_repo.upsert_status(
                    blob_name=blob.name,
                    storage_container=self._settings.azure_storage_container or "",
                    etag=blob.etag,
                    last_modified_iso=blob.last_modified_iso,
                    status="error",
                    message=str(e)[:500],
                )

        return count
