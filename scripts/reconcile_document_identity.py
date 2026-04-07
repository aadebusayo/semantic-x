from __future__ import annotations

import os
from collections import defaultdict
import hashlib

from dotenv import load_dotenv

from config import get_settings
from services.azure_ai_search import AzureAISearchService
from services.cosmos_repositories import CosmosEntityRepository, CosmosIngestionRepository
from services.document_catalog import DocumentCatalogService


def _safe_chunk_id(document_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"chunk_{digest}_{chunk_index}"


async def _collect_search_documents(search_service: AzureAISearchService) -> dict[str, dict[str, object]]:
    client = search_service._get_client()
    if client is None:
        raise RuntimeError("Azure AI Search is not configured")

    pageable = await client.search(
        search_text="*",
        top=1000,
        query_type="simple",
        select=["id", "documentId", "documentName", "content", "chunkIndex", "chunkStart", "chunkEnd"],
    )

    documents: dict[str, dict[str, str]] = {}
    async for item in pageable:
        row = dict(item)
        key = row.get("id")
        if not key:
            continue
        documents[key] = row
    return documents


async def main() -> None:
    load_dotenv()
    settings = get_settings()

    ingestion_repo = CosmosIngestionRepository.from_settings(settings)
    entity_repo = CosmosEntityRepository.from_settings(settings)
    search_service = AzureAISearchService.from_settings(settings)
    catalog = DocumentCatalogService(
        ingestion_repo=ingestion_repo,
        entity_repo=entity_repo,
        blob_source=None,
    )

    await ingestion_repo.open()
    await entity_repo.open()

    try:
        search_docs = await _collect_search_documents(search_service)
        updates_by_id: dict[str, dict[str, object]] = {}
        deletes = []
        grouped = defaultdict(int)
        for key, item in search_docs.items():
            metadata = await catalog.resolve_metadata(
                raw_document_id=item.get("documentId"),
                raw_document_name=item.get("documentName"),
            )
            document_id = metadata.id or item.get("documentId")
            document_name = metadata.name or item.get("documentName")
            chunk_index = int(item.get("chunkIndex") or 0)
            new_key = _safe_chunk_id(document_id, chunk_index)

            if (
                document_id == item.get("documentId")
                and document_name == item.get("documentName")
                and new_key == key
            ):
                continue

            updates_by_id[new_key] = {
                "id": new_key,
                "documentId": document_id,
                "documentName": document_name,
                "content": item.get("content") or "",
                "chunkIndex": chunk_index,
                "chunkStart": int(item.get("chunkStart") or 0),
                "chunkEnd": int(item.get("chunkEnd") or 0),
            }
            if new_key != key:
                deletes.append({"id": key})
            grouped[f"{item.get('documentId')} -> {document_id}"] += 1

        updates = list(updates_by_id.values())
        if not updates and not deletes:
            print("No search identity updates required")
            return

        client = search_service._get_client()
        batch_size = 200
        for idx in range(0, len(updates), batch_size):
            await client.merge_or_upload_documents(documents=updates[idx : idx + batch_size])

        for idx in range(0, len(deletes), batch_size):
            await client.delete_documents(documents=deletes[idx : idx + batch_size])

        print(f"Upserted {len(updates)} canonical search documents")
        print(f"Deleted {len(deletes)} legacy search documents")
        for transition, count in grouped.items():
            print(f"{transition}: {count}")
    finally:
        await entity_repo.close()
        await ingestion_repo.close()
        await search_service.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())