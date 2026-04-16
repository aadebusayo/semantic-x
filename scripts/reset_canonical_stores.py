from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.cosmos_repositories import CosmosEntityRepository, CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService
from services.ingestion_worker import IngestionWorker


async def _count_blobs(blob_source: BlobStorageSource) -> int:
    count = 0
    async for blob in blob_source.list_blobs(limit=None):
        if blob.name.endswith("/"):
            continue
        count += 1
    return count


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset derived document stores and rebuild them from Blob using canonical Entity ids."
    )
    parser.add_argument("--skip-search-clear", action="store_true", help="Do not delete existing Azure AI Search chunks")
    parser.add_argument(
        "--skip-cosmos-clear",
        action="store_true",
        help="Do not delete existing ingestion and suggestions rows before rebuilding",
    )
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("ENTITY_FILE_OBJECT_TYPE", "1")
    get_settings.cache_clear()
    settings = get_settings()

    if not settings.azure_storage_connection_string or not settings.azure_storage_container:
        raise RuntimeError("Blob storage is not configured")

    search_service = AzureAISearchService.from_settings(settings)
    blob_source = BlobStorageSource(
        connection_string=settings.azure_storage_connection_string,
        container=settings.azure_storage_container,
        prefix=settings.azure_storage_prefix,
    )
    ingestion_repo = CosmosIngestionRepository.from_settings(settings)
    suggestions_repo = CosmosSuggestionsRepository.from_settings(settings)
    entity_repo = CosmosEntityRepository.from_settings(settings)
    document_catalog = DocumentCatalogService(
        ingestion_repo=ingestion_repo,
        entity_repo=entity_repo,
        blob_source=blob_source,
    )
    worker = IngestionWorker(
        settings=settings,
        blob_source=blob_source,
        search_service=search_service,
        ingestion_repo=ingestion_repo,
        suggestions_repo=suggestions_repo,
        document_catalog=document_catalog,
    )

    await ingestion_repo.open()
    await suggestions_repo.open()
    await entity_repo.open()
    await blob_source.open()

    try:
        if args.skip_search_clear:
            print("Skipped clearing search index")
        else:
            removed = await search_service.delete_all_chunks()
            print(f"Deleted {removed} search chunks")

        if args.skip_cosmos_clear:
            print("Skipped clearing ingestion and suggestions containers")
        else:
            deleted_ingestion = await ingestion_repo.clear_all()
            deleted_suggestions = await suggestions_repo.clear_all()
            document_catalog.invalidate()
            print(f"Deleted {deleted_ingestion} ingestion rows")
            print(f"Deleted {deleted_suggestions} suggestion rows")

        blob_count = await _count_blobs(blob_source)
        print(f"Discovered {blob_count} blobs to rebuild")

        processed = await worker.run_once(limit=None, force=True)
        print(f"Rebuilt {processed} blobs")
    finally:
        await blob_source.close()
        await entity_repo.close()
        await suggestions_repo.close()
        await ingestion_repo.close()
        await search_service.close()


if __name__ == "__main__":
    asyncio.run(main())