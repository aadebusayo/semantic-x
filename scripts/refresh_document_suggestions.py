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
from services.blob_storage import BlobStorageSource
from services.cosmos_repositories import CosmosEntityRepository, CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService
from services.llm_service import AzureOpenAILLMService
from services.suggestions_engine import build_suggestion_subject, build_suggestion_title, generate_quick_questions
from services.text_extraction import extract_text


async def _refresh_document_suggestions(limit: int | None) -> None:
    load_dotenv()
    os.environ.setdefault("ENTITY_FILE_OBJECT_TYPE", "1")
    get_settings.cache_clear()
    settings = get_settings()

    if not settings.azure_storage_connection_string or not settings.azure_storage_container:
        raise RuntimeError("Blob storage is not configured")

    blob_source = BlobStorageSource(
        connection_string=settings.azure_storage_connection_string,
        container=settings.azure_storage_container,
        prefix=settings.azure_storage_prefix,
    )
    ingestion_repo = CosmosIngestionRepository.from_settings(settings)
    suggestions_repo = CosmosSuggestionsRepository.from_settings(settings)
    entity_repo = CosmosEntityRepository.from_settings(settings)
    llm_service = AzureOpenAILLMService.from_settings(settings)
    document_catalog = DocumentCatalogService(
        ingestion_repo=ingestion_repo,
        entity_repo=entity_repo,
        blob_source=blob_source,
    )

    await ingestion_repo.open()
    await suggestions_repo.open()
    await entity_repo.open()
    await llm_service.open()
    await blob_source.open()

    processed = 0
    skipped = 0

    try:
        documents = await document_catalog.list_documents(limit=limit or 100000)
        print(f"Found {len(documents)} indexed documents")

        for document in documents:
            blob_name = await document_catalog.resolve_blob_name(document)
            if not blob_name:
                skipped += 1
                print(f"Skipping {document.name}: no blob mapping")
                continue

            content = await blob_source.download(blob_name=blob_name)
            text = extract_text(blob_name=blob_name, content=content)
            if not text:
                skipped += 1
                print(f"Skipping {document.name}: unsupported or empty content")
                continue

            snippet = text[:5000]
            title_hint = build_suggestion_title(document_name=document.name, text=snippet)
            suggestion_subject = build_suggestion_subject(document_name=document.name, text=snippet, title=title_hint)
            questions = await llm_service.generate_suggestions(
                document_name=suggestion_subject,
                text=snippet,
                limit=3,
            )
            if not questions:
                questions = generate_quick_questions(
                    document_name=suggestion_subject,
                    text=snippet,
                    limit=3,
                )

            title = await llm_service.generate_suggestion_title(
                document_name=document.name,
                text=snippet,
            )
            title = (title or "").strip() or title_hint

            await suggestions_repo.upsert_questions(
                document_id=document.id,
                document_name=document.name,
                title=title,
                blob_name=blob_name,
                questions=questions,
            )

            processed += 1
            print(f"[{processed}] Refreshed suggestions for {document.name}")

        print(f"Completed suggestion refresh. processed={processed} skipped={skipped}")
    finally:
        await blob_source.close()
        await llm_service.close()
        await entity_repo.close()
        await suggestions_repo.close()
        await ingestion_repo.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate quick-question suggestions for all available documents.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of documents to refresh")
    args = parser.parse_args()
    await _refresh_document_suggestions(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())