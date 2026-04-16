from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from models.infosearch_api import DocumentRef
from services.azure_ai_search import AzureAISearchService
from services.cosmos_repositories import CosmosEntityRepository, CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import document_name_from_id, strip_file_extension


def _safe_chunk_id(document_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"chunk_{digest}_{chunk_index}"


def _blob_name_from_search_document_id(document_id: Optional[str]) -> Optional[str]:
    text = (document_id or "").strip()
    return text if "/" in text else None


def _entity_filename(entity: Dict[str, Any]) -> Optional[str]:
    name = str(entity.get("Name") or "").strip()
    extension = str(entity.get("DocExtension") or "").strip()
    if not name:
        return None
    return f"{name}{extension}" if extension else name


def _entity_id_candidates(*values: Optional[str]) -> List[str]:
    candidates: List[str] = []
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        candidate = strip_file_extension(text)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _filename_candidates(
    *,
    raw_document_id: Optional[str],
    raw_document_name: Optional[str],
    blob_name: Optional[str],
) -> List[str]:
    candidates: List[str] = []
    for value in (
        raw_document_name,
        document_name_from_id(blob_name),
        document_name_from_id(raw_document_id),
    ):
        text = (value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


@dataclass(frozen=True)
class IdentityBackfillSummary:
    scanned: int = 0
    updated: int = 0
    unresolved: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class IdentityChange:
    old_document_id: Optional[str]
    old_document_name: Optional[str]
    blob_name: Optional[str]
    new_document_id: str
    new_document_name: str


@dataclass
class EntityLookupIndex:
    by_id: Dict[str, Dict[str, Any]]
    by_filename: Dict[str, Dict[str, Any]]


async def resolve_entity_metadata(
    entity_repo: CosmosEntityRepository,
    *,
    raw_document_id: Optional[str],
    raw_document_name: Optional[str],
    blob_name: Optional[str],
    entity_index: Optional[EntityLookupIndex] = None,
) -> Optional[DocumentRef]:
    for entity_id in _entity_id_candidates(raw_document_id, blob_name):
        entity = entity_index.by_id.get(entity_id) if entity_index is not None else await entity_repo.get_document(entity_id=entity_id)
        if entity is None:
            continue
        filename = _entity_filename(entity)
        resolved_id = str(entity.get("id") or "").strip() or entity_id
        return DocumentRef(id=resolved_id, name=filename or raw_document_name)

    for filename in _filename_candidates(
        raw_document_id=raw_document_id,
        raw_document_name=raw_document_name,
        blob_name=blob_name,
    ):
        entity = entity_index.by_filename.get(filename.casefold()) if entity_index is not None else await entity_repo.find_document_by_filename(filename=filename)
        if entity is None:
            continue
        resolved_id = str(entity.get("id") or "").strip()
        resolved_name = _entity_filename(entity)
        if resolved_id and resolved_name:
            return DocumentRef(id=resolved_id, name=resolved_name)

    return None


def needs_identity_update(
    *,
    raw_document_id: Optional[str],
    raw_document_name: Optional[str],
    canonical: DocumentRef,
) -> bool:
    return raw_document_id != canonical.id or raw_document_name != canonical.name


async def _collect_cosmos_items(container, query: str, parameters: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    async for item in container.query_items(query=query, parameters=parameters or []):
        items.append(item)
    return items


async def build_entity_lookup_index(entity_repo: CosmosEntityRepository) -> EntityLookupIndex:
    items = await _collect_cosmos_items(
        entity_repo._container,
        "SELECT c.id, c.Name, c.DocExtension, c.ObjectType FROM c WHERE c.ObjectType = @objectType",
        [{"name": "@objectType", "value": entity_repo._file_object_type}],
    )

    by_id: Dict[str, Dict[str, Any]] = {}
    by_filename: Dict[str, Dict[str, Any]] = {}
    duplicate_filenames: set[str] = set()

    for item in items:
        entity_id = str(item.get("id") or "").strip()
        if entity_id:
            by_id[entity_id] = item

        filename = _entity_filename(item)
        key = filename.casefold() if filename else ""
        if not key:
            continue
        if key in duplicate_filenames:
            continue
        if key in by_filename:
            duplicate_filenames.add(key)
            del by_filename[key]
            continue
        by_filename[key] = item

    return EntityLookupIndex(by_id=by_id, by_filename=by_filename)


def _escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


async def _collect_search_documents_for_filter(
    search_service: AzureAISearchService,
    *,
    filter_expr: str,
) -> List[Dict[str, Any]]:
    client = search_service._get_client()
    if client is None:
        raise RuntimeError("Azure AI Search is not configured")

    page_size = 1000
    skip = 0
    documents: List[Dict[str, Any]] = []

    while True:
        page = await client.search(
            search_text="*",
            top=page_size,
            skip=skip,
            query_type="simple",
            filter=filter_expr,
            select=["id", "documentId", "documentName", "content", "chunkIndex", "chunkStart", "chunkEnd"],
        )

        batch: List[Dict[str, Any]] = []
        async for item in page:
            batch.append(dict(item))

        documents.extend(batch)
        if len(batch) < page_size:
            break
        skip += page_size

    return documents


async def backfill_ingestion_identities(
    *,
    ingestion_repo: CosmosIngestionRepository,
    entity_repo: CosmosEntityRepository,
    entity_index: EntityLookupIndex,
    storage_container: str,
    dry_run: bool,
) -> IdentityBackfillSummary:
    items = await _collect_cosmos_items(ingestion_repo._container, "SELECT * FROM c")
    summary = IdentityBackfillSummary(scanned=len(items))

    updated = unresolved = 0
    for item in items:
        if item.get("source") != "blob":
            continue

        canonical = await resolve_entity_metadata(
            entity_repo,
            raw_document_id=item.get("documentId") or item.get("document_id"),
            raw_document_name=item.get("documentName") or item.get("document_name"),
            blob_name=item.get("blobName") or item.get("documentId") or item.get("document_id"),
            entity_index=entity_index,
        )
        if canonical is None or not canonical.id or not canonical.name:
            unresolved += 1
            continue

        if not needs_identity_update(
            raw_document_id=item.get("documentId") or item.get("document_id"),
            raw_document_name=item.get("documentName") or item.get("document_name"),
            canonical=canonical,
        ):
            continue

        updated += 1
        if dry_run:
            continue

        await ingestion_repo.upsert_status(
            document_id=canonical.id,
            document_name=canonical.name,
            blob_name=item.get("blobName") or item.get("documentId") or item.get("document_id") or "",
            storage_container=item.get("container") or storage_container,
            etag=item.get("etag") or "",
            last_modified_iso=item.get("lastModified") or item.get("last_modified") or "",
            status=item.get("status") or "processed",
            message=item.get("message"),
        )

    return IdentityBackfillSummary(
        scanned=summary.scanned,
        updated=updated,
        unresolved=unresolved,
    )


async def backfill_suggestion_identities(
    *,
    suggestions_repo: CosmosSuggestionsRepository,
    entity_repo: CosmosEntityRepository,
    entity_index: EntityLookupIndex,
    dry_run: bool,
) -> IdentityBackfillSummary:
    items = await _collect_cosmos_items(suggestions_repo._container, "SELECT * FROM c")
    updated = unresolved = deleted = 0

    for item in items:
        canonical = await resolve_entity_metadata(
            entity_repo,
            raw_document_id=item.get("documentId") or item.get("document_id"),
            raw_document_name=item.get("documentName") or item.get("document_name"),
            blob_name=item.get("blobName"),
            entity_index=entity_index,
        )
        if canonical is None or not canonical.id or not canonical.name:
            unresolved += 1
            continue

        old_document_id = item.get("documentId") or item.get("document_id")
        old_document_name = item.get("documentName") or item.get("document_name")
        if not needs_identity_update(
            raw_document_id=old_document_id,
            raw_document_name=old_document_name,
            canonical=canonical,
        ):
            continue

        updated += 1
        if dry_run:
            continue

        await suggestions_repo.upsert_questions(
            document_id=canonical.id,
            document_name=canonical.name,
            title=item.get("title"),
            questions=item.get("questions") or [],
            blob_name=item.get("blobName"),
        )

        if old_document_id != canonical.id or old_document_name != canonical.name:
            await suggestions_repo.delete_questions(
                document_id=old_document_id,
                document_name=old_document_name,
            )
            deleted += 1

    return IdentityBackfillSummary(
        scanned=len(items),
        updated=updated,
        unresolved=unresolved,
        deleted=deleted,
    )


async def collect_ingestion_identity_changes(
    *,
    ingestion_repo: CosmosIngestionRepository,
    entity_repo: CosmosEntityRepository,
    entity_index: EntityLookupIndex,
) -> List[IdentityChange]:
    items = await _collect_cosmos_items(ingestion_repo._container, "SELECT * FROM c")
    changes: List[IdentityChange] = []
    seen: set[Tuple[Optional[str], Optional[str], Optional[str], str, str]] = set()

    for item in items:
        if item.get("source") != "blob":
            continue
        canonical = await resolve_entity_metadata(
            entity_repo,
            raw_document_id=item.get("documentId") or item.get("document_id"),
            raw_document_name=item.get("documentName") or item.get("document_name"),
            blob_name=item.get("blobName") or item.get("documentId") or item.get("document_id"),
            entity_index=entity_index,
        )
        if canonical is None or not canonical.id or not canonical.name:
            continue
        old_document_id = item.get("documentId") or item.get("document_id")
        old_document_name = item.get("documentName") or item.get("document_name")
        if not needs_identity_update(
            raw_document_id=old_document_id,
            raw_document_name=old_document_name,
            canonical=canonical,
        ):
            continue
        key = (old_document_id, old_document_name, item.get("blobName"), canonical.id, canonical.name)
        if key in seen:
            continue
        seen.add(key)
        changes.append(
            IdentityChange(
                old_document_id=old_document_id,
                old_document_name=old_document_name,
                blob_name=item.get("blobName"),
                new_document_id=canonical.id,
                new_document_name=canonical.name,
            )
        )

    return changes


async def collect_suggestion_identity_changes(
    *,
    suggestions_repo: CosmosSuggestionsRepository,
    entity_repo: CosmosEntityRepository,
    entity_index: EntityLookupIndex,
) -> List[IdentityChange]:
    items = await _collect_cosmos_items(suggestions_repo._container, "SELECT * FROM c")
    changes: List[IdentityChange] = []
    seen: set[Tuple[Optional[str], Optional[str], Optional[str], str, str]] = set()

    for item in items:
        canonical = await resolve_entity_metadata(
            entity_repo,
            raw_document_id=item.get("documentId") or item.get("document_id"),
            raw_document_name=item.get("documentName") or item.get("document_name"),
            blob_name=item.get("blobName"),
            entity_index=entity_index,
        )
        if canonical is None or not canonical.id or not canonical.name:
            continue
        old_document_id = item.get("documentId") or item.get("document_id")
        old_document_name = item.get("documentName") or item.get("document_name")
        if not needs_identity_update(
            raw_document_id=old_document_id,
            raw_document_name=old_document_name,
            canonical=canonical,
        ):
            continue
        key = (old_document_id, old_document_name, item.get("blobName"), canonical.id, canonical.name)
        if key in seen:
            continue
        seen.add(key)
        changes.append(
            IdentityChange(
                old_document_id=old_document_id,
                old_document_name=old_document_name,
                blob_name=item.get("blobName"),
                new_document_id=canonical.id,
                new_document_name=canonical.name,
            )
        )

    return changes


async def backfill_search_identities(
    *,
    search_service: AzureAISearchService,
    identity_changes: Sequence[IdentityChange],
    dry_run: bool,
) -> IdentityBackfillSummary:
    search_docs_by_key: Dict[str, Dict[str, Any]] = {}
    updates_by_id: Dict[str, Dict[str, Any]] = {}
    deletes: List[Dict[str, str]] = []
    scanned = 0

    for change in identity_changes:
        filters: List[str] = []
        if change.old_document_id:
            filters.append(f"documentId eq '{_escape_odata_string(change.old_document_id)}'")
        if change.old_document_name:
            filters.append(f"documentName eq '{_escape_odata_string(change.old_document_name)}'")
        if change.blob_name:
            filters.append(f"documentId eq '{_escape_odata_string(change.blob_name)}'")

        for filter_expr in dict.fromkeys(filters):
            for item in await _collect_search_documents_for_filter(search_service, filter_expr=filter_expr):
                search_docs_by_key[str(item.get("id"))] = item

    for item in search_docs_by_key.values():
        matching_change = None
        for change in identity_changes:
            if item.get("documentId") in {change.old_document_id, change.blob_name} or item.get("documentName") == change.old_document_name:
                matching_change = change
                break
        if matching_change is None:
            continue

        scanned += 1
        chunk_index = int(item.get("chunkIndex") or 0)
        new_key = _safe_chunk_id(matching_change.new_document_id, chunk_index)
        if (
            item.get("documentId") == matching_change.new_document_id
            and item.get("documentName") == matching_change.new_document_name
            and item.get("id") == new_key
        ):
            continue

        updates_by_id[new_key] = {
            "id": new_key,
            "documentId": matching_change.new_document_id,
            "documentName": matching_change.new_document_name,
            "content": item.get("content") or "",
            "chunkIndex": chunk_index,
            "chunkStart": int(item.get("chunkStart") or 0),
            "chunkEnd": int(item.get("chunkEnd") or 0),
        }
        if item.get("id") != new_key:
            deletes.append({"id": item.get("id")})

    if not dry_run and (updates_by_id or deletes):
        client = search_service._get_client()
        batch_size = 200
        updates = list(updates_by_id.values())
        for idx in range(0, len(updates), batch_size):
            await client.merge_or_upload_documents(documents=updates[idx : idx + batch_size])
        for idx in range(0, len(deletes), batch_size):
            await client.delete_documents(documents=deletes[idx : idx + batch_size])

    return IdentityBackfillSummary(
        scanned=scanned,
        updated=len(updates_by_id),
        deleted=len(deletes),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill canonical document ids and names from Entity metadata into Cosmos and Azure AI Search.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing updates")
    args = parser.parse_args()

    load_dotenv()
    settings = get_settings()

    entity_repo = CosmosEntityRepository.from_settings(settings)
    ingestion_repo = CosmosIngestionRepository.from_settings(settings)
    suggestions_repo = CosmosSuggestionsRepository.from_settings(settings)
    search_service = AzureAISearchService.from_settings(settings)

    await entity_repo.open()
    await ingestion_repo.open()
    await suggestions_repo.open()

    try:
        entity_index = await build_entity_lookup_index(entity_repo)
        ingestion_changes = await collect_ingestion_identity_changes(
            ingestion_repo=ingestion_repo,
            entity_repo=entity_repo,
            entity_index=entity_index,
        )
        suggestion_changes = await collect_suggestion_identity_changes(
            suggestions_repo=suggestions_repo,
            entity_repo=entity_repo,
            entity_index=entity_index,
        )
        ingestion_summary = await backfill_ingestion_identities(
            ingestion_repo=ingestion_repo,
            entity_repo=entity_repo,
            entity_index=entity_index,
            storage_container=settings.azure_storage_container or "",
            dry_run=args.dry_run,
        )
        suggestions_summary = await backfill_suggestion_identities(
            suggestions_repo=suggestions_repo,
            entity_repo=entity_repo,
            entity_index=entity_index,
            dry_run=args.dry_run,
        )
        search_summary = await backfill_search_identities(
            search_service=search_service,
            identity_changes=[*ingestion_changes, *suggestion_changes],
            dry_run=args.dry_run,
        )

        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "ingestion": ingestion_summary.__dict__,
                    "suggestions": suggestions_summary.__dict__,
                    "search": search_summary.__dict__,
                },
                indent=2,
            )
        )
    finally:
        await search_service.close()
        await suggestions_repo.close()
        await ingestion_repo.close()
        await entity_repo.close()


if __name__ == "__main__":
    asyncio.run(main())