from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from services.blob_storage import BlobInfo, BlobStorageSource
from services.cosmos_repositories import CosmosEntityRepository
from services.document_catalog import strip_file_extension


def _blob_filename(blob_name: str) -> str:
    return blob_name.rsplit("/", 1)[-1]


def _blob_extension(blob_name: str) -> str:
    filename = _blob_filename(blob_name)
    if "." not in filename:
        return ""
    return f".{filename.rsplit('.', 1)[1]}"


def _blob_entity_id(blob_name: str) -> str:
    return strip_file_extension(_blob_filename(blob_name)) or ""


def _entity_name(entity: Dict[str, Any]) -> str:
    return str(entity.get("Name") or "").strip()


def _entity_extension(entity: Dict[str, Any]) -> str:
    return str(entity.get("DocExtension") or "").strip()


def _blob_matches_file_entity(blob_name: str, entity: Dict[str, Any]) -> bool:
    entity_id = str(entity.get("id") or "").strip()
    if not entity_id or _blob_entity_id(blob_name) != entity_id:
        return False

    entity_ext = _entity_extension(entity).lower()
    blob_ext = _blob_extension(blob_name).lower()

    if entity_ext and blob_ext:
        return entity_ext == blob_ext
    if entity_ext and not blob_ext:
        return False
    return True


def build_reconciliation_plan(
    *,
    blob_names: Sequence[str],
    file_entities: Sequence[Dict[str, Any]],
    all_entity_ids: Sequence[str],
    directory_like_blob_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    entity_ids = {str(entity_id).strip() for entity_id in all_entity_ids if str(entity_id).strip()}
    directory_like_blob_ids = {
        str(blob_name).strip() for blob_name in (directory_like_blob_names or []) if str(blob_name).strip()
    }
    file_entities_by_id = {
        str(entity.get("id") or "").strip(): entity
        for entity in file_entities
        if str(entity.get("id") or "").strip()
    }

    delete_blobs: List[Dict[str, Any]] = []
    delete_entities: List[Dict[str, Any]] = []
    kept_blob_count = 0
    kept_entity_count = 0
    delete_directory_like_blob_count = 0

    for blob_name in blob_names:
        blob_id = _blob_entity_id(blob_name)
        blob_ext = _blob_extension(blob_name)
        is_directory_like = blob_name in directory_like_blob_ids
        if not blob_id:
            delete_blobs.append({"blobName": blob_name, "reason": "missing_blob_id"})
            continue

        file_entity = file_entities_by_id.get(blob_id)
        if file_entity is not None:
            if not _entity_name(file_entity):
                delete_blobs.append({"blobName": blob_name, "entityId": blob_id, "reason": "entity_missing_name"})
                continue
            if not _blob_matches_file_entity(blob_name, file_entity):
                delete_blobs.append(
                    {
                        "blobName": blob_name,
                        "entityId": blob_id,
                        "reason": "entity_extension_mismatch",
                        "entityExtension": _entity_extension(file_entity),
                        "blobExtension": blob_ext,
                    }
                )
                continue

            kept_blob_count += 1
            continue

        if not blob_ext and blob_id in entity_ids:
            kept_blob_count += 1
            continue

        if is_directory_like:
            delete_blobs.append(
                {
                    "blobName": blob_name,
                    "entityId": blob_id,
                    "reason": "directory_like_without_entity",
                    "isDirectory": True,
                }
            )
            delete_directory_like_blob_count += 1
            continue

        delete_blobs.append({"blobName": blob_name, "entityId": blob_id, "reason": "no_matching_entity"})

    for entity in file_entities:
        entity_id = str(entity.get("id") or "").strip()
        if not entity_id:
            continue
        entity_name = _entity_name(entity)
        matching_blob_exists = any(_blob_matches_file_entity(blob_name, entity) for blob_name in blob_names)

        if not entity_name:
            delete_entities.append({"entityId": entity_id, "reason": "missing_name"})
            continue

        if not matching_blob_exists:
            delete_entities.append(
                {
                    "entityId": entity_id,
                    "name": entity_name,
                    "docExtension": _entity_extension(entity),
                    "basePathId": str(entity.get("BasePathId") or "").strip() or None,
                    "reason": "no_matching_blob",
                }
            )
            continue

        kept_entity_count += 1

    return {
        "summary": {
            "blobCount": len(blob_names),
            "fileEntityCount": len(file_entities_by_id),
            "deleteBlobCount": len(delete_blobs),
            "deleteDirectoryLikeBlobCount": delete_directory_like_blob_count,
            "deleteEntityCount": len(delete_entities),
            "skippedDirectoryLikeBlobCount": 0,
            "keptBlobCount": kept_blob_count,
            "keptEntityCount": kept_entity_count,
        },
        "deleteBlobs": delete_blobs,
        "deleteEntities": delete_entities,
        "skippedDirectoryLikeBlobs": [],
    }


async def _load_blob_infos(blob_source: BlobStorageSource) -> List[BlobInfo]:
    blobs: List[BlobInfo] = []
    async for blob in blob_source.list_blobs(limit=None):
        blobs.append(blob)
    return blobs


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile Blob storage against Entity metadata and delete source-level orphans."
    )
    parser.add_argument("--apply", action="store_true", help="Apply deletions instead of only reporting them")
    parser.add_argument(
        "--report",
        default="reports/blob_entity_source_reconciliation.json",
        help="Path to write the reconciliation report JSON",
    )
    args = parser.parse_args()

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
    entity_repo = CosmosEntityRepository.from_settings(settings)

    await blob_source.open()
    await entity_repo.open()

    try:
        blob_infos = await _load_blob_infos(blob_source)
        file_entities = await entity_repo.list_file_entities()
        entity_ids = await entity_repo.list_entity_ids()

        plan = build_reconciliation_plan(
            blob_names=[blob.name for blob in blob_infos],
            file_entities=file_entities,
            all_entity_ids=entity_ids,
            directory_like_blob_names=[blob.name for blob in blob_infos if blob.is_directory],
        )

        deleted_blobs = 0
        deleted_entities = 0
        if args.apply:
            for item in plan["deleteBlobs"]:
                try:
                    await blob_source.delete_path(
                        path_name=item["blobName"],
                        is_directory=bool(item.get("isDirectory")),
                    )
                    deleted_blobs += 1
                except Exception as exc:
                    item["deleteError"] = str(exc)

            for item in plan["deleteEntities"]:
                if await entity_repo.delete_entity(
                    entity_id=item["entityId"],
                    partition_key=item.get("basePathId") or item["entityId"],
                ):
                    deleted_entities += 1
                else:
                    item["deleteError"] = "delete_failed"

        plan["summary"]["apply"] = args.apply
        plan["summary"]["deletedBlobCount"] = deleted_blobs
        plan["summary"]["deletedEntityCount"] = deleted_entities

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan["summary"], indent=2))
        print(f"Report written to {report_path}")
    finally:
        await entity_repo.close()
        await blob_source.close()


if __name__ == "__main__":
    asyncio.run(main())