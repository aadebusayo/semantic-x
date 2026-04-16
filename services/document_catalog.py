from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from models.infosearch_api import Citation, DocumentRef, IndexedDocument
from services.blob_storage import BlobStorageSource
from services.cosmos_repositories import CosmosEntityRepository, CosmosIngestionRepository


def document_name_from_id(document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    return document_id.rsplit("/", 1)[-1]


def strip_file_extension(value: Optional[str]) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    filename = text.rsplit("/", 1)[-1]
    if "." not in filename:
        return filename
    return filename.rsplit(".", 1)[0]


def build_entity_filename(entity: Dict[str, object]) -> Optional[str]:
    name = str(entity.get("Name") or "").strip()
    ext = str(entity.get("DocExtension") or "").strip()
    if not name:
        return None
    return f"{name}{ext}" if ext else name


def normalize_storage_display_name(raw_name: Optional[str]) -> Optional[str]:
    return document_name_from_id(raw_name)


def iter_entity_id_candidates(value: Optional[str]) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    direct = strip_file_extension(text)
    if direct:
        candidates.append(direct)

    normalized = text + ("=" * (-len(text) % 4))
    try:
        decoded = base64.b64decode(normalized, validate=False).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        decoded = ""

    decoded_candidate = strip_file_extension(decoded)
    if decoded_candidate and decoded_candidate not in candidates:
        candidates.append(decoded_candidate)

    return candidates


@dataclass(frozen=True)
class CatalogDocument:
    document_id: str
    document_name: str
    blob_name: Optional[str]
    status: Optional[str]
    updated_at: Optional[datetime]


class DocumentCatalogService:
    def __init__(
        self,
        *,
        ingestion_repo: CosmosIngestionRepository,
        entity_repo: CosmosEntityRepository,
        blob_source: BlobStorageSource | None,
    ):
        self._ingestion_repo = ingestion_repo
        self._entity_repo = entity_repo
        self._blob_source = blob_source
        self._documents_cache: Optional[List[IndexedDocument]] = None
        self._records_cache: Optional[List[CatalogDocument]] = None
        self._records_by_id: Dict[str, CatalogDocument] = {}
        self._records_by_blob: Dict[str, CatalogDocument] = {}
        self._records_by_name: Dict[str, CatalogDocument] = {}
        self._entity_cache: Dict[str, Optional[Dict[str, object]]] = {}
        self._entity_filename_cache: Dict[str, Optional[Dict[str, object]]] = {}

    def invalidate(self) -> None:
        self._documents_cache = None
        self._records_cache = None
        self._records_by_id.clear()
        self._records_by_blob.clear()
        self._records_by_name.clear()
        self._entity_cache.clear()
        self._entity_filename_cache.clear()

    async def _get_entity(self, canonical_id: Optional[str]) -> Optional[Dict[str, object]]:
        key = strip_file_extension(canonical_id)
        if not key:
            return None
        if key not in self._entity_cache:
            self._entity_cache[key] = await self._entity_repo.get_document(entity_id=key)
        return self._entity_cache[key]

    async def _get_entity_by_filename(self, filename: Optional[str]) -> Optional[Dict[str, object]]:
        key = (filename or "").strip()
        if not key:
            return None
        if key not in self._entity_filename_cache:
            self._entity_filename_cache[key] = await self._entity_repo.find_document_by_filename(filename=key)
        return self._entity_filename_cache[key]

    @staticmethod
    def _normalized_name_key(value: Optional[str]) -> Optional[str]:
        text = (value or "").strip()
        return text.casefold() if text else None

    @staticmethod
    def _filename_candidates(
        *,
        raw_document_id: Optional[str],
        raw_document_name: Optional[str],
        blob_name: Optional[str],
    ) -> List[str]:
        candidates: List[str] = []
        for candidate in (
            raw_document_name,
            normalize_storage_display_name(blob_name),
            normalize_storage_display_name(raw_document_id),
            document_name_from_id(blob_name),
            document_name_from_id(raw_document_id),
        ):
            text = (candidate or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        return candidates

    async def _build_record(
        self,
        *,
        raw_document_id: Optional[str],
        raw_document_name: Optional[str],
        blob_name: Optional[str],
        status: Optional[str],
        updated_at_raw: Optional[str],
    ) -> Optional[CatalogDocument]:
        metadata = await self._resolve_metadata_uncached(
            raw_document_id=raw_document_id,
            raw_document_name=raw_document_name,
            blob_name=blob_name,
        )
        if not metadata.id or not metadata.name:
            return None

        updated_at = datetime.fromisoformat(updated_at_raw) if updated_at_raw else None
        return CatalogDocument(
            document_id=metadata.id,
            document_name=metadata.name,
            blob_name=blob_name,
            status=status,
            updated_at=updated_at,
        )

    async def _load_records(self) -> List[CatalogDocument]:
        if self._records_cache is not None:
            return self._records_cache

        items = await self._ingestion_repo.list_documents()
        records: List[CatalogDocument] = []
        self._records_by_id.clear()
        self._records_by_blob.clear()
        self._records_by_name.clear()

        ordered_items = sorted(items, key=lambda value: value.get("updatedAt") or "", reverse=True)
        for item in ordered_items:
            if item.get("source") != "blob":
                continue

            blob_name = item.get("blobName") or item.get("blob_name")
            raw_document_id = item.get("documentId") or item.get("document_id")
            raw_document_name = item.get("documentName") or item.get("document_name")

            if not blob_name and raw_document_id:
                blob_name = raw_document_id
                raw_document_id = None

            record = await self._build_record(
                raw_document_id=raw_document_id,
                raw_document_name=raw_document_name,
                blob_name=blob_name,
                status=item.get("status"),
                updated_at_raw=item.get("updatedAt"),
            )
            if record is None:
                continue

            if record.document_id not in self._records_by_id:
                self._records_by_id[record.document_id] = record
                records.append(record)

            if record.blob_name and record.blob_name not in self._records_by_blob:
                self._records_by_blob[record.blob_name] = record

            name_key = self._normalized_name_key(record.document_name)
            if name_key and name_key not in self._records_by_name:
                self._records_by_name[name_key] = record

        self._records_cache = records
        return records

    async def _record_from_inputs(
        self,
        *,
        raw_document_id: Optional[str],
        raw_document_name: Optional[str],
        blob_name: Optional[str],
    ) -> Optional[CatalogDocument]:
        await self._load_records()

        if raw_document_id:
            by_id = self._records_by_id.get(raw_document_id)
            if by_id is not None:
                return by_id

            by_blob = self._records_by_blob.get(raw_document_id)
            if by_blob is not None:
                return by_blob

        if blob_name:
            by_blob = self._records_by_blob.get(blob_name)
            if by_blob is not None:
                return by_blob

        for filename_candidate in self._filename_candidates(
            raw_document_id=raw_document_id,
            raw_document_name=raw_document_name,
            blob_name=blob_name,
        ):
            name_key = self._normalized_name_key(filename_candidate)
            if not name_key:
                continue
            by_name = self._records_by_name.get(name_key)
            if by_name is not None:
                return by_name

        return None

    async def _resolve_metadata_uncached(
        self,
        *,
        raw_document_id: Optional[str],
        raw_document_name: Optional[str],
        blob_name: Optional[str] = None,
    ) -> DocumentRef:
        storage_name = normalize_storage_display_name(blob_name or raw_document_id or raw_document_name)
        entity = None

        entity_id_candidates: List[str] = []
        for candidate_source in (raw_document_id, blob_name):
            for candidate in iter_entity_id_candidates(candidate_source):
                if candidate not in entity_id_candidates:
                    entity_id_candidates.append(candidate)

        for entity_id_candidate in entity_id_candidates:
            entity = await self._get_entity(entity_id_candidate)
            if entity is not None:
                break

        if entity is None:
            for filename_candidate in self._filename_candidates(
                raw_document_id=raw_document_id,
                raw_document_name=raw_document_name,
                blob_name=blob_name,
            ):
                entity = await self._get_entity_by_filename(filename_candidate)
                if entity is not None:
                    break

        if entity is not None:
            resolved_name = build_entity_filename(entity) or raw_document_name or storage_name
            resolved_id = str(entity.get("id") or (entity_id_candidates[0] if entity_id_candidates else "") or "")
            return DocumentRef(id=resolved_id or None, name=resolved_name or None)

        fallback_id = (entity_id_candidates[0] if entity_id_candidates else None) or strip_file_extension(raw_document_id or blob_name or raw_document_name)
        fallback_name = raw_document_name or storage_name or raw_document_id or blob_name
        return DocumentRef(id=fallback_id or None, name=fallback_name or None)

    async def resolve_metadata(
        self,
        *,
        raw_document_id: Optional[str],
        raw_document_name: Optional[str] = None,
        blob_name: Optional[str] = None,
    ) -> DocumentRef:
        record = await self._record_from_inputs(
            raw_document_id=raw_document_id,
            raw_document_name=raw_document_name,
            blob_name=blob_name,
        )
        if record is not None:
            return DocumentRef(id=record.document_id, name=record.document_name)

        return await self._resolve_metadata_uncached(
            raw_document_id=raw_document_id,
            raw_document_name=raw_document_name,
            blob_name=blob_name,
        )

    async def list_documents(self, *, limit: int) -> List[IndexedDocument]:
        if self._documents_cache is None:
            documents: List[IndexedDocument] = []
            for record in await self._load_records():
                if record.status != "processed":
                    continue
                if self._blob_source is not None and record.blob_name:
                    try:
                        if not await self._blob_source.exists(blob_name=record.blob_name):
                            continue
                    except Exception:
                        pass
                documents.append(
                    IndexedDocument(
                        id=record.document_id,
                        name=record.document_name,
                        status=record.status,
                        updated_at=record.updated_at,
                    )
                )
            self._documents_cache = documents

        documents = self._documents_cache
        return documents[:limit]

    async def resolve_blob_name(self, document: DocumentRef) -> Optional[str]:
        record = await self._record_from_inputs(
            raw_document_id=document.id,
            raw_document_name=document.name,
            blob_name=None,
        )
        return None if record is None else record.blob_name

    async def normalize_document_ref(self, document: DocumentRef) -> DocumentRef:
        return await self.resolve_metadata(
            raw_document_id=document.id,
            raw_document_name=document.name,
        )

    async def normalize_citations(self, *, citations: List[Citation]) -> List[Citation]:
        normalized: List[Citation] = []
        for citation in citations:
            metadata = await self.resolve_metadata(
                raw_document_id=citation.document_id,
                raw_document_name=citation.document_name,
            )
            normalized.append(
                citation.model_copy(
                    update={
                        "document_name": metadata.name,
                        "document_id": metadata.id,
                    }
                )
            )
        return normalized