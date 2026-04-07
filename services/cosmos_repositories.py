from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from azure.cosmos.aio import CosmosClient

from config import Settings

logger = logging.getLogger(__name__)


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    trimmed = value.strip()
    return (not trimmed) or ("<" in trimmed and ">" in trimmed)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _InMemoryContainer:
    def __init__(self, pk_field: str):
        self._pk_field = pk_field
        self._items: Dict[str, Dict[str, Any]] = {}

    def all_items(self) -> List[Dict[str, Any]]:
        return list(self._items.values())

    async def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        self._items[item["id"]] = item
        return item

    async def read_item(self, item: str, partition_key: str) -> Dict[str, Any]:
        it = self._items.get(item)
        if not it or it.get(self._pk_field) != partition_key:
            raise KeyError("not found")
        return it

    async def delete_item(self, item: str, partition_key: str) -> None:
        it = self._items.get(item)
        if not it or it.get(self._pk_field) != partition_key:
            raise KeyError("not found")
        del self._items[item]

    async def query_items(
        self,
        query: str,
        parameters: List[Dict[str, Any]],
        enable_cross_partition_query: bool = False,
    ):
        # Minimal query support for our two query patterns.
        # 1) List chats by user
        # 2) List recent suggestions
        if "FROM c" not in query:
            return []

        param_map = {p["name"]: p["value"] for p in parameters}

        if "WHERE c.userId" in query:
            user_id = param_map.get("@userId")
            items = [v for v in self._items.values() if v.get("userId") == user_id]
            items.sort(key=lambda x: x.get("updatedAt", ""), reverse=True)
            return items

        if "ORDER BY c.createdAt DESC" in query:
            items = list(self._items.values())
            items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
            return items

        return []


class _CosmosBase:
    def __init__(self, *, endpoint: Optional[str], key: Optional[str], database: str, container: str):
        self._endpoint = endpoint
        self._key = key
        self._database_name = database
        self._container_name = container

        self._client: Optional[CosmosClient] = None
        self._container = None

    async def open(self) -> None:
        if _is_placeholder(self._endpoint) or _is_placeholder(self._key):
            return
        try:
            self._client = CosmosClient(self._endpoint, credential=self._key)
            db = self._client.get_database_client(self._database_name)
            self._container = db.get_container_client(self._container_name)
        except (ModuleNotFoundError, ImportError) as exc:
            logger.warning("Cosmos async transport dependency missing; falling back to in-memory mode: %s", exc)
            self._client = None
            self._container = None
        except Exception as exc:
            logger.warning("Cosmos initialization failed; falling back to in-memory mode: %s", exc)
            self._client = None
            self._container = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._container = None


class CosmosChatRepository(_CosmosBase):
    """Stores chats in Cosmos.

    Expected container partition key: /id
    Item shape:
      - id == chatId
      - userId
      - title
      - createdAt, updatedAt (ISO)
      - turns: [{role, content, createdAt}]
    """

    def __init__(self, *, endpoint: Optional[str], key: Optional[str], database: str, container: str, default_user_id: str, max_chat_messages: int):
        super().__init__(endpoint=endpoint, key=key, database=database, container=container)
        self._default_user_id = default_user_id
        self._max_chat_messages = max_chat_messages
        self._mem = _InMemoryContainer(pk_field="userId")

    @classmethod
    def from_settings(cls, settings: Settings) -> "CosmosChatRepository":
        return cls(
            endpoint=settings.cosmos_endpoint,
            key=settings.cosmos_key,
            database=settings.cosmos_database,
            container=settings.cosmos_chat_container,
            default_user_id=settings.default_user_id,
            max_chat_messages=settings.max_chat_messages,
        )

    def _user(self, user_id: Optional[str]) -> str:
        return user_id or self._default_user_id

    async def _upsert(self, item: Dict[str, Any]) -> None:
        if self._container is not None:
            await self._container.upsert_item(item)
        else:
            await self._mem.upsert_item(item)

    async def _read(self, chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        if self._container is not None:
            try:
                return await self._container.read_item(item=chat_id, partition_key=chat_id)
            except Exception:
                return None
        try:
            return await self._mem.read_item(item=chat_id, partition_key=user_id)
        except Exception:
            return None

    async def append_turns(self, *, chat_id: str, user_id: Optional[str], title: str, user_message: str, assistant_message: str) -> Dict[str, Any]:
        user = self._user(user_id)
        now = utcnow()

        existing = await self._read(chat_id, user)
        if existing is None:
            existing = {
                "id": chat_id,
                "chatId": chat_id,
                "userId": user,
                "title": title,
                "createdAt": now.isoformat(),
                "updatedAt": now.isoformat(),
                "turns": [],
            }

        existing["title"] = title or existing.get("title") or "New chat"
        existing["updatedAt"] = now.isoformat()

        turns = list(existing.get("turns") or [])
        turns.append({"role": "user", "content": user_message, "createdAt": now.isoformat()})
        turns.append({"role": "assistant", "content": assistant_message, "createdAt": now.isoformat()})

        # Trim history to keep items small.
        max_turns = self._max_chat_messages * 2
        if len(turns) > max_turns:
            turns = turns[-max_turns:]

        existing["turns"] = turns
        await self._upsert(existing)
        return existing

    async def list_chats(self, *, user_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        user = self._user(user_id)

        if self._container is None:
            items = await self._mem.query_items(
                "SELECT * FROM c WHERE c.userId = @userId ORDER BY c.updatedAt DESC",
                parameters=[{"name": "@userId", "value": user}],
            )
            return items[:limit]

        query = "SELECT c.id, c.chatId, c.title, c.updatedAt FROM c WHERE c.userId = @userId ORDER BY c.updatedAt DESC"
        items_iter = self._container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": user}],
        )
        out: List[Dict[str, Any]] = []
        async for it in items_iter:
            out.append(it)
            if len(out) >= limit:
                break
        return out

    async def get_chat(self, *, chat_id: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return await self._read(chat_id, self._user(user_id))

    async def rename_chat(
        self,
        *,
        chat_id: str,
        user_id: Optional[str],
        new_title: str,
    ) -> Optional[Dict[str, Any]]:
        """Update the title of an existing chat. Returns None if not found."""
        item = await self._read(chat_id, self._user(user_id))
        if item is None:
            return None
        item["title"] = new_title.strip()[:200]
        item["updatedAt"] = utcnow().isoformat()
        await self._upsert(item)
        return item

    async def save_reaction(
        self,
        *,
        chat_id: str,
        user_id: Optional[str],
        turn_index: int,
        reaction: str,
    ) -> Optional[Dict[str, Any]]:
        """Persist a thumbs-up/down reaction on a specific turn. Returns None if not found."""
        item = await self._read(chat_id, self._user(user_id))
        if item is None:
            return None
        turns = list(item.get("turns") or [])
        if turn_index < 0 or turn_index >= len(turns):
            return None
        turns[turn_index]["reaction"] = reaction
        item["turns"] = turns
        item["updatedAt"] = utcnow().isoformat()
        await self._upsert(item)
        return item


class CosmosSuggestionsRepository(_CosmosBase):
    """Stores per-document quick questions.

        Expected container partition key: /document_id
        Item shape:
        - id: stable unique key (documentKey)
            - document_id
            - documentKey
        - documentId (optional, canonical document id)
        - documentName (optional, display name)
                - title (optional, short display label)
        - blobName (optional, storage locator)
      - questions: [str]
      - createdAt (ISO)
    """

    def __init__(self, *, endpoint: Optional[str], key: Optional[str], database: str, container: str):
        super().__init__(endpoint=endpoint, key=key, database=database, container=container)
        self._mem = _InMemoryContainer(pk_field="document_id")

    @classmethod
    def from_settings(cls, settings: Settings) -> "CosmosSuggestionsRepository":
        return cls(
            endpoint=settings.cosmos_endpoint,
            key=settings.cosmos_key,
            database=settings.cosmos_database,
            container=settings.cosmos_suggestions_container,
        )

    @staticmethod
    def _doc_key(document_id: Optional[str], document_name: Optional[str]) -> str:
        if document_id:
            return f"id:{document_id}"
        if document_name:
            return f"name:{document_name}"
        return "unknown"

    async def upsert_questions(
        self,
        *,
        document_id: Optional[str],
        document_name: Optional[str],
        title: Optional[str],
        questions: List[str],
        blob_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = self._doc_key(document_id, document_name)
        now = utcnow()
        item = {
            "id": key,
            "document_id": key,
            "documentKey": key,
            "documentId": document_id,
            "documentName": document_name,
            "title": title,
            "blobName": blob_name,
            "questions": questions,
            "createdAt": now.isoformat(),
        }

        if self._container is not None:
            await self._container.upsert_item(item)
        else:
            await self._mem.upsert_item(item)

        return item

    async def list_recent(self, *, limit: int) -> List[Dict[str, Any]]:
        if self._container is None:
            items = await self._mem.query_items(
                "SELECT * FROM c ORDER BY c.createdAt DESC",
                parameters=[],
            )
            return items[:limit]

        query = (
            "SELECT TOP @limit c.documentId, c.documentName, c.title, c.blobName, c.questions, c.createdAt "
            "FROM c ORDER BY c.createdAt DESC"
        )
        items_iter = self._container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
        )
        out: List[Dict[str, Any]] = []
        async for it in items_iter:
            out.append(it)
        return out

    async def delete_questions(self, *, document_id: Optional[str], document_name: Optional[str] = None) -> None:
        key = self._doc_key(document_id, document_name)
        if self._container is not None:
            try:
                await self._container.delete_item(item=key, partition_key=key)
            except Exception:
                return
            return
        try:
            await self._mem.delete_item(item=key, partition_key=key)
        except Exception:
            return


class CosmosEntityRepository(_CosmosBase):
    _fallback_file_object_type = 1

    def __init__(self, *, endpoint: Optional[str], key: Optional[str], database: str, container: str, file_object_type: int):
        super().__init__(endpoint=endpoint, key=key, database=database, container=container)
        self._file_object_type = file_object_type
        self._mem = _InMemoryContainer(pk_field="id")

    @classmethod
    def from_settings(cls, settings: Settings) -> "CosmosEntityRepository":
        return cls(
            endpoint=settings.cosmos_endpoint,
            key=settings.cosmos_key,
            database=settings.cosmos_database,
            container=settings.cosmos_entity_container,
            file_object_type=settings.entity_file_object_type,
        )

    @staticmethod
    def _filename_parts(filename: str) -> tuple[str, str]:
        value = filename.strip()
        if not value:
            return "", ""
        if "." not in value:
            return value, ""
        name, ext = value.rsplit(".", 1)
        return name, f".{ext}"

    async def _query_document_by_id(self, *, entity_id: str, object_type: int) -> Optional[Dict[str, Any]]:
        if self._container is None:
            try:
                item = await self._mem.read_item(item=entity_id, partition_key=entity_id)
            except Exception:
                return None
            if item.get("ObjectType") != object_type:
                return None
            return item

        query = (
            "SELECT TOP 1 c.id, c.Name, c.DocExtension, c.BasePathId, c.ObjectType, c.ContentType "
            "FROM c WHERE c.id = @id AND c.ObjectType = @objectType"
        )
        items_iter = self._container.query_items(
            query=query,
            parameters=[
                {"name": "@id", "value": entity_id},
                {"name": "@objectType", "value": object_type},
            ],
        )
        async for item in items_iter:
            return item
        return None

    async def get_document(self, *, entity_id: str) -> Optional[Dict[str, Any]]:
        if not entity_id:
            return None

        item = await self._query_document_by_id(entity_id=entity_id, object_type=self._file_object_type)
        if item is not None:
            return item

        if self._file_object_type != self._fallback_file_object_type:
            return await self._query_document_by_id(entity_id=entity_id, object_type=self._fallback_file_object_type)
        return None

    async def find_document_by_filename(self, *, filename: str) -> Optional[Dict[str, Any]]:
        name, extension = self._filename_parts(filename)
        if not name:
            return None

        candidates: List[Dict[str, Any]] = []
        object_types = [self._file_object_type]
        if self._file_object_type != self._fallback_file_object_type:
            object_types.append(self._fallback_file_object_type)

        if self._container is None:
            for item in self._mem.all_items():
                if item.get("ObjectType") not in object_types:
                    continue
                if str(item.get("Name") or "") != name:
                    continue
                item_ext = str(item.get("DocExtension") or "")
                if item_ext.lower() != extension.lower():
                    continue
                candidates.append(item)
                if len(candidates) >= 2:
                    break
        else:
            query = (
                "SELECT TOP 2 c.id, c.Name, c.DocExtension, c.BasePathId, c.ObjectType, c.ContentType "
                "FROM c WHERE c.Name = @name AND LOWER(c.DocExtension) = @extension AND ARRAY_CONTAINS(@objectTypes, c.ObjectType)"
            )
            items_iter = self._container.query_items(
                query=query,
                parameters=[
                    {"name": "@name", "value": name},
                    {"name": "@extension", "value": extension.lower()},
                    {"name": "@objectTypes", "value": object_types},
                ],
            )
            async for item in items_iter:
                candidates.append(item)
                if len(candidates) >= 2:
                    break

        if len(candidates) != 1:
            return None
        return candidates[0]


class CosmosIngestionRepository(_CosmosBase):
    """Tracks which blobs have been processed.

        Expected container partition key: /document_id
        Item shape:
        - id: stable hash of blob name
            - document_id: canonical document id
            - document_name: canonical display name
            - blobName: storage blob path
            - source: "blob"
      - container: storage container name
      - etag: last processed ETag
      - lastModified: ISO
      - status: processed|skipped|error
      - message: optional error
      - updatedAt: ISO
    """

    def __init__(self, *, endpoint: Optional[str], key: Optional[str], database: str, container: str):
        super().__init__(endpoint=endpoint, key=key, database=database, container=container)
        self._mem = _InMemoryContainer(pk_field="document_id")

    @staticmethod
    def _safe_id(blob_name: str) -> str:
        digest = hashlib.sha256(blob_name.encode("utf-8")).hexdigest()
        return f"blob_{digest}"

    @classmethod
    def from_settings(cls, settings: Settings) -> "CosmosIngestionRepository":
        return cls(
            endpoint=settings.cosmos_endpoint,
            key=settings.cosmos_key,
            database=settings.cosmos_database,
            container=settings.cosmos_ingestion_container,
        )

    async def get(self, *, blob_name: str) -> Optional[Dict[str, Any]]:
        if self._container is not None:
            query = "SELECT TOP 1 * FROM c WHERE c.blobName = @blobName OR (NOT IS_DEFINED(c.blobName) AND c.document_id = @blobName)"
            items_iter = self._container.query_items(
                query=query,
                parameters=[{"name": "@blobName", "value": blob_name}],
            )
            async for item in items_iter:
                return item
            return None

        for item in self._mem.all_items():
            if item.get("blobName") == blob_name:
                return item
            if "blobName" not in item and item.get("document_id") == blob_name:
                return item
        return None

    async def upsert_status(
        self,
        *,
        document_id: str,
        document_name: Optional[str],
        blob_name: str,
        storage_container: str,
        etag: str,
        last_modified_iso: str,
        status: str,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = utcnow().isoformat()
        item_id = self._safe_id(blob_name)
        existing = await self.get(blob_name=blob_name)
        if existing is not None and existing.get("document_id") != document_id:
            if self._container is not None:
                try:
                    await self._container.delete_item(item=existing["id"], partition_key=existing["document_id"])
                except Exception:
                    pass
            else:
                try:
                    await self._mem.delete_item(item=existing["id"], partition_key=existing["document_id"])
                except Exception:
                    pass

        item = {
            "id": item_id,
            "document_id": document_id,
            "document_name": document_name,
            "documentId": document_id,
            "documentName": document_name,
            "blobName": blob_name,
            "source": "blob",
            "container": storage_container,
            "etag": etag,
            "lastModified": last_modified_iso,
            "status": status,
            "message": message,
            "updatedAt": now,
        }

        if self._container is not None:
            await self._container.upsert_item(item)
        else:
            await self._mem.upsert_item(item)

        return item

    async def list_documents(self) -> List[Dict[str, Any]]:
        if self._container is None:
            return self._mem.all_items()

        query = (
            "SELECT c.document_id, c.document_name, c.documentId, c.documentName, c.blobName, "
            "c.source, c.status, c.updatedAt FROM c"
        )
        items_iter = self._container.query_items(query=query, parameters=[])
        out: List[Dict[str, Any]] = []
        async for it in items_iter:
            out.append(it)
        return out

    async def delete(self, *, blob_name: str) -> None:
        existing = await self.get(blob_name=blob_name)
        if existing is None:
            return

        if self._container is not None:
            try:
                await self._container.delete_item(item=existing["id"], partition_key=existing["document_id"])
            except Exception:
                return
            return

        try:
            await self._mem.delete_item(item=existing["id"], partition_key=existing["document_id"])
        except Exception:
            return
