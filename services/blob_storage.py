from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

from azure.storage.blob.aio import BlobServiceClient


@dataclass(frozen=True)
class BlobInfo:
    name: str
    etag: str
    last_modified_iso: str
    size: int


class BlobStorageSource:
    def __init__(self, *, connection_string: str, container: str, prefix: Optional[str] = None):
        self._connection_string = connection_string
        self._container_name = container
        self._prefix = prefix

        self._svc: Optional[BlobServiceClient] = None

    async def open(self) -> None:
        if self._svc is None:
            self._svc = BlobServiceClient.from_connection_string(self._connection_string)

    async def close(self) -> None:
        if self._svc is not None:
            await self._svc.close()
        self._svc = None

    def _container(self):
        assert self._svc is not None
        return self._svc.get_container_client(self._container_name)

    async def list_blobs(self, *, limit: Optional[int] = 500) -> AsyncIterator[BlobInfo]:
        container = self._container()
        count = 0
        async for blob in container.list_blobs(name_starts_with=self._prefix):
            # blob has properties: name, etag, last_modified, size
            etag = getattr(blob, "etag", None) or ""
            last_modified = getattr(blob, "last_modified", None)
            last_modified_iso = last_modified.isoformat() if last_modified else ""
            size = int(getattr(blob, "size", 0) or 0)
            yield BlobInfo(name=blob.name, etag=str(etag), last_modified_iso=last_modified_iso, size=size)
            count += 1
            if limit is not None and count >= limit:
                break

    async def download(self, *, blob_name: str) -> bytes:
        container = self._container()
        client = container.get_blob_client(blob_name)
        stream = await client.download_blob()
        return await stream.readall()

    async def exists(self, *, blob_name: str) -> bool:
        container = self._container()
        client = container.get_blob_client(blob_name)
        return bool(await client.exists())
