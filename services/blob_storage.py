from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional

from azure.storage.blob.aio import BlobServiceClient
from azure.storage.filedatalake.aio import DataLakeServiceClient


@dataclass(frozen=True)
class BlobInfo:
    name: str
    etag: str
    last_modified_iso: str
    size: int
    is_directory: bool = False


def _looks_like_directory(name: str, *, size: int, metadata: Optional[dict] = None) -> bool:
    metadata = metadata or {}
    if str(metadata.get("hdi_isfolder") or "").lower() in {"true", "1"}:
        return True
    if name.endswith("/"):
        return True

    filename = name.rsplit("/", 1)[-1]
    return bool(filename) and "." not in filename and size == 0


class BlobStorageSource:
    def __init__(self, *, connection_string: str, container: str, prefix: Optional[str] = None):
        self._connection_string = connection_string
        self._container_name = container
        self._prefix = prefix

        self._svc: Optional[BlobServiceClient] = None
        self._datalake_svc: Optional[DataLakeServiceClient] = None

    async def open(self) -> None:
        if self._svc is None:
            self._svc = BlobServiceClient.from_connection_string(self._connection_string)
        if self._datalake_svc is None:
            self._datalake_svc = DataLakeServiceClient.from_connection_string(self._connection_string)

    async def close(self) -> None:
        if self._svc is not None:
            await self._svc.close()
        if self._datalake_svc is not None:
            await self._datalake_svc.close()
        self._svc = None
        self._datalake_svc = None

    def _container(self):
        assert self._svc is not None
        return self._svc.get_container_client(self._container_name)

    def _file_system(self):
        assert self._datalake_svc is not None
        return self._datalake_svc.get_file_system_client(self._container_name)

    async def list_blobs(self, *, limit: Optional[int] = 500) -> AsyncIterator[BlobInfo]:
        container = self._container()
        count = 0
        async for blob in container.list_blobs(name_starts_with=self._prefix):
            # blob has properties: name, etag, last_modified, size
            etag = getattr(blob, "etag", None) or ""
            last_modified = getattr(blob, "last_modified", None)
            last_modified_iso = last_modified.isoformat() if last_modified else ""
            size = int(getattr(blob, "size", 0) or 0)
            metadata = getattr(blob, "metadata", None) or {}
            yield BlobInfo(
                name=blob.name,
                etag=str(etag),
                last_modified_iso=last_modified_iso,
                size=size,
                is_directory=_looks_like_directory(blob.name, size=size, metadata=metadata),
            )
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

    async def delete(self, *, blob_name: str) -> None:
        container = self._container()
        client = container.get_blob_client(blob_name)
        await client.delete_blob(delete_snapshots="include")

    async def delete_directory(self, *, directory_name: str) -> None:
        file_system = self._file_system()
        client = file_system.get_directory_client(directory_name)
        await client.delete_directory()

    async def delete_path(self, *, path_name: str, is_directory: bool = False) -> None:
        if is_directory:
            await self.delete_directory(directory_name=path_name)
            return
        await self.delete(blob_name=path_name)
