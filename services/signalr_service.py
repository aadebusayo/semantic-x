from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from config import Settings

logger = logging.getLogger(__name__)


class SignalRService:
    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: Optional[str],
        access_token: Optional[str],
        hub: str,
        target: str,
    ):
        self._enabled = enabled and bool(endpoint)
        self._endpoint = (endpoint or "").rstrip("/")
        self._access_token = access_token
        self._hub = hub
        self._target = target
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "SignalRService":
        return cls(
            enabled=settings.signalr_enabled,
            endpoint=settings.signalr_endpoint,
            access_token=settings.signalr_access_token,
            hub=settings.signalr_hub,
            target=settings.signalr_target,
        )

    async def open(self) -> None:
        if self._enabled and self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def publish(self, *, user_id: Optional[str], event: str, payload: Dict[str, Any]) -> None:
        if not self._enabled or self._client is None:
            return

        url = f"{self._endpoint}/api/v1/hubs/{self._hub}"
        body: Dict[str, Any] = {
            "target": self._target,
            "arguments": [
                {
                    "event": event,
                    "userId": user_id,
                    **payload,
                }
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            response = await self._client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                logger.warning("SignalR publish failed status=%s body=%s", response.status_code, response.text)
        except Exception as exc:
            logger.warning("SignalR publish error: %s", exc)
