"""Azure Text-to-Speech service using the Cognitive Services REST API.

Audio is synthesised on demand and cached in-memory keyed by the sha256
of the (voice, text) pair.  Entries expire after TTL seconds so the cache
doesn't grow unbounded across a long-running session.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from config import Settings

logger = logging.getLogger(__name__)

_SSML_TEMPLATE = (
    "<speak version='1.0' xml:lang='en-US'>"
    "<voice name='{voice}'>{text}</voice>"
    "</speak>"
)


def _escape_ssml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


class _TTSCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, tuple[bytes, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[bytes]:
        entry = self._store.get(key)
        if entry is None:
            return None
        data, ts = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return data

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = (data, time.monotonic())

    def evict_chat(self, chat_id: str) -> None:
        """Drop all entries tagged with *chat_id* (used on session expiry)."""
        keys = [k for k in self._store if k.startswith(f"chat:{chat_id}:")]
        for k in keys:
            del self._store[k]

    def _chat_key(self, chat_id: str, turn_index: int) -> str:
        return f"chat:{chat_id}:{turn_index}"

    def get_by_turn(self, chat_id: str, turn_index: int) -> Optional[bytes]:
        return self.get(self._chat_key(chat_id, turn_index))

    def put_by_turn(self, chat_id: str, turn_index: int, data: bytes) -> None:
        self.put(self._chat_key(chat_id, turn_index), data)


class AzureTTSService:
    """Thin async wrapper around the Azure Cognitive Services TTS REST endpoint."""

    def __init__(
        self,
        *,
        key: Optional[str],
        region: Optional[str],
        voice: str = "en-US-AvaMultilingualNeural",
        cache_ttl: int = 3600,
    ):
        self._key = key or ""
        self._region = region or ""
        self._voice = voice
        self._cache = _TTSCache(ttl_seconds=cache_ttl)
        self._enabled = bool(self._key.strip() and self._region.strip())

    @classmethod
    def from_settings(cls, settings: Settings) -> "AzureTTSService":
        return cls(
            key=getattr(settings, "azure_speech_key", None),
            region=getattr(settings, "azure_speech_region", None),
            voice=getattr(settings, "azure_speech_voice", "en-US-AvaMultilingualNeural"),
        )

    @property
    def enabled(self) -> bool:  # noqa: D102
        return self._enabled

    def _content_key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._voice}:{text}".encode()).hexdigest()[:20]
        return f"content:{digest}"

    async def synthesize_for_turn(
        self,
        *,
        text: str,
        chat_id: str,
        turn_index: int,
    ) -> Optional[bytes]:
        """Return MP3 bytes for *text*, using per-turn cache key."""
        cached = self._cache.get_by_turn(chat_id, turn_index)
        if cached is not None:
            return cached

        data = await self._synthesize(text)
        if data:
            self._cache.put_by_turn(chat_id, turn_index, data)
        return data

    async def _synthesize(self, text: str) -> Optional[bytes]:
        if not self._enabled:
            return None

        # Check content-level cache (same text reused across turns)
        ck = self._content_key(text)
        cached = self._cache.get(ck)
        if cached is not None:
            return cached

        ssml = _SSML_TEMPLATE.format(
            voice=self._voice,
            text=_escape_ssml(text[:3000]),
        )
        url = (
            f"https://{self._region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        }

        try:
            import httpx  # available via openai SDK dependency

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, content=ssml.encode(), headers=headers)
                resp.raise_for_status()
                data = resp.content
                self._cache.put(ck, data)
                return data
        except Exception as exc:
            logger.warning("TTS synthesis failed: %s", exc)
            return None

    def evict_chat(self, chat_id: str) -> None:
        self._cache.evict_chat(chat_id)
